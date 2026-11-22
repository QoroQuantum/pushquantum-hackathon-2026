"""Slot formulation: a constraint-free, log-encoded single-period portfolio model for PCE.

A period's portfolio is K <= B "slots"; slot k names one unit: (asset, long/short) or nothing,
with b = 1 + ceil(log2(n + 1)) bits (sign bit + asset-rank bits).  Then
    L_i = #long slots on asset i,  S_i = #short slots on asset i,   sum (L + S) <= K <= B
so the budget holds for EVERY bitstring (and every fractional relaxation) - no slack, no penalty.
The referee's cost on (L, S) is exactly pairwise between slots:
    E = const + sum_k E1[c_k] + sum_{k<l} E2[c_k, c_l]
    E1[c] = lin.e_c + absw.|e_c| + negw.[short] + e_c' R e_c,   E2[c, c'] = 2 e_c' R e_c'
(e_c = +-1 on its asset).  Decoding to the net position v = L - S (clipped to +-ub) is never worse
(dominance), and is what we score with the referee.

Correlation-aware code ("encoding in correlations"): the asset-rank bits are a Gray code over assets
sorted along the leading eigenvector of the correlation matrix, so a single bit flip moves a unit to a
neighbouring, similar asset; the sign bit flips long <-> short.  Rank codes >= n mean "no unit".

PCE relaxation: with bit marginals x_kj = (1 + tanh(alpha z_kj)) / 2, the multilinear extension of the
slot HUBO is the mean-field energy  const + sum_k p_k.E1 + sum_{k<l} p_k' E2 p_l,  p_k = product
distribution over slot k's 2^b codes.  SlotObjective evaluates it (and its gradient) in closed form,
O(K^2 4^b) instead of expanding the degree-2b HUBO.  Optional pairwise penalties: capital bounds
(unbalanced, slack-free) and a duplicate-option discount/penalty (ub = 3 per asset-direction).
"""
from __future__ import annotations

import math

import numpy as np

from divi.qprog.algorithms._pce import ExpectationObjective, _soft_relaxation

import formulations as fm


def gray(r: int) -> int:
    return r ^ (r >> 1)


def asset_order(P: fm.PeriodProblem, how: str = "spectral", rng=None) -> np.ndarray:
    n = P.n
    if how == "index":
        return np.arange(n)
    if how == "random":
        return (rng or np.random.default_rng(0)).permutation(n)
    R = P.R.astype(float)
    d = np.sqrt(np.maximum(np.diag(R), 1e-12))
    Cr = R / np.outer(d, d)
    w, U = np.linalg.eigh(Cr)
    return np.argsort(U[:, -1])  # leading factor loading: neighbours in rank = similar exposure


class SlotModel:
    def __init__(self, P: fm.PeriodProblem, n_slots: int | None = None, order: str = "spectral",
                 capital: str = "auto", cap_weight: float = 1.0, dup_weight: float = 0.0, rng=None,
                 base: np.ndarray | None = None):
        """base: fixed net portfolio v0 (units that stay); the K slots add units on top of it
        (exact referee cost on L = L0 + L_slots, S = S0 + S_slots)."""
        self.P = P
        n = P.n
        self.base = np.zeros(n, dtype=np.int64) if base is None else np.asarray(base, dtype=np.int64)
        used0 = int(np.abs(self.base).sum())
        self.K = n_slots or (P.budget - used0)
        assert self.K + used0 <= P.budget
        self.rb = max(1, math.ceil(math.log2(n + 1)))  # rank bits (one rank pattern at least is "none")
        self.b = 1 + self.rb
        self.ncode = 2 ** self.b
        self.n_vars = self.K * self.b
        self.order = asset_order(P, order, rng)
        # code -> (asset, sign); bit 0 of a slot (MSB) = sign, bits 1.. = Gray(rank)
        asset = -np.ones(self.ncode, dtype=np.int64)
        sign = np.zeros(self.ncode, dtype=np.int64)
        inv_gray = {gray(r): r for r in range(2 ** self.rb)}
        for c in range(self.ncode):
            s = (c >> self.rb) & 1
            r = inv_gray[c & (2 ** self.rb - 1)]
            if r < n:
                asset[c] = self.order[r]
                sign[c] = -1 if s else 1
        self.asset, self.sign = asset, sign
        E = np.zeros((self.ncode, n))
        ok = asset >= 0
        E[np.flatnonzero(ok), asset[ok]] = sign[ok]
        self.Evec = E
        R = P.R.astype(float)
        self.E1 = (E @ P.lin + np.abs(E) @ P.absw + np.maximum(-E, 0) @ P.negw + np.einsum("ci,ij,cj->c", E, R, E)
                   + 2.0 * E @ (R @ self.base))
        self.E2 = 2.0 * E @ R @ E.T
        self.const = float(P.energy(self.base)[0])
        net0 = int(self.base.sum())
        # capital: lo <= net0 + sum s_k <= hi; binds only if K units can reach it
        self.cap_active = (net0 + self.K > P.net_hi) or (net0 - self.K < P.net_lo)
        if capital == "auto":
            capital = "unbalanced" if self.cap_active else "none"
        self.capital = capital
        if capital == "unbalanced":
            # -a h + b h^2 on h = hi - net >= 0 and on h' = net - lo >= 0, in units of a typical unit gain
            W = cap_weight * float(np.abs(self.E1[ok]).max())
            s = self.sign.astype(float)
            for h_lin, h0 in ((-s, P.net_hi - net0), (s, net0 - P.net_lo)):  # h = h0 + sum_k h_lin[c_k]
                width = max(P.net_hi - P.net_lo, 1)
                a, bq = W, W / width
                # -a h + bq h^2 = -a h0 - a sum h_k + bq (h0^2 + 2 h0 sum h_k + sum h_k^2 + 2 sum_{k<l} h_k h_l)
                self.const += -a * h0 + bq * h0 * h0
                self.E1 += -a * h_lin + bq * (2 * h0 * h_lin + h_lin**2)
                self.E2 += 2 * bq * np.outer(h_lin, h_lin)
        if dup_weight:
            Dm = np.eye(self.ncode) * dup_weight * float(np.abs(self.E1[ok]).max())
            Dm[~ok, ~ok] = 0
            self.E2 += Dm
        self.bits = ((np.arange(self.ncode)[:, None] >> np.arange(self.b - 1, -1, -1)[None, :]) & 1).astype(float)

    # ── exact ────────────────────────────────────────────────────────────
    def codes(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).reshape(len(np.atleast_2d(X)), self.K, self.b)
        w = 2 ** np.arange(self.b - 1, -1, -1)
        return (X * w).sum(-1).astype(np.int64)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """bits (S, K*b) -> net positions v (S, n), clipped to +-ub."""
        C = self.codes(X)
        v = np.repeat(self.base[None], len(C), 0).astype(np.int64)
        for k in range(self.K):
            a, s = self.asset[C[:, k]], self.sign[C[:, k]]
            ok = a >= 0
            np.add.at(v, (np.flatnonzero(ok), a[ok]), s[ok])
        return np.clip(v, -self.P.ub, self.P.ub)

    def slot_energy(self, X: np.ndarray) -> np.ndarray:
        C = self.codes(X)
        e = self.const + self.E1[C].sum(1)
        for k in range(self.K):
            for l in range(k + 1, self.K):
                e += self.E2[C[:, k], C[:, l]]
        return e

    def encode(self, v: np.ndarray) -> np.ndarray:
        """net v -> slot bits (units assigned to slots in asset order; remaining slots 'none')."""
        inv = {}
        for c in range(self.ncode):
            if self.asset[c] >= 0:
                inv.setdefault((int(self.asset[c]), int(self.sign[c])), c)
        none = int(np.flatnonzero(self.asset < 0)[0])
        cs = []
        for i in np.flatnonzero(v):
            cs += [inv[(int(i), int(np.sign(v[i])))]] * int(abs(v[i]))
        assert len(cs) <= self.K
        cs += [none] * (self.K - len(cs))
        return self.bits[cs].reshape(-1).astype(np.uint8)

    # ── mean-field relaxation (= multilinear extension of the slot HUBO) ──
    def relaxed(self, x: np.ndarray):
        """x (K*b,) bit marginals -> (energy, dE/dx)."""
        x = np.clip(x.reshape(self.K, self.b), 0.0, 1.0)
        B = self.bits  # (ncode, b)
        F = B[None] * x[:, None, :] + (1 - B[None]) * (1 - x[:, None, :])  # (K, ncode, b) factor per bit
        p = F.prod(-1)  # (K, ncode)
        Q = p @ self.E2  # (K, ncode)   sum_c' p_k(c') E2[c', c]
        tot = Q.sum(0)
        e = self.const + (p * self.E1).sum() + 0.5 * ((p * tot).sum() - (p * Q).sum())
        G = self.E1[None] + tot[None] - Q  # dE/dp_k(c)
        # leave-one-out products: dp_k(c)/dx_kj = (2 B_cj - 1) prod_{j' != j} F
        loo = np.empty_like(F)
        for j in range(self.b):
            loo[:, :, j] = np.prod(np.delete(F, j, axis=2), axis=2)
        dx = np.einsum("kc,kcj->kj", G, loo * (2 * B[None] - 1))
        return float(e), dx.reshape(-1)


class SlotObjective(ExpectationObjective):
    """PCE loss = slot mean-field energy at x = (1 + tanh(alpha z)) / 2 (normalised by `scale`)."""

    def __init__(self, model: SlotModel, alpha: float = 2.0, scale: float | None = None):
        self.model, self.alpha = model, alpha
        if scale is None:  # energy spread of random bitstrings -> O(1) loss variation, any λ
            X = np.random.default_rng(0).integers(0, 2, size=(2000, model.n_vars))
            scale = float(np.std(model.slot_energy(X))) or 1.0
        self.scale = scale

    def score(self, z, problem) -> float:
        x, _ = _soft_relaxation(self.alpha * np.asarray(z))
        return self.model.relaxed(x)[0] / self.scale

    def gradient(self, z, problem):
        x, decay = _soft_relaxation(self.alpha * np.asarray(z))
        _, dx = self.model.relaxed(x)
        sech2 = 4.0 * decay / (1.0 + decay) ** 2
        return dx * 0.5 * self.alpha * sech2 / self.scale


def placeholder_problem(n_vars: int):
    from divi.qprog.problems import BinaryOptimizationProblem
    return BinaryOptimizationProblem({(i,): 1e-9 * (i + 1) for i in range(n_vars)})


def check(P: fm.PeriodProblem, n_samples: int = 2000, seed: int = 0, **kw) -> str:
    """slot energy == referee (L,S) cost; relaxed == exact at vertices; gradient == finite differences."""
    rng = np.random.default_rng(seed)
    M = SlotModel(P, **kw)
    X = rng.integers(0, 2, size=(n_samples, M.n_vars))
    C = M.codes(X)
    Lm = np.zeros((n_samples, P.n), dtype=np.int64)
    Sm = np.zeros_like(Lm)
    for k in range(M.K):
        a, s = M.asset[C[:, k]], M.sign[C[:, k]]
        for r in np.flatnonzero(a >= 0):
            (Lm if s[r] > 0 else Sm)[r, a[r]] += 1
    V = Lm - Sm
    ref = (np.einsum("ki,ij,kj->k", V, P.R, V) + V @ P.lin + (Lm + Sm) @ P.absw + Sm @ P.negw + P.const)
    pen_free = M.capital == "none"
    e = M.slot_energy(X)
    mism = int((np.abs(e - ref) > 1e-6).sum()) if pen_free else -1
    vert = max(abs(M.relaxed(X[i].astype(float))[0] - e[i]) for i in range(20))
    x = rng.uniform(0.1, 0.9, M.n_vars)
    _, g = M.relaxed(x)
    h = 1e-6
    fd = np.array([(M.relaxed(x + h * np.eye(M.n_vars)[j])[0] - M.relaxed(x - h * np.eye(M.n_vars)[j])[0]) / (2 * h)
                   for j in range(M.n_vars)])
    dec_better = int((P.energy(M.decode(X)) > e + 1e-6).sum()) if pen_free else -1
    return (f"slots K={M.K} b={M.b} vars={M.n_vars} capital={M.capital}: energy mismatches {mism}/{n_samples}, "
            f"vertex |relaxed-exact| {vert:.2e}, grad rel err {np.abs(fd - g).max() / (np.abs(g).max() + 1e-12):.1e}, "
            f"net-decode worse than slot energy: {dec_better} (expect 0 unless ub clipping)")


if __name__ == "__main__":
    for inst, lam in (("po_a010_t10_orig", "1e-4"), ("po_a050_t10_s00", "1e-2")):
        m = fm.pc.make_model(inst, lam)
        P = fm.period_problems(m)[1]
        print(inst, check(P, n_samples=500))
