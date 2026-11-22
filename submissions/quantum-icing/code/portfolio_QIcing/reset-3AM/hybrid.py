"""Q-accelerated LNS: a classical large-neighbourhood search that calls a QUBO sub-solver (PCE) as an accelerator.

The classical loop owns the problem; the sub-solver only ever sees a small, *exact*, *constraint-free* QUBO:

    move  = a period block [ta, tb) and m "move bits", each touching a distinct asset set:
              swap  i -> j : one unit leaves held asset i, one unit (same sign) enters asset j   (budget/net neutral)
              drop  i      : one unit leaves held asset i
              add   j, d   : one unit enters asset j with sign d          (only while there is free budget)
            the same delta is applied to every period of the block (multi-day hedges move together).
    QUBO  = E(s) - E(v) = h.s + sum_{k<l} J_kl s_k s_l, EXACT (risk is quadratic, every |.| / transaction term
            touches one bit only, because bits are asset-disjoint); every bitstring is feasible by construction
            (budget: #add bits <= free budget; net bounds: checked for the worst-case subset).

Single bits are the moves a local search already sees; the value of the sub-solver is in the joint subsets
(factor-neutral baskets: a lone swap costs, a coordinated set of swaps pays).  Sub-solvers, same contract:

    exact  enumeration (Gray code, m <= 24)           -- the ideal, for reference
    sa     simulated annealing on the QUBO           -- the classical accelerator ("tabu-search slot")
    tabu   1-flip tabu search on the QUBO
    pce    Divi PCE (MultiBasis-2, L-BFGS), harvested expectation roundings + final assignment
    rand   uniform bitstrings, same count            -- the null control

Every candidate is re-scored on the exact multi-period objective (= referee, asserted). No repair anywhere.
Optional classical polish between moves: exact asset-subset DP LNS (lns.lns) for --polish-sec seconds.

    python hybrid.py --instance po_a050_t10_orig --lam 1e-3 --solver pce --time-limit 1200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
PF = HERE.parent
for p in (str(PF), str(PF / "formulation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import lns as L  # noqa: E402
import pf_core as pc  # noqa: E402

LOGS = HERE / "logs"


# ── the move QUBO ─────────────────────────────────────────────────────────


def block_delta(fm: L.FastModel, v, ta, tb, D):
    """Exact objective change of applying each row of D (M, n) to every period in [ta, tb)."""
    D = np.atleast_2d(D).astype(np.int64)
    out = np.zeros(len(D), dtype=np.int64)
    for t in range(ta, tb):
        x = v[t]
        X1 = x[None, :] + D
        out += 2 * D @ (fm.R[t] @ x) + np.einsum("ki,ij,kj->k", D, fm.R[t], D) + D @ fm.lin[t]
        out += (np.abs(X1) - np.abs(x)) @ fm.absw[t] + (np.maximum(-X1, 0) - np.maximum(-x, 0)) @ fm.negw[t]
    if ta >= 1:  # entry boundary
        p, x = v[ta - 1], v[ta]
        out += (np.abs(x + D - p) - np.abs(x - p)) @ fm.tw[ta]
    if tb <= fm.T - 1:  # exit boundary
        q, x = v[tb], v[tb - 1]
        out += (np.abs(q - x - D) - np.abs(q - x)) @ fm.tw[tb]
    return out


class MoveQUBO:
    """m asset-disjoint move bits on a period block; E(s) = E0 + h.s + 1/2 s'Js exactly."""

    def __init__(self, fm, v, ta, tb, D, kinds, cons=()):
        self.fm, self.ta, self.tb = fm, ta, tb
        self.D, self.kinds = np.asarray(D, dtype=np.int64), kinds
        self.m = len(self.D)
        self.h_obj = block_delta(fm, v, ta, tb, self.D)
        Rsum = fm.R[ta:tb].sum(0)
        J = 2 * self.D @ Rsum @ self.D.T
        np.fill_diagonal(J, 0)
        ov = np.argwhere(np.triu((self.D != 0).astype(int) @ (self.D != 0).T.astype(int), 1))
        if len(ov):  # bits sharing an asset: exact pairwise term from the true 2-bit change (<= 2 bits/asset)
            both = block_delta(fm, v, ta, tb, self.D[ov[:, 0]] + self.D[ov[:, 1]])
            J[ov[:, 0], ov[:, 1]] = J[ov[:, 1], ov[:, 0]] = both - self.h_obj[ov[:, 0]] - self.h_obj[ov[:, 1]]
        self.J_obj = J.astype(np.int64)
        self.h, self.J = self.h_obj.copy(), self.J_obj.copy()
        self.cons = [(np.asarray(c, dtype=np.int64), int(b)) for c, b in cons]
        self.P = int(2 * (np.abs(self.h_obj).max() + np.abs(self.J_obj).sum(1).max()) + 1) if cons else 0
        for c, b in self.cons:  # exact penalty P (c.s - b)^2: zero iff the linear equality holds
            self.h += self.P * (c * c - 2 * b * c)
            Jp = 2 * self.P * np.outer(c, c)
            np.fill_diagonal(Jp, 0)
            self.J += Jp

    def delta(self, S):
        """Exact objective change (no penalty)."""
        S = np.atleast_2d(S).astype(np.int64)
        return S @ self.h_obj + np.einsum("ki,ij,kj->k", S, self.J_obj, S) // 2

    def ok(self, S):
        S = np.atleast_2d(S).astype(np.int64)
        good = np.ones(len(S), bool)
        for c, b in self.cons:
            good &= S @ c == b
        return good

    def energy(self, S):
        """Sub-solver energy (objective change + penalty, minus the constant P tau^2)."""
        S = np.atleast_2d(S).astype(np.int64)
        return S @ self.h + np.einsum("ki,ij,kj->k", S, self.J, S) // 2

    def upper(self):
        """Upper-triangular float QUBO matrix (x'Qx) for Divi."""
        Q = np.triu(self.J.astype(float), 1)
        Q[np.diag_indices(self.m)] = self.h
        return Q

    def apply(self, v, s):
        w = v.copy()
        w[self.ta:self.tb] += (np.asarray(s, dtype=np.int64) @ self.D)[None, :]
        return w


def build_free_move(fm, v, ta, tb, rng, m_max=20, p_cluster=0.5, spread=3, p_double=0.4):
    """Independent drop bits (held assets) and add bits (asset, sign), <= 2 same-direction bits per asset, with
    #adds - #drops = tau (<= free budget) and net change = nu as exact penalties. The sub-solver does the pairing."""
    x = v[ta:tb]
    n, ub = fm.n, fm.ub
    sg = np.sign(x)
    held = np.flatnonzero((sg == sg[0]).all(0) & (sg[0] != 0))
    if len(held) == 0:
        return None
    n_drop = int(min(len(held), rng.integers(2, m_max // 2 + 1)))
    if rng.random() < p_cluster:
        seed = int(rng.choice(held))
        drops = held[np.argsort(-np.abs(fm.corr[seed, held]) + 1e-9 * rng.random(len(held)))][:n_drop]
    else:
        drops = rng.choice(held, size=n_drop, replace=False)
    used = np.zeros(n, bool)
    used[drops] = True
    ok = {d: ((x == 0) | (np.sign(x) == d)).all(0) & (np.abs(x + d) <= ub).all(0) & ~used for d in (1, -1)}
    cand = [(j, d) for d in (1, -1) for j in np.flatnonzero(ok[d])]
    if not cand:
        return None
    Da = np.zeros((len(cand), n), np.int64)
    for r, (j, d) in enumerate(cand):
        Da[r, j] = d
    dl = block_delta(fm, v, ta, tb, Da)
    pool = np.argsort(dl)[: spread * (m_max - n_drop)]
    rows, kinds = [], []
    mag = np.abs(x).min(0)
    for i in drops:  # 2 drop bits when the asset holds >= 2 units in every period (no zero crossing)
        r = np.zeros(n, np.int64)
        r[i] = -sg[0, i]
        for _ in range(2 if mag[i] >= 2 and rng.random() < p_double else 1):
            rows.append(r), kinds.append(("drop", int(i)))
    room = ub - np.abs(x).max(0)
    for r in rng.permutation(pool):
        j, d = cand[r]
        if used[j] or len(rows) >= m_max:
            continue
        used[j] = True
        for _ in range(2 if room[j] >= 2 and len(rows) + 1 < m_max and rng.random() < p_double else 1):
            rows.append(Da[r]), kinds.append(("add", int(j), int(d)))
    g = int(fm.B - np.abs(x).sum(1).max())
    tau = g if rng.random() < 0.7 else max(g - 1, -n_drop)
    if len(rows) < 3:
        return None
    D = np.array(rows)
    card = np.array([1 if k[0] == "add" else -1 for k in kinds])
    net = x.sum(1)  # net change nu must keep every period in [netmin, C]; prefer nu = 0 (net is usually tight)
    lo, hi = int(fm.netmin - net.min()), int(fm.C - net.max())
    nvec = D.sum(1)
    lo, hi = max(lo, int(nvec.clip(max=0).sum())), min(hi, int(nvec.clip(0).sum()))
    if lo > hi:
        return None
    nu = 0 if lo <= 0 <= hi and rng.random() < 0.6 else int(rng.integers(lo, hi + 1))
    return MoveQUBO(fm, v, ta, tb, D, kinds, cons=[(card, tau), (nvec, nu)])


def build_period_move(fm, v, ta, tb, rng, m_max=None, **_):
    """Whole-block re-optimisation around the incumbent: EVERY asset gets <= 2 bits (delta in -2..2), so the
    sub-QUBO is the block's own portfolio problem centred on x (all-zero bitstring = incumbent).
        held |x| >= 2 : 2 drop bits;   held |x| = 1 : 1 drop + 1 grow bit (if room);   empty : 2 bits in the
        better single-unit direction.  Budget (#units = tau) and net change (= nu) are exact equality penalties."""
    x = v[ta:tb]
    n, ub = fm.n, fm.ub
    sg = np.sign(x)
    s0 = sg.max(0) + sg.min(0)  # +-1 if the asset is never signed the other way in the block, 0 if empty/mixed
    held = (sg == sg[0]).all(0) & (sg[0] != 0)
    empty = (x == 0).all(0)
    amin, amax = np.abs(x).min(0), np.abs(x).max(0)
    rows, kinds, card = [], [], []

    def bit(i, d, kind, c):
        r = np.zeros(n, np.int64)
        r[i] = d
        rows.append(r), kinds.append((kind, int(i), int(d))), card.append(c)

    E = np.eye(n, dtype=np.int64)
    dpos, dneg = block_delta(fm, v, ta, tb, E), block_delta(fm, v, ta, tb, -E)
    for i in range(n):
        if held[i]:
            d = int(sg[0, i])
            if amin[i] >= 2:
                bit(i, -d, "drop", -1), bit(i, -d, "drop", -1)
            else:
                bit(i, -d, "drop", -1)
                if amax[i] < ub:
                    bit(i, d, "grow", 1)
        elif empty[i]:
            d = 1 if dpos[i] <= dneg[i] else -1
            bit(i, d, "add", 1), bit(i, d, "add", 1)
        elif s0[i] != 0 and amax[i] + 1 <= ub:  # zero in some periods, signed s0 in others: grow only
            bit(i, int(s0[i]), "grow", 1)
    if len(rows) < 3:
        return None
    D, card = np.array(rows), np.array(card)
    tau = int(fm.B - np.abs(x).sum(1).max())
    tau = tau if rng.random() < 0.8 else tau - 1  # sometimes shed a unit (high risk aversion)
    net = x.sum(1)
    lo, hi = int(fm.netmin - net.min()), int(fm.C - net.max())
    nu = 0 if lo <= 0 <= hi and rng.random() < 0.6 else int(rng.integers(lo, hi + 1)) if lo <= hi else None
    if nu is None:
        return None
    return MoveQUBO(fm, v, ta, tb, D, kinds, cons=[(card, tau), (D.sum(1), nu)])


def build_move(fm, v, ta, tb, rng, m_max=20, p_cluster=0.5, top_targets=4, p_drop=0.15, max_add=4):
    """Randomised asset-disjoint move set for block [ta, tb). Returns MoveQUBO or None."""
    x = v[ta:tb]
    n, ub = fm.n, fm.ub
    sg = np.sign(x)
    src = np.flatnonzero((sg == sg[0]).all(0) & (sg[0] != 0))
    if len(src) == 0:
        return None
    if len(src) > m_max:
        if rng.random() < p_cluster:  # correlated cluster around a random held asset
            seed = int(rng.choice(src))
            order = src[np.argsort(-np.abs(fm.corr[seed, src]) + 1e-9 * rng.random(len(src)))]
            src = order[:m_max]
        else:
            src = rng.choice(src, size=m_max, replace=False)
    src = rng.permutation(src)
    # target eligibility per direction d: every period of the block is 0 or already signed d, and room to grow
    ok = {d: ((x == 0) | (np.sign(x) == d)).all(0) & (np.abs(x + d) <= ub).all(0) for d in (1, -1)}
    used = np.zeros(n, bool)
    used[src] = True
    rows, kinds = [], []
    for i in src:
        d = int(sg[0, i])
        if rng.random() < p_drop:
            r = np.zeros(n, np.int64)
            r[i] = -d
            rows.append(r), kinds.append(("drop", int(i)))
            continue
        tg = np.flatnonzero(ok[d] & ~used)
        if len(tg) == 0:
            continue
        D = np.zeros((len(tg), n), np.int64)
        D[:, i] = -d
        D[np.arange(len(tg)), tg] = d
        dl = block_delta(fm, v, ta, tb, D)
        pick = np.argsort(dl)[:top_targets]
        j = int(tg[rng.choice(pick)])
        used[j] = True
        rows.append(D[np.flatnonzero(tg == j)[0]]), kinds.append(("swap", int(i), j))
    free = int(min(fm.B - np.abs(x).sum(1).min(), max_add))
    if free > 0:  # add bits are safe while #adds <= free budget in every period
        cand = [(j, d) for d in (1, -1) for j in np.flatnonzero(ok[d] & ~used)]
        if cand:
            D = np.zeros((len(cand), n), np.int64)
            for r, (j, d) in enumerate(cand):
                D[r, j] = d
            dl = block_delta(fm, v, ta, tb, D)
            for r in np.argsort(dl)[: 3 * free]:
                j, d = cand[r]
                if used[j] or sum(k[0] == "add" for k in kinds) >= free:
                    continue
                used[j] = True
                rows.append(D[r]), kinds.append(("add", int(j), int(d)))
    if not rows:
        return None
    D = np.array(rows)
    net = x.sum(1)
    while len(D):  # worst-case net bounds over all subsets, every period
        hi, lo = net.max() + D.sum(1).clip(0).sum(), net.min() + D.sum(1).clip(max=0).sum()
        if hi <= fm.C and lo >= fm.netmin:
            break
        bad = np.flatnonzero(D.sum(1) > 0) if hi > fm.C else np.flatnonzero(D.sum(1) < 0)
        k = int(rng.choice(bad))
        D = np.delete(D, k, 0)
        kinds.pop(k)
    if len(D) < 2:
        return None
    return MoveQUBO(fm, v, ta, tb, D, kinds)


# ── sub-solvers (all return candidate bitstrings, (K, m) uint8) ──────────


@njit(cache=True)
def _exact(h, J, K):
    m = len(h)
    s = np.zeros(m, np.int8)
    f = h.copy()
    E = np.int64(0)
    code = np.int64(0)
    topE = np.full(K, np.int64(1) << 60)
    topC = np.zeros(K, np.int64)
    topE[0], topC[0] = 0, 0
    for step in range(1, np.int64(1) << m):
        k = 0
        while not (step >> k) & 1:
            k += 1
        if s[k] == 0:
            E += f[k]
            s[k] = 1
            for l in range(m):
                f[l] += J[l, k]
        else:
            s[k] = 0
            for l in range(m):
                f[l] -= J[l, k]
            E -= f[k]
        code ^= np.int64(1) << k
        if E < topE[K - 1]:
            p = K - 1
            while p > 0 and topE[p - 1] > E:
                topE[p], topC[p] = topE[p - 1], topC[p - 1]
                p -= 1
            topE[p], topC[p] = E, code
    return topE, topC


def solve_exact(Qm: MoveQUBO, K=8, **_):
    _, C = _exact(Qm.h, Qm.J, K)
    return ((C[:, None] >> np.arange(Qm.m)[None, :]) & 1).astype(np.uint8)


@njit(cache=True)
def _sa(h, J, n_reads, sweeps, T0, T1, seed):
    np.random.seed(seed)
    m = len(h)
    out = np.zeros((n_reads, m), np.uint8)
    for r in range(n_reads):
        s = (np.random.random(m) < 0.5).astype(np.int64)
        f = h.copy()  # flip-on gain for bit k (when s_k = 0)
        for k in range(m):
            for l in range(m):
                f[k] += J[k, l] * s[l]
        best = np.int64(0)
        for k in range(m):
            best += s[k] * h[k]
        E = best
        for a in range(m):
            for b in range(a + 1, m):
                E += J[a, b] * s[a] * s[b]
        best = E
        bs = s.copy()
        for w in range(sweeps):
            temp = T0 * (T1 / T0) ** (w / max(sweeps - 1, 1))
            for k in range(m):
                dE = f[k] if s[k] == 0 else -f[k]
                if dE <= 0 or np.random.random() < np.exp(-dE / temp):
                    sgn = 1 if s[k] == 0 else -1
                    s[k] += sgn
                    E += dE
                    for l in range(m):
                        f[l] += sgn * J[l, k]
                    if E < best:
                        best = E
                        bs[:] = s
        out[r] = bs
    return out


def solve_sa(Qm: MoveQUBO, n_reads=16, sweeps=200, seed=0, **_):
    nz = Qm.J[Qm.J != 0]
    scale = float(np.median(np.abs(Qm.h))) + (float(np.median(np.abs(nz))) if len(nz) else 0.0) + 1.0
    return _sa(Qm.h, Qm.J, n_reads, sweeps, 2.0 * scale, 0.02 * scale, seed)


@njit(cache=True)
def _tabu(h, J, restarts, iters, tenure, seed):
    np.random.seed(seed)
    m = len(h)
    out = np.zeros((restarts, m), np.uint8)
    for r in range(restarts):
        s = (np.random.random(m) < 0.5).astype(np.int64)
        f = h.copy()
        for k in range(m):
            for l in range(m):
                f[k] += J[k, l] * s[l]
        E = np.int64(0)
        for k in range(m):
            E += s[k] * h[k]
        for a in range(m):
            for b in range(a + 1, m):
                E += J[a, b] * s[a] * s[b]
        best, bs = E, s.copy()
        tabu = np.zeros(m, np.int64)
        for it in range(1, iters + 1):
            bk, bd = -1, np.int64(1) << 60
            for k in range(m):
                dE = f[k] if s[k] == 0 else -f[k]
                if (tabu[k] < it or E + dE < best) and dE < bd:
                    bk, bd = k, dE
            if bk < 0:
                continue
            sgn = 1 if s[bk] == 0 else -1
            s[bk] += sgn
            E += bd
            for l in range(m):
                f[l] += sgn * J[l, bk]
            tabu[bk] = it + tenure
            if E < best:
                best = E
                bs[:] = s
        out[r] = bs
    return out


def solve_tabu(Qm: MoveQUBO, restarts=8, iters=None, seed=0, **_):
    return _tabu(Qm.h, Qm.J, restarts, iters or 20 * Qm.m, max(2, Qm.m // 4), seed)


def solve_rand(Qm: MoveQUBO, count=64, seed=0, **_):
    return np.random.default_rng(seed).integers(0, 2, size=(count, Qm.m)).astype(np.uint8)


_PCE_CACHE = {}


def solve_pce(Qm: MoveQUBO, n_layers=4, iters=60, alpha=None, n_qubits=None, seed=0, **_):
    from divi.backends import MaestroSimulator
    from divi.qprog.algorithms import MultiBasisEncoding, SoftObjective
    from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem
    import pce_tools as pt

    Q = Qm.upper()
    Q /= np.abs(Q).max() or 1.0
    kw = dict(objective=SoftObjective(alpha=alpha)) if alpha else {}
    prog = pt.HarvestPCE(BinaryOptimizationProblem(Q), encoding=MultiBasisEncoding(order=2, n_qubits=n_qubits),
                         n_layers=n_layers, optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
                         max_iterations=iters, backend=MaestroSimulator(), seed=seed, **kw)
    prog.run()
    top = prog.get_top_solutions(n=1, sort_by="energy")
    X = [np.array([int(c) for c in top[0].bitstring], np.uint8)] if top else []
    if prog.harvest:
        X += list((np.array(prog.harvest) > 0).astype(np.uint8))
    _PCE_CACHE["qubits"], _PCE_CACHE["circuits"] = prog.n_qubits, prog.total_circuit_count
    return np.unique(np.array(X, np.uint8), axis=0)


BUILDERS = dict(period=build_period_move, free=build_free_move, pairs=build_move)
SOLVERS = dict(exact=solve_exact, sa=solve_sa, tabu=solve_tabu, rand=solve_rand, pce=solve_pce)


# ── outer loop ────────────────────────────────────────────────────────────


def load_start(model, instance, lam, start):
    T, n = model.n_periods, len(model.symbols)
    if start == "empty":
        return np.zeros((T, n), np.int64), "empty"
    if start != "best":
        return pc.positions_to_v(model, pc.qt.read_portfolio_solution(start)[1]), start
    best = None
    for f in glob.glob(str(PF / "solutions" / f"{instance}_b{model.budget:03d}_l{lam}" / "*.sol")):
        vv = pc.positions_to_v(model, pc.qt.read_portfolio_solution(f)[1])
        o = pc.score_v(model, vv)
        if o.feasible and (best is None or o.objective < best[0]):
            best = (o.objective, vv, f)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a050_t10_orig")
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--start", default="best", help="'best' held .sol, 'empty', or a .sol path")
    ap.add_argument("--solver", default="pce", choices=list(SOLVERS))
    ap.add_argument("--m-max", type=int, default=20, help="max move bits (QUBO size)")
    ap.add_argument("--max-block", type=int, default=3, help="max periods per block")
    ap.add_argument("--moves", default="period:1", help="move mix, e.g. period:0.6,free:0.3,pairs:0.1")
    ap.add_argument("--top", type=int, default=8, help="candidates re-scored exactly per call")
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n-qubits", type=int, default=None)
    ap.add_argument("--rand-count", type=int, default=64)
    ap.add_argument("--reference", action="store_true", help="also solve each QUBO exactly (m<=22) to log sub-optimality")
    ap.add_argument("--polish-sec", type=float, default=0.0, help="classical asset-DP LNS after each accept")
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--max-calls", type=int, default=10**9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    model = pc.make_model(a.instance, a.lam)
    fm = L.FastModel(model)
    T = fm.T
    v, start_file = load_start(model, a.instance, a.lam, a.start)
    cur = fm.objective(v)
    start_obj = cur
    assert fm.feasible(v) and cur == pc.score_v(model, v).objective
    bkv = pc.load_bkv().get((a.instance, model.budget, a.lam), (None, ""))[0]
    gap = lambda o: f"{(o - bkv) / abs(bkv):+.4%}" if bkv else "n/a"
    print(f"{a.instance} λ={a.lam}: start {cur} ({gap(cur)} vs BKV {bkv}) solver={a.solver} m_max={a.m_max} "
          f"| {start_file}", flush=True)
    solve = SOLVERS[a.solver]
    mix = dict((k, float(p)) for k, p in (it.split(":") for it in a.moves.split(",")))
    mode_names, mode_p = list(mix), np.array(list(mix.values())) / sum(mix.values())
    kw = dict(n_layers=a.n_layers, iters=a.iters, alpha=a.alpha, n_qubits=a.n_qubits, count=a.rand_count)

    t0 = time.time()
    calls = accepts = circuits = max_q = 0
    stats = dict(neg_opt=0, hit_opt=0, ref_calls=0, sub_ratio=[])
    hist = []
    while time.time() - t0 < a.time_limit and calls < a.max_calls:
        ta = int(rng.integers(0, T - 1))
        tb = min(T - 1, ta + int(rng.integers(1, a.max_block + 1)))
        mode = str(rng.choice(mode_names, p=mode_p))
        Qm = BUILDERS[mode](fm, v, ta, tb, rng, m_max=a.m_max)
        if Qm is None:
            continue
        calls += 1
        tq = time.time()
        X = solve(Qm, seed=int(rng.integers(1 << 30)), **kw)
        sec = time.time() - tq
        if a.solver == "pce":
            circuits += _PCE_CACHE["circuits"]
            max_q = max(max_q, _PCE_CACHE["qubits"])
        X = X[Qm.ok(X)]  # penalty-free candidates only (always all of them for 'pairs'); no repair
        dE = Qm.delta(X)
        order = np.argsort(dE)[: a.top]
        X, dE = X[order], dE[order]
        ebest = int(dE[0]) if len(dE) else 0
        if a.reference:  # exact for m <= 22, else a long SA run (64 reads x 2000 sweeps) as the reference
            Xo = solve_exact(Qm, K=4) if Qm.m <= 22 else solve_sa(Qm, n_reads=64, sweeps=2000, seed=12345)
            Xo = Xo[Qm.ok(Xo)]
            eopt = min(0, int(Qm.delta(Xo).min())) if len(Xo) else 0
            stats["ref_calls"] += 1
            stats["neg_opt"] += eopt < 0
            stats["hit_opt"] += ebest <= eopt
            if eopt < 0:
                stats["sub_ratio"].append(max(-ebest, 0) / -eopt)  # fraction of the available gain found
        else:
            eopt = None
        acc = False
        for s, e in zip(X, dE):
            if e >= 0:
                break
            w = Qm.apply(v, s)
            if not fm.feasible(w):  # 'free' moves can still hit the net (capital) bounds
                continue
            new = fm.objective(w)
            assert new == cur + e, (new, cur, e)  # the move QUBO is exact
            v, cur, acc, ebest = w, new, True, int(e)
            break
        if acc:
            accepts += 1
            if a.polish_sec > 0:
                v, cur, _ = L.lns(fm, v0=v, time_limit=a.polish_sec, seed=int(rng.integers(1 << 30)))
        hist.append(dict(call=calls, mode=mode, ta=ta, tb=tb, m=Qm.m, ebest=ebest, eopt=eopt, acc=acc, cur=cur,
                         sec=round(sec, 3), t=round(time.time() - t0, 1)))
        if acc or calls % 50 == 0:
            ref = f" opt {eopt}" if eopt is not None else ""
            print(f"call {calls:5d} [{ta},{tb}) m={Qm.m:2d} best ΔE {ebest:>7}{ref} {'ACCEPT' if acc else '      '} "
                  f"cur {cur} ({gap(cur)}) [{time.time() - t0:.0f}s]", flush=True)

    sc = pc.score_v(model, v)
    assert sc.feasible and sc.objective == cur
    rs = stats["sub_ratio"]
    summ = dict(args=vars(a), start=start_obj, final=cur, bkv=bkv, gap=gap(cur), calls=calls, accepts=accepts,
                circuits=circuits, max_qubits=max_q, sec=round(time.time() - t0, 1),
                ref_calls=stats["ref_calls"], qubo_improvable=stats["neg_opt"], hit_opt=stats["hit_opt"],
                mean_gain_fraction=float(np.mean(rs)) if rs else None, start_file=start_file)
    print("FINAL " + json.dumps({k: v_ for k, v_ in summ.items() if k != "args"}), flush=True)
    LOGS.mkdir(exist_ok=True)
    stem = f"hyb_{a.instance}_l{a.lam}_{a.solver}_m{a.m_max}_s{a.seed}{a.tag}"
    (LOGS / f"{stem}.json").write_text(json.dumps(dict(summary=summ, hist=hist), indent=1, default=float))
    if a.record and cur < start_obj:
        settings = (f"Q-accelerated LNS: swap/drop/add move-QUBO (m<={a.m_max}, blocks<={a.max_block}) sub-solver "
                    f"{a.solver}" + (f" MultiBasis-2 n_layers={a.n_layers} iters={a.iters}" if a.solver == "pce" else "")
                    + f", polish {a.polish_sec}s/accept, calls={calls} accepts={accepts} start={start_file} ({start_obj})")
        _, path, _ = pc.record_solution(model, pc.v_to_positions(model, v),
                                        method=f"Q-accelerated LNS ({a.solver} sub-QUBO solver)", settings=settings,
                                        n_qubo_vars=f"<={a.m_max}/call", max_qubits=max_q or "",
                                        quantum=a.solver == "pce", tag=f"hyb-{a.solver}")
        print("recorded", path, flush=True)


if __name__ == "__main__":
    main()
