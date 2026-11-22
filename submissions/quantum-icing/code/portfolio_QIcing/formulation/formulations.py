"""Alternative binary formulations of the QOBLIB 06 portfolio problem, built for PCE.

Three independent reductions, combinable:

1. Period decoupling (exact).  The multi-period objective is a chain:
       sum_t stage_t(v_t) + sum_t tc_t . |v_t - v_{t-1}|
   so a per-period candidate pool + exact chain DP over the pool recovers the full optimum as soon
   as every period's optimal state is in its pool.  On every a010 instance tested the optimal state
   is among the 4 lowest single-period states (see README), so the quantum sub-problem shrinks from
   the 630-variable multi-period QUBO to T independent single-period problems of ~30 variables.

2. Cell encodings (per asset, net position v in -U..U; holding long AND short is dominated):
       unary    : U long + U short bits, v = sum l - sum s              (official; 2U bits, QUBO)
       signmag  : sign bit + ceil(log2(U+1)) magnitude bits,
                  v = (1 - 2 sigma) * M,  |v| = M  (LINEAR)             (3 bits for U = 3, HUBO deg 4)
       ternary  : U = 1 unary (l, s)                                    (2 bits, QUBO)
       ternsm   : U = 1 sign-magnitude (sigma, m)                       (2 bits, HUBO deg 4)
   PCE's loss is a classical polynomial of the per-variable expectations, so HUBO terms cost no
   qubits and no gates (QAOA would need many-body ZZZZ rotations).

3. Constraint handling for sum|v| <= B and C - 15 <= sum v <= C:
       slack       : binary slack register + squared equality penalty (official style)
       unbalanced  : no slack;  penalty  -a*g + b*g^2  on g = B - sum|v|  (Montanez-Barrera et al.)
       none        : objective only; feasibility is restored at decode time (greedy repair)

Everything here is exact integer arithmetic against the referee's coefficient tables; see
`validate()` for the energy == referee checks.
"""
from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PF = HERE.parent
for p in (str(PF), str(PF.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import pf_core as pc  # noqa: E402

qt = pc.qt
CELL_BITS = {"unary": lambda U: 2 * U, "signmag": lambda U: 1 + max(1, math.ceil(math.log2(U + 1))),
             "ternary": lambda U: 2, "ternsm": lambda U: 2}


# ── single-period reduced problem ────────────────────────────────────────


@dataclass
class PeriodProblem:
    """stage_t(v) = v'Rv + lin.v + absw.|v| + negw.max(-v,0) + const, with sum|v| <= B, net bounds."""

    R: np.ndarray
    lin: np.ndarray
    absw: np.ndarray
    negw: np.ndarray
    const: float
    budget: int
    net_lo: int
    net_hi: int
    ub: int
    t: int = 0

    @property
    def n(self) -> int:
        return len(self.lin)

    def energy(self, V: np.ndarray) -> np.ndarray:
        """Stage cost of each row of V (k, n) (exact integers)."""
        V = np.atleast_2d(V).astype(np.int64)
        risk = np.einsum("ki,ij,kj->k", V, self.R, V)
        return risk + V @ self.lin + np.abs(V) @ self.absw + np.maximum(-V, 0) @ self.negw + self.const

    def feasible(self, V: np.ndarray) -> np.ndarray:
        V = np.atleast_2d(V)
        s, a = V.sum(1), np.abs(V).sum(1)
        return (a <= self.budget) & (s <= self.net_hi) & (s >= self.net_lo) & (np.abs(V).max(1) <= self.ub)

    def repair(self, v: np.ndarray) -> np.ndarray:
        """Greedy: while infeasible, drop the unit whose removal lowers the energy most."""
        v = np.clip(np.asarray(v, dtype=np.int64), -self.ub, self.ub).copy()
        while not self.feasible(v)[0]:
            s = v.sum()
            if s > self.net_hi:
                cand = np.flatnonzero(v > 0)
            elif s < self.net_lo:
                cand = np.flatnonzero(v < 0)
            else:
                cand = np.flatnonzero(v)
            trial = np.repeat(v[None], len(cand), 0)
            trial[np.arange(len(cand)), cand] -= np.sign(v[cand])
            v = trial[int(np.argmin(self.energy(trial)))]
        return v


def period_problems(model: qt.PortfolioModel) -> list[PeriodProblem]:
    lin, absw, negw, const = pc.stage_tables(model)
    T = model.n_periods
    max_cash = 2**model.cash_slack_bits - 1
    return [PeriodProblem(model.risk[t].astype(np.int64), lin[t], absw[t], negw[t], float(const[t]), model.budget,
                          model.capital - max_cash, model.capital, model.ub, t) for t in range(T)]


def chain_dp(model: qt.PortfolioModel, pools: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Exact optimum of the multi-period objective restricted to v_t in pools[t]. Returns (v, objective)."""
    T = model.n_periods
    probs = period_problems(model)
    best = probs[0].energy(pools[0]).astype(np.int64)
    backs = []
    for t in range(1, T):
        w = model.tc[t].astype(np.int64) if t < T - 1 else np.zeros(len(model.symbols), dtype=np.int64)
        D = (np.abs(pools[t - 1][:, None, :] - pools[t][None, :, :]) * w).sum(-1) + best[:, None]
        backs.append(D.argmin(0))
        best = D.min(0) + probs[t].energy(pools[t]).astype(np.int64)
    k = int(best.argmin())
    chain = [k]
    for a in reversed(backs):
        chain.append(int(a[chain[-1]]))
    chain.reverse()
    v = np.array([pools[t][chain[t]] for t in range(T)], dtype=np.int64)
    return v, int(best[k])


# ── polynomial helper ────────────────────────────────────────────────────


class Poly:
    """Multilinear polynomial over binary variables: {sorted tuple of var ids: coef}."""

    def __init__(self, terms=None, const=0.0):
        self.t: dict[tuple, float] = dict(terms or {})
        self.const = float(const)

    @staticmethod
    def var(i):
        return Poly({(i,): 1.0})

    @staticmethod
    def lin_comb(pairs, const=0.0):  # [(var, coef)]
        p = Poly(const=const)
        for v, c in pairs:
            p.t[(v,)] = p.t.get((v,), 0.0) + c
        return p

    def __add__(self, o):
        if not isinstance(o, Poly):
            return Poly(self.t, self.const + o)
        r = Poly(self.t, self.const + o.const)
        for k, c in o.t.items():
            r.t[k] = r.t.get(k, 0.0) + c
        return r

    __radd__ = __add__

    def iadd(self, o, scale=1.0):
        """In-place self += scale * o."""
        if not isinstance(o, Poly):
            self.const += scale * o
            return self
        self.const += scale * o.const
        t = self.t
        for k, c in o.t.items():
            t[k] = t.get(k, 0.0) + scale * c
        return self

    def __mul__(self, o):
        if not isinstance(o, Poly):
            return Poly({k: c * o for k, c in self.t.items()}, self.const * o)
        r = Poly(const=self.const * o.const)
        for k, c in self.t.items():
            r.t[k] = r.t.get(k, 0.0) + c * o.const
        for k, c in o.t.items():
            r.t[k] = r.t.get(k, 0.0) + c * self.const
        rt = r.t
        for k1, c1 in self.t.items():
            s1 = set(k1)
            for k2, c2 in o.t.items():
                k = k1 if k1 == k2 else tuple(sorted(s1.union(k2)))  # x^2 = x
                rt[k] = rt.get(k, 0.0) + c1 * c2
        return r

    __rmul__ = __mul__

    def clean(self, tol=1e-12):
        self.t = {k: c for k, c in self.t.items() if abs(c) > tol}
        return self

    @property
    def degree(self):
        return max((len(k) for k in self.t), default=0)

    def compiled(self):
        """[(idx (m, d) int array, coefs (m,))] grouped by degree, cached."""
        key = (len(self.t), self.const)
        if getattr(self, "_ckey", None) != key:
            groups: dict[int, tuple[list, list]] = {}
            for k, c in self.t.items():
                g = groups.setdefault(len(k), ([], []))
                g[0].append(k)
                g[1].append(c)
            self._comp = [(np.array(ix, dtype=np.int64), np.array(cs)) for d, (ix, cs) in sorted(groups.items())]
            self._ckey = key
        return self._comp

    def evaluate(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.full(len(X), self.const)
        comp = self.compiled()
        for s in range(0, len(X), chunk):
            Xc = X[s: s + chunk].astype(bool)
            e = out[s: s + chunk]
            for idx, cs in comp:
                e += Xc[:, idx].all(-1) @ cs
        return out

    def as_divi_dict(self, n_vars: int) -> dict:
        """HUBO dict for BinaryOptimizationProblem; keeps every variable present (tiny linear term)."""
        d = {k: c for k, c in self.t.items()}
        present = {v for k in d for v in k}
        for v in range(n_vars):
            if v not in present:
                d[(v,)] = 1e-9
        return d


# ── per-period encodings ─────────────────────────────────────────────────


@dataclass
class PeriodFormulation:
    """Binary encoding of one PeriodProblem.

    cell: unary | signmag | ternary | ternsm ;  constraints: slack | unbalanced | none
    Variable layout: asset-major cells, then budget slack, then net (cash) slack.
    """

    prob: PeriodProblem
    cell: str = "signmag"
    constraints: str = "slack"
    penalty_weight: float | None = None
    assets: np.ndarray | None = None  # restrict to a subset (others fixed at 0)
    unbalanced: tuple[float, float] = (1.0, 1.0)  # (a, b) multipliers of W for -a*g + b*g^2
    obj: Poly = field(init=False)
    pen: Poly = field(init=False)

    def __post_init__(self):
        P = self.prob
        self.U = 1 if self.cell in ("ternary", "ternsm") else P.ub
        self.assets = np.arange(P.n) if self.assets is None else np.asarray(self.assets)
        self.k = CELL_BITS[self.cell](self.U)
        na = len(self.assets)
        self.n_cell_vars = na * self.k
        maxunits = min(P.budget, na * self.U)
        self.budget_bits = max(1, math.ceil(math.log2(P.budget + 1))) if self.constraints == "slack" else 0
        self.net_active = (P.net_hi < maxunits) or (P.net_lo > -maxunits)
        self.net_bits = (max(1, math.ceil(math.log2(P.net_hi - P.net_lo + 1)))
                         if (self.constraints == "slack" and self.net_active) else 0)
        self.n_vars = self.n_cell_vars + self.budget_bits + self.net_bits
        if self.penalty_weight is None:
            self.penalty_weight = self.auto_weight()
        self._build()

    # value / |value| / neg-part of each cell as polynomials
    def cell_polys(self, a: int):
        b = a * self.k
        if self.cell in ("unary", "ternary"):
            L = Poly.lin_comb([(b + m, 1.0) for m in range(self.U)])
            S = Poly.lin_comb([(b + self.U + m, 1.0) for m in range(self.U)])
            return L + S * -1.0, L + S, S  # v, units (budget count), short units
        nm = self.k - 1
        sigma = Poly.var(b)
        M = Poly.lin_comb([(b + 1 + j, float(2**j)) for j in range(nm)])
        v = M + (sigma * M) * -2.0
        return v, M, sigma * M

    def auto_weight(self) -> float:
        """A unit is never worth more than its best linear gain + its best risk-diversification gain."""
        P = self.prob
        R = P.R[np.ix_(self.assets, self.assets)].astype(float)
        gain = np.abs(P.lin[self.assets]) + np.abs(R).sum(1) * 2 * self.U
        return float(2.0 * gain.max() + 1.0)

    def _build(self):
        P, W = self.prob, self.penalty_weight
        cells = [self.cell_polys(a) for a in range(len(self.assets))]
        A = self.assets
        obj = Poly(const=P.const)
        for ia, (v, units, neg) in enumerate(cells):
            i = A[ia]
            obj.iadd(v, float(P.lin[i])).iadd(units, float(P.absw[i])).iadd(neg, float(P.negw[i]))
        R = P.R[np.ix_(A, A)].astype(float)
        for a, b in itertools.product(range(len(A)), repeat=2):
            if R[a, b] != 0 and b >= a:
                c = R[a, b] if a == b else R[a, b] + R[b, a]
                obj.iadd(cells[a][0] * cells[b][0], float(c))
        units, net = Poly(), Poly()
        for c in cells:
            units.iadd(c[1])
            net.iadd(c[0])
        pen = Poly()
        if self.constraints == "slack":
            s0 = self.n_cell_vars
            slack = Poly.lin_comb([(s0 + c, float(2**c)) for c in range(self.budget_bits)])
            g = units + slack + (-P.budget)
            pen = pen + g * g
            if self.net_bits:
                s1 = s0 + self.budget_bits
                ns = Poly.lin_comb([(s1 + c, float(2**c)) for c in range(self.net_bits)])
                h = net + ns + (-P.net_hi)  # net + slack = net_hi, slack in 0..(hi-lo)
                pen = pen + h * h
        elif self.constraints == "unbalanced":
            a, b = self.unbalanced
            g = units * -1.0 + P.budget  # >= 0 when feasible
            pen = pen + g * (-a) + (g * g) * (b / max(P.budget, 1))
            if self.net_active:
                h = net * -1.0 + P.net_hi
                pen = pen + h * (-a) + (h * h) * (b / max(P.net_hi - P.net_lo, 1))
        self.obj = obj.clean()
        self.pen = (pen * W).clean() if pen.t or pen.const else pen
        self.full = (self.obj + self.pen).clean()

    # bits <-> v
    def decode(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X)).astype(np.int64)
        na, k, U = len(self.assets), self.k, self.U
        C = X[:, : na * k].reshape(len(X), na, k)
        if self.cell in ("unary", "ternary"):
            v = C[:, :, :U].sum(-1) - C[:, :, U:].sum(-1)
        else:
            M = (C[:, :, 1:] * (2 ** np.arange(k - 1))).sum(-1)
            v = np.where(C[:, :, 0] == 1, -M, M)
        out = np.zeros((len(X), self.prob.n), dtype=np.int64)
        out[:, self.assets] = v
        return out

    def encode(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.int64)
        P = self.prob
        x = np.zeros(self.n_vars, dtype=np.int8)
        for ia, i in enumerate(self.assets):
            b, val = ia * self.k, int(v[i])
            if self.cell in ("unary", "ternary"):
                off = 0 if val > 0 else self.U
                x[b + off: b + off + min(abs(val), self.U)] = 1
            else:
                x[b] = 1 if val < 0 else 0
                for j in range(self.k - 1):
                    x[b + 1 + j] = (abs(val) >> j) & 1
        s0 = self.n_cell_vars
        if self.budget_bits:
            sl = P.budget - int(np.abs(v).sum())
            for c in range(self.budget_bits):
                x[s0 + c] = (sl >> c) & 1
        if self.net_bits:
            sl = P.net_hi - int(v.sum())
            for c in range(self.net_bits):
                x[s0 + self.budget_bits + c] = (sl >> c) & 1
        return x

    def to_divi(self):
        from divi.qprog.problems import BinaryOptimizationProblem
        scale = max(abs(c) for c in self.full.t.values())
        d = {k: c / scale for k, c in self.full.as_divi_dict(self.n_vars).items()}
        return BinaryOptimizationProblem(d), scale

    def summary(self) -> str:
        return (f"{self.cell}/{self.constraints}: {self.n_vars} vars ({self.n_cell_vars} cell + "
                f"{self.budget_bits} budget + {self.net_bits} net slack), degree {self.full.degree}, "
                f"{len(self.full.t)} terms, W={self.penalty_weight:.0f}")


def validate(model, cells=("unary", "signmag", "ternary", "ternsm"), n_samples=300, seed=0) -> list[str]:
    """Polynomial energy == referee stage cost on feasible v; penalty 0 there; > 0 on infeasible."""
    rng = np.random.default_rng(seed)
    out = []
    for t, P in enumerate(period_problems(model)[: 3]):
        for cell in cells:
            for cons in ("slack", "unbalanced", "none"):
                F = PeriodFormulation(P, cell=cell, constraints=cons)
                U = F.U
                bad = 0
                for _ in range(n_samples):
                    v = np.zeros(P.n, dtype=np.int64)
                    for i in rng.choice(P.n, size=rng.integers(0, min(P.n, P.budget) + 1), replace=False):
                        v[i] = rng.integers(-U, U + 1)
                    while np.abs(v).sum() > P.budget:
                        j = np.flatnonzero(v)[0]
                        v[j] -= np.sign(v[j])
                    if not P.feasible(v)[0]:
                        continue
                    x = F.encode(v)
                    assert (F.decode(x)[0] == v).all()
                    e_obj = F.obj.evaluate(x)[0]
                    e_ref = P.energy(v)[0]
                    pen = F.pen.evaluate(x)[0] if F.pen.t else 0.0
                    ok = abs(e_obj - e_ref) < 1e-6 and (cons != "slack" or abs(pen) < 1e-6)
                    bad += not ok
                out.append(f"t={t} {F.summary()}  mismatches {bad}/{n_samples}")
    return out


# ── multi-period variable counts ─────────────────────────────────────────


def variable_counts(model) -> dict:
    """Variables of each full multi-period formulation (no decomposition) and per-period sub-problem."""
    n, T, B = len(model.symbols), model.n_periods, model.budget
    U = model.ub
    budget_bits_off, cash_bits_off = model.count_slack_bits, model.cash_slack_bits
    bb = max(1, math.ceil(math.log2(B + 1)))
    cash_needed = B > model.capital - 0 or B > (2**model.cash_slack_bits - 1 - model.capital)
    cb = model.cash_slack_bits if B > 5 else 0
    out = {
        "official (unary, 7+4 slack bits)": n * 2 * U * T + (budget_bits_off + cash_bits_off) * T,
        "compact (unary, log slack, last period kept)": n * 2 * U * T + (bb + cb) * T,
    }
    Te = T - 1  # last period is always empty in an optimum
    for cell in ("unary", "signmag", "ternary", "ternsm"):
        k = CELL_BITS[cell](1 if cell in ("ternary", "ternsm") else U)
        out[f"{cell}, last period dropped, slack"] = n * k * Te + (bb + cb) * Te
        out[f"{cell}, last period dropped, slack-free"] = n * k * Te
    for cell in ("unary", "signmag", "ternary", "ternsm"):
        k = CELL_BITS[cell](1 if cell in ("ternary", "ternsm") else U)
        out[f"per-period sub-problem, {cell}, slack"] = n * k + bb + cb
        out[f"per-period sub-problem, {cell}, slack-free"] = n * k
    return out


def poly_qubits(n_vars: int) -> int:
    q = (math.isqrt(8 * n_vars + 1) - 1) // 2
    return q if q * (q + 1) // 2 >= n_vars else q + 1


if __name__ == "__main__":
    m = pc.make_model("po_a010_t10_orig", "1e-4")
    for line in validate(m):
        print(line)
    m50 = pc.make_model("po_a050_t10_s00", "1e-2")
    for line in validate(m50, n_samples=60):
        print(line)
