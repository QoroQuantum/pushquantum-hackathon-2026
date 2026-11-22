"""Core of the portfolio quantum LNS: exact move-selection QUBOs on the v = long - short model.

A *window* is a list of candidate unit moves m = (t, i, d) with d = +1 or -1.  One binary
x_m per move says whether the move is applied.  For a fixed incumbent v the change of the
official objective is, exactly,

    dE(x) = sum_m a_m x_m + sum_{m < m'} b_mm' x_m x_m'

as long as every slot (t, i) carries at most one move, or two opposite moves on a window
whose days are all different for that asset (both on = no change).  The day budget
sum_i |v_t,i| <= B and the capital rule C - 15 <= net_t <= C are linear in x and become
Divi ``LinearConstraint``s (slack qubits) for the quantum solver; the oracle checks them
exactly.  Problem tables and the objective come from portfolio_search.Problem, whose
``full_obj`` equals ``qoblib_tools.PortfolioModel.score`` (its ``selftest``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import portfolio_search as ps  # noqa: E402
import qoblib_tools as qt  # noqa: E402


class State:
    """Incumbent v with cached per-day risk gradients G[t] = RS[t] v[t], nets and budget use."""

    def __init__(self, p: ps.Problem, v: np.ndarray):
        self.p = p
        self.v = np.ascontiguousarray(v.astype(np.int64))
        self.G, self.net, self.absv = ps.make_state(self.v, p.RS, p.ub)
        self.obj = p.objective(self.v)

    def apply(self, moves, x) -> None:
        for (t, i, d), on in zip(moves, x):
            if on:
                ps.apply_value(t, i, self.v[t, i] + d, self.v, self.G, self.net, self.absv, self.p.RS)
        self.obj = self.p.objective(self.v)


def slot_cost(p: ps.Problem, v: np.ndarray, t: int, i: int, x: int, skip_prev=False, skip_next=False) -> int:
    """Separable part of the objective that depends on v[t, i] = x (others fixed at v)."""
    c = p.lin[t, i] * x + p.sc[t, i] * max(-x, 0) + p.absc[t, i] * abs(x)
    if t > 0 and not skip_prev:
        c += p.w[t, i] * abs(x - v[t - 1, i])
    if t + 1 < p.T and not skip_next:
        c += p.w[t + 1, i] * abs(v[t + 1, i] - x)
    return int(c)


def build_qubo(s: State, moves):
    """Exact (a, b) of dE for a move list; also per-move budget and net coefficients."""
    p, v = s.p, s.v
    M = len(moves)
    a = np.zeros(M, dtype=np.int64)
    b = np.zeros((M, M), dtype=np.int64)
    bud = np.zeros(M, dtype=np.int64)
    for m, (t, i, d) in enumerate(moves):
        x0 = int(v[t, i])
        a[m] = (d * s.G[t, i] + p.R[t, i, i]
                + slot_cost(p, v, t, i, x0 + d) - slot_cost(p, v, t, i, x0) + 10 * d)
        bud[m] = abs(x0 + d) - abs(x0)
    for m in range(M):
        t, i, d = moves[m]
        for k in range(m + 1, M):
            t2, j, d2 = moves[k]
            if t2 == t and j != i:
                b[m, k] = d * d2 * p.RS[t, i, j]
            elif t2 == t and j == i:  # opposite moves on one slot: both on = no change
                b[m, k] = -(a[m] + a[k])
            elif j == i and abs(t2 - t) == 1:  # rebalancing between consecutive days
                ta, da, tb, db = (t, d, t2, d2) if t < t2 else (t2, d2, t, d)
                va, vb, w = int(v[ta, i]), int(v[tb, i]), int(p.w[tb, i])
                b[m, k] = w * (abs(vb + db - va - da) - abs(vb + db - va) - abs(vb - va - da) + abs(vb - va))
    return a, b, bud


def window_days(s: State, moves):
    """Per window day: index map, budget room and net bounds."""
    days = sorted({t for t, _, _ in moves})
    di = {t: k for k, t in enumerate(days)}
    p = s.p
    room = np.array([p.budget - s.absv[t] for t in days], dtype=np.int64)
    lo = np.array([p.C - p.maxcash - s.net[t] for t in days], dtype=np.int64)
    hi = np.array([p.C - s.net[t] for t in days], dtype=np.int64)
    md = np.array([di[t] for t, _, _ in moves], dtype=np.int64)
    dd = np.array([d for _, _, d in moves], dtype=np.int64)
    return days, md, dd, room, lo, hi


@nb.njit(cache=True)
def _oracle(a, b, bud, md, dd, room, lo, hi, any_sign):
    """Exact minimum of dE over all 2^M choices that keep every day feasible (Gray code).

    any_sign=False: only strict improvements over x = 0 count (returns 0, x = 0 otherwise).
    any_sign=True: best feasible choice whatever its sign (+inf if none is feasible)."""
    M = a.shape[0]
    nd = room.shape[0]
    h = a.copy().astype(np.float64)
    x = np.zeros(M, dtype=np.int64)
    bu = np.zeros(nd, dtype=np.int64)
    nt = np.zeros(nd, dtype=np.int64)
    E = 0.0
    best = 0.0
    if any_sign:
        best = 1e300
        ok0 = True
        for w in range(nd):
            if 0 > room[w] or 0 < lo[w] or 0 > hi[w]:
                ok0 = False
        if ok0:
            best = 0.0
    bestcode = np.int64(0)
    code = np.int64(0)
    bs = b + b.T
    for step in range(1, np.int64(1) << M):
        k = 0
        while not (step >> k) & 1:
            k += 1
        if x[k] == 0:
            E += h[k]
            x[k] = 1
            sgn = 1
        else:
            E -= h[k]
            x[k] = 0
            sgn = -1
        for j in range(M):
            h[j] += sgn * bs[k, j]
        bu[md[k]] += sgn * bud[k]
        nt[md[k]] += sgn * dd[k]
        code ^= np.int64(1) << k
        if E < best - 1e-9:
            ok = True
            for w in range(nd):
                if bu[w] > room[w] or nt[w] < lo[w] or nt[w] > hi[w]:
                    ok = False
                    break
            if ok:
                best = E
                bestcode = code
    out = np.zeros(M, dtype=np.int64)
    for k in range(M):
        out[k] = (bestcode >> k) & 1
    return best, out


def oracle(s: State, moves, any_sign=False):
    """Exact window optimum (classical reference for the quantum sub-solver)."""
    a, b, bud = build_qubo(s, moves)
    _, md, dd, room, lo, hi = window_days(s, moves)
    best, x = _oracle(a.astype(np.float64), b.astype(np.float64), bud, md, dd, room, lo, hi, any_sign)
    if best > 1e299:
        return None, x
    return int(round(best)), x


def energy(a, b, x) -> int:
    x = np.asarray(x, dtype=np.int64)
    return int(a @ x + x @ b @ x)


def feasible_x(s: State, moves, x) -> bool:
    _, md, dd, room, lo, hi = window_days(s, moves)
    a, b, bud = None, None, None
    nd = len(room)
    bu = np.zeros(nd, dtype=np.int64)
    nt = np.zeros(nd, dtype=np.int64)
    for k, on in enumerate(x):
        if on:
            t, i, d = moves[k]
            bu[md[k]] += abs(s.v[t, i] + d) - abs(s.v[t, i])
            nt[md[k]] += d
    return bool(((bu <= room) & (nt >= lo) & (nt <= hi)).all())


def check_with_referee(p: ps.Problem, v: np.ndarray):
    """Score v with the notebook's official referee (qoblib_tools.PortfolioModel)."""
    inst = qt.load_portfolio_instance(p.inst)
    model = qt.PortfolioModel(inst, budget=p.budget, lam=p.lam)
    return model, model.score(p.positions(v))


def best_known_file(stem: str) -> Path | None:
    """Best existing .sol for a configuration across the repo's earlier (classical) result sets."""
    cands = [ROOT / "deliverables" / "portfolio" / f"{stem}.sol"]
    cands += list((ROOT / "portfolio_QIcing" / "solutions").glob(f"{stem}/*.sol"))
    best, bp = None, None
    for c in cands:
        if c.exists():
            o = ps.read_objective(c)
            if o is not None and (best is None or o < best):
                best, bp = o, c
    return bp


def load_v(p: ps.Problem, path: Path) -> np.ndarray:
    _, pos = qt.read_portfolio_solution(path)
    return p.v_from_positions(pos)
