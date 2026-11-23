"""Exact solver for the small portfolio instances: dynamic programming over days.

Days interact only through the trading cost between consecutive days (``day_swap.day_form``:
day t's cost given its neighbours is ``v A_t v + Σ_i f_i(v_i)``, and the neighbours enter
only through ``|L_t - L_{t-1}| + |S_t - S_{t-1}|`` terms). So the whole problem is a shortest
path through the layered graph of per-day holdings:

    J_t(v) = u_t(v) + min_w [ J_{t-1}(w) + p_t(w, v) ]

with ``u_t`` the day's own cost (risk, return, short and interest terms, entry/exit
trading on the first and last day) and ``p_t`` the trading cost between days t-1 and t.
On a010 (10 assets, at most 4 units a day) a day has 8,341 feasible net holdings, so the
DP is exact and takes seconds per instance (numba min-plus with pruning). Holdings are net
positions (long or short, not both), which loses nothing: holding both legs of one asset
only adds costs. The result is re-scored with the official referee; the DP value and the
referee differ by a constant that is checked on random solutions before it is trusted.

    python exact_dp.py --instances po_a010_t10_orig po_a010_t15_s02 --lams 0 1e-3 1e-2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

import qoblib_tools as qt
from asset_groups import BlockEvaluator
from day_swap import day_form
from grid_runs import LAM_TAGS, load_bkv


def ball(n: int, B: int, ub: int = 3) -> np.ndarray:
    """Every net vector in Z^n with |v_i| <= ub and sum|v_i| <= B."""
    pts = [np.zeros(n, np.int64)]
    for k in range(1, B + 1):
        for idx in itertools.combinations(range(n), k):
            for mags in itertools.product(range(1, ub + 1), repeat=k):
                if sum(mags) > B:
                    continue
                for sg in itertools.product((1, -1), repeat=k):
                    v = np.zeros(n, np.int64)
                    v[list(idx)] = np.array(mags) * np.array(sg)
                    pts.append(v)
    return np.array(pts)


@njit(parallel=True, cache=True)
def minplus(J, order, V, tc, D):
    """out[s] = min_w J[w] + Σ_i tc[i] D[V[w,i], V[s,i]]  (D >= 0, so prune on J sorted by ``order``)."""
    S, n = V.shape
    out = np.empty(S)
    arg = np.empty(S, np.int64)
    for s in prange(S):
        best = np.inf
        ba = -1
        for k in range(S):
            w = order[k]
            c = J[w]
            if c >= best:
                break  # J is sorted, so no later w can win
            for i in range(n):
                c += tc[i] * D[V[w, i], V[s, i]]
                if c >= best:
                    break
            if c < best:
                best = c
                ba = w
        out[s] = best
        arg[s] = ba
    return out, arg


@njit(parallel=True, cache=True)
def minplus2(J, order, Vp, Vc, tc, D):
    """minplus with different state sets on the two days (rows of Vp, Vc)."""
    S = Vc.shape[0]
    n = Vc.shape[1]
    out = np.empty(S)
    arg = np.empty(S, np.int64)
    for s in prange(S):
        best = np.inf
        ba = -1
        for k in range(Vp.shape[0]):
            w = order[k]
            c = J[w]
            if c >= best:
                break
            for i in range(n):
                c += tc[i] * D[Vp[w, i], Vc[s, i]]
                if c >= best:
                    break
            if c < best:
                best = c
                ba = w
        out[s] = best
        arg[s] = ba
    return out, arg


def _unary(model, ev, states, t):
    T, n = model.n_periods, len(model.symbols)
    Z = np.zeros((T, n), np.int64)
    F = day_form(ev, Z, Z, t)
    Lg, Sg = np.maximum(states, 0), np.maximum(-states, 0)
    c = np.einsum("si,ij,sj->s", states, F["A"], states) + F["f"][np.arange(n)[None, :], states + 3].sum(1)
    if 1 <= t <= T - 2:
        c = c - (Lg + Sg) @ model.tc[t]
    if t + 1 <= T - 2:
        c = c - (Lg + Sg) @ model.tc[t + 1]
    return c.astype(np.float64)


def recombine(model: qt.PortfolioModel, Vs: list[np.ndarray]):
    """Best mix of days across solutions: exact DP over each day's candidate holdings
    (the distinct holdings the given solutions use on that day)."""
    ev = BlockEvaluator(model)
    T = model.n_periods
    vals = np.arange(-3, 4)
    D = (np.abs(np.maximum(vals, 0)[:, None] - np.maximum(vals, 0)[None, :])
         + np.abs(np.maximum(-vals, 0)[:, None] - np.maximum(-vals, 0)[None, :])).astype(np.float64)
    cand = [np.unique(np.stack([V[t] for V in Vs]), axis=0).astype(np.int64) for t in range(T)]
    J = _unary(model, ev, cand[0], 0)
    back = []
    for t in range(1, T):
        if 1 <= t <= T - 2:
            order = np.argsort(J, kind="stable")
            m, arg = minplus2(J, order, (cand[t - 1] + 3).astype(np.int8), (cand[t] + 3).astype(np.int8),
                              model.tc[t].astype(np.float64), D)
        else:
            arg = np.full(len(cand[t]), int(np.argmin(J)))
            m = np.full(len(cand[t]), J.min())
        back.append(arg)
        J = _unary(model, ev, cand[t], t) + m
    s = int(np.argmin(J))
    path = [s]
    for arg in reversed(back):
        s = int(arg[s])
        path.append(s)
    path.reverse()
    return np.stack([cand[t][path[t]] for t in range(T)]), [len(c) for c in cand]


def solve(model: qt.PortfolioModel, states=None):
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    if states is None:
        states = ball(n, model.budget, model.ub)
    net = states.sum(1)
    states = states[(net >= model.capital - ev.max_cash) & (net <= model.capital)]
    Z = np.zeros((T, n), np.int64)
    Lg, Sg = np.maximum(states, 0), np.maximum(-states, 0)
    U = []
    for t in range(T):
        F = day_form(ev, Z, Z, t)
        c = np.einsum("si,ij,sj->s", states, F["A"], states) + F["f"][np.arange(n)[None, :], states + 3].sum(1)
        # day_form with empty neighbours charges trading to and from zero: take that out,
        # it is added back exactly by the pairwise terms
        if 1 <= t <= T - 2:
            c = c - (Lg + Sg) @ model.tc[t]
        if t + 1 <= T - 2:
            c = c - (Lg + Sg) @ model.tc[t + 1]
        U.append(c.astype(np.float64))
    vals = np.arange(-3, 4)
    D = (np.abs(np.maximum(vals, 0)[:, None] - np.maximum(vals, 0)[None, :])
         + np.abs(np.maximum(-vals, 0)[:, None] - np.maximum(-vals, 0)[None, :])).astype(np.float64)
    Vi = (states + 3).astype(np.int8)
    J = U[0].copy()
    back = []
    for t in range(1, T):
        if 1 <= t <= T - 2:
            order = np.argsort(J, kind="stable")
            m, arg = minplus(J, order, Vi, model.tc[t].astype(np.float64), D)
        else:  # no trading term between the last two days
            arg = np.full(len(J), int(np.argmin(J)))
            m = np.full(len(J), J.min())
        back.append(arg)
        J = U[t] + m
    s = int(np.argmin(J))
    path = [s]
    for arg in reversed(back):
        s = int(arg[s])
        path.append(s)
    path.reverse()
    V = states[path]
    return V, float(J.min()), states, U


def dp_value(model, U_states, V):
    """DP objective of holdings V (T, n): Σ u_t + Σ trading, for the decomposition check."""
    states, U = U_states
    idx = {tuple(v): k for k, v in enumerate(states)}
    T = model.n_periods
    tot = sum(U[t][idx[tuple(V[t])]] for t in range(T))
    for t in range(1, T - 1):
        L1, S1 = np.maximum(V[t], 0), np.maximum(-V[t], 0)
        L0, S0 = np.maximum(V[t - 1], 0), np.maximum(-V[t - 1], 0)
        tot += float((np.abs(L1 - L0) + np.abs(S1 - S0)) @ model.tc[t])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+",
                    default=[f"po_a010_{t}_{s}" for t in ("t10", "t15") for s in ("orig", "s00", "s01", "s02")])
    ap.add_argument("--lams", nargs="+", default=LAM_TAGS)
    ap.add_argument("--out", default="results/exact_dp")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bkv = load_bkv()
    rng = np.random.default_rng(0)
    states_cache = {}
    for inst in a.instances:
        pi = qt.load_portfolio_instance(inst)
        for lam in a.lams:
            t0 = time.time()
            model = qt.PortfolioModel(pi, budget=4 if "a010" in inst else 20, lam=lam)
            key = (len(model.symbols), model.budget)
            if key not in states_cache:
                states_cache[key] = ball(len(model.symbols), model.budget, model.ub)
            V, val, states, U = solve(model, states_cache[key])
            L, S = np.maximum(V, 0), np.maximum(-V, 0)
            sc = model.score_arrays(L, S)
            assert sc.feasible
            # the referee and the DP must differ by one constant: check on random solutions too
            const = sc.objective - val
            for _ in range(5):
                W = states[rng.integers(len(states), size=model.n_periods)]
                s2 = model.score_arrays(np.maximum(W, 0), np.maximum(-W, 0))
                assert s2.feasible and abs(s2.objective - dp_value(model, (states, U), W) - const) < 1e-6, \
                    (inst, lam, s2.objective, dp_value(model, (states, U), W), const)
            stem = f"{inst}_b{model.budget:03d}_l{lam}"
            sol = out / f"{stem}.sol"
            model.write_solution(sol, model.to_positions(L, S),
                                 comment=f"exact_dp.py: exact DP over the {len(states)} feasible day holdings (proven optimal)")
            bk = bkv.get(stem)
            gap = None if bk is None else (sc.objective - bk) / abs(bk) * 100
            rec = dict(instance=inst, budget=model.budget, lam=lam, score=int(sc.objective), bkv=bk, gap=gap,
                       states=len(states), seconds=round(time.time() - t0, 1), solution_file=str(sol),
                       optimal=True)
            (out / f"{stem}.json").write_text(json.dumps(rec, indent=1))
            g = "—" if gap is None else f"{gap:+.3f}%"
            print(f"{stem:32s} optimum {int(sc.objective):>9}  bkv {bk}  gap {g}  ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
