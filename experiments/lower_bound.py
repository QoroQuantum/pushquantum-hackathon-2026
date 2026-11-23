"""Certified lower bound on a portfolio instance's optimum (useful where QOBLIB has no BKV).

Every term of the official objective except risk is linear in the unit counts
``(L, S)``: return, transaction (|ΔL| + |ΔS|, linearised), short cost, liquidation and
cash interest (the 4 slack bits weigh 10·2^k, so interest = -10·cash). Risk ``vᵀR_t v`` is
bounded below by ``λ_min(R_t)·Σ v_i² ≥ 3·min(0, λ_min)·Σ|v_i|`` (|v_i| ≤ 3), which is linear
in units too. Relaxing integrality gives an LP whose optimum is ≤ every feasible score.

    python experiments/lower_bound.py --instance po_a400_t15_orig --budget 100 --lam 1e-3 \
        --sol handin/po_a400_t15_orig_b100_l1e-3.sol
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import qoblib_tools as qt


def lp_bound(model: qt.PortfolioModel, risk_floor: bool = True):
    T, n, ub = model.n_periods, len(model.symbols), model.ub
    max_cash, max_count = 2 ** model.cash_slack_bits - 1, 2 ** model.count_slack_bits - 1
    r = int(model.interest[0])
    assert np.array_equal(model.interest, r * 2 ** np.arange(model.cash_slack_bits)), "interest not linear"
    lmin = np.array([np.linalg.eigvalsh(model.risk[t].astype(float)).min() for t in range(T)])
    rfl = 3 * np.minimum(lmin, 0) if risk_floor else np.zeros(T)
    nv = T * n
    iL = lambda t, i: t * n + i  # noqa: E731
    iS = lambda t, i: nv + t * n + i  # noqa: E731
    nd = (T - 2) * n  # transitions 0->1 .. T-3->T-2 are charged (T-2 -> T-1 is liquidation)
    iDL = lambda k, i: 2 * nv + k * n + i  # noqa: E731
    iDS = lambda k, i: 2 * nv + nd + k * n + i  # noqa: E731
    N = 2 * nv + 2 * nd
    c = np.zeros(N)
    tc, sc, ret = model.tc.astype(float), model.sc.astype(float), model.ret.astype(float)
    for t in range(T):
        cl = np.full(n, rfl[t] + r)  # interest = -r*(capital - net) = const + r*net
        cs = np.full(n, rfl[t] - r) + sc[t]
        if t < T - 1:
            cl -= ret[t]
            cs += ret[t]
        if t == 0:
            cl += tc[0]
            cs += tc[0]
        if t == T - 1:
            cl += tc[T - 1]
            cs += tc[T - 1]
        c[t * n:(t + 1) * n] = cl
        c[nv + t * n: nv + (t + 1) * n] = cs
    for k in range(T - 2):
        c[2 * nv + k * n: 2 * nv + (k + 1) * n] = tc[k + 1]
        c[2 * nv + nd + k * n: 2 * nv + nd + (k + 1) * n] = tc[k + 1]
    const = -r * model.capital * T
    rows, cols, vals, b = [], [], [], []
    row = 0
    ar = np.arange(n)
    for k in range(T - 2):  # ±(X[k+1] - X[k]) - D[k] <= 0 for X in (L, S)
        for ix, idd in ((iL, iDL), (iS, iDS)):
            for sgn in (1, -1):
                rr = row + ar
                rows += [rr, rr, rr]
                cols += [ix(k + 1, ar), ix(k, ar), idd(k, ar)]
                vals += [np.full(n, sgn), np.full(n, -sgn), np.full(n, -1.0)]
                b.append(np.zeros(n))
                row += n
    for t in range(T):
        for coefL, coefS, rhs in ((1, 1, model.budget), (-1, -1, -(model.budget - max_count)),
                                  (1, -1, model.capital), (-1, 1, -(model.capital - max_cash))):
            rows += [np.full(n, row), np.full(n, row)]
            cols += [iL(t, ar), iS(t, ar)]
            vals += [np.full(n, coefL, float), np.full(n, coefS, float)]
            b.append(np.array([rhs], float))
            row += 1
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(row, N))
    bounds = [(0, ub)] * (2 * nv) + [(0, None)] * (2 * nd)
    t0 = time.time()
    res = linprog(c, A_ub=A, b_ub=np.concatenate(b), bounds=bounds, method="highs")
    assert res.status == 0, res.message
    x = res.x
    Lx, Sx = x[:nv].reshape(T, n), x[nv:2 * nv].reshape(T, n)
    return dict(bound=res.fun + const, seconds=round(time.time() - t0, 1), lmin=lmin, L=Lx, S=Sx,
                c=c, const=const, nv=nv, nd=nd)


def linear_part(model, L, S, info):
    """LP objective at an integer solution (to check the LP against the official score)."""
    T, n = L.shape
    x = np.concatenate([L.ravel(), S.ravel(), np.abs(np.diff(L, axis=0))[:T - 2].ravel(),
                        np.abs(np.diff(S, axis=0))[:T - 2].ravel()]).astype(float)
    return float(info["c"] @ x + info["const"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a400_t15_orig")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--sol", default=None)
    args = ap.parse_args()
    model = qt.PortfolioModel(qt.load_portfolio_instance(args.instance), budget=args.budget, lam=args.lam)
    info = lp_bound(model)
    lb = info["bound"]
    print(f"{args.instance} B={args.budget} λ={args.lam}: LP lower bound {lb:.0f} ({info['seconds']}s); "
          f"min eig of risk per day {info['lmin'].min():.1f}..{info['lmin'].max():.1f}")
    frac = np.abs(info["L"] - np.round(info["L"])).max(), np.abs(info["S"] - np.round(info["S"])).max()
    print(f"  LP solution max fractionality L {frac[0]:.3f}, S {frac[1]:.3f}")
    if args.sol:
        _, pos = qt.read_portfolio_solution(args.sol)
        L, S = model.to_arrays(pos)
        sc = model.score_arrays(L, S)
        # check: official score = linear part (without the risk floor) + risk
        lin = linear_part(model, L, S, info)
        floor = sum(3 * min(info["lmin"][t], 0) * (L[t] + S[t]).sum() for t in range(model.n_periods))
        assert abs(lin - floor + sc.breakdown["risk"] - sc.objective) < 1e-6 * abs(sc.objective) + 1, \
            (lin, floor, sc.breakdown, sc.objective)
        print(f"  solution {sc.objective} (feasible {sc.feasible}); gap to bound "
              f"{(sc.objective - lb) / abs(lb):.2%} — the optimum is at most this far below the solution")


if __name__ == "__main__":
    main()
