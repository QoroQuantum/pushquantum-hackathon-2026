"""Continuous relaxation of one day of the portfolio model, neighbours fixed (numba).

    min_v  v'R_t v + sum_i [ (lin+10) v_i + sc max(-v_i,0) + absc |v_i|
                             + w_t |v_i - v_{t-1,i}| + w_{t+1} |v_{t+1,i} - v_i| ]
    s.t.   |v_i| <= ub,  sum_i |v_i| <= B

Exact coordinate descent (each 1-D piece is a convex piecewise quadratic) on the Lagrangian
with a budget multiplier mu, found by bisection.  The capital rule is left to the integer
stage.  The support and the fractional values of the result seed the rounding QUBO.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _min1d(q, l, kinks, weights, sc, ub):
    """argmin over [-ub, ub] of q x^2 + l x + sum_k weights[k] |x - kinks[k]| + sc max(-x, 0)."""
    # candidate points: bounds, kinks, and stationary points of each smooth piece
    pts = np.empty(kinks.shape[0] + 2)
    pts[0] = -ub
    pts[1] = ub
    for k in range(kinks.shape[0]):
        pts[k + 2] = min(max(kinks[k], -ub), ub)
    pts.sort()
    best_x = 0.0
    best_f = 1e300
    for seg in range(pts.shape[0] - 1):
        a, b = pts[seg], pts[seg + 1]
        mid = 0.5 * (a + b)
        # slope of the piecewise-linear part on (a, b)
        s = l
        for k in range(kinks.shape[0]):
            s += weights[k] if mid > kinks[k] else -weights[k]
        if mid < 0:
            s -= sc
        if q > 1e-12:
            x = -s / (2 * q)
            x = min(max(x, a), b)
        else:
            x = a if s > 0 else b
        for cand in (a, b, x):
            f = q * cand * cand + l * cand + sc * max(-cand, 0.0)
            for k in range(kinks.shape[0]):
                f += weights[k] * abs(cand - kinks[k])
            if f < best_f:
                best_f = f
                best_x = cand
    return best_x


@nb.njit(cache=True)
def _cd(R, lin, sc, absc, wprev, aprev, wnext, bnext, mu, ub, v, sweeps, tol):
    n = R.shape[0]
    RS = R + R.T
    G = RS @ v
    kinks = np.empty(3)
    weights = np.empty(3)
    for sw in range(sweeps):
        delta = 0.0
        for i in range(n):
            q = R[i, i]
            h = G[i] - 2 * R[i, i] * v[i]
            kinks[0] = 0.0
            weights[0] = absc[i] + mu
            kinks[1] = aprev[i]
            weights[1] = wprev[i]
            kinks[2] = bnext[i]
            weights[2] = wnext[i]
            x = _min1d(q, h + lin[i], kinks, weights, sc[i], ub)
            d = x - v[i]
            if d != 0.0:
                for k in range(n):
                    G[k] += d * RS[k, i]
                v[i] = x
                delta = max(delta, abs(d))
        if delta < tol:
            break
    return v


def solve_day(p, v_int, t, sweeps=400, tol=1e-6, iters=40, R_override=None, tilt=None, budget=None, ub_mask=None, capital=True):
    """Relaxed optimum of day t with days t-1, t+1 fixed at v_int.  Returns float vector."""
    n, ub = p.n, float(p.ub)
    B = float(p.budget if budget is None else budget)
    R = (p.R[t] if R_override is None else R_override).astype(np.float64)
    lin = (p.lin[t] + 10).astype(np.float64)
    if tilt is not None:
        lin = lin + tilt
    if ub_mask is not None:  # excluded assets: a huge |v| cost keeps them at 0
        absc_extra = np.where(ub_mask, 0.0, 1e12)
    else:
        absc_extra = np.zeros(n)
    sc = p.sc[t].astype(np.float64)
    absc = p.absc[t].astype(np.float64) + absc_extra
    wprev = p.w[t].astype(np.float64) if t > 0 else np.zeros(n)
    aprev = v_int[t - 1].astype(np.float64) if t > 0 else np.zeros(n)
    wnext = p.w[t + 1].astype(np.float64) if t + 1 < p.T else np.zeros(n)
    bnext = v_int[t + 1].astype(np.float64) if t + 1 < p.T else np.zeros(n)
    v0 = v_int[t].astype(np.float64).copy()

    def budget_solve(shift, v_start, sweeps_, iters_):
        ln = lin + shift
        v = _cd(R, ln, sc, absc, wprev, aprev, wnext, bnext, 0.0, ub, v_start.copy(), sweeps_, tol)
        if np.abs(v).sum() <= B + 1e-6:
            return v, 0.0
        lo, hi = 0.0, 1.0
        while True:
            w = _cd(R, ln, sc, absc, wprev, aprev, wnext, bnext, hi, ub, v.copy(), sweeps_, tol)
            if np.abs(w).sum() <= B:
                break
            lo, hi = hi, hi * 4
        best = w
        for _ in range(iters_):
            mid = 0.5 * (lo + hi)
            w = _cd(R, ln, sc, absc, wprev, aprev, wnext, bnext, mid, ub, best.copy(), sweeps_, tol)
            if np.abs(w).sum() <= B:
                hi, best = mid, w
            else:
                lo = mid
            if hi - lo < 1e-3 * max(1.0, hi):
                break
        return best, hi

    v, mu = budget_solve(0.0, v0, sweeps, iters)
    if not capital:
        return v, mu
    # capital rule C - maxcash <= sum v <= C: a multiplier nu on sum v, found by bisection, keeps the
    # relaxed day inside it (with one unit of margin), so its rounding window has feasible points
    cap_hi, cap_lo = p.C - 1.0, p.C - p.maxcash + 1.0
    net = v.sum()
    if cap_lo <= net <= cap_hi:
        return v, mu
    sgn = 1.0 if net > cap_hi else -1.0
    target = cap_hi if sgn > 0 else cap_lo
    lo_nu, hi_nu, best = 0.0, 50.0, None
    for _ in range(12):  # grow until the net is on the right side
        w, m = budget_solve(sgn * hi_nu, v, sweeps // 2, 20)
        if (w.sum() <= target) if sgn > 0 else (w.sum() >= target):
            best = (w, m)
            break
        lo_nu, hi_nu = hi_nu, hi_nu * 3
    if best is None:
        return v, mu
    for _ in range(10):
        mid = 0.5 * (lo_nu + hi_nu)
        w, m = budget_solve(sgn * mid, best[0], sweeps // 2, 20)
        if (w.sum() <= target) if sgn > 0 else (w.sum() >= target):
            hi_nu, best = mid, (w, m)
        else:
            lo_nu = mid
    return best
