"""Continuous all-period guide for integer portfolio search."""
from __future__ import annotations

import numba as nb
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@nb.njit(cache=True)
def _volume(positive, negative, gross_price, net_price, limit, sign):
    total = 0.0
    for i in range(len(positive)):
        total += min(limit, max(positive[i] - gross_price - net_price, 0.0))
        total += sign * min(limit, max(negative[i] - gross_price + net_price, 0.0))
    return total


@nb.njit(cache=True)
def _multiplier(positive, negative, limit, target, gross):
    lo = 0.0 if gross else min(-1.0, -negative.max())
    hi = max(1.0, positive.max(), negative.max() if gross else 0.0)
    for _ in range(27):
        mid = (lo + hi) / 2
        if gross:
            value = _volume(positive, negative, mid, 0.0, limit, 1)
        else:
            value = _volume(positive, negative, 0.0, mid, limit, -1)
        if value > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@nb.njit(cache=True)
def _side_multiplier(values, target, limit):
    lo, hi = values.min() - limit, values.max()
    for _ in range(27):
        mid = (lo + hi) / 2
        total = 0.0
        for value in values:
            total += min(limit, max(0.0, value - mid))
        if total > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@nb.njit(cache=True)
def _project(values, holding, short, net_min, net_max, budget, limit):
    result = np.empty_like(values)
    for t in range(len(values)):
        positive = values[t] - holding[t]
        negative = -values[t] - holding[t] - short[t]
        gross_price, net_price = 0.0, 0.0
        if _volume(positive, negative, 0.0, 0.0, limit, 1) > budget:
            gross_price = _multiplier(positive, negative, limit, budget, True)
        net = _volume(positive, negative, gross_price, 0.0, limit, -1)
        if not net_min <= net <= net_max:
            net = min(net_max, max(net_min, net))
            gross_price = 0.0
            net_price = _multiplier(positive, negative, limit, net, False)
            if _volume(positive, negative, 0.0, net_price, limit, 1) > budget:
                long_price = _side_multiplier(positive, (budget + net) / 2, limit)
                short_price = _side_multiplier(negative, (budget - net) / 2, limit)
                gross_price = (long_price + short_price) / 2
                net_price = (long_price - short_price) / 2
        result[t] = np.minimum(limit, np.maximum(positive - gross_price - net_price, 0))
        result[t] -= np.minimum(limit, np.maximum(negative - gross_price + net_price, 0))
    return result


def trajectory_guide(p, start, steps=700, rho=500.0):
    """Use ADMM on risk, constrained positions and successive-day turnover.

    Diagonal shifts make the rounded covariance matrices positive semidefinite.
    The shift is used only to generate this fractional guide.
    """
    identity = np.eye(p.n)
    risk = p.RS.astype(float) / 2
    shifts = np.maximum(0, np.ceil(-np.linalg.eigvalsh(risk)[:, 0]))
    risk += shifts[:, None, None] * identity
    edges = np.any(p.w[1:] != 0, axis=1)
    degree = np.r_[edges, False].astype(int) + np.r_[False, edges]
    factors = []
    inverse = None
    for t in range(p.T):
        matrix = 2 * risk[t] + rho * (1 + degree[t]) * identity
        if t > 0 and edges[t - 1]:
            matrix -= rho**2 * inverse
        factor = cho_factor(matrix, check_finite=False)
        factors.append(factor)
        inverse = cho_solve(factor, identity, check_finite=False)
    positions = start.astype(float)
    position_dual = np.zeros_like(positions)
    differences = np.diff(positions, axis=0)
    turnover_dual = np.zeros_like(differences)
    holding, short, turnover = p.absc / rho, p.sc / rho, p.w[1:] / rho

    for _ in range(steps):
        rhs = rho * (positions - position_dual) - (p.lin + 10)
        edge_rhs = rho * (differences - turnover_dual) * edges[:, None]
        rhs[:-1] -= edge_rhs
        rhs[1:] += edge_rhs
        for t in range(1, p.T):
            if edges[t - 1]:
                rhs[t] += rho * cho_solve(factors[t - 1], rhs[t - 1], check_finite=False)
        for t in range(p.T - 1, -1, -1):
            if t + 1 < p.T and edges[t]:
                rhs[t] += rho * rhs[t + 1]
            rhs[t] = cho_solve(factors[t], rhs[t], check_finite=False)
        relaxed = 1.6 * rhs - 0.6 * positions
        positions = _project(
            relaxed + position_dual, holding, short, p.C - p.maxcash, p.C, p.budget, p.ub
        )
        position_dual += relaxed - positions
        relaxed_diff = 1.6 * np.diff(rhs, axis=0) - 0.6 * differences + turnover_dual
        differences = np.sign(relaxed_diff) * np.maximum(abs(relaxed_diff) - turnover, 0)
        turnover_dual = relaxed_diff - differences
    return positions
