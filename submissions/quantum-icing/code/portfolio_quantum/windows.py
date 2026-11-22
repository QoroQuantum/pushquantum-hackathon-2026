"""Window (neighbourhood) generators for the portfolio quantum LNS.

Every generator returns a list of unit moves (t, i, d).  Single-day windows may carry both
directions for an asset (2 qubits); multi-day windows carry one move per slot, so that the
move-selection QUBO stays exact (see qlns_core.build_qubo).
"""
from __future__ import annotations

import numpy as np


def _both(s, t, i):
    x0 = int(s.v[t, i])
    return [(t, int(i), d) for d in (1, -1) if abs(x0 + d) <= s.p.ub]


def single_gain(s, t):
    """Exact dE of each +1 / -1 move on day t alone (inf where not allowed)."""
    p, v = s.p, s.v
    n = p.n
    g = np.full((n, 2), np.inf)
    for i in range(n):
        x0 = int(v[t, i])
        for k, d in enumerate((1, -1)):
            if abs(x0 + d) > p.ub:
                continue
            c = d * s.G[t, i] + p.R[t, i, i] + 10 * d
            c += p.lin[t, i] * d + p.sc[t, i] * (max(-(x0 + d), 0) - max(-x0, 0))
            c += p.absc[t, i] * (abs(x0 + d) - abs(x0))
            if t > 0:
                c += p.w[t, i] * (abs(x0 + d - v[t - 1, i]) - abs(x0 - v[t - 1, i]))
            if t + 1 < p.T:
                c += p.w[t + 1, i] * (abs(v[t + 1, i] - x0 - d) - abs(v[t + 1, i] - x0))
            g[i, k] = c
    return g


def corr_matrix(p):
    """|correlation| per day from the risk tables (lambda > 0), or None."""
    d = np.sqrt(np.maximum(np.einsum("tii->ti", p.R).astype(float), 1.0))
    if not p.R.any():
        return None
    return np.abs(p.R / (d[:, :, None] * d[:, None, :]))


def day_window(s, rng, K, corr=None, t=None, mode=None):
    """K assets on one day, both directions each (about 2K qubits)."""
    p = s.p
    T, n = p.T, p.n
    if t is None:
        t = int(rng.integers(0, T - 1))  # the last day is always empty in an optimum
    held = np.flatnonzero(s.v[t] != 0)
    g = single_gain(s, t).min(axis=1)
    mode = mode or rng.choice(["corr", "grad", "held", "rand"], p=[0.4, 0.25, 0.2, 0.15])
    chosen: list[int] = []
    if mode == "corr" and corr is not None:
        pool = held if len(held) and rng.random() < 0.5 else np.arange(n)
        pivot = int(rng.choice(pool))
        order = np.argsort(-corr[t, pivot] + 1e-3 * rng.random(n))
        chosen = [pivot] + [int(j) for j in order if j != pivot][: K - 1]
    elif mode == "grad":
        k1 = min(len(held), K // 2)
        chosen = [int(j) for j in rng.choice(held, size=k1, replace=False)] if k1 else []
        order = np.argsort(g + 50.0 * rng.random(n))
        chosen += [int(j) for j in order if j not in chosen][: K - len(chosen)]
    elif mode == "held" and len(held):
        k1 = min(len(held), K)
        chosen = [int(j) for j in rng.choice(held, size=k1, replace=False)]
        rest = [j for j in rng.permutation(n) if j not in chosen]
        chosen += [int(j) for j in rest][: K - len(chosen)]
    else:
        chosen = [int(j) for j in rng.choice(n, size=K, replace=False)]
    moves = []
    for i in chosen[:K]:
        moves += _both(s, t, i)
    return moves, f"day:{mode}"


def block_window(s, rng, K, D, corr=None):
    """K assets over D consecutive days, one move per slot (direction from the single-move gain)."""
    p = s.p
    T, n = p.T, p.n
    t0 = int(rng.integers(0, max(1, T - D)))
    days = list(range(t0, min(t0 + D, T - 1)))
    if corr is not None and rng.random() < 0.6:
        pivot = int(rng.integers(n))
        order = np.argsort(-corr[days[0], pivot] + 1e-3 * rng.random(n))
        chosen = [pivot] + [int(j) for j in order if j != pivot][: K - 1]
    else:
        chosen = [int(j) for j in rng.choice(n, size=K, replace=False)]
    moves = []
    for t in days:
        g = single_gain(s, t)
        for i in chosen:
            opts = [(k, d) for k, d in enumerate((1, -1)) if np.isfinite(g[i, k])]
            if not opts:
                continue
            if len(opts) == 2 and rng.random() < 0.7:
                k, d = opts[int(np.argmin([g[i, kk] for kk, _ in opts]))]
            else:
                k, d = opts[int(rng.integers(len(opts)))]
            moves.append((t, int(i), int(d)))
    return moves, f"block{len(days)}"
