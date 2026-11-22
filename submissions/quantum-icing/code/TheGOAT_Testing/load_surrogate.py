"""A cheap surrogate that models the referee's objective instead of a bound on it.

Every surrogate used so far (`hops`, `bound`, the team sub-QUBO) is the weak-duality bound
sum_ij d_ij dist_w(i,j) / sum_a w_a: a SUM over paths. The referee minimises a MAX over links.
That difference is the whole remaining margin on network24: the hand-in's score is 6.7% above
its own hop bound and 23 of its 48 links sit at the maximum load, so a surrogate that cannot
see imbalance cannot see what separates one good topology from another. Measured: the hop
count ranks block rewirings at rho 0.85-0.89, but among already-good topologies it collapses
to 0.17 (hop_search.py).

This module routes the demand and takes the maximum link load, which is what the LP reports.
Routing is the expensive part of the LP, so it is approximated:

* `sp_loads`  all-pairs shortest paths under arc weights, demand split evenly over every
  shortest path (the usual ECMP rule), loads accumulated per arc. One call is ~1-3 ms.
* `max_load`  iterates that, raising the weight of loaded arcs so the next routing avoids
  them, and keeps the best maximum seen. This is a congestion-aware heuristic routing, so
  its maximum is a FEASIBLE routing's maximum and therefore an UPPER bound on the LP, where
  the hop bound is a lower one. Six iterations cost ~10 ms against ~120 ms for the exact LP.

Being an upper bound matters for honesty: a topology whose surrogate says 660 really does
admit a routing with maximum load 660. Nothing here replaces the referee; it ranks candidates
before spending exact LPs on them.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path

SCALE = 1000.0          # qoblib_tools.INTSCALE: the referee reports max load * 1000


def sp_loads(demand: np.ndarray, tail: np.ndarray, head: np.ndarray,
             weight: np.ndarray) -> np.ndarray:
    """Per-arc load when every demand splits evenly over the shortest paths under `weight`.

    Returns an array of len(tail); all-inf if some demand has no path.
    """
    n = len(demand)
    g = csr_array((weight, (tail, head)), shape=(n, n))
    dist = shortest_path(g, method="D", directed=True)
    if not np.isfinite(dist[demand > 0]).all():
        return np.full(len(tail), np.inf)

    load = np.zeros(len(tail))
    for dst in range(n):
        d = dist[:, dst]                              # distance of every node to dst
        # An arc is on a shortest path to dst when it closes the triangle exactly.
        on = np.isclose(d[tail], weight + d[head], rtol=0, atol=1e-9) & (tail != dst)
        if not on.any():
            continue
        idx = np.flatnonzero(on)
        # Even split: each node divides what it carries among its own shortest next hops.
        share = np.zeros(len(idx))
        for node in np.unique(tail[idx]):
            here = tail[idx] == node
            share[here] = 1.0 / here.sum()
        # Push flow in order of decreasing distance: a node is settled after everything
        # further from dst has already pushed into it.
        carried = demand[:, dst].astype(float).copy()
        carried[dst] = 0.0
        for node in np.argsort(-d):
            if node == dst or not np.isfinite(d[node]) or carried[node] <= 0:
                continue
            here = idx[tail[idx] == node]
            if len(here) == 0:
                continue
            flow = carried[node] * share[np.searchsorted(idx, here)]
            load[here] += flow
            np.add.at(carried, head[here], flow)
            carried[node] = 0.0
    return load


def max_load(demand: np.ndarray, arcs, iters: int = 6, alpha: float = 1.0) -> float:
    """Best maximum link load found by congestion-aware shortest-path routing.

    An upper bound on the LP optimum: every iteration is a feasible routing. `iters=1` is plain
    ECMP on hop count.
    """
    tail, head = np.asarray(arcs, dtype=int).T
    w = np.ones(len(tail))
    best = np.inf
    for _ in range(max(1, iters)):
        load = sp_loads(demand, tail, head, w)
        if not np.isfinite(load).all():
            return np.inf
        best = min(best, float(load.max()))
        peak = load.max()
        if peak <= 0:
            break
        # Price the congested arcs so the next routing walks around them.
        w = 1.0 + alpha * (load / peak) ** 2
    return best * SCALE


def load_profile(demand: np.ndarray, arcs, iters: int = 6) -> dict:
    """max load, how many arcs sit within 1% of it, and the mean load: the imbalance picture."""
    tail, head = np.asarray(arcs, dtype=int).T
    w = np.ones(len(tail))
    best, best_load = np.inf, None
    for _ in range(max(1, iters)):
        load = sp_loads(demand, tail, head, w)
        if not np.isfinite(load).all():
            return {"max": np.inf}
        if load.max() < best:
            best, best_load = float(load.max()), load
        w = 1.0 + (load / max(load.max(), 1e-12)) ** 2
    return {"max": best * SCALE, "mean": float(best_load.mean()) * SCALE,
            "n_at_max": int((best_load >= 0.99 * best).sum()), "n_arcs": len(tail),
            "imbalance": float(best / max(best_load.mean(), 1e-12))}
