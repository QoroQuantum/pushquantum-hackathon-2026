"""Tests of the weak-duality bound (bound_prune.py).

The bound is the one piece of the certificate machinery that could silently make a *proof*
wrong: `certify_*` may only skip an LP where the bound already reaches the incumbent, so if the
bound ever exceeded the true objective the certificate would refute a topology that is in fact
better. These tests pin the two facts that make it safe.

    bound(arcs, w) <= z_LP(arcs)   for every w >= 0        (validity, the one that matters)
    bound(arcs, w*) == z_LP(arcs)  for the LP's own duals   (strong duality, i.e. w* is right)

network05 is the instance: 5 nodes, 10 arcs, an LP in milliseconds, so random topologies can be
tested in bulk.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

import bound_prune as bp  # noqa: E402

N = 5
DEMAND = qt.load_network_demand(N)


def random_tops(count, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        arcs = sorted(set(qt.random_topology(N, rng)))
        if qt.is_valid_topology(N, arcs):
            out.append(arcs)
    return out


def test_strong_duality_holds_at_the_topology_the_duals_came_from():
    for arcs in random_tops(8, seed=1):
        z_lp, w = bp.dual_weights(DEMAND, arcs)
        if not math.isfinite(z_lp):
            continue
        assert w.min() >= -1e-9, "a capacity dual came back negative"
        assert abs(w.sum() - 1.0) < 1e-6, "the capacity duals should sum to 1"
        assert abs(bp.flow_bound(DEMAND, arcs, w) / z_lp - 1.0) < 1e-6


def test_the_bound_never_exceeds_the_lp_for_any_weights():
    """The property the certificate rests on, over random topologies and random weights."""
    rng = np.random.default_rng(2)
    for arcs in random_tops(10, seed=2):
        z_lp = qt.route_min_congestion(DEMAND, arcs).z_lp
        if not math.isfinite(z_lp):
            continue
        weights = [np.ones(len(arcs)),
                   rng.random(len(arcs)),
                   rng.random(len(arcs)) ** 4,                 # very uneven
                   (rng.random(len(arcs)) < 0.3).astype(float)]  # mostly zeros
        for w in weights:
            if w.sum() <= 0:
                continue
            assert bp.flow_bound(DEMAND, arcs, w) <= z_lp + 1e-6


def test_the_duals_of_one_topology_still_bound_a_different_one():
    """How the certificate would use it: weights from the incumbent, applied to a candidate."""
    tops = random_tops(6, seed=3)
    _, w_star = bp.dual_weights(DEMAND, tops[0])
    index = {a: i for i, a in enumerate(tops[0])}
    for cand in tops[1:]:
        z_lp = qt.route_min_congestion(DEMAND, cand).z_lp
        if not math.isfinite(z_lp):
            continue
        for fill in (0.0, float(w_star.max())):
            w = np.array([w_star[index[a]] if a in index else fill for a in cand])
            if w.sum() <= 0:
                continue
            assert bp.flow_bound(DEMAND, cand, w) <= z_lp + 1e-6


def test_an_unroutable_topology_bounds_to_infinity():
    """Nothing reaches node 0, so some demand has no path: bound and LP must both be infinite.

    The bound has to agree with the LP here, not merely stay below it: a finite bound on an
    infeasible candidate would be valid but would send it to the LP, and an infinite one lets
    the certificate discard it for free.
    """
    arcs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 1), (4, 1), (4, 2)]
    assert len(set(arcs)) == len(arcs) and all(i != j for i, j in arcs)
    assert sum(1 for _, j in arcs if j == 0) == 0, "node 0 should be unreachable"
    assert DEMAND[:, 0].sum() > 0, "the instance must actually want to reach node 0"
    assert bp.flow_bound(DEMAND, arcs, np.ones(len(arcs))) == math.inf
    assert not math.isfinite(qt.route_min_congestion(DEMAND, arcs).z_lp)


def test_zero_weights_give_a_vacuous_but_valid_bound():
    arcs = random_tops(1, seed=4)[0]
    assert bp.flow_bound(DEMAND, arcs, np.zeros(len(arcs))) == 0.0
