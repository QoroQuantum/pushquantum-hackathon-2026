"""Tests of the instance lower bound (instance_bound.py).

This is the one number in the repo that claims something about *every* topology, so a mistake
would not be a worse result, it would be a false statement in the hand-in. The tests check it
from both ends: against QOBLIB's published values and proven optima (it must never exceed one),
and against real topologies, where the quantity it bounds can be computed exactly.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.sparse.csgraph import shortest_path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

import instance_bound as ib  # noqa: E402
from network_design.pipeline import BKV, PROVEN_OPTIMAL  # noqa: E402


@pytest.mark.parametrize("n", sorted(BKV))
def test_the_bound_never_exceeds_the_published_value(n):
    """On a proven optimum this is exact; on a best-known value it is still necessary."""
    d = qt.load_network_demand(n).astype(float)
    assert ib.bounds(d)["best"] <= BKV[n] + 1e-9


@pytest.mark.parametrize("n", sorted(PROVEN_OPTIMAL))
def test_each_part_of_the_bound_holds_against_a_proven_optimum(n):
    d = qt.load_network_demand(n).astype(float)
    for name, value in ib.bounds(d).items():
        assert value <= BKV[n] + 1e-9, f"{name} exceeds the proven optimum of network{n:02d}"


def test_moore_levels_fill_the_graph_at_doubling_width():
    assert ib.moore_levels(5) == [2, 2]                 # 4 others: 2 at distance 1, 2 at 2
    assert ib.moore_levels(8) == [2, 4, 1]
    assert ib.moore_levels(24) == [2, 4, 8, 9]
    for n in range(5, 25):
        lv = ib.moore_levels(n)
        assert sum(lv) == n - 1
        assert lv[0] == 2 and all(lv[i] <= 2 ** (i + 1) for i in range(len(lv)))


@pytest.mark.parametrize("n", [5, 18, 24])
def test_the_hop_bound_is_below_a_real_topology_s_weighted_hops(n):
    """The quantity bounded in step (1)-(2), computed exactly on our own deliverable."""
    sol = ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    if not sol.exists():
        pytest.skip(f"no deliverable at {sol}")
    d = qt.load_network_demand(n).astype(float)
    _, arcs, _ = qt.read_network_solution(sol)
    arcs = [tuple(map(int, a)) for a in arcs]
    g = sps.csr_matrix((np.ones(len(arcs)), ([i for i, _ in arcs], [j for _, j in arcs])),
                       shape=(n, n))
    dist = shortest_path(g, method="D", directed=True, unweighted=True)
    actual = float(np.where(d > 0, d * dist, 0.0).sum())
    assert ib.hop_lower_bound(d) <= actual + 1e-9
    assert ib.hop_lower_bound(d.T) <= actual + 1e-9


@pytest.mark.parametrize("n", [5, 18, 24])
def test_the_bound_is_below_the_referee_s_score_of_a_real_topology(n):
    """End to end: the bound against the exact LP of a topology that exists."""
    sol = ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    if not sol.exists():
        pytest.skip(f"no deliverable at {sol}")
    d = qt.load_network_demand(n).astype(float)
    _, arcs, _ = qt.read_network_solution(sol)
    z = qt.route_min_congestion(d, [tuple(map(int, a)) for a in arcs]).z_lp
    assert math.isfinite(z)
    assert ib.bounds(d)["best"] <= z + 1e-9


def test_the_bound_is_below_random_topologies_too():
    """Random valid topologies are far worse than the hand-ins, so this is a weak check --
    it exists to catch a bound that is accidentally tied to the polished solutions."""
    n = 5
    d = qt.load_network_demand(n).astype(float)
    b = ib.bounds(d)["best"]
    rng = np.random.default_rng(0)
    seen = 0
    while seen < 10:
        arcs = sorted(set(qt.random_topology(n, rng)))
        if not qt.is_valid_topology(n, arcs):
            continue
        z = qt.route_min_congestion(d, arcs).z_lp
        if math.isfinite(z):
            assert b <= z + 1e-9
            seen += 1
