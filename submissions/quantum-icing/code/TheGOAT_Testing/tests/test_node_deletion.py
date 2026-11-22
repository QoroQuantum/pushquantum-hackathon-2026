"""Tests of the node deletion that re-derives the chain (node_deletion.py).

Two things are worth pinning. First the premise: the whole descending chain is only meaningful
because the QOBLIB demand matrices are exactly nested, so network(n) really is network(n+1)
without its last node. Second the mechanics: a deletion must leave a *valid* topology, which
means the degree repair has to be right, and deleting the last node must leave every surviving
arc at the label it already had.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

from node_deletion import deletions  # noqa: E402

INSTANCES = [18, 19, 20, 21, 22, 23, 24]


@pytest.mark.parametrize("n", INSTANCES[:-1])
def test_the_demand_matrices_are_exactly_nested(n):
    """The premise of the chain: network(n) is network(n+1) with the last row and column cut."""
    assert np.array_equal(qt.load_network_demand(n), qt.load_network_demand(n + 1)[:n, :n])


@pytest.fixture(scope="module")
def parent():
    sol = ROOT / "deliverables/network18/network18.sol"
    if not sol.exists():
        pytest.skip(f"no deliverable at {sol}")
    _, arcs, _ = qt.read_network_solution(sol)
    return sorted({tuple(map(int, a)) for a in arcs})


def test_every_deletion_is_a_valid_topology_one_node_smaller(parent):
    out = deletions(parent, 18)
    assert out, "no deletion survived the repair"
    for c in out:
        arcs = [tuple(a) for a in c["arcs"]]
        assert qt.is_valid_topology(17, arcs), f"deleting node {c['node']} left an invalid topology"
        assert len(arcs) == 2 * 17
        assert max(max(a) for a in arcs) == 16, "labels should be 0..16 after relabelling"


def test_each_node_offers_at_most_the_two_repairs(parent):
    out = deletions(parent, 18)
    per_node = {}
    for c in out:
        per_node.setdefault(c["node"], []).append(tuple(c["repair"]))
    assert set(per_node) <= set(range(18))
    for node, repairs in per_node.items():
        assert len(repairs) == len(set(repairs)) <= 2, f"node {node} repeated a repair"
        assert set(repairs) <= {(0, 1), (1, 0)}


def test_deleting_the_last_node_keeps_the_other_arcs_where_they_were(parent):
    """With the identity relabelling, only the repair arcs are new."""
    out = [c for c in deletions(parent, 18) if c["node"] == 17]
    assert out, "the chain's own deletion produced nothing"
    survivors = {a for a in parent if 17 not in a}
    for c in out:
        arcs = {tuple(a) for a in c["arcs"]}
        assert survivors <= arcs, "an untouched arc moved or vanished"
        added = arcs - survivors
        assert len(added) == 2 and added == {tuple(a) for a in c["added"]}


def test_the_added_arcs_carry_the_deficient_tails_and_heads(parent):
    """The repair must join exactly the nodes the deleted node was serving."""
    for c in deletions(parent, 18):
        v = c["node"]
        want_tails = sorted(i for i, j in parent if j == v)
        want_heads = sorted(j for i, j in parent if i == v)
        shift = lambda x: x - 1 if x > v else x  # noqa: E731
        got_tails = sorted(i for i, j in c["added"])
        got_heads = sorted(j for i, j in c["added"])
        assert got_tails == sorted(shift(x) for x in want_tails)
        assert got_heads == sorted(shift(x) for x in want_heads)
