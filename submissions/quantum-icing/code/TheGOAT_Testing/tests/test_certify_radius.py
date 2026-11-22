"""Tests of the radius certificate (certify_radius.py).

The certificate rests on one claim: in a digraph where every node has out-degree 2 and
in-degree 2, two topologies differing in exactly k arcs always differ by a DERANGEMENT of the
heads of those k arcs. If that is false the enumeration misses moves and certifies nothing, so
it is checked here by brute force on network05, where all 216 valid topologies fit in a test.
"""

import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

from certify_radius import apply_perm, derangements  # noqa: E402

N = 5


@pytest.fixture(scope="module")
def topologies():
    """Every valid 2-in / 2-out topology on 5 nodes (216 of them, per notebook 02)."""
    rows = [list(itertools.combinations([j for j in range(N) if j != i], 2)) for i in range(N)]
    out = []
    for choice in itertools.product(*rows):
        arcs = [(i, j) for i, pair in enumerate(choice) for j in pair]
        if qt.is_valid_topology(N, arcs):
            out.append(sorted(arcs))
    return out


# ------------------------------------------------------------------- the counting argument


def test_derangement_counts():
    assert [len(derangements(k)) for k in range(2, 7)] == [1, 2, 9, 44, 265]
    for k in range(2, 7):
        for p in derangements(k):
            assert sorted(p) == list(range(k))
            assert all(p[i] != i for i in range(k))


def test_radius_move_counts_match_the_docstring():
    import math
    assert math.comb(48, 2) * 1 == 1128
    assert math.comb(48, 3) * 2 == 34592
    assert math.comb(48, 4) * 9 == 1751220


# ------------------------------------------------ the claim the certificate depends on


def test_every_nearby_topology_is_a_derangement_of_the_differing_arcs(topologies):
    """Brute force: for every pair at distance k <= 5, some derangement of the k differing
    arcs of one gives the other. A single counterexample would void the certificate."""
    assert len(topologies) == 216
    ref = topologies[0]
    checked = 0
    for other in topologies:
        diff = [i for i, a in enumerate(ref) if a not in set(other)]
        k = len(diff)
        if k == 0 or k > 5:
            continue
        hit = any(apply_perm(ref, diff, p) is not None
                  and sorted(apply_perm(ref, diff, p)) == other
                  for p in derangements(k))
        assert hit, f"topology at distance {k} not reachable by a derangement: {other}"
        checked += 1
    assert checked > 30, f"only {checked} pairs exercised"


def test_the_tail_and_head_multisets_are_preserved(topologies):
    """The reason the claim holds: degrees force removed and added arcs to share endpoints."""
    from collections import Counter

    ref = set(topologies[0])
    for other in topologies[1:60]:
        removed, added = ref - set(other), set(other) - ref
        assert len(removed) == len(added)
        assert Counter(t for t, _ in removed) == Counter(t for t, _ in added)
        assert Counter(h for _, h in removed) == Counter(h for _, h in added)


# ------------------------------------------------------------------------- apply_perm


def test_apply_perm_permutes_only_the_heads():
    arcs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    out = apply_perm(arcs, [0, 2], (1, 0))      # swap the heads of arcs 0 and 2
    assert out is not None
    assert [t for t, _ in out] == [t for t, _ in arcs]
    assert out[0] == (0, 3) and out[2] == (2, 1)


def test_apply_perm_rejects_a_self_loop():
    arcs = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assert apply_perm(arcs, [0, 1], (1, 0)) is None     # would make (0,0) and (1,1)


def test_apply_perm_rejects_a_duplicate_arc():
    arcs = [(0, 1), (2, 3), (0, 3), (2, 1)]
    assert apply_perm(arcs, [0, 1], (1, 0)) is None     # (0,3) and (2,1) already present


def test_apply_perm_rejects_a_move_that_rebuilds_the_same_topology():
    """Two arcs out of the same node: swapping their heads reorders the list, nothing else."""
    arcs = [(0, 1), (0, 2), (1, 0), (2, 0)]
    assert apply_perm(arcs, [0, 1], (1, 0)) is None
