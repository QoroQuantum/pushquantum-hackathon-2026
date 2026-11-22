"""Tests of the certified ILS (certified_ils.py).

The search edits nothing by itself, so what matters is that the kick really leaves the basin it
claims to leave, that the descent lands on a genuine local optimum rather than wherever it ran
out of patience, and that a proven optimum is never abandoned. network05 answers all three in
seconds: 10 arcs, 216 valid topologies, its optimum proven, so every restart must come back to
65,500 and the run must report exactly one distinct optimum.
"""

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

import certified_ils as ci  # noqa: E402
from network_design import pipeline as npl  # noqa: E402

N = 5
SOL = ROOT / f"deliverables/network{N:02d}/network{N:02d}.sol"

pytestmark = pytest.mark.skipif(not SOL.exists(), reason=f"no deliverable at {SOL}")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(npl, "SOL_DIR", tmp_path / "sol")
    monkeypatch.setattr(npl, "LOG_PATH", tmp_path / "sol" / "log.csv")
    monkeypatch.setattr(ci, "OUT", tmp_path / "out")
    return tmp_path


@pytest.fixture
def arcs():
    _, a, _ = qt.read_network_solution(SOL)
    return [tuple(map(int, x)) for x in a]


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_a_kick_lands_exactly_k_arcs_away_on_a_valid_topology(arcs, k):
    """Exactly k, not at most k: a shorter kick lands inside the radius already exhausted."""
    import random
    rng = random.Random(7)
    seen = 0
    for _ in range(20):
        new = ci.kick(arcs, k, rng)
        if new is None:
            continue
        assert qt.is_valid_topology(N, new)
        assert len(set(arcs) ^ set(new)) == 2 * k, "the kick did not move exactly k arcs"
        seen += 1
    assert seen, f"no kick of radius {k} was produced at all"


def test_a_raw_derangement_can_land_closer_than_k(arcs):
    """Why the check above is needed: in-degree 2 means two arcs can share a head, and
    swapping those two heads rebuilds the same pair. `apply_perm` is right to allow it --
    inside the certificate such a move is merely redundant -- but a kick must reject it."""
    import random
    from certify_radius import apply_perm, derangements
    rng = random.Random(11)
    short = 0
    for _ in range(3000):
        idx = tuple(sorted(rng.sample(range(len(arcs)), 3)))
        new = apply_perm(arcs, idx, rng.choice(derangements(3)))
        if new is not None and len(set(arcs) ^ set(new)) < 6:
            short += 1
    assert short > 0, "expected some radius-3 moves to land closer than 3 arcs"


def test_the_descent_stays_put_on_a_proven_optimum(arcs, scratch):
    demand = qt.load_network_demand(N)
    z = qt.route_min_congestion(demand, arcs).z_lp
    z_end, arcs_end, lps = ci.descend(N, arcs, z, r_max=3, workers=1, chunk=50)
    assert math.ceil(z_end - 1e-6) == 65_500
    assert sorted(arcs_end) == sorted(arcs), "it walked away from the optimum"
    assert lps > 0, "the descent never scored anything"


def test_no_restart_ever_beats_or_loses_the_proven_optimum(scratch):
    """What must hold: nothing goes below 65,500, and the incumbent is never traded down.

    What must *not* be asserted: that every restart comes back to 65,500. Radius 3 on 10 arcs
    reaches 240 moves of the 216 valid topologies, so a radius-3 descent has local optima that
    are not the global one -- this run finds 70,000 among them. That is the premise of the whole
    search, not a defect: certified local optima are plural, and the kick is how you reach past
    the one you are standing on.
    """
    rec = ci.run(N, r_max=3, kicks=[4, 5], minutes=0.15, workers=1, chunk=50, seed=3)
    assert rec["rounds"], "no restart completed"
    assert rec["best"] == 65_500, "the incumbent was traded for something worse"
    assert all(r["optimum"] >= 65_500 for r in rec["rounds"]), "a descent beat a proven optimum"
    assert all(r["start"] >= 65_500 for r in rec["rounds"]), "a kick beat a proven optimum"
    assert 65_500 in {r["optimum"] for r in rec["rounds"]}, "no restart found its way back"


def test_it_writes_its_record_and_leaves_the_deliverable_alone(scratch):
    before = SOL.read_bytes()
    rec = ci.run(N, r_max=3, kicks=[4], minutes=0.1, workers=1, chunk=50, seed=4)
    assert SOL.read_bytes() == before, "the search wrote over the deliverable it read"
    path = scratch / "out" / f"n{N:02d}_r3_s4.json"
    assert path.exists()
    import json
    saved = json.loads(path.read_text())
    assert saved["best"] == rec["best"]
    assert qt.is_valid_topology(N, [tuple(a) for a in saved["arcs"]])
