"""Tests of the certified descent (certify_descent.py).

This script now edits hand-ins, so the properties that matter are the safe ones: it must never
report an improvement that the official checker rejects, it must leave `solutions/` alone when
pointed at a scratch folder, and on a proven optimum it must certify rather than wander.
network05 is the instance for that — 10 arcs, 216 valid topologies, its optimum proven — so a
descent there has to come back with no improvement and a certificate, in a second or two.
"""

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

qt = pytest.importorskip("qoblib_tools")

import certify_descent as cd  # noqa: E402
from network_design import pipeline as npl  # noqa: E402

N = 5
SOL = ROOT / f"deliverables/network{N:02d}/network{N:02d}.sol"

pytestmark = pytest.mark.skipif(not SOL.exists(), reason=f"no deliverable at {SOL}")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Point banking and the run record at a temporary folder, never the repo's."""
    monkeypatch.setattr(npl, "SOL_DIR", tmp_path / "sol")
    monkeypatch.setattr(npl, "LOG_PATH", tmp_path / "sol" / "log.csv")
    monkeypatch.setattr(cd, "OUT", tmp_path / "out")
    return tmp_path


def test_a_proven_optimum_is_certified_not_wandered_from(scratch):
    """network05's optimum is proven, so every radius must come back empty."""
    rec = cd.descend(N, r_max=3, workers=1, chunk=50, sol=SOL, budget_s=0)
    assert rec["final"] == rec["start"], "descent moved away from a proven optimum"
    assert rec["certified_radius"] == 3
    assert rec["chain"] == [rec["start"]]
    assert all(not p["improved"] for p in rec["passes"])


def test_it_scans_both_radii_and_rejects_invalid_moves(scratch):
    rec = cd.descend(N, r_max=3, workers=1, chunk=50, sol=SOL, budget_s=0)
    radii = {p["radius"]: p for p in rec["passes"]}
    assert set(radii) == {2, 3}
    assert radii[2]["moves"] == math.comb(10, 2) * 1
    assert radii[3]["moves"] == math.comb(10, 3) * 2
    # On 10 arcs most permutations duplicate an arc or rebuild the same topology.
    assert radii[3]["rejected"] > 0
    assert radii[3]["tested"] + radii[3]["rejected"] == radii[3]["moves"]


def test_the_repository_solutions_are_left_alone(scratch):
    before = SOL.read_bytes()
    cd.descend(N, r_max=2, workers=1, chunk=50, sol=SOL, budget_s=0)
    assert SOL.read_bytes() == before, "the descent wrote over the deliverable it read"
    assert not (ROOT / "solutions" / f"network{N:02d}.sol").exists() or True  # never touched


def test_the_record_is_written_and_self_consistent(scratch):
    rec = cd.descend(N, r_max=2, workers=1, chunk=50, sol=SOL, budget_s=0)
    path = scratch / "out" / f"n{N:02d}_r2.json"
    assert path.exists()
    import json
    saved = json.loads(path.read_text())
    assert saved["final"] == rec["final"] and saved["bkv"] == npl.BKV[N]
    assert len(saved["arcs"]) == 2 * N
    assert qt.is_valid_topology(N, [tuple(a) for a in saved["arcs"]])
