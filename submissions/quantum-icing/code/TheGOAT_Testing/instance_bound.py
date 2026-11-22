"""A lower bound on the *instance*, not on a topology: how far from optimal can a hand-in be?

Everything else here bounds one topology. This bounds the whole problem: no 2-in/2-out topology
on n nodes, however it is found, can score below this number. It costs microseconds and it is
the only statement in the repo that limits the optimum from below rather than from above.

Three facts compose it.

1. Every unit of demand from i to j traverses at least `dist(i, j)` arcs, so the total load on
   all arcs is at least `sum_ij d_ij dist(i, j)`.
2. Out-degree 2 caps how near the rest of the graph can be: at most 2 nodes lie at distance 1
   from i, at most 4 more at distance 2, 8 at distance 3, and so on -- the Moore bound for
   directed graphs. For a fixed multiset of distances the sum in (1) is smallest when the
   largest demands take the shortest distances (rearrangement), so sorting row i of the demand
   matrix downwards and assigning it to 1, 1, 2, 2, 2, 2, 3, ... gives a lower bound on
   `sum_j d_ij dist(i, j)` that holds for *every* topology. In-degree 2 gives the same argument
   per destination, i.e. on the columns, and the two are both valid, so take the larger.
3. The objective is the *maximum* arc load, which is at least the average: total load / 2n.

A fourth, independent bound: all demand out of node i leaves on its 2 arcs, so the busiest arc
carries at least `sum_j d_ij / 2`, and likewise for the 2 arcs into a node. On these instances
it is weaker than (1)-(3) but it is free, and the maximum of valid bounds is valid.

What the numbers say, and the honest reading of them. The bound holds on all twenty instances
with a published value, and QOBLIB proves the optimum for n = 5..10, so its looseness can be
measured where the truth is known: it lands 4%, 1%, 7%, 10%, 12% and 18% below the optimum at
n = 5, 6, 7, 8, 9, 10 -- tight at the bottom, decaying smoothly as n grows, because both the
Moore bound and the max-versus-average step weaken with n. From n = 11 it flattens at 0.74-0.77
of the best-known value and stays there to n = 24.

That flatness is informative but it is not a proof of near-optimality. It says the published
best-known values across 11-24 sit at one uniform distance from this bound, so the large
instances are not a different regime from the small ones; it cannot tell a uniformly good family
from a uniformly bad one. The rigorous claim is the weaker one: **every network hand-in is
provably within 28.5% to 32.3% of the true optimum** -- and `network05`, 4.0% above the bound,
is in fact optimal, which is exactly how much room the bound leaves at n = 5.

    venv/bin/python TheGOAT_Testing/instance_bound.py
    venv/bin/python TheGOAT_Testing/instance_bound.py --n 24 --verbose
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import qoblib_tools as qt  # noqa: E402
from qoblib_tools import INTSCALE  # noqa: E402
from network_design.pipeline import BKV, PROVEN_OPTIMAL  # noqa: E402

OUT = Path(os.environ.get("GOAT_RUN_DIR", ROOT / "runs/goat")) / "instancebound"


def moore_levels(n: int, degree: int = 2) -> list[int]:
    """How many nodes can sit at distance 1, 2, 3, ... from one node at this out-degree."""
    levels, reach, width = [], 0, degree
    while reach < n - 1:
        levels.append(min(width, n - 1 - reach))
        reach += levels[-1]
        width *= degree
    return levels


def hop_lower_bound(demand: np.ndarray, degree: int = 2) -> float:
    """Lower bound on ``sum_ij d_ij dist(i,j)`` over every topology of this degree.

    Rows use out-degree, columns use in-degree; both are valid, so the caller takes the max.
    """
    n = len(demand)
    levels = moore_levels(n, degree)
    total = 0.0
    off = ~np.eye(n, dtype=bool)
    for i in range(n):
        row = np.sort(demand[i][off[i]])[::-1]        # biggest demand to the nearest level
        pos = 0
        for dist, width in enumerate(levels, start=1):
            total += dist * float(row[pos:pos + width].sum())
            pos += width
    return total


def bounds(demand: np.ndarray, degree: int = 2) -> dict[str, float]:
    """Every bound, in QOBLIB objective units. The max of them is the one to quote."""
    n = len(demand)
    n_arcs = degree * n
    d = np.asarray(demand, dtype=float)
    out = {
        "hops_by_source": INTSCALE * hop_lower_bound(d, degree) / n_arcs,
        "hops_by_target": INTSCALE * hop_lower_bound(d.T, degree) / n_arcs,
        "node_out": INTSCALE * float(d.sum(axis=1).max()) / degree,
        "node_in": INTSCALE * float(d.sum(axis=0).max()) / degree,
    }
    out["best"] = max(out.values())
    return out


def hand_in(n: int, demand: np.ndarray) -> int | None:
    """The checked objective of our own deliverable for this instance, if there is one."""
    sol = ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    if not sol.exists():
        return None
    _, arcs, _ = qt.read_network_solution(sol)
    z = qt.route_min_congestion(demand, [tuple(map(int, a)) for a in arcs]).z_lp
    return math.ceil(z - 1e-6) if math.isfinite(z) else None


def table(instances, verbose=False) -> dict:
    rows = []
    print(f"{'n':>3} {'reference':>10} {'proven':>7} {'bound':>10} {'bound/ref':>10} "
          f"{'hand-in':>10} {'provable gap':>13}")
    for n in instances:
        if n not in BKV:
            continue
        d = qt.load_network_demand(n).astype(float)
        b = bounds(d)
        ref = BKV[n]
        assert b["best"] <= ref + 1e-9, (
            f"network{n:02d}: the bound {b['best']:.0f} exceeds the published {ref}, "
            "so one of the three steps is wrong")
        ours = hand_in(n, d)
        assert ours is None or b["best"] <= ours + 1e-9, (
            f"network{n:02d}: the bound {b['best']:.0f} exceeds our own {ours}")
        rows.append({"n": n, "reference": ref, "proven": n in PROVEN_OPTIMAL,
                     "bound": math.floor(b["best"]), "ratio": b["best"] / ref,
                     "hand_in": ours,
                     "provable_gap": None if ours is None else ours / b["best"] - 1,
                     "parts": {k: v for k, v in b.items() if k != "best"}})
        print(f"{n:>3} {ref:>10,} {'yes' if n in PROVEN_OPTIMAL else '-':>7} "
              f"{math.floor(b['best']):>10,} {b['best'] / ref:>10.3f} "
              f"{'-' if ours is None else format(ours, ',') :>10} "
              f"{'-' if ours is None else format(ours / b['best'] - 1, '.1%'):>13}")
        if verbose:
            for k, v in b.items():
                if k != "best":
                    print(f"      {k:>16}: {v:>12,.0f}")

    proven = [r for r in rows if r["proven"]]
    rest = [r for r in rows if not r["proven"]]
    ours = [r for r in rows if r["provable_gap"] is not None]
    rec = {"rows": rows,
           "our_gap_range": [min(r["provable_gap"] for r in ours),
                             max(r["provable_gap"] for r in ours)] if ours else [],
           "proven_ratio_range": [min(r["ratio"] for r in proven),
                                  max(r["ratio"] for r in proven)] if proven else [],
           "unproven_ratio_range": [min(r["ratio"] for r in rest),
                                    max(r["ratio"] for r in rest)] if rest else []}
    if proven:
        print(f"\nwhere QOBLIB proves the optimum (n = {proven[0]['n']}..{proven[-1]['n']}): "
              f"the bound is {rec['proven_ratio_range'][0]:.3f} to "
              f"{rec['proven_ratio_range'][1]:.3f} of the truth")
    if rest:
        print(f"where it does not (n = {rest[0]['n']}..{rest[-1]['n']}): "
              f"{rec['unproven_ratio_range'][0]:.3f} to {rec['unproven_ratio_range'][1]:.3f} "
              f"of the best known value")
    if ours:
        lo, hi = rec["our_gap_range"]
        print(f"our own hand-ins are provably within {lo:.1%} to {hi:.1%} of the true optimum "
              f"({len(ours)} instances)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bounds.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, nargs="+", default=list(range(5, 25)))
    p.add_argument("--verbose", action="store_true", help="show each bound separately")
    a = p.parse_args()
    table(a.n, a.verbose)


if __name__ == "__main__":
    main()
