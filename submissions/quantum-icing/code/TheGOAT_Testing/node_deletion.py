"""Shrink network(n+1) to network(n) by deleting a node, every node and every degree repair.

`node_insertion.py` goes up; this goes down, the direction the team's chain ran (network23 ->
network18, "delete the last node and descend again"). The QOBLIB demand matrices are exactly
nested -- `load_network_demand(n)` is `load_network_demand(n+1)[:n, :n]`, verified -- so
deleting the *last* node is the one deletion that keeps every remaining node's demand attached
to the same label. That is the deletion the chain used, and the only one it ever tried.

Deleting any other node is still legal: relabelling the survivors in increasing order gives a
valid 2-in/2-out topology on n nodes, which the referee will happily score against network(n)'s
own demand. The demands no longer line up with the structure, so these starts should be worse
-- but they are starts in *different basins*, and the chain never looked at them. This script
scores all of them with the exact LP so the question is answered by measurement.

Deleting node v removes its two out-arcs (v -> b1, v -> b2) and its two in-arcs (a1 -> v,
a2 -> v). The survivors a1, a2 are then one out-arc short and b1, b2 one in-arc short, so the
repair is a perfect matching between the two pairs: (a1 -> b1, a2 -> b2) or (a1 -> b2,
a2 -> b1). A repair is dropped when it would make a self-loop or duplicate an existing arc.

    venv/bin/python TheGOAT_Testing/node_deletion.py --n 21 --from deliverables/network22/network22.sol
    venv/bin/python TheGOAT_Testing/node_deletion.py --chain 18 19 20 21 22 23
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import qoblib_tools as qt  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402

OUT = Path(os.environ.get("GOAT_RUN_DIR", ROOT / "runs/goat")) / "deletion"


def deletions(arcs: list[tuple[int, int]], n_parent: int) -> list[dict]:
    """Every (node, repair) deletion of `arcs`, relabelled onto 0..n_parent-2."""
    out = []
    for v in range(n_parent):
        kept = [a for a in arcs if v not in a]
        tails = [i for i, j in arcs if j == v]      # lost an out-arc
        heads = [j for i, j in arcs if i == v]      # lost an in-arc
        if len(tails) != 2 or len(heads) != 2:
            continue                                 # not 2-in/2-out: nothing to repair
        relabel = lambda x: x - 1 if x > v else x    # noqa: E731
        for perm in ((0, 1), (1, 0)):
            new = list(kept) + [(tails[0], heads[perm[0]]), (tails[1], heads[perm[1]])]
            if any(i == j for i, j in new) or len(set(new)) != len(new):
                continue
            relabelled = sorted((relabel(i), relabel(j)) for i, j in new)
            if not qt.is_valid_topology(n_parent - 1, relabelled):
                continue
            out.append({"node": v, "repair": list(perm), "arcs": relabelled,
                        "added": [[relabel(tails[k]), relabel(heads[perm[k]])] for k in (0, 1)]})
    return out


def run(n: int, parent_sol: Path, keep: int = 5, bank: bool = False) -> dict:
    """Score every deletion of `parent_sol` as a network(n) topology."""
    demand = qt.load_network_demand(n)
    _, p_arcs, _ = qt.read_network_solution(parent_sol)
    p_arcs = sorted({tuple(map(int, a)) for a in p_arcs})
    n_parent = 1 + max(max(a) for a in p_arcs)
    cands = deletions(p_arcs, n_parent)

    incumbent_sol = ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    incumbent = None
    if incumbent_sol.exists():
        _, i_arcs, _ = qt.read_network_solution(incumbent_sol)
        incumbent = math.ceil(qt.route_min_congestion(
            demand, [tuple(map(int, a)) for a in i_arcs]).z_lp - 1e-6)

    t0 = time.time()
    scored = []
    for c in cands:
        z = qt.route_min_congestion(demand, [tuple(a) for a in c["arcs"]]).z_lp
        scored.append({**c, "z": z, "score": math.ceil(z - 1e-6) if math.isfinite(z) else None})
    scored.sort(key=lambda r: (r["z"], r["node"]))
    routable = [r for r in scored if r["score"] is not None]

    last = n_parent - 1
    chain_choice = next((r for r in routable if r["node"] == last), None)
    best = routable[0] if routable else None
    rec = {"n": n, "parent": str(parent_sol), "n_parent": n_parent,
           "candidates": len(cands), "routable": len(routable),
           "incumbent": incumbent, "bkv": BKV[n],
           "seconds": round(time.time() - t0, 1),
           "best": [{k: r[k] for k in ("node", "repair", "score", "added", "arcs")}
                    for r in routable[:keep]],
           "chain_choice": None if chain_choice is None else chain_choice["score"],
           "quantiles": [float(np.quantile([r["z"] for r in routable], q))
                         for q in (0.0, 0.25, 0.5, 1.0)] if routable else []}

    print(f"network{n:02d} from {parent_sol.name} ({n_parent} nodes): "
          f"{len(cands)} deletions, {len(routable)} routable, {rec['seconds']}s")
    if incumbent is not None:
        print(f"  current hand-in {incumbent:,} (gap {(incumbent - BKV[n]) / BKV[n]:+.3%})")
    if chain_choice is not None:
        print(f"  the chain's own choice (delete node {last}): {chain_choice['score']:,} "
              f"({(chain_choice['score'] / incumbent - 1) * 100:+.1f}% vs the hand-in)")
    for r in routable[:keep]:
        mark = "  <- the chain's node" if r["node"] == last else ""
        rel = "" if incumbent is None else f" ({(r['score'] / incumbent - 1) * 100:+.1f}%)"
        print(f"  delete {r['node']:>2} repair {r['repair']}: {r['score']:>9,}{rel}{mark}")
    if bank and best is not None:
        banked, checked, msg = npl.bank(demand=demand, n=n, arcs=[tuple(a) for a in best["arcs"]],
                                        method=f"node_deletion from {parent_sol.name}")
        rec["banked"] = {"banked": banked, "checked": checked, "message": msg,
                         "path": str(npl.banked_path(n))}
        print(f"  bank -> {msg}  ({npl.banked_path(n)})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"n{n:02d}_from_n{n_parent:02d}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, help="the instance to build")
    p.add_argument("--from", dest="parent", type=Path, help="the network(n+1) solution to shrink")
    p.add_argument("--chain", type=int, nargs="+", help="run for each n, parent = deliverable n+1")
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--bank", action="store_true",
                   help="write the best deletion into NETWORK_SOL_DIR, checker-validated")
    a = p.parse_args()
    if a.chain:
        for n in a.chain:
            run(n, ROOT / f"deliverables/network{n + 1:02d}/network{n + 1:02d}.sol", a.keep, a.bank)
    else:
        run(a.n, a.parent or ROOT / f"deliverables/network{a.n + 1:02d}/network{a.n + 1:02d}.sol",
            a.keep, a.bank)


if __name__ == "__main__":
    main()
