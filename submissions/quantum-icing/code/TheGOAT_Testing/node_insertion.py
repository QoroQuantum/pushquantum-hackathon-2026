"""Grow network(n) from a strong network(n-1) topology by inserting the missing node.

The team built network18-23 downwards, deleting the last node and descending again; network23
reached 639,739 (-2.98% vs best known) while the network24 hand-in is at 657,121 (-0.99%). This
goes the other way: insert node n-1 into the network(n-1) topology in every degree-preserving
way, remove two arcs (a -> b), (c -> d) and add (a -> v), (v -> b), (c -> v), (v -> d), so that
every node, the new one included, has 2 in / 2 out. Every insertion is scored with the exact LP
and the hop count, and the best ones are written as starting points for a descent.

    venv/bin/python TheGOAT_Testing/node_insertion.py --n 24 --from deliverables/network23/network23.sol
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import qoblib_tools as qt  # noqa: E402
from diagnostics import rank_agreement  # noqa: E402
from goat_swaps import hop_cost  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402
from network_design.sa_search import lp_eval  # noqa: E402

OUT = ROOT / "runs" / "goat" / "insertion"
_D = None


def _init(d):
    global _D
    _D = d


def _score(arcs):
    return lp_eval(_D, arcs)[0]


def insertions(arcs, v, n):
    out = []
    for i, j in combinations(range(len(arcs)), 2):
        (a, b), (c, d) = arcs[i], arcs[j]
        if a == c or b == d:
            continue                                   # would duplicate (a, v) or (v, b)
        new = [x for k, x in enumerate(arcs) if k not in (i, j)] + [(a, v), (v, b), (c, v), (v, d)]
        if len(set(new)) == len(new) and qt.is_valid_topology(n, new):
            out.append(((i, j), new))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--from", dest="src", default="deliverables/network23/network23.sol")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--keep", type=int, default=20)
    a = p.parse_args()

    t0 = time.time()
    _, small, _ = qt.read_network_solution(ROOT / a.src)
    small = [tuple(x) for x in small]
    v = a.n - 1
    cands = insertions(small, v, a.n)
    demand = qt.load_network_demand(a.n)
    with Pool(a.workers, initializer=_init, initargs=(demand,)) as pool:
        z = np.array(pool.map(_score, [c for _, c in cands], chunksize=16))
    hops = np.array([hop_cost(demand, c) for _, c in cands])
    order = np.argsort(z)
    agree = rank_agreement(hops, z, model_lower_is_better=True, referee_lower_is_better=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"source": a.src, "n": a.n, "n_insertions": len(cands), "seconds": round(time.time() - t0, 1),
           "hops_rho": agree.rho, "z_quantiles": np.quantile(z, [0, 0.01, 0.1, 0.5]).tolist(),
           "best": [{"z": float(z[k]), "score": math.ceil(z[k] - 1e-6), "removed": cands[k][0], "arcs": cands[k][1]}
                    for k in order[: a.keep]]}
    path = OUT / f"n{a.n:02d}_from_{Path(a.src).stem}.json"
    path.write_text(json.dumps(rec))
    print(f"{len(cands)} insertions scored in {rec['seconds']}s; hop-count rho vs LP {agree.rho:+.2f}")
    print(f"LP quantiles min/1%/10%/50%: " + " / ".join(f"{q:.0f}" for q in rec["z_quantiles"]))
    for b in rec["best"][:10]:
        print(f"  {b['score']}  gap {(b['score'] - BKV[a.n]) / BKV[a.n]:+.3%}  removed arcs {b['removed']}")
    print(f"written {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
