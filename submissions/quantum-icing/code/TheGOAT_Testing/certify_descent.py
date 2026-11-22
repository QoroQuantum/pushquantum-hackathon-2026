"""Descend by certified radius: take the best move of the whole radius, repeat, end with a proof.

`certify_radius.py` answers one question — is anything within r arcs better? This turns that
into a search. Each pass enumerates every radius-2..r move (degrees force each to be a
derangement of the heads of k arcs, see that module), takes the single best improving one,
banks it through the official checker, and starts again from there. When a pass finds nothing,
the run stops *with a certificate*: the solution it leaves is a proven radius-r local optimum,
not a place a heuristic got stuck in.

That is the difference from the team's ILS, which samples 2-exchanges, 3-rotations and cycle
reversals and stops when it stops improving. Here the neighbourhood is exhausted, so "no
improvement" is a fact about the instance rather than about the search.

Banking goes through `pipeline.bank`, so every accepted topology is written as integer flows
and validated by QOBLIB's own checker before it is kept. Set `NETWORK_SOL_DIR` to a scratch
folder to leave `solutions/` and `deliverables/` untouched.

    NETWORK_SOL_DIR=runs/goat/descent venv/bin/python TheGOAT_Testing/certify_descent.py \
        --n 18 19 20 --r 3 --workers 1
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from itertools import combinations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from certify_radius import _init, _scan, derangements  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402
from network_ils import WarmLP  # noqa: E402

OUT = ROOT / "runs" / "goat" / "descent"


def best_in_radius(n, arcs, z0, r, workers, chunk):
    """Scan every radius-r move of `arcs`; return (z, arcs, stats) with z = inf when none is better."""
    subsets = list(combinations(range(len(arcs)), r))
    chunks = [subsets[i:i + chunk] for i in range(0, len(subsets), chunk)]
    tested = rejected = cut = 0
    best = (math.inf, None)
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(n, arcs, z0)) as pool:
        for t, rj, c, b in pool.imap_unordered(_scan, chunks):
            tested += t
            rejected += rj
            cut += c
            if b[0] < best[0]:
                best = b
    return best[0], best[1], {"radius": r, "moves": len(subsets) * len(derangements(r)),
                              "tested": tested, "rejected": rejected, "cut": cut,
                              "seconds": round(time.time() - t0, 1)}


def descend(n, r_max, workers, chunk, sol, budget_s):
    demand = qt.load_network_demand(n)
    _, arcs, _ = qt.read_network_solution(sol)
    arcs = [tuple(map(int, x)) for x in arcs]
    lp = WarmLP(demand, arcs)
    lp.h.resetGlobalScheduler(True)
    z = lp.solve(arcs)[0]
    start = math.ceil(z - 1e-6)
    print(f"network{n:02d}: {len(arcs)} arcs, start {start} "
          f"(gap {(start - BKV[n]) / BKV[n]:+.3%})", flush=True)

    chain, passes, t0 = [start], [], time.time()
    certified = 0
    while True:
        improved = False
        for r in range(2, r_max + 1):
            zb, ab, st = best_in_radius(n, arcs, z, r, workers, chunk)
            st["improved"] = zb < z - 0.5
            passes.append(st)
            if st["improved"]:
                arcs, z = ab, zb
                score = math.ceil(z - 1e-6)
                banked, checked, msg = npl.bank(n, demand, arcs, f"certify_descent r={r}")
                chain.append(checked if checked > 0 else score)
                print(f"  r={r}: {st['tested']:,} LPs in {st['seconds']}s -> {score} "
                      f"(gap {(score - BKV[n]) / BKV[n]:+.3%}) | bank: {msg}", flush=True)
                improved = True
                break                                  # restart the radius ladder from 2
            print(f"  r={r}: {st['tested']:,} LPs in {st['seconds']}s -> none better", flush=True)
            certified = r
        if not improved:
            break
        if budget_s and time.time() - t0 > budget_s:
            print(f"  budget reached after {time.time() - t0:.0f}s: stopping, "
                  f"certificate is only up to the last complete pass", flush=True)
            certified = 0
            break

    final = math.ceil(z - 1e-6)
    rec = {"n": n, "sol": str(sol), "start": start, "final": final, "chain": chain,
           "certified_radius": certified, "bkv": BKV[n],
           "gap_start": (start - BKV[n]) / BKV[n], "gap_final": (final - BKV[n]) / BKV[n],
           "passes": passes, "seconds": round(time.time() - t0, 1),
           "arcs": [list(a) for a in arcs]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"n{n:02d}_r{r_max}.json").write_text(json.dumps(rec))
    arrow = " -> ".join(str(c) for c in chain)
    print(f"network{n:02d} done in {rec['seconds']}s: {arrow} "
          f"(gap {rec['gap_final']:+.3%}), certified radius {certified}\n", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, nargs="+", default=[18])
    p.add_argument("--r", type=int, default=3)
    p.add_argument("--sol", default=None, help="start file; default: the deliverable of each n")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--minutes", type=float, default=0.0, help="per instance (0 = no limit)")
    a = p.parse_args()

    results = []
    for n in a.n:
        sol = Path(a.sol) if a.sol else ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
        if not sol.exists():
            print(f"network{n:02d}: no solution at {sol}, skipped", flush=True)
            continue
        results.append(descend(n, a.r, a.workers, a.chunk, sol, a.minutes * 60))

    print("| instance | start | final | gap | certified radius |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        mark = " **improved**" if r["final"] < r["start"] else ""
        print(f"| network{r['n']:02d} | {r['start']:,} | {r['final']:,}{mark} | "
              f"{r['gap_final']:+.3%} | {r['certified_radius']} |")


if __name__ == "__main__":
    main()
