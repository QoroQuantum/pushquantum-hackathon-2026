"""Certify that a solution is optimal within Hamming distance r of itself.

Why this is a certificate and the block enumerations are not. Removing k arcs and adding k
arcs while keeping every degree forces the added arcs to carry exactly the removed arcs'
heads, so the whole radius-k neighbourhood is "pick k arcs, permute their heads". A
permutation that fixes an arc is a smaller move, so radius k adds exactly the DERANGEMENTS of
each k-subset. That makes the neighbourhood finite and enumerable:

    r = 2   C(48,2) * 1          1,128 moves
    r = 3   C(48,3) * 2         34,592
    r = 4   C(48,4) * 9      1,751,220

`block_fidelity.py` enumerated 6 particular blocks at r = 6-7, which is 6 subsets out of the
C(48,6) ~ 12.3 million there are: strong evidence, but not a certificate. This enumerates
*every* subset up to r and so proves the absence of an improvement, or finds one.

Each candidate goes to a warm-started dual simplex with `objective_bound` set at the
incumbent, so HiGHS abandons a candidate as soon as it can prove it is no better instead of
solving it to optimality. Candidates that create a duplicate arc or a self-loop are rejected
before the LP.

    venv/bin/python TheGOAT_Testing/certify_radius.py --n 24 --r 3 --workers 2
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
from itertools import combinations, permutations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402
from network_ils import WarmLP  # noqa: E402

OUT = ROOT / "runs" / "goat" / "certify"

_S: dict = {}


def derangements(k: int) -> list[tuple[int, ...]]:
    """Permutations of range(k) with no fixed point: the moves that really need all k arcs."""
    return [p for p in permutations(range(k)) if all(p[i] != i for i in range(k))]


def apply_perm(arcs, idx, perm):
    """arcs with the heads of `idx` permuted by `perm`; None when that is not a new topology.

    Rejected: a repeated arc, a self-loop, and the degenerate case where the permutation only
    reorders arcs that share a tail, which rebuilds the same topology and would cost an LP for
    nothing.
    """
    new = list(arcs)
    for pos, f in enumerate(idx):
        new[f] = (arcs[f][0], arcs[idx[perm[pos]]][1])
    if len(set(new)) != len(new):
        return None
    for i, j in new:
        if i == j:
            return None
    if set(new) == set(arcs):
        return None
    return new


def _init(n, arcs, z0):
    lp = WarmLP(qt.load_network_demand(n), arcs)
    lp.h.resetGlobalScheduler(True)
    lp.solve(arcs)
    _S.update(arcs=arcs, lp=lp, z0=z0, basis=lp.h.getBasis())


def _scan(chunk):
    """Test one chunk of index subsets; return (tested, rejected, cut, best_found)."""
    arcs, lp, z0, basis = _S["arcs"], _S["lp"], _S["z0"], _S["basis"]
    perms = derangements(len(chunk[0]))
    tested = rejected = cut = 0
    best = (math.inf, None)
    for idx in chunk:
        for p in perms:
            new = apply_perm(arcs, idx, p)
            if new is None:
                rejected += 1
                continue
            tested += 1
            z, _ = lp.solve(new, cutoff=z0 - 0.5, basis=basis)
            if z is None:                      # proved >= z0 - 0.5 without solving
                cut += 1
            elif z < best[0]:
                best = (float(z), list(new))
    return tested, rejected, cut, best


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--sol", default=None, help="default: deliverables/network<n>/network<n>.sol")
    p.add_argument("--r", type=int, default=3, help="certify every radius up to this one")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--chunk", type=int, default=200, help="subsets per task")
    p.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0 = no limit)")
    a = p.parse_args()

    sol = ROOT / (a.sol or f"deliverables/network{a.n:02d}/network{a.n:02d}.sol")
    _, arcs, _ = qt.read_network_solution(sol)
    arcs = [tuple(map(int, x)) for x in arcs]
    demand = qt.load_network_demand(a.n)
    lp0 = WarmLP(demand, arcs)
    lp0.h.resetGlobalScheduler(True)
    z0 = lp0.solve(arcs)[0]
    m = len(arcs)
    print(f"network{a.n:02d}: {m} arcs, incumbent {math.ceil(z0 - 1e-6)} "
          f"(gap {(math.ceil(z0 - 1e-6) - BKV[a.n]) / BKV[a.n]:+.3%})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    record = {"n": a.n, "sol": str(sol.relative_to(ROOT)), "z0": z0, "radii": {}}
    certified = 0
    t_all = time.time()
    for r in range(2, a.r + 1):
        n_sub, n_der = math.comb(m, r), len(derangements(r))
        print(f"r = {r}: {n_sub:,} subsets x {n_der} derangements = {n_sub * n_der:,} moves", flush=True)
        subsets = list(combinations(range(m), r))
        chunks = [subsets[i:i + a.chunk] for i in range(0, len(subsets), a.chunk)]
        t0 = time.time()
        tested = rejected = cut = 0
        best = (math.inf, None)
        done = 0
        stopped = False
        with Pool(a.workers, initializer=_init, initargs=(a.n, arcs, z0)) as pool:
            for t, rj, c, b in pool.imap_unordered(_scan, chunks):
                tested += t
                rejected += rj
                cut += c
                done += 1
                if b[0] < best[0]:
                    best = b
                if a.minutes and time.time() - t0 > a.minutes * 60:
                    stopped = True
                    pool.terminate()
                    break
        dt = time.time() - t0
        found = best[1] is not None and best[0] < z0 - 0.5
        rec = {"subsets": n_sub, "moves": n_sub * n_der, "tested": tested, "rejected": rejected,
               "cut_by_bound": cut, "seconds": round(dt, 1),
               "ms_per_move": round(1000 * dt / max(tested, 1), 2),
               "coverage": round(done / len(chunks), 4), "complete": not stopped,
               "improvement": {"z": best[0], "arcs": best[1]} if found else None}
        record["radii"][str(r)] = rec
        status = ("IMPROVED to %.0f" % best[0]) if found else (
            "certified: no better topology" if not stopped else f"incomplete ({100 * done / len(chunks):.0f}%)")
        print(f"  {tested:,} LPs in {dt:.0f}s ({rec['ms_per_move']} ms each, "
              f"{100 * cut / max(tested, 1):.0f}% cut by the bound, {rejected:,} invalid) -> {status}", flush=True)
        if found:
            break
        if not stopped:
            certified = r
    record["certified_radius"] = certified
    record["total_seconds"] = round(time.time() - t_all, 1)
    path = OUT / f"n{a.n:02d}_r{a.r}.json"
    path.write_text(json.dumps(record))
    print(f"\ncertified radius: {certified} "
          f"(every topology differing from the hand-in in at most {certified} arcs is no better)")
    print(f"written {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
