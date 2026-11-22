"""Iterated local search whose local search is a *proof*: kick, exhaust radius 2..r, repeat.

The team's ILS samples 2-exchanges, 3-rotations and cycle reversals and stops when it stops
improving, so "stuck" is a statement about the sampler. `certify_descent.py` replaced the
sampling with enumeration: every topology within r arcs is scored, so "stuck" becomes a fact
about the instance. That costs more per step and buys a certificate -- and a certificate is
exactly what an ILS needs, because it tells the kick how far it has to reach.

One pass: kick the incumbent by a random derangement on a random k-subset of its arcs (k above
the certified radius, so the kick cannot be undone by the descent it is about to trigger), then
descend by exhausting radius 2..r until a pass comes back empty. That lands on a *certified*
radius-r local optimum. Keep it if it beats the incumbent, otherwise keep the incumbent and kick
again.

The by-product is a measurement the team's ILS cannot make: the list of distinct certified local
optima and how far apart they are. network21 gave the first two -- 562,362 from the hybrid chain
and 565,343 from the re-derived chain start, both proven radius-3 optima, 3,086 apart -- which
is why this is worth running: the landscape has many of them and the question is only how to
reach a better one.

Nothing here touches `deliverables/`. Banking goes through `pipeline.bank`, so every accepted
topology passes QOBLIB's own checker first; set `NETWORK_SOL_DIR` to a scratch folder.

    NETWORK_SOL_DIR=runs/goat/ils venv/bin/python TheGOAT_Testing/certified_ils.py \\
        --n 18 --r 3 --kick 5 6 --minutes 180 --workers 1
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from certify_descent import best_in_radius  # noqa: E402
from certify_radius import apply_perm, derangements  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402
from network_ils import WarmLP  # noqa: E402

OUT = Path(os.environ.get("GOAT_RUN_DIR", ROOT / "runs/goat")) / "ils"


def kick(arcs, k, rng, tries=2000):
    """A random valid topology exactly k arcs away: random k-subset, random derangement.

    The distance has to be checked, not assumed. Every node has in-degree 2, so two arcs in the
    chosen subset can share a head; swapping those two heads rebuilds the same two arcs and the
    move lands closer than k. That is harmless inside the certificate -- a move that lands at
    distance j < k was already enumerated at radius j -- but it defeats a kick, whose whole job
    is to leave a neighbourhood that has been exhausted up to r. So reject it and draw again.
    """
    d = derangements(k)
    want = 2 * k
    old = set(arcs)
    for _ in range(tries):
        idx = tuple(sorted(rng.sample(range(len(arcs)), k)))
        new = apply_perm(arcs, idx, rng.choice(d))
        if new is not None and len(old ^ set(new)) == want:
            return sorted(new)
    return None


def descend(n, arcs, z, r_max, workers, chunk):
    """Exhaust radius 2..r_max from `arcs`, taking the best move each pass. Returns the optimum."""
    lps = 0
    while True:
        improved = False
        for r in range(2, r_max + 1):
            zb, ab, st = best_in_radius(n, arcs, z, r, workers, chunk)
            lps += st["tested"]
            if zb < z - 0.5:
                arcs, z, improved = ab, zb, True
                break                       # restart the ladder from radius 2
        if not improved:
            return z, arcs, lps


def run(n, r_max, kicks, minutes, workers, chunk, seed, sol=None):
    demand = qt.load_network_demand(n)
    sol = Path(sol) if sol else ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    _, arcs, _ = qt.read_network_solution(sol)
    arcs = [tuple(map(int, a)) for a in arcs]
    lp = WarmLP(demand, arcs)
    lp.h.resetGlobalScheduler(True)
    z = lp.solve(arcs)[0]
    best_z, best_arcs = z, list(arcs)
    rng = random.Random(seed)

    print(f"network{n:02d}: {len(arcs)} arcs, incumbent {math.ceil(z - 1e-6):,} "
          f"(gap {(math.ceil(z - 1e-6) - BKV[n]) / BKV[n]:+.3%}), exhausting radius 2..{r_max}, "
          f"kicks of {kicks}, {minutes:g} min", flush=True)

    t0, optima, rounds = time.time(), {}, []
    while time.time() - t0 < minutes * 60:
        k = rng.choice(kicks)
        start = kick(best_arcs, k, rng)
        if start is None:
            continue
        z_start = lp.solve(start)[0]
        if z_start is None or not math.isfinite(z_start):
            continue
        t1 = time.time()
        z_end, arcs_end, lps = descend(n, start, z_start, r_max, workers, chunk)
        score = math.ceil(z_end - 1e-6)
        optima[score] = optima.get(score, 0) + 1
        rounds.append({"kick": k, "start": math.ceil(z_start - 1e-6), "optimum": score,
                       "lps": lps, "seconds": round(time.time() - t1, 1)})
        tag = ""
        if z_end < best_z - 0.5:
            banked, checked, msg = npl.bank(n, demand, arcs_end, f"certified_ils kick={k}")
            best_z, best_arcs = z_end, list(arcs_end)
            tag = f"  **better** | bank: {msg}"
        print(f"  kick {k} -> start {math.ceil(z_start - 1e-6):,} -> optimum {score:,} "
              f"({lps:,} LPs, {rounds[-1]['seconds']:.0f}s){tag}", flush=True)

    best = math.ceil(best_z - 1e-6)
    rec = {"n": n, "r_max": r_max, "kicks": kicks, "seed": seed,
           "incumbent": math.ceil(z - 1e-6), "best": best, "bkv": BKV[n],
           "gap": (best - BKV[n]) / BKV[n], "rounds": rounds,
           "distinct_optima": {str(k): v for k, v in sorted(optima.items())},
           "seconds": round(time.time() - t0, 1),
           "arcs": [list(a) for a in best_arcs]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"n{n:02d}_r{r_max}_s{seed}.json").write_text(json.dumps(rec))
    spread = (max(optima) - min(optima)) if optima else 0
    print(f"\nnetwork{n:02d}: {len(rounds)} restarts in {rec['seconds']:.0f}s, "
          f"{len(optima)} distinct certified radius-{r_max} optima, spread {spread:,}")
    print(f"best {best:,} (gap {rec['gap']:+.3%}), incumbent was {rec['incumbent']:,}")
    for score, hits in sorted(optima.items())[:10]:
        print(f"  {score:>9,}  x{hits}")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--r", type=int, default=3, help="radius exhausted by the local search")
    p.add_argument("--kick", type=int, nargs="+", default=[5, 6],
                   help="kick radii to draw from; keep them above --r")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sol", default=None)
    a = p.parse_args()
    if min(a.kick) <= a.r:
        print(f"warning: kick {min(a.kick)} <= exhausted radius {a.r}; "
              "the descent can undo the kick in one move", flush=True)
    run(a.n, a.r, a.kick, a.minutes, a.workers, a.chunk, a.seed, a.sol)


if __name__ == "__main__":
    main()
