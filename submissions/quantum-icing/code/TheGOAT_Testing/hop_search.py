"""Global search on the demand-weighted hop count, then the exact LP on the best topologies.

The hop count H = sum_ij d_ij hops(i, j) ranks rewirings like the exact LP (Spearman 0.85-0.89
on network24 blocks, block_fidelity.py) and costs ~0.5 ms instead of ~120 ms per evaluation, so
it can explore regions far from the hand-in that an LP-driven search never reaches. It is also a
bound: any routing loads the links with at least H in total, so max load >= H / (2n).

Each worker runs simulated annealing over degree-preserving 2-exchanges, restarting from random
2-in/2-out topologies (and once from the hand-in, as a reference), and keeps a pool of the best
distinct topologies by H. The merged pool's best are then scored with the exact LP.

    venv/bin/python TheGOAT_Testing/hop_search.py --n 24 --minutes 6 --workers 2 --lp 150
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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

OUT = ROOT / "runs" / "goat" / "hopsearch"
HAND_IN = ROOT / "deliverables" / "network24" / "network24.sol"


def anneal(demand, arcs, rng, steps, pool, keep):
    """SA on H over 2-exchanges (a,b),(c,d) -> (a,d),(c,b); the pool keeps the best distinct."""
    arcs = list(arcs)
    present = set(arcs)
    h = hop_cost(demand, arcs)
    T0 = 0.01 * h
    m = len(arcs)
    for t in range(steps):
        T = T0 * (0.002 ** (t / steps))                     # geometric cooling to 0.2% of T0
        i, j = rng.choice(m, 2, replace=False)
        (a, b), (c, d) = arcs[i], arcs[j]
        if a == c or b == d or a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        arcs[i], arcs[j] = (a, d), (c, b)
        h2 = hop_cost(demand, arcs)                         # inf if no longer strongly connected
        if h2 <= h or (math.isfinite(h2) and rng.random() < math.exp(-(h2 - h) / T)):
            present -= {(a, b), (c, d)}
            present |= {(a, d), (c, b)}
            h = h2
            key = tuple(sorted(arcs))
            if key not in pool and (len(pool) < keep or h < -pool_heap_min(pool)):
                add_to_pool(pool, key, h, keep)
        else:
            arcs[i], arcs[j] = (a, b), (c, d)
    return h


def pool_heap_min(pool):
    return pool["_heap"][0][0] if pool["_heap"] else -math.inf


def add_to_pool(pool, key, h, keep):
    heap = pool["_heap"]                                    # max-heap on h via negation
    if len(heap) < keep:
        heapq.heappush(heap, (-h, key))
        pool[key] = h
    elif h < -heap[0][0]:
        _, old = heapq.heapreplace(heap, (-h, key))
        pool.pop(old, None)
        pool[key] = h


def worker(job):
    n, seed, seconds, steps, keep, with_hand_in = job
    rng = np.random.default_rng(seed)
    demand = qt.load_network_demand(n)
    pool = {"_heap": []}
    t0, restarts, ends = time.time(), 0, []
    if with_hand_in:
        _, arcs, _ = qt.read_network_solution(HAND_IN)
        ends.append(("hand-in", anneal(demand, [tuple(a) for a in arcs], rng, steps // 4, pool, keep)))
    while time.time() - t0 < seconds:
        start = [tuple(a) for a in qt.random_topology(n, rng)]
        ends.append(("random", anneal(demand, start, rng, steps, pool, keep)))
        restarts += 1
    items = [(h, list(k)) for k, h in pool.items() if k != "_heap"]
    return {"seed": seed, "restarts": restarts, "ends": ends, "pool": items}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--minutes", type=float, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=20000, help="SA steps per restart")
    p.add_argument("--keep", type=int, default=300, help="pool size per worker")
    p.add_argument("--lp", type=int, default=150, help="pool topologies scored with the exact LP")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    demand = qt.load_network_demand(a.n)
    _, hand, _ = qt.read_network_solution(HAND_IN)
    hand = [tuple(x) for x in hand]
    h_hand, z_hand = hop_cost(demand, hand), lp_eval(demand, hand)[0]
    jobs = [(a.n, a.seed * 100 + w, a.minutes * 60, a.steps, a.keep, w == 0) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        results = pool.map(worker, jobs)

    merged = {}
    for r in results:
        for h, arcs in r["pool"]:
            merged.setdefault(tuple(sorted(map(tuple, arcs))), h)
    ranked = sorted(merged.items(), key=lambda kv: kv[1])[: a.lp]
    zs = np.array([lp_eval(demand, list(k))[0] for k, _ in ranked])
    hs = np.array([h for _, h in ranked])
    agree = rank_agreement(hs, zs, model_lower_is_better=True, referee_lower_is_better=True) if len(zs) >= 3 else None
    best = int(np.argmin(zs))
    # how far each pool topology is from the hand-in, in arcs
    hand_set = set(hand)
    dist = [len(set(k) - hand_set) for k, _ in ranked]

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"n": a.n, "minutes": a.minutes, "workers": a.workers, "restarts": sum(r["restarts"] for r in results),
           "hand_in": {"hop": h_hand, "z": z_hand},
           "pool_best_hop": float(hs.min()), "pool_lp_best": float(zs[best]), "pool_lp_best_hop": float(hs[best]),
           "pool_lp_best_arcs_from_hand_in": dist[best], "rho_hop_vs_lp_in_pool": agree.rho if agree else None,
           "ends": [e for r in results for e in r["ends"]],
           "top": [{"hop": float(hs[i]), "z": float(zs[i]), "arcs_from_hand_in": dist[i], "arcs": list(ranked[i][0])}
                   for i in np.argsort(zs)[:20]]}
    path = OUT / f"n{a.n:02d}_s{a.seed}.json"
    path.write_text(json.dumps(rec))
    print(f"{rec['restarts']} SA restarts in {a.minutes} min x {a.workers} workers")
    print(f"hand-in:            hop {h_hand:,.0f}  LP {z_hand:,.0f}")
    print(f"best hop in pool:   hop {hs.min():,.0f}  ({100 * (hs.min() / h_hand - 1):+.2f}% vs hand-in)")
    print(f"best LP in pool:    LP {zs[best]:,.0f} (gap {(math.ceil(zs[best] - 1e-6) - BKV[a.n]) / BKV[a.n]:+.3%}), "
          f"hop {hs[best]:,.0f}, {dist[best]} arcs away from the hand-in")
    if agree:
        print(f"hop vs LP inside the pool: rho {agree.rho:+.2f}")
    print(f"written {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
