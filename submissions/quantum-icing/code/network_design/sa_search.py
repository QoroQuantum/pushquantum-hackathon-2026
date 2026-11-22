"""Network design: parallel simulated annealing over valid topologies, exact LP as referee.

State   : a valid topology (every node 2 in / 2 out).
Moves   : degree-preserving head rewires — 2-swap (i,j),(k,l) -> (i,l),(k,j) and
          3-rotation of heads among three arcs. The first arc is drawn from the LP's
          *bottleneck* arcs (positive dual on the capacity row) most of the time.
Energy  : exact LP optimum z_lp (fractional, so smoother than the rounded score).

    python -m network_design.sa_search --n 24 --workers 14 --minutes 20

Each worker writes its best topology to runs/network<NN>/seed<S>.json; the main
process then writes + checks solutions/network<NN>.sol via network_design.pipeline.bank().
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

warnings.filterwarnings("ignore", category=SyntaxWarning)
import qoblib_tools as qt  # noqa: E402

RUN_DIR = Path(__file__).resolve().parents[1] / "runs"


def lp_eval(demand, arcs):
    """(z_lp, bottleneck arc indices) for a topology given as a list of arcs."""
    order = sorted(range(len(arcs)), key=lambda a: arcs[a])
    sarcs = [arcs[a] for a in order]
    _, c, A_ub, A_eq, b_eq = qt._routing_lp(demand, sarcs)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq,
                bounds=(0, None), method="highs")
    if r.status != 0:
        return math.inf, []
    dual = -r.ineqlin.marginals
    return float(r.fun), [order[a] for a in np.flatnonzero(dual > 1e-9)]


def propose(arcs, hot, rng):
    """Degree-preserving rewire; None if it creates a self-loop or duplicate arc."""
    m = len(arcs)
    a = int(rng.choice(hot)) if hot and rng.random() < 0.7 else int(rng.integers(m))
    if rng.random() < 0.6:
        b = int(rng.integers(m))
        if b == a:
            return None
        idx = [a, b]
        heads = [arcs[b][1], arcs[a][1]]
    else:
        b, c = rng.choice(m, 2, replace=False)
        if a in (b, c):
            return None
        idx = [a, int(b), int(c)]
        heads = [arcs[idx[1]][1], arcs[idx[2]][1], arcs[idx[0]][1]]
    new = list(arcs)
    for i, h in zip(idx, heads):
        t = arcs[i][0]
        if t == h:
            return None
        new[i] = (t, h)
    return new if len(set(new)) == m else None


def anneal(args):
    n, seed, seconds, start, t0_frac = args
    rng = np.random.default_rng(seed)
    demand = qt.load_network_demand(n)
    out = RUN_DIR / f"network{n:02d}" / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    cur = list(start) if start else qt.random_topology(n, rng)
    cur_e, hot = lp_eval(demand, cur)
    best, best_e = cur, cur_e
    T0 = t0_frac * cur_e
    t_start = time.time()
    evals = 0
    while (el := time.time() - t_start) < seconds:
        T = T0 * (1e-3 ** (el / seconds))  # geometric cooling to ~0
        cand = propose(cur, hot, rng)
        if cand is None:
            continue
        e, h = lp_eval(demand, cand)
        evals += 1
        if e <= cur_e or rng.random() < math.exp(-(e - cur_e) / T):
            cur, cur_e, hot = cand, e, h
            if e < best_e - 1e-9:
                best, best_e = cand, e
                out.write_text(json.dumps({"z_lp": best_e, "score": math.ceil(best_e - 1e-6),
                                           "arcs": best, "evals": evals, "t": round(el)}))
    return seed, math.ceil(best_e - 1e-6), evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--t0", type=float, default=0.02, help="start temperature, fraction of energy")
    p.add_argument("--from-best", action="store_true", help="start from our banked best")
    a = p.parse_args()

    start = None
    if a.from_best:
        f = Path(__file__).resolve().parents[1] / "solutions" / f"network{a.n:02d}.sol"
        _, start, _ = qt.read_network_solution(f)
    jobs = [(a.n, a.seed0 + w, a.minutes * 60, start, a.t0) for w in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        for seed, score, evals in pool.imap_unordered(anneal, jobs):
            print(f"seed {seed}: best {score}  ({evals} evals)", flush=True)

    from network_design import pipeline as npipe
    runs = [json.loads(f.read_text()) for f in (RUN_DIR / f"network{a.n:02d}").glob("seed*.json")]
    runs = [r for r in runs if isinstance(r, dict) and "z_lp" in r]
    if not runs:
        print("no improving run; nothing to bank")
        return
    top = min(runs, key=lambda r: r["z_lp"])
    print(f"overall best {top['score']}  gap {npipe.gap(a.n, top['score']):+.3%}")
    print(" ", npipe.bank(a.n, qt.load_network_demand(a.n), [tuple(x) for x in top["arcs"]], "sa_search")[2])


if __name__ == "__main__":
    main()
