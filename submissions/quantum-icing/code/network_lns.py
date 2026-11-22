"""Network design: exact large-neighbourhood search (LNS).

Each step frees r arcs of the current topology (biased to LP bottleneck arcs),
keeps their tails, and lets a MILP re-assign their heads *jointly with the
routing*: y[f,h] = 1 if freed arc f now points to head h. Every node keeps
2 in / 2 out by construction. The MILP minimises the max link load z, so a step
never gets worse; accepted steps are re-scored by the official LP.

    python network_lns.py --n 24 --workers 14 --minutes 20 [--start solutions|greedy|runs]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.optimize import Bounds, LinearConstraint, milp

warnings.filterwarnings("ignore", category=SyntaxWarning)
import qoblib_tools as qt  # noqa: E402
from network_design.sa_search import RUN_DIR, lp_eval  # noqa: E402


def rewire_milp(demand, arcs, free, z_ub, time_limit):
    """Optimal head re-assignment of arcs[free] + routing. Returns (z, new_arcs) or None."""
    n = len(demand)
    free = list(free)
    fixed = [a for i, a in enumerate(arcs) if i not in set(free)]
    fixed_set = set(fixed)
    tails = [arcs[f][0] for f in free]
    mult = Counter(arcs[f][1] for f in free)
    heads = sorted(mult)
    # assignment vars y[(fi, h)]
    ys = [(fi, h) for fi, t in enumerate(tails) for h in heads if h != t and (t, h) not in fixed_set]
    cand = sorted({(tails[fi], h) for fi, h in ys})
    all_arcs = fixed + cand
    a_idx = {a: i for i, a in enumerate(all_arcs)}

    col = 0
    fvar = {}
    for k in range(n):
        for a, (i, j) in enumerate(all_arcs):
            if j != k:
                fvar[k, a] = col
                col += 1
    zc = col
    col += 1
    ycol = {y: col + i for i, y in enumerate(ys)}
    ncol = col + len(ys)

    R, C, V, lo, hi = [], [], [], [], []
    row = 0

    def add(entries, l, u):
        nonlocal row
        for c, v in entries:
            R.append(row), C.append(c), V.append(v)
        lo.append(l), hi.append(u)
        row += 1

    # flow conservation
    out_of = [[] for _ in range(n)]
    into = [[] for _ in range(n)]
    for a, (i, j) in enumerate(all_arcs):
        out_of[i].append(a)
        into[j].append(a)
    for k in range(n):
        for v in range(n):
            if v == k:
                continue
            e = [(fvar[k, a], 1.0) for a in into[v] if (k, a) in fvar]
            e += [(fvar[k, a], -1.0) for a in out_of[v] if (k, a) in fvar]
            b = float(demand[k, v] * qt.INTSCALE)
            add(e, b, b)
    # capacity: load <= z on every arc; candidate arcs also <= z_ub * (chosen)
    by_cand = {c: [] for c in cand}
    for (fi, h) in ys:
        by_cand[tails[fi], h].append(ycol[fi, h])
    for a, arc in enumerate(all_arcs):
        load = [(fvar[k, a], 1.0) for k in range(n) if (k, a) in fvar]
        add(load + [(zc, -1.0)], -np.inf, 0.0)
        if arc in by_cand:
            add(load + [(c, -z_ub) for c in by_cand[arc]], -np.inf, 0.0)
            if len(by_cand[arc]) > 1:  # no duplicate arcs
                add([(c, 1.0) for c in by_cand[arc]], 0.0, 1.0)
    # each freed arc gets one head; each head regains its multiplicity
    for fi in range(len(free)):
        add([(ycol[y], 1.0) for y in ys if y[0] == fi], 1.0, 1.0)
    for h in heads:
        add([(ycol[y], 1.0) for y in ys if y[1] == h], mult[h], mult[h])

    A = sps.csr_array((V, (R, C)), shape=(row, ncol))
    c = np.zeros(ncol)
    c[zc] = 1.0
    integ = np.zeros(ncol)
    integ[col:] = 1
    ub = np.full(ncol, np.inf)
    ub[zc] = z_ub
    ub[col:] = 1.0
    res = milp(c, constraints=[LinearConstraint(A, lo, hi)], integrality=integ,
               bounds=Bounds(np.zeros(ncol), ub), options={"time_limit": time_limit})
    if res.x is None:
        return None
    new = list(arcs)
    for (fi, h), cc in ycol.items():
        if res.x[cc] > 0.5:
            new[free[fi]] = (tails[fi], h)
    return res.fun, new


def pick_free(arcs, hot, r, rng):
    """Half bottleneck arcs, plus arcs sharing their endpoints, plus random."""
    m = len(arcs)
    chosen = set()
    hot = list(hot)
    rng.shuffle(hot)
    for a in hot[: max(1, r // 2)]:
        chosen.add(a)
    nodes = {x for a in chosen for x in arcs[a]}
    near = [a for a in range(m) if a not in chosen and (arcs[a][0] in nodes or arcs[a][1] in nodes)]
    rng.shuffle(near)
    for a in near:
        if len(chosen) >= r - r // 4:
            break
        chosen.add(a)
    while len(chosen) < r:
        chosen.add(int(rng.integers(m)))
    return sorted(chosen)


def load_start(n, how, seed):
    if how == "greedy":
        return qt.repair_topology(n, [], qt.load_network_demand(n))
    if how == "published":  # QOBLIB's best-known topology (warm start; disclose if used)
        kind = "opt" if n <= 10 else "bst"
        _, arcs, _ = qt.read_network_solution(qt.fetch_qoblib_file(f"08-network/solutions/network{n:02d}.{kind}.sol"))
        return arcs
    if how == "solutions":
        _, arcs, _ = qt.read_network_solution(Path(__file__).resolve().parent / "solutions" / f"network{n:02d}.sol")
        return arcs
    runs = sorted((json.loads(f.read_text()) for f in (RUN_DIR / f"network{n:02d}").glob("*.json")),
                  key=lambda r: r["z_lp"])
    return [tuple(a) for a in runs[seed % len(runs)]["arcs"]]


def worker(args):
    n, seed, seconds, start_how, r0, rmax, tl = args
    rng = np.random.default_rng(seed)
    demand = qt.load_network_demand(n)
    out = RUN_DIR / f"network{n:02d}" / f"lns{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cur = load_start(n, start_how, seed)
    cur_e, hot = lp_eval(demand, cur)
    best_e = cur_e
    r, stall, steps = r0, 0, 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        free = pick_free(cur, hot, r, rng)
        got = rewire_milp(demand, cur, free, cur_e + 1.0, tl)
        steps += 1
        if got is not None:
            e, h = lp_eval(demand, got[1])
            if e < cur_e - 1e-6:
                cur, cur_e, hot, stall = got[1], e, h, 0
                r = r0
            else:
                if e <= cur_e + 1e-6:  # sideways move: new bottleneck structure
                    cur, hot = got[1], h
                stall += 1
        else:
            stall += 1
        if stall >= 15:
            r, stall = min(rmax, r + 2), 0
        if cur_e < best_e - 1e-6:
            best_e = cur_e
            out.write_text(json.dumps({"z_lp": cur_e, "score": math.ceil(cur_e - 1e-6), "arcs": cur,
                                       "steps": steps, "t": round(time.time() - t0)}))
            print(f"[lns{seed}] {math.ceil(cur_e - 1e-6)}  r={r}  step {steps}  {time.time() - t0:.0f}s", flush=True)
    return seed, math.ceil(best_e - 1e-6), steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--start", default="greedy", choices=["greedy", "solutions", "runs", "published"])
    p.add_argument("--r", type=int, default=8, help="arcs freed per step (start)")
    p.add_argument("--rmax", type=int, default=16)
    p.add_argument("--tl", type=float, default=20, help="MILP time limit per step, s")
    p.add_argument("--seed0", type=int, default=100)
    a = p.parse_args()
    jobs = [(a.n, a.seed0 + w, a.minutes * 60, a.start, a.r, a.rmax, a.tl) for w in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        for seed, score, steps in pool.imap_unordered(worker, jobs):
            print(f"lns{seed}: best {score} ({steps} steps)", flush=True)

    import network_pipeline as npipe
    runs = [json.loads(f.read_text()) for f in (RUN_DIR / f"network{a.n:02d}").glob("*.json")]
    runs = [r for r in runs if isinstance(r, dict) and "z_lp" in r]
    if not runs:
        print("no improving run; nothing to bank")
        return
    top = min(runs, key=lambda r: r["z_lp"])
    print(f"overall best {top['score']}  gap {npipe.gap(a.n, top['score']):+.3%}")
    print(" ", npipe.bank(a.n, qt.load_network_demand(a.n), [tuple(x) for x in top["arcs"]], "network_lns")[2])


if __name__ == "__main__":
    main()
