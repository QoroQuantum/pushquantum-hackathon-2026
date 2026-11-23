"""Recombine days across all solutions of an instance (exact DP), then polish with day_swap.

For every instance the given folders hold several solutions of (different seeds, warm
starts, QAOA loops). ``exact_dp.recombine`` picks, by an exact DP over days, the best mix
of the day holdings they use (the "combine samples" stitching step); ``day_swap.run``
then polishes the mix. Records go to ``--out`` in the format ``grid_runs.py report`` and
``make_handin.py`` read.

    python recombine_runs.py --match a050 --dirs results/grid results/grid_polish2 handin
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import qoblib_tools as qt
from grid_runs import lam_tag, load_bkv


def _job(args):
    key, files, out, tlim = args
    import day_swap  # noqa: PLC0415
    from exact_dp import recombine  # noqa: PLC0415

    h, _ = qt.read_portfolio_solution(files[0])
    m = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]), lam=h["lambda"])
    Vs, scores = [], []
    for f in files:
        _, pos = qt.read_portfolio_solution(f)
        L, S = m.to_arrays(pos)
        sc = m.score_arrays(L, S)
        if sc.feasible:
            Vs.append((L - S).astype(np.int64))
            scores.append(int(sc.objective))
    best_in = min(scores)
    V, sizes = recombine(m, Vs)
    L, S = np.maximum(V, 0), np.maximum(-V, 0)
    rec_sc = m.score_arrays(L, S)
    assert rec_sc.feasible and rec_sc.objective <= best_in, (key, rec_sc.objective, best_in)
    if rec_sc.objective >= best_in:  # the best input already uses the best mix of days: nothing to polish
        print(f"{key:32s} {len(Vs)} inputs, best {best_in}: no better mix of days", flush=True)
        return None
    r = day_swap.run(m, L, S, tlim, 0, verbose=False, stall=min(60, tlim), window=4)
    sol = Path(out) / f"{key}_recomb.sol"
    m.write_solution(sol, m.to_positions(r["L"], r["S"]),
                     comment=f"recombine_runs.py: exact DP over the day holdings of {len(Vs)} solutions, "
                             f"then day_swap.py polish ({tlim:.0f} s, window moves)")
    rec = dict(instance=h["instance"], budget=int(h["budget"]), lam=lam_tag(h["lambda"]), init="recombine",
               inputs=len(Vs), best_input=best_in, recombined=int(rec_sc.objective), score=int(r["score"]),
               candidates_per_day=sizes, solution_file=str(sol))
    (Path(out) / f"{key}_recomb.json").write_text(json.dumps(rec, indent=1))
    print(f"{key:32s} {len(Vs)} inputs, best {best_in} -> recombined {rec_sc.objective} -> polished {r['score']}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/grid", "results/grid_polish", "results/grid_polish2",
                                                   "results/grid_fresh", "results/warm_day_qaoa", "handin"])
    ap.add_argument("--match", nargs="*", default=["a050"])
    ap.add_argument("--out", default="results/recombine")
    ap.add_argument("--time", type=float, default=90)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    bkv = load_bkv()
    groups = defaultdict(list)
    for d in a.dirs:
        for p in Path(d).rglob("*.sol"):
            if p.name.endswith(".ckpt.sol"):
                continue
            h, _ = qt.read_portfolio_solution(p)
            if not {"instance", "budget", "lambda"} <= h.keys():
                continue
            key = f"{h['instance']}_b{int(h['budget']):03d}_l{lam_tag(h['lambda'])}"
            if a.match and not any(mm in key for mm in a.match):
                continue
            groups[key].append(str(p))
    jobs = [(k, sorted(set(v)), a.out, a.time) for k, v in sorted(groups.items()) if len(set(v)) >= 2 and k in bkv]
    print(f"{len(jobs)} instances to recombine", flush=True)
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_job, jobs):
            pass


if __name__ == "__main__":
    main()
