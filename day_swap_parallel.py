"""Day-sharded parallel rounds of ``day_swap.py``.

Days couple only through rebalancing costs, so each round splits the days at random into
``--workers`` shards and runs one ``day_swap.py --days ...`` process per shard on the
current best solution (the other days stay fixed in that process). Merge: take each
shard's days one at a time and keep a day only if the official score improves, so the
merged solution is never worse than the round's start.

    python day_swap_parallel.py --init results/day_swap/a400_best.sol --workers 6 \
        --round 300 --rounds 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import qoblib_tools as qt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a400_t15_orig")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--init", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--round", type=float, default=300, help="seconds per round")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--start-round", type=int, default=0, help="first round number (for restarts)")
    ap.add_argument("--out", default="results/day_swap_par")
    ap.add_argument("--window", type=int, default=2, help="longest multi-day move, in days")
    ap.add_argument("--two-day", type=float, default=0.2, help="share of multi-day moves")
    args = ap.parse_args()
    
    if args.workers < 1 or args.rounds < 1:
        ap.error("--workers and --rounds must be positive")


    model = qt.PortfolioModel(qt.load_portfolio_instance(args.instance), budget=args.budget, lam=args.lam)
    _, pos = qt.read_portfolio_solution(args.init)
    L, S = (a.astype(np.int64) for a in model.to_arrays(pos))
    V = L - S
    split = lambda V: (np.maximum(V, 0), np.maximum(-V, 0))  # noqa: E731
    sc = model.score_arrays(*split(V))
    assert sc.feasible
    best = sc.objective
    T = model.n_periods
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instance}_b{args.budget:03d}_l{args.lam}"
    best_sol = out / f"{stem}.sol"
    rng = np.random.default_rng(args.seed)
    rec = dict(instance=args.instance, budget=args.budget, lam=args.lam, init=args.init, start=best,
                seed=args.seed, top=args.top, start_round=args.start_round, solution_file=str(best_sol),
               workers=args.workers, round_seconds=args.round, rounds=[])
    t0 = time.time()
    print(f"{args.instance}: start {best}", flush=True)
    for r in range(args.start_round, args.start_round + args.rounds):
        start = out / f"round{r:02d}_start.sol"
        model.write_solution(start, model.to_positions(*split(V)), comment=f"round {r} start {best}")
        # contiguous runs of days (rotated at random each round) so that two-day moves stay
        # inside one worker; the shard boundaries move from round to round
        shift = int(rng.integers(T))
        shards = [np.sort((np.asarray(sh) + shift) % T) for sh in np.array_split(np.arange(T), args.workers)]
        procs = []
        for k, sh in enumerate(shards):
            if len(sh) == 0:
                continue
            seed = args.seed * 100_000 + 1000 * r + k
            od = out / f"round{r:02d}" / f"w{k}"
            od.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", "day_swap.py", "--instance", args.instance, "--budget", str(args.budget),
                   "--lam", args.lam, "--init", str(start), "--time", str(args.round), "--seed", str(seed),
                   "--top", str(args.top), "--quiet", "--out", str(od), "--window", str(args.window),
                   "--two-day", str(args.two_day), "--days", *map(str, sh)]
            log = open(od / "run.log", "w")
            procs.append((sh, od / f"{stem}_seed{seed}.sol", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        for _, _, p in procs:
            p.wait()
        before = best
        kept = []
        for sh, sol, p in procs:
            if not sol.exists():
                print(f"  round {r}: worker for days {list(sh)} wrote nothing (exit {p.returncode})", flush=True)
                continue
            _, pk = qt.read_portfolio_solution(sol)
            Lk, Sk = model.to_arrays(pk)
            Vk = (Lk - Sk).astype(np.int64)
            W = V.copy()  # the whole shard at once first (keeps joint multi-day gains)
            W[sh] = Vk[sh]
            s2 = model.score_arrays(*split(W))
            if s2.feasible and s2.objective < best:
                V, best = W, s2.objective
                kept += [int(t) for t in sh]
                continue
            for t in sh:
                if (Vk[t] != V[t]).any():
                    W = V.copy()
                    W[t] = Vk[t]
                    s2 = model.score_arrays(*split(W))
                    if s2.feasible and s2.objective < best:
                        V, best = W, s2.objective
                        kept.append(int(t))
        final = model.score_arrays(*split(V))
        assert final.feasible and final.objective == best
        model.write_solution(best_sol, model.to_positions(*split(V)),
                             comment=f"day_swap_parallel.py: {r + 1} rounds x {args.workers} day shards, "
                                     f"pair-move descent + per-day ILS, from {args.init}")
        rec["rounds"].append(dict(round=r, start=before, score=best, kept_days=kept,
                                  elapsed=round(time.time() - t0, 1)))
        rec["score"] = best
        (out / f"{stem}.json").write_text(json.dumps(rec, indent=2))
        print(f"  round {r}: {before} -> {best} (kept days {sorted(kept)}), {time.time() - t0:.0f}s", flush=True)
    print("AUTORUNNER_RESULT: " + json.dumps({"score": best, "extra": rec}), flush=True)


if __name__ == "__main__":
    main()
