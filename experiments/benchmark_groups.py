"""Run the grouping pipeline over a grid of instances and summarise the gaps.

    python experiments/benchmark_groups.py --instances po_a050_t10_orig po_a050_t10_s00 ... \
        --budget 20 --lams 1e-4 1e-3 1e-2 --time 120

Each run is ``asset_groups.py pipeline`` (SA start -> blocks -> iterated search), in
its own process; results are read back from ``results/pipeline_*.json``.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--lams", nargs="+", required=True)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--kick-blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", default="results/benchmark_groups.json")
    args = ap.parse_args()

    rows = []
    for inst in args.instances:
        for lam in args.lams:
            cmd = [sys.executable, "asset_groups.py", "pipeline", "--instance", inst,
                   "--budget", str(args.budget), "--lam", lam, "--size", str(args.size),
                   "--methods", "random", "--time", str(args.time),
                   "--kick-blocks", str(args.kick_blocks), "--seed", str(args.seed)]
            print(f"== {inst} λ={lam}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            for line in proc.stdout.splitlines()[1:]:
                print(line, flush=True)
            if proc.returncode:
                print(proc.stderr[-2000:], flush=True)
                continue
            rec = json.loads(Path(f"results/pipeline_{inst}_B{args.budget}_lam{lam}_seed{args.seed}.json").read_text())
            rows.append({k: rec.get(k) for k in
                         ("instance", "lam", "bkv", "start", "blocks", "score", "gap", "seconds")})

    print("\n| instance | λ | best known | SA start | + blocks | + iterated search | gap |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        g = "—" if r["gap"] is None else f"{r['gap']:+.2%}"
        print(f"| {r['instance']} | {r['lam']} | {r['bkv']} | {r['start']} | {r['blocks']} | {r['score']} | {g} |")
    Path(args.summary).write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved {args.summary}")


if __name__ == "__main__":
    main()
