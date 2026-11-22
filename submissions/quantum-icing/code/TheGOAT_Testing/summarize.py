"""Tables for the report, from what block_fidelity.py and goat_lns.py left under runs/goat/.

    venv/bin/python TheGOAT_Testing/summarize.py            # markdown tables on stdout

Head-to-head: on every block, each proposer's best LP value from the same number of LP solves.
A proposer "wins" a block when its value is strictly the lowest, "ties" when it shares the lowest.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from network_design.pipeline import BKV  # noqa: E402

RUNS = ROOT / "runs" / "goat"
PROPOSERS = ("quantum", "quantum+ls", "random+ls", "sa", "hungarian", "greedy", "sampled", "random")
START = {"greedy": "greedy 768375", "solutions": "incumbent 657121"}     # network24 LP values


def fidelity_table():
    files = sorted((RUNS / "fidelity").glob("n*_r*_s*_b*.json"))
    if not files:
        return "no fidelity runs"
    names = ["team", "bound", "hops", "fit_lin", "fit_quad"]
    rows = ["| block | r | valid rewirings | improving | " + " | ".join(names) + " |",
            "|---|---|---|---|" + "---|" * len(names)]
    per = {s: [] for s in names}
    for f in files:
        rec = json.loads(f.read_text())
        cells = []
        for s in names:
            v = rec["surrogates"].get(s)
            cells.append(f"{v['rho']:+.2f}" if v else "-")
            if v:
                per[s].append(v["rho"])
        rows.append(f"| {rec['block']} | {rec['r']} | {rec['n_valid']} | {rec['n_improving']} | " + " | ".join(cells) + " |")
    rows.append("| **range** | | | | " + " | ".join(
        f"**{min(per[s]):+.2f} to {max(per[s]):+.2f}**" if per[s] else "-" for s in names) + " |")
    return "\n".join(rows)


def load_runs():
    out = {}
    for kind in ("lns", "swaps"):
        root = RUNS / kind
        for d in sorted(root.iterdir()) if root.exists() else []:
            log = d / "steps.jsonl"
            if log.exists():
                steps = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
                if steps:
                    args = json.loads((d / "args.json").read_text())
                    args.setdefault("solver", "pce (swap variables)")
                    out[d.name] = (args, steps)
    return out


def runs_table(runs):
    rows = ["| run | solver | qubits | r | steps | start | end | gap vs BKV | s/step | credits used |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (args, steps) in runs.items():
        n = args["n"]
        end = math.ceil(min(s["z_cur"] for s in steps) - 1e-6)
        cr = [s["credits"] for s in steps if "credits" in s]
        used = f"{cr[0] - cr[-1]:.1f} (+ step 1)" if len(cr) > 1 else ("-" if not cr else "step 1 only")
        rows.append(f"| {name} | {args['solver']} | {steps[0]['qubits']} | {args['r']} | {len(steps)} | "
                    f"{START.get(args['start'], args['start'])} | {end} | "
                    f"{(end - BKV[n]) / BKV[n]:+.2%} | {np.mean([s['t_step'] for s in steps]):.0f} | {used} |")
    return "\n".join(rows)


def head_to_head(runs):
    others = [p for p in PROPOSERS if p != "quantum"]
    rows = ["| run | blocks | " + " | ".join(f"{p} wins / ties" for p in PROPOSERS)
            + " | " + " | ".join(f"quantum vs {p} (W/T/L)" for p in others) + " |",
            "|---|---|" + "---|" * (len(PROPOSERS) + len(others))]
    for name, (_, steps) in runs.items():
        present = [p for p in PROPOSERS if p in steps[-1]]       # older runs lack some controls
        blocks = [s for s in steps if all(p in s for p in present)]
        wins = {p: 0 for p in present}
        ties = {p: 0 for p in present}
        for s in blocks:
            z = {p: s[p]["z"] for p in present}
            lo = min(z.values())
            best = [p for p in present if z[p] <= lo + 1e-6]
            for p in best:
                (wins if len(best) == 1 else ties)[p] += 1

        def wtl(other):
            if other not in present:
                return "-"
            w = sum(s["quantum"]["z"] < s[other]["z"] - 1e-6 for s in blocks)
            t = sum(abs(s["quantum"]["z"] - s[other]["z"]) <= 1e-6 for s in blocks)
            return f"{w} / {t} / {len(blocks) - w - t}"

        rows.append(f"| {name} | {len(blocks)} | "
                    + " | ".join(f"{wins[p]} / {ties[p]}" if p in present else "-" for p in PROPOSERS)
                    + " | " + " | ".join(wtl(p) for p in others) + " |")
    return "\n".join(rows)


def main():
    runs = load_runs()
    print("## Block fidelity (Spearman vs exact LP, every valid rewiring of the block)\n")
    print(fidelity_table())
    print("\n## LNS runs\n")
    print(runs_table(runs) if runs else "no LNS runs")
    print("\n## Head-to-head on the same blocks, same LP budget\n")
    print(head_to_head(runs) if runs else "no LNS runs")


if __name__ == "__main__":
    main()
