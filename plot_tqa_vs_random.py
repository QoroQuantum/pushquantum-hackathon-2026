#!/usr/bin/env python
"""Plot of the random vs TQA comparison, from the run.json files tqa_vs_random.py stored.

Reads only AutoData/<problem>/l<λ>/<init><seed>/run.json, so it can be edited and rerun
without executing anything again:

    python plot_tqa_vs_random.py AutoData/po_a010_t10_orig_b004

Writes into that folder: gap_vs_lambda.png (300 dpi) and .pdf, gap_vs_lambda.csv
(mean and std of the relative gap per λ and init) and all_runs.csv (one row per run).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RISKS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]  # λ grid, in plot order
INITS = ["random", "tqa"]
STYLE = {"random": dict(color="#d62728", marker="o", label="Random"),
         "tqa": dict(color="#1f77b4", marker="s", label="TQA")}
RUN_FIELDS = ["instance", "lam", "init", "seed", "objective", "feasible", "best_known", "gap", "wall_s",
              "exit_code"]


def collect(problem_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(problem_dir.glob("l*/*/run.json"))]


def summarise(runs: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    """λ values present, mean/std gap (%) per (λ, init), and one flat row per run."""
    lams = [lam for lam in RISKS if any(r["lam"] == lam for r in runs)]
    inits = [i for i in INITS if any(r["init"] == i for r in runs)]
    all_rows = [{k: r.get(k) for k in RUN_FIELDS}
                | {"dt_chosen_median": (r.get("solver") or {}).get("dt_chosen_median")} for r in runs]
    summary = []
    for lam in lams:
        bkv = next((r["best_known"] for r in runs if r["lam"] == lam), None)
        for init in inits:
            gaps = [100 * r["gap"] for r in runs if r["lam"] == lam and r["init"] == init and r["gap"] is not None]
            summary.append({"lambda": lam, "init": init, "n": len(gaps), "best_known": bkv,
                            "mean_gap_percent": float(np.mean(gaps)) if gaps else None,
                            "std_gap_percent": (float(np.std(gaps, ddof=1)) if len(gaps) > 1
                                                else 0.0 if gaps else None)})
    return lams, summary, all_rows


def lam_label(lam: str) -> str:
    """'0' -> 0, '1e-4' -> 10^-4, '5e-5' -> 5x10^-5 (mathtext)."""
    if lam == "0":
        return "0"
    mant, exp = lam.split("e")
    return f"$10^{{{int(exp)}}}$" if mant == "1" else f"${mant}\\times10^{{{int(exp)}}}$"


def draw(problem_dir: Path, lams: list[str], summary: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11,
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(lams))
    inits = list(dict.fromkeys(s["init"] for s in summary))
    for k, init in enumerate(inits):
        pts = [s for s in summary if s["init"] == init]
        mean = np.array([np.nan if s["mean_gap_percent"] is None else s["mean_gap_percent"] for s in pts])
        std = np.array([np.nan if s["std_gap_percent"] is None else s["std_gap_percent"] for s in pts])
        n = max((s["n"] for s in pts), default=0)
        st = STYLE[init]
        ax.errorbar(x + (k - (len(inits) - 1) / 2) * 0.18, mean, yerr=std, fmt=st["marker"], color=st["color"], ms=6,
                    mfc=st["color"], mec="black", mew=0.6, ecolor=st["color"], elinewidth=1.1, capsize=3,
                    capthick=1.1, label=f"{st['label']} ($n={n}$)")
    ax.set_xticks(x)
    ax.set_xticklabels([lam_label(lam) for lam in lams])
    ax.set_xlabel(r"Risk aversion $\lambda$")
    ax.set_ylabel(r"Relative gap $(E - E_{\mathrm{best}})/|E_{\mathrm{best}}|$ (%)")
    done = [s for s in summary if s["mean_gap_percent"] is not None]
    tops = [s["mean_gap_percent"] + (s["std_gap_percent"] or 0) for s in done]
    lows = [s["mean_gap_percent"] - (s["std_gap_percent"] or 0) for s in done]
    if tops and max(tops) > 50:  # gaps spanning decades: linear below 1 %, logarithmic above
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(min(min(lows) * 1.5, -0.3), max(tops) * 3)
    elif tops:
        ax.set_ylim(min(min(lows) * 1.5, -0.1 * max(1.0, max(tops))), max(tops) * 1.15 + 0.1)
    ax.set_xlim(-0.5, len(lams) - 0.5)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, title=problem_dir.name,
              title_fontsize=9, fontsize=10)
    fig.tight_layout()
    png = problem_dir / "gap_vs_lambda.png"
    for ext in ("png", "pdf"):
        fig.savefig(problem_dir / f"gap_vs_lambda.{ext}", dpi=300)
    plt.close(fig)
    return png


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(problem_dir: Path) -> None:
    runs = collect(problem_dir)
    if not runs:
        raise SystemExit(f"no run.json under {problem_dir}")
    lams, summary, all_rows = summarise(runs)
    png = draw(problem_dir, lams, summary)
    write_csv(problem_dir / "gap_vs_lambda.csv", summary)
    write_csv(problem_dir / "all_runs.csv", all_rows)
    print(f"wrote {png} (+ .pdf, gap_vs_lambda.csv, all_runs.csv)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem_dir", help="AutoData/<problem> folder written by tqa_vs_random.py")
    plot(Path(ap.parse_args().problem_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
