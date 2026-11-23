"""Small multiples of the hybrid-loop runs: warm-start QAOA vs its γ = 0 twin, per instance.

Reads ``results/warm_day_qaoa/loops.jsonl`` (one record per ``warm_day_qaoa.py loop`` run)
and writes ``handin/qaoa_loops.png``. Lower objective is better; the dashed line is
QOBLIB's best-known value.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from grid_runs import load_bkv, lam_tag  # noqa: E402

QAOA, BASE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: CVD ΔE 24.7, contrast >= 3:1)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main(src="results/warm_day_qaoa/loops.jsonl", out="handin/qaoa_loops.png"):
    bkv = load_bkv()
    runs = {}
    for line in Path(src).read_text().splitlines():
        r = json.loads(line)
        if r.get("cluster") == "local":
            continue
        key = (r["instance"], lam_tag(str(r["lam"])))
        if key not in runs or len(r["history"]) > len(runs[key]["history"]):
            runs[key] = r
    keys = sorted(runs, key=lambda k: (k[1] != "1e-3", k))
    ncol = min(5, len(keys))
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow + 0.4), squeeze=False, facecolor=SURFACE)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axes.flat, keys):
        r = runs[key]
        h = r["history"]
        x = [e["round"] for e in h]
        q = [e["qaoa"] / 1000 for e in h]
        b = [e["base"] / 1000 for e in h]
        ax.set_facecolor(SURFACE)
        ax.plot(x, q, color=QAOA, lw=2, marker="o", ms=4, label="warm-start QAOA (150 qubits)")
        ax.plot(x, b, color=BASE, lw=2, marker="o", ms=4, label="γ = 0 twin (same shots)")
        bk = bkv.get(f"{key[0]}_b{int(r['budget']):03d}_l{key[1]}")
        if bk is not None:
            ax.axhline(bk / 1000, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.text(x[0], bk / 1000, " best known", color=MUTED, fontsize=8, va="bottom")
        ax.text(x[-1], q[-1], f" {h[-1]['qaoa']:,}", color=INK, fontsize=8, va="center")
        ax.text(x[-1], b[-1], f" {h[-1]['base']:,}", color=INK, fontsize=8, va="center")
        ax.set_title(f"{key[0].replace('po_', '')}  λ {key[1]}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("round (one merged qoro_cpu job)", fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.set_xlim(0, x[-1] + max(2, 0.25 * x[-1]))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for row in axes:
        row[0].set_ylabel("objective (thousands; lower = better)", fontsize=8, color=MUTED)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.998))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (3.1 * nrow + 0.4)))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out} ({len(keys)} instances)")


if __name__ == "__main__":
    main()
