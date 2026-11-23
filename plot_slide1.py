"""Slide 1: hero numbers on the left, the two contested a400 boards on the right.

Numbers are computed, not typed: hand-in file count from ``handin/*.sol``, the largest instance
from ``handin/RESULTS.md``, the win/tie/loss record and the a400 bars from the Aquora leaderboard
(``results/leaderboard_2026-11-22.csv``, our team merged; see ``plot_head_to_head.py``).

    .venv/bin/python plot_slide1.py            # handin/slide1_dark.png
    .venv/bin/python plot_slide1.py --light    # handin/slide1_light.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_head_to_head import THEMES, load_boards  # noqa: E402

A400 = [("po_a400_t15_orig_b100_l1e-3", "a400_t15_orig"), ("po_a400_t15_s00_b100_l1e-3", "a400_t15_s00")]


def largest_instance():
    best = (0, "")
    for line in open("handin/RESULTS.md"):
        if line.startswith("| `po_"):
            c = [x.strip() for x in line.split("|")]
            best = max(best, (int(c[3].replace(",", "")), c[1].strip("`").split("_b")[0]))
    return best


def millions(x):
    return f"{x / 1e6:.2f}M".replace("-", "−")


def main(theme="dark"):
    t = THEMES[theme]
    n_files = len(list(Path("handin").glob("*.sol")))
    n_vars, big = largest_instance()
    boards = load_boards()
    wins = sum(us < them for *_, us, them in boards)
    ties = sum(us == them for *_, us, them in boards)
    losses = sum(us > them for *_, us, them in boards)
    by_inst = {i: (us, them) for i, _, us, them in boards}

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=t["surface"])

    # ---- left: hero numbers
    tiles = [
        (f"{n_files}", "feasible solution files", "every one re-scored by the official referee"),
        (f"{n_vars:,}", "binary variables", f"largest instance solved: {big}"),
        (f"{wins} · {ties} · {losses}", "wins · ties · losses", "head-to-head vs every other team on Aquora"),
    ]
    x0, y = 0.06, 0.80
    for big_txt, label, note in tiles:
        fig.text(x0, y, big_txt, color=t["us"], fontsize=54, fontweight="bold", ha="left", va="baseline")
        fig.text(x0, y - 0.065, label, color=t["ink"], fontsize=17, ha="left", va="baseline")
        fig.text(x0, y - 0.105, note, color=t["ink2"], fontsize=12, ha="left", va="baseline")
        y -= 0.27

    # ---- right: a400 head-to-head
    ax = fig.add_axes([0.56, 0.16, 0.40, 0.62], facecolor=t["surface"])
    w, gap = 0.36, 0.04
    for k, (inst, name) in enumerate(A400):
        us, them = by_inst[inst]
        best = min(us, them)
        for dx, val, raw, c in [(-(w + gap) / 2, us / best * 100, us, t["us"]),
                                ((w + gap) / 2, them / best * 100, them, t["other"])]:
            ax.bar(k + dx, val, w, color=c, zorder=3)
            ax.text(k + dx, val + 2, f"{val:.1f}%" if val < 100 else "100%", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color=t["ink"])
            ax.text(k + dx, 4, millions(raw), ha="center", va="bottom", fontsize=11,
                    color="white" if c == t["us"] or theme == "dark" else t["ink"], zorder=4)
    ax.set_xticks(range(len(A400)))
    ax.set_xticklabels([n for _, n in A400], fontsize=13, color=t["ink"])
    ax.set_ylim(0, 115)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=t["muted"], lw=1, zorder=4)
    ax.set_xlim(-0.6, len(A400) - 0.4)

    fig.text(0.56, 0.885, "The largest instance: 400 assets × 15 days", color=t["ink"], fontsize=17,
             fontweight="bold", ha="left")
    fig.text(0.56, 0.845, "Score as % of the best on each Aquora board (higher is better)", color=t["ink2"],
             fontsize=12, ha="left")
    for k, (lab, c) in enumerate([("our team", t["us"]), ("best other team", t["other"])]):
        xl = 0.56 + k * 0.12
        fig.patches.append(plt.Rectangle((xl, 0.793), 0.013, 0.024, color=c, transform=fig.transFigure))
        fig.text(xl + 0.019, 0.805, lab, color=t["ink2"], fontsize=12, va="center")

    out = f"handin/slide1_{theme}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    print("wrote", out, f"({n_files} files, {n_vars} vars, {wins}W {ties}T {losses}L)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--light", action="store_true")
    main("light" if p.parse_args().light else "dark")
