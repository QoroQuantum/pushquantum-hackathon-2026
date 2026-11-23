"""Slide-1 plot: head-to-head on the Aquora leaderboard, our team vs the best other team.

One bar pair per board where our team and at least one other team both submitted, ordered by
instance size. Bar = score as a % of the board's best score (objectives are negative, so this is
score / best); our team in blue, the best other team in grey. Ties on a010 are merged into one pair.

    .venv/bin/python plot_head_to_head.py            # handin/head_to_head_dark.png
    .venv/bin/python plot_head_to_head.py --light    # handin/head_to_head_light.png
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAM = {"Jules02", "Sahilkar", "Hugo_Tadeo", "Akuji"}  # our team's Aquora accounts
LEADERBOARD = "results/leaderboard_2026-11-22.csv"

THEMES = {
    "dark": dict(surface="#111113", ink="#f0efec", ink2="#c3c2b7", muted="#8a897f", grid="#2a2a2c",
                 us="#3987e5", other="#5f5e58"),
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a897f", grid="#e6e5e1",
                  us="#2a78d6", other="#bdbcb5"),
}


def load_boards():
    """[(instance, vars, ours, best_other)] for boards with both our team and another team."""
    best, nv = defaultdict(dict), {}
    with open(LEADERBOARD) as f:
        for r in csv.DictReader(f):
            side = "us" if r["user"] in TEAM else "them"
            i, o = r["instance"], int(r["objective"])
            nv[i] = int(r["variables"])
            best[i][side] = min(best[i].get(side, o), o)
    return sorted(((i, nv[i], d["us"], d["them"]) for i, d in best.items() if len(d) == 2),
                  key=lambda x: (x[1], x[0]))


def short(inst):
    m = re.match(r"po_(a\d{3})_(t\d{2})_(\w+?)_b\d{3}_l(.+)", inst)
    a, t, var, lam = m.groups()
    return f"{t}_{var}\nλ = {lam}"


def main(theme="dark"):
    t = THEMES[theme]
    boards = load_boards()
    wins = sum(us < them for _, _, us, them in boards)
    ties = sum(us == them for _, _, us, them in boards)
    losses = sum(us > them for _, _, us, them in boards)

    # merge the a010 ties into one pair
    a010 = [b for b in boards if b[1] == 710]
    rest = [b for b in boards if b[1] != 710]
    pairs = []
    if a010:
        assert all(us == them for _, _, us, them in a010)
        pairs.append((f"{len(a010)} boards\n(all tied)", 710, 100.0, 100.0))
    for inst, n, us, them in rest:
        b = min(us, them)
        pairs.append((short(inst), n, us / b * 100, them / b * 100))

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(13.33, 7.0), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.2)

    # x positions with an extra gap between size groups
    xs, x, prev = [], 0.0, None
    for _, n, _, _ in pairs:
        if prev is not None:
            x += 1.0 if n == prev else 1.7
        xs.append(x)
        prev = n
    w = 0.38
    for xi, (_, _, us, them) in zip(xs, pairs):
        ax.bar(xi - w / 2 - 0.02, us, w, color=t["us"], zorder=3)
        ax.bar(xi + w / 2 + 0.02, them, w, color=t["other"], zorder=3)
        lab = ("tie" if us == them else f"{them:.2f}%" if them >= 99 else f"{them:.1f}%" if them >= 90
               else f"{them:.0f}%")
        ax.text(xi + w / 2 + 0.02, them + 1.5, lab, ha="center", va="bottom", fontsize=12,
                color=t["ink2"], fontweight="bold" if lab != "tie" else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=10.5, color=t["ink2"])

    # size-group labels under the ticks
    groups = defaultdict(list)
    for xi, (_, n, _, _) in zip(xs, pairs):
        groups[n].append(xi)
    names = {710: "a010", 3110: "a050_t10", 4665: "a050_t15", 12110: "a200_t10", 18165: "a200_t15", 36165: "a400"}
    for n, gx in groups.items():
        cx = (min(gx) + max(gx)) / 2
        ax.annotate(f"{names.get(n, n)}  ·  {n:,} vars", xy=(cx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -58), textcoords="offset points", ha="center", va="top", fontsize=12,
                    fontweight="bold", color=t["ink"])
        if len(gx) > 1:
            ax.annotate("", xy=(min(gx) - 0.45, 0), xycoords=("data", "axes fraction"),
                        xytext=(max(gx) + 0.45, 0), textcoords=("data", "axes fraction"))
            line_y = -0.175
            ax.plot([min(gx) - 0.45, max(gx) + 0.45], [line_y, line_y], transform=ax.get_xaxis_transform(),
                    color=t["muted"], lw=1, clip_on=False)

    ax.set_ylim(0, 112)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=t["ink2"], labelsize=10.5, length=0)
    ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.06, 0.915, f"{wins} wins  ·  {ties} ties  ·  {losses} losses", color=t["us"], fontsize=34,
             fontweight="bold", ha="left")
    fig.text(0.06, 0.855, "against every other team on the Aquora leaderboard, on each board where both submitted",
             color=t["ink"], fontsize=15, ha="left")
    fig.text(0.06, 0.805, "Bar height: score as % of the best score on that board (higher is better).",
             color=t["ink2"], fontsize=11.5, ha="left")
    # legend, top right
    for k, (lab, c) in enumerate([("our team", t["us"]), ("best other team", t["other"])]):
        y = 0.905 - k * 0.045
        fig.patches.append(plt.Rectangle((0.80, y - 0.012), 0.014, 0.026, color=c, transform=fig.transFigure))
        fig.text(0.82, y, lab, color=t["ink2"], fontsize=12, va="center")

    out = f"handin/head_to_head_{theme}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    print("wrote", out, f"({wins}W {ties}T {losses}L)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--light", action="store_true")
    main("light" if p.parse_args().light else "dark")
