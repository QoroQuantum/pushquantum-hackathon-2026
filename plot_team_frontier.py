"""Scale x quality frontier, one marker per team on the Aquora leaderboard.

x = largest instance (binary variables) the team has entered; y = share of contested boards
(boards with at least one other team) that the team won or tied. Marker area ~ contested boards.
Our team's accounts are merged (``TEAM`` in plot_head_to_head.py); other teams are not named.

    .venv/bin/python plot_team_frontier.py            # handin/team_frontier_dark.png
    .venv/bin/python plot_team_frontier.py --light    # handin/team_frontier_light.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from plot_head_to_head import LEADERBOARD, TEAM, THEMES  # noqa: E402


def team_stats():
    board, nv = defaultdict(dict), {}
    with open(LEADERBOARD) as f:
        for r in csv.DictReader(f):
            u = "us" if r["user"] in TEAM else r["user"]
            i, o = r["instance"], int(r["objective"])
            nv[i] = int(r["variables"])
            board[i][u] = min(board[i].get(u, o), o)
    stats = {}
    for u in {u for d in board.values() for u in d}:
        entered = [i for i in board if u in board[i]]
        contested = [i for i in entered if len(board[i]) > 1]
        rivals = {i: min(o for v, o in board[i].items() if v != u) for i in contested}
        wins = sum(board[i][u] < rivals[i] for i in contested)
        ties = sum(board[i][u] == rivals[i] for i in contested)
        stats[u] = dict(x=max(nv[i] for i in entered), n=len(contested), wins=wins, ties=ties,
                        y=100 * (wins + ties) / len(contested) if contested else 0.0)
    return stats


def main(theme="dark"):
    t = THEMES[theme]
    st = team_stats()
    us = st.pop("us")

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.09, right=0.95, top=0.78, bottom=0.14)

    size = lambda n: 180 + 120 * n  # noqa: E731

    def desc(s):
        parts = [f"{s['wins']} win{'s' * (s['wins'] != 1)}", f"{s['ties']} tie{'s' * (s['ties'] != 1)}"]
        return f"{s['n']} contested board{'s' * (s['n'] != 1)}: " + ", ".join(parts)

    # other teams
    for k, s in enumerate(sorted(st.values(), key=lambda s: (s["x"], s["y"]))):
        ax.scatter(s["x"], s["y"], s=size(s["n"]), color=t["other"], zorder=3, edgecolor=t["surface"], lw=2)
        right = s["x"] < 5000
        ax.annotate(f"Other team\n{desc(s)}", xy=(s["x"], s["y"]), xytext=(28 if right else -28, -6),
                    textcoords="offset points", ha="left" if right else "right", va="top", fontsize=13,
                    color=t["ink2"], linespacing=1.35)

    # us, with glow
    for r, a in [(5.0, 0.06), (3.2, 0.10), (2.0, 0.16)]:
        ax.scatter(us["x"], us["y"], s=size(us["n"]) * r, color=t["us"], alpha=a, lw=0, zorder=4, clip_on=False)
    ax.scatter(us["x"], us["y"], s=size(us["n"]), color=t["us"], zorder=5, edgecolor=t["surface"], lw=2)
    ax.annotate(f"Our team\n{desc(us)}", xy=(us["x"], us["y"]), xytext=(-34, -8), textcoords="offset points",
                ha="right", va="top", fontsize=15, fontweight="bold", color=t["us"], linespacing=1.35,
                path_effects=[pe.withStroke(linewidth=4, foreground=t["surface"])])

    # frontier: the best win/tie share reached at or beyond each scale
    ax.plot([500, us["x"]], [us["y"], us["y"]], color=t["us"], lw=2, ls=(0, (4, 3)), alpha=0.6, zorder=2)
    ax.text(560, us["y"] + 2.5, "frontier: every contested board won or tied", color=t["us"], fontsize=11,
            alpha=0.9, va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(500, 60000)
    ax.set_ylim(-5, 112)
    ax.xaxis.set_major_locator(FixedLocator([710, 3110, 12110, 36165]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["710\na010", "3,110\na050", "12,110\na200", "36,165\na400"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("largest instance entered on Aquora  (binary variables, log scale)  →", color=t["ink2"],
                  fontsize=12, labelpad=10)
    ax.set_ylabel("contested boards won or tied  →", color=t["ink2"], fontsize=12, labelpad=10)
    ax.tick_params(colors=t["ink2"], labelsize=11.5, length=0, pad=8)
    ax.grid(color=t["grid"], lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.09, 0.905, "The largest instances, and the best results on them", color=t["ink"], fontsize=26,
             fontweight="bold", ha="left")
    fig.text(0.09, 0.85, "Every team on the Aquora leaderboard. A board is contested when at least two teams "
             "submitted to it. Marker size = contested boards.", color=t["ink2"], fontsize=12.5, ha="left")

    out = f"handin/team_frontier_{theme}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    print("wrote", out, us, st)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--light", action="store_true")
    main("light" if p.parse_args().light else "dark")
