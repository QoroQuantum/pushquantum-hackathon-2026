"""Slide-1 "frontier" plot: solution quality vs instance size, us vs the leaderboard.

x = binary variables of the instance (log). y = leaderboard-style gap (score - ref) / |ref| in %
(symlog, linear within ±10%, axis inverted so negative = better is at the top): the reference is QOBLIB's best-known value (BKV) where one exists, otherwise
(a200 / a400) the best score anyone has, on the leaderboard or in our local results.

Ours = ``handin/RESULTS.md`` (all 130 hand-in files) plus local runs not yet on the leaderboard
(``EXTRA_LOCAL``). Others = ``results/leaderboard_2026-11-22.csv`` minus our own rows.

    .venv/bin/python plot_frontier.py            # dark, handin/frontier_dark.png
    .venv/bin/python plot_frontier.py --light    # light, handin/frontier_light.png
"""

from __future__ import annotations

import argparse
import csv
import re
import zlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

TEAM = {"Jules02", "Sahilkar", "Hugo_Tadeo", "Akuji"}  # our team's Aquora accounts
LEADERBOARD = "results/leaderboard_2026-11-22.csv"
# local .sol files not (yet) on Aquora: instance -> candidate files, best feasible one is used
EXTRA_LOCAL = {
    "po_a400_t15_orig_b100_l1e-2": ["results/day_swap_par_l1e-2/po_a400_t15_orig_b100_l1e-2.sol",
                                    "results/day_swap_par_l1e-2/warm_l1e-3_start.sol"],
}
NVARS = {"a010_t10": 710, "a010_t15": 1065, "a050_t10": 3110, "a050_t15": 4665,
         "a200_t10": 12110, "a200_t15": 18165, "a400_t10": 24110, "a400_t15": 36165}

THEMES = {
    "dark": dict(surface="#111113", ink="#f0efec", ink2="#c3c2b7", muted="#8a897f", grid="#2a2a2c",
                 us="#3987e5", other="#6e6d66", band="#18181b"),
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a897f", grid="#e6e5e1",
                  us="#2a78d6", other="#b3b2ab", band="#f3f2ee"),
}


def stem(name: str) -> str:
    return re.sub(r"\.(sol|json)$", "", name.strip("` "))


def local_score(path):
    """Official re-score of a local .sol (None if missing or infeasible)."""
    import qoblib_tools as qt
    if not Path(path).exists():
        return None
    h, pos = qt.read_portfolio_solution(path)
    sc = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]),
                           lam=h["lambda"]).score(pos)
    return sc.objective if sc.feasible else None


def load_ours(board):
    """instance -> (vars, score, bkv or None, not_on_aquora). RESULTS.md and our leaderboard rows
    count as submitted; EXTRA_LOCAL files count as not submitted when they beat those."""
    ours = {}
    for line in open("handin/RESULTS.md"):
        if not line.startswith("| `po_"):
            continue
        c = [x.strip() for x in line.split("|")]
        bkv = None if c[5] in ("—", "") else int(c[5])
        ours[stem(c[1])] = (int(c[3].replace(",", "")), int(c[4]), bkv, False)
    for inst, users in board.items():
        mine = [o for u, o in users.items() if u in TEAM]
        if mine and (inst not in ours or min(mine) < ours[inst][1]):
            nv = NVARS[re.search(r"a\d{3}_t\d{2}", inst).group()]
            ours[inst] = (nv, min(mine), ours[inst][2] if inst in ours else None, False)
    for inst, paths in EXTRA_LOCAL.items():
        scores = [s for s in map(local_score, paths) if s is not None]
        if scores and (inst not in ours or min(scores) < ours[inst][1]):
            nv = NVARS[re.search(r"a\d{3}_t\d{2}", inst).group()]
            ours[inst] = (nv, min(scores), None, True)
    return ours


def load_board():
    """instance -> {user: best objective}, instance -> (vars, bkv or None)."""
    best, meta = defaultdict(dict), {}
    with open(LEADERBOARD) as f:
        for r in csv.DictReader(f):
            inst, user, obj = r["instance"], r["user"], int(r["objective"])
            meta[inst] = (int(r["variables"]), int(r["best known"]) if r["best known"] else None)
            if user not in best[inst] or obj < best[inst][user]:
                best[inst][user] = obj
    return best, meta


def gap_pct(score, ref):
    """Leaderboard gap in %: (score - ref) / |ref|; negative beats the reference."""
    return (score - ref) / abs(ref) * 100.0


def jitter(x, key, spread=0.07):
    rng = np.random.default_rng(zlib.crc32(repr(key).encode()))
    return x * 10 ** rng.uniform(-spread, spread)


def main(theme="dark", out=None):
    t = THEMES[theme]
    out = out or f"handin/frontier_{theme}.png"
    board, meta = load_board()
    ours = load_ours(board)

    # reference per instance: BKV, else best score anyone has (board or ours)
    insts = set(ours) | set(board)
    ref = {}
    for i in insts:
        bkv = ours[i][2] if i in ours and ours[i][2] is not None else (meta.get(i, (0, None))[1])
        if bkv is not None:
            ref[i] = bkv
        else:
            cands = list(board.get(i, {}).values()) + ([ours[i][1]] if i in ours else [])
            ref[i] = min(cands)

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(13.33, 6.6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.075, right=0.775, top=0.80, bottom=0.12)

    # no-BKV region
    ax.axvspan(9000, 60000, color=t["band"], zorder=0, lw=0)
    ax.text(10000, -380, "no QOBLIB best-known value:\nscored against the best result anywhere", color=t["muted"],
            fontsize=9.5, va="top", ha="left", linespacing=1.3)

    ax.axhline(0, color=t["muted"], lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(560, -0.3, "QOBLIB best-known value", color=t["muted"], fontsize=9, va="bottom")

    # other teams
    others = []
    for inst, users in board.items():
        for user, obj in users.items():
            if user in TEAM:
                continue
            nv = meta[inst][0]
            others.append((user, inst, nv, gap_pct(obj, ref[inst])))
    ox = [jitter(nv, ("o", u, i)) for u, i, nv, _ in others]
    oy = [y for *_, y in others]
    ax.scatter(ox, oy, s=46, color=t["other"], edgecolor=t["surface"], linewidth=1.2, zorder=3,
               label="other teams")

    # ours
    pts = [(i, nv, gap_pct(s, ref[i]), local) for i, (nv, s, _b, local) in ours.items()]
    sub = [p for p in pts if not p[3]]
    loc = [p for p in pts if p[3]]
    glow = [pe.withStroke(linewidth=6, foreground=t["us"], alpha=0.18)]
    ax.scatter([jitter(nv, ("u", i), 0.07 if nv < 9000 else 0) for i, nv, _, _ in sub], [y for _, _, y, _ in sub], s=40, color=t["us"],
               edgecolor=t["surface"], linewidth=1.0, zorder=5, alpha=0.95, label="ours, submitted on Aquora")
    ax.scatter([nv * 1.12 for _, nv, _, _ in loc], [y for _, _, y, _ in loc], s=90, facecolor=t["surface"],
               edgecolor=t["us"], linewidth=2.2, zorder=6, label="ours, not yet submitted on Aquora")

    # frontier: our worst result at each size, carried out to the largest instance
    by_n = defaultdict(list)
    for _, nv, y, _ in pts:
        by_n[nv].append(y)
    xs = sorted(by_n)
    ys = [max(by_n[n]) for n in xs]
    ax.plot(xs, ys, color=t["us"], lw=2.4, zorder=4, solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=9, foreground=t["us"], alpha=0.15)])

    # annotations (numbers recomputed from the data)
    def ahead(inst):
        mine = ours[inst][1]
        rivals = [o for u, o in board.get(inst, {}).items() if u not in TEAM]
        return (min(rivals) - mine) / abs(min(rivals)) * 100 if rivals else None

    a400, a400s, a200 = "po_a400_t15_orig_b100_l1e-3", "po_a400_t15_s00_b100_l1e-3", "po_a200_t10_orig_b050_l1e-3"
    a400h = "po_a400_t15_orig_b100_l1e-2"
    kw = dict(fontsize=10.5, color=t["ink"], linespacing=1.35, zorder=8)
    ax.annotate(f"a400_t15 · 36,165 vars\n#1 on orig and s00: {ahead(a400):.1f}% and\n{ahead(a400s):.1f}% better than the next team",
                xy=(36165, 0), xytext=(40500, -6.5), ha="left", va="center", **kw,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
    on_board = min(o for u, o in board[a400h].items() if u in TEAM)
    ax.annotate(f"a400_t15 at λ = 1e-2 (not yet submitted on Aquora): {ours[a400h][1] / 1e6:.2f}M "
                f"vs our {on_board / 1e6:.2f}M on the board".replace("-", "−"),
                xy=(36165 * 1.12, 0), xytext=(36165, 2.2), ha="right", va="center", **kw,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
    ax.annotate("a200_t10 and a200_t15 · up to 18,165 vars\nno other team has entered",
                xy=(12110, 0), xytext=(13500, -9), ha="left", va="center", **kw,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
    # bracket: our a400 margin over the next team, in gap points on this axis
    rival_gap = min(gap_pct(o, ref[a400]) for u, o in board[a400].items() if u not in TEAM)
    bx = 36165 * 1.3
    ax.annotate("", xy=(bx, 0), xytext=(bx, rival_gap), zorder=7,
                arrowprops=dict(arrowstyle="<->", color=t["us"], lw=1.6, shrinkA=0, shrinkB=0))
    ax.text(bx * 1.04, rival_gap / 2, f"+{rival_gap:.1f}%\nnext team", color=t["us"], fontsize=10.5,
            fontweight="bold", ha="left", va="center", linespacing=1.2, zorder=8)

    beats = [y for _, _, y, _ in pts if y < -0.005]
    ax.text(880, min(beats) * 1.05, f"{len(beats)} instances beat the BKV\ngaps down to −{abs(min(beats)):.0f}% (λ = 1e-2)",
            ha="left", va="center", **kw)


    # axes
    ax.set_xscale("log")
    ax.set_xlim(560, 60000)
    ax.xaxis.set_major_locator(FixedLocator([710, 1065, 3110, 4665, 12110, 18165, 36165]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["710\na010_t10", "1,065\na010_t15", "3,110\na050_t10", "4,665\na050_t15",
                        "12,110\na200_t10", "18,165\na200_t15", "36,165\na400_t15"])
    ax.set_yscale("symlog", linthresh=10, linscale=2.5)  # linear in ±10%: the contested range
    ax.set_ylim(90, -420)  # inverted: negative gap (better) at the top
    ax.yaxis.set_major_locator(FixedLocator([60, 30, 10, 5, 0, -5, -10, -100, -300]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["+60%", "+30%", "+10%", "+5%", "0%", "−5%", "−10%", "−100%", "−300%"])
    ax.set_xlabel("binary variables in the instance (log scale)", color=t["ink2"], fontsize=11, labelpad=8)
    ax.set_ylabel("gap to best known  →  (negative = better)", color=t["ink2"], fontsize=11)
    ax.tick_params(colors=t["ink2"], labelsize=9.5, length=0)
    ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.075, 0.93, "Best-known quality at every size, out to 36,165 variables", color=t["ink"],
             fontsize=21, fontweight="bold", ha="left")
    fig.text(0.075, 0.875, "Gap to the best-known value (BKV), as on the leaderboard: negative beats it. Every instance and λ; "
             "line = our worst gap at each size (+0.05%).", color=t["ink2"], fontsize=11, ha="left")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10, labelcolor=t["ink2"],
                    handletextpad=0.5, borderaxespad=0.6)
    for h in leg.legend_handles:
        h.set_sizes([60])
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--light", action="store_true")
    a = p.parse_args()
    main("light" if a.light else "dark")
