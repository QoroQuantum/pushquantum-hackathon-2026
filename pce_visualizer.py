#!/usr/bin/env python3
"""Real-instance Pauli Correlation Encoding (PCE) QUBO Graph Transformation Visualizer.

Visualizes a real QOBLIB portfolio optimization QUBO graph before PCE and animates
its exact algebraic transformation under Divi's Pauli Correlation Encoding
(``PolyEncoding``, ``DenseEncoding``, or ``MultiBasisEncoding``) into the compressed
post-PCE qubit interaction graph.

By default, uses the smallest real QOBLIB portfolio problem:
``po_a003_t02_orig`` (3 stocks: AAPL, NVDA, MSFT; B=1 unit -> 6 binary variables,
15 quadratic edges -> 3 qubits under PCE) and highlights a SINGLE tracked edge so the
viewer can easily follow its exact Pauli-product rewiring from pre-PCE to post-PCE.

Usage:
    python pce_visualizer.py
    python pce_visualizer.py --spotlight-edge 0
    python pce_visualizer.py --encoding dense
    python pce_visualizer.py --budget 3 --no-budget-equality
    python pce_visualizer.py --save pce_transform.gif
    python pce_visualizer.py --snapshots ./pce_frames
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# Ensure sibling modules (qoblib_tools, portfolio_compact, per_day_pce) are importable
# regardless of the current working directory.
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Avoid slow / unwritable ~/.matplotlib cache warnings in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import dimod
import numpy as np

from divi.qprog.algorithms import DenseEncoding, MultiBasisEncoding, PolyEncoding
from divi.qprog.problems import BinaryOptimizationProblem

import qoblib_tools as qt
from portfolio_compact import CompactPortfolioQUBO


# ── Real instance loading ────────────────────────────────────────────────


def load_local_or_cached_instance(name: str) -> qt.PortfolioInstance:
    """Load a QOBLIB portfolio instance from local ``instances/`` if available,
    otherwise fall back to ``qoblib_tools.load_portfolio_instance``."""
    local_dir = REPO_DIR / "instances" / name
    prices_path = local_dir / "stock_prices.txt.gz"
    cov_path = local_dir / "covariance_matrices.txt.gz"
    if prices_path.exists() and cov_path.exists():
        symbols: list[str] = []
        index: dict[str, int] = {}
        entries = []
        for day, sym, price in qt._data_rows(prices_path):
            if sym not in index:
                index[sym] = len(symbols)
                symbols.append(sym)
            entries.append((index[sym], int(day), Fraction(price)))
        n_periods = max(t for _, t, _ in entries) + 1
        prices: list[list[Fraction | None]] = [[None] * n_periods for _ in symbols]
        for i, t, p in entries:
            prices[i][t] = p
        cov = {}
        for day, s1, s2, value in qt._data_rows(cov_path):
            t = int(day)
            if s1 in index and s2 in index and t < n_periods:
                cov[index[s1], index[s2], t] = Fraction(value)
        return qt.PortfolioInstance(name, symbols, n_periods, prices, cov)
    return qt.load_portfolio_instance(name)


def format_var_label(raw_label) -> tuple[str, str]:
    """Return ``(short_label, group_name)`` for a CompactPortfolioQUBO variable label.

    Uses 1-indexed Long/Short unit copy labels (e.g. ``AAPL L1``, ``AAPL S1``)
    instead of 0-indexed ``+0``/``-0`` for immediate readability.
    """
    if isinstance(raw_label, tuple):
        if raw_label[0] == "x":
            _, sym, copy_idx, tau, day = raw_label
            direction = "L" if tau > 0 else "S"
            return f"{sym} {direction}{copy_idx + 1}", str(sym)
        if raw_label[0] == "s":
            _, bit_idx, day = raw_label
            return f"slack{bit_idx + 1}", "Slack"
        if raw_label[0] == "y":
            _, bit_idx, day = raw_label
            return f"cash{bit_idx + 1}", "Cash"
    return str(raw_label), "Var"


def build_real_qubo(
    instance_name: str,
    budget: int,
    lam: str,
    budget_equality: bool = True,
    day: int | None = None,
) -> tuple[dimod.BinaryQuadraticModel, list[str], list[str], str]:
    """Build a real QOBLIB portfolio optimization QUBO (full or single-day)."""
    inst = load_local_or_cached_instance(instance_name)
    model = qt.PortfolioModel(inst, budget=budget, lam=lam)
    compact = CompactPortfolioQUBO(model, budget_equality=budget_equality)

    if day is None:
        bqm = compact.objective.copy()
        labels, groups = zip(*(format_var_label(lab) for lab in compact.labels))
        desc = f"{instance_name} (B={budget}, λ={lam})"
        return bqm, list(labels), list(groups), desc

    from per_day_pce import day_problem

    n_days = model.risk.shape[0]
    L0 = np.zeros((n_days, len(model.symbols)), dtype=int)
    S0 = np.zeros((n_days, len(model.symbols)), dtype=int)
    dp = day_problem(compact, day, L0, S0)
    labels = []
    groups = []
    for asset_idx, copy_idx, tau in dp.slot:
        sym = model.symbols[asset_idx]
        direction = "L" if tau > 0 else "S"
        labels.append(f"{sym} {direction}{copy_idx + 1}")
        groups.append(str(sym))
    desc = f"{instance_name} day {day} (B={budget}, λ={lam})"
    return dp.bqm, labels, groups, desc


# ── Exact Divi PCE Algebraic Transformation ──────────────────────────────


@dataclass
class PCETransformation:
    """Exact pre-PCE QUBO and post-PCE qubit graph computed via Divi."""

    instance_desc: str
    encoding_name: str
    is_parity_encoding: bool
    n_vars: int
    n_qubits: int
    var_labels: list[str]
    var_groups: list[str]

    # Pre-PCE Ising representation: E(s) = const + sum_i h_i s_i + sum_{u<v} J_{uv} s_u s_v
    h_pre: np.ndarray
    edges_pre: list[tuple[int, int, float]]  # (u, v, J_uv)

    # Exact Divi PCE variable mapping
    pauli_strings: list[tuple[str, tuple[int, ...]]]
    pauli_labels: list[str]
    var_supports: list[tuple[int, ...]]

    # Exact per-edge algebraic image under PCE
    edge_target_supports: list[tuple[int, ...]]

    # Exact post-PCE Hamiltonian terms keyed by sorted qubit tuple
    post_1body: dict[int, float]
    post_2body: dict[tuple[int, int], float]
    post_higher: dict[tuple[int, ...], float]  # |S| >= 3
    edge_weight_counts: dict[int, int]
    unique_term_counts: dict[int, int]


SUB_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_pauli(letters: str, qubits: tuple[int, ...]) -> str:
    """Format e.g. ('ZZ', (1, 4)) -> 'Z₁Z₄'."""
    return "".join(f"{ch}{str(q).translate(SUB_MAP)}" for ch, q in zip(letters, qubits))


def format_z_support(qubits: tuple[int, ...]) -> str:
    """Format e.g. (0, 1) -> 'Z₀Z₁'."""
    return "".join(f"Z{str(q).translate(SUB_MAP)}" for q in qubits)


def compute_pce_transformation(
    bqm: dimod.BinaryQuadraticModel,
    var_labels: list[str],
    var_groups: list[str],
    instance_desc: str,
    encoding_key: str = "poly",
) -> PCETransformation:
    """Run Divi's real PCE encoding on ``bqm`` and compute the exact pre- and post-PCE graphs."""
    _ = BinaryOptimizationProblem(bqm)

    n_vars = bqm.num_variables
    if encoding_key == "poly":
        enc = PolyEncoding()
        enc_name = "PolyEncoding (Divi)"
        is_parity = True
    elif encoding_key == "dense":
        enc = DenseEncoding()
        enc_name = "DenseEncoding (Divi)"
        is_parity = True
    elif encoding_key == "multibasis":
        enc = MultiBasisEncoding(order=2)
        enc_name = "MultiBasisEncoding(k=2) (Divi)"
        is_parity = False
    else:
        raise ValueError(f"Unknown encoding: {encoding_key}")

    n_qubits = enc.min_qubits(n_vars)
    pauli_strings = list(enc.strings(n_vars, n_qubits))
    pauli_labels = [format_pauli(letters, qubits) for letters, qubits in pauli_strings]
    var_supports = [tuple(sorted(qubits)) for _, qubits in pauli_strings]

    # Exact QUBO -> Ising conversion using x_i = (1 - s_i) / 2
    h_pre = np.zeros(n_vars, dtype=float)
    edge_map: dict[tuple[int, int], float] = defaultdict(float)
    for i in range(n_vars):
        h_pre[i] -= 0.5 * float(bqm.linear.get(i, 0.0))
    for (u, v), w in bqm.quadratic.items():
        w = float(w)
        if abs(w) < 1e-12:
            continue
        if u > v:
            u, v = v, u
        h_pre[u] -= 0.25 * w
        h_pre[v] -= 0.25 * w
        edge_map[(u, v)] += 0.25 * w

    edges_pre = [(u, v, w) for (u, v), w in sorted(edge_map.items()) if abs(w) > 1e-12]

    # Exact PCE operator substitution
    post_terms: dict[tuple[int, ...], float] = defaultdict(float)
    for i in range(n_vars):
        post_terms[var_supports[i]] += h_pre[i]

    edge_target_supports: list[tuple[int, ...]] = []
    for u, v, w in edges_pre:
        su, sv = set(var_supports[u]), set(var_supports[v])
        if is_parity:
            target = tuple(sorted(su ^ sv))
        else:
            target = tuple(sorted(su | sv))
        edge_target_supports.append(target)
        if is_parity:
            post_terms[target] += w
        else:
            for a_idx in range(len(target)):
                for b_idx in range(a_idx + 1, len(target)):
                    post_terms[(target[a_idx], target[b_idx])] += w / max(1, len(target) - 1)

    post_1body = {k[0]: v for k, v in post_terms.items() if len(k) == 1 and abs(v) > 1e-12}
    post_2body = {k: v for k, v in post_terms.items() if len(k) == 2 and abs(v) > 1e-12}
    post_higher = {k: v for k, v in post_terms.items() if len(k) >= 3 and abs(v) > 1e-12}

    edge_weight_counts = dict(sorted(Counter(len(s) for s in edge_target_supports).items()))
    unique_term_counts = dict(
        sorted(Counter(len(k) for k, v in post_terms.items() if abs(v) > 1e-12).items())
    )

    return PCETransformation(
        instance_desc=instance_desc,
        encoding_name=enc_name,
        is_parity_encoding=is_parity,
        n_vars=n_vars,
        n_qubits=n_qubits,
        var_labels=var_labels,
        var_groups=var_groups,
        h_pre=h_pre,
        edges_pre=edges_pre,
        pauli_strings=pauli_strings,
        pauli_labels=pauli_labels,
        var_supports=var_supports,
        edge_target_supports=edge_target_supports,
        post_1body=post_1body,
        post_2body=post_2body,
        post_higher=post_higher,
        edge_weight_counts=edge_weight_counts,
        unique_term_counts=unique_term_counts,
    )


def pick_spotlight_edge(tx: PCETransformation, requested: int | None = None) -> int:
    """Pick a single clear edge to track through the animation.

    Prefers the strongest edge whose exact Pauli product maps to a 2-qubit edge
    (and ideally involves 2-qubit correlation variables so the cancellation is vivid).
    """
    if not tx.edges_pre:
        return -1
    if requested is not None:
        return max(0, min(len(tx.edges_pre) - 1, requested))

    # Prefer edges mapping to a 2-qubit target with both endpoints on 2-qubit variables
    candidates = [
        (abs(tx.edges_pre[i][2]), i)
        for i, tgt in enumerate(tx.edge_target_supports)
        if len(tgt) == 2
        and len(tx.var_supports[tx.edges_pre[i][0]]) >= 2
        and len(tx.var_supports[tx.edges_pre[i][1]]) >= 2
    ]
    if candidates:
        return max(candidates)[1]

    # Fallback: strongest edge mapping to any 2-qubit edge
    candidates_2 = [
        (abs(tx.edges_pre[i][2]), i)
        for i, tgt in enumerate(tx.edge_target_supports)
        if len(tgt) == 2
    ]
    if candidates_2:
        return max(candidates_2)[1]

    return int(np.argmax([abs(e[2]) for e in tx.edges_pre]))


# ── Layout & Easing Helpers ──────────────────────────────────────────────


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def compute_layouts(tx: PCETransformation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute crisp circular layouts for pre-PCE variables, post-PCE qubits, and
    the exact landing coordinate of each QUBO variable in the post-PCE graph."""
    angles_pre = np.linspace(math.pi / 2, math.pi / 2 - 2 * math.pi, tx.n_vars, endpoint=False)
    pos_pre = np.column_stack([1.02 * np.cos(angles_pre), 1.02 * np.sin(angles_pre)])

    angles_q = np.linspace(math.pi / 2, math.pi / 2 - 2 * math.pi, tx.n_qubits, endpoint=False)
    pos_qubits = np.column_stack([0.78 * np.cos(angles_q), 0.78 * np.sin(angles_q)])

    pair_slots: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, sup in enumerate(tx.var_supports):
        pair_slots[sup].append(i)

    pos_var_post = np.zeros((tx.n_vars, 2), dtype=float)
    for i, sup in enumerate(tx.var_supports):
        pts = pos_qubits[list(sup)]
        center = pts.mean(axis=0)
        siblings = pair_slots[sup]
        if len(sup) == 2 and np.linalg.norm(center) < 0.08:
            center = 0.34 * pts[0] + 0.66 * pts[1]
        if len(siblings) > 1 and len(sup) == 2:
            idx_in_sib = siblings.index(i)
            offset_k = idx_in_sib - 0.5 * (len(siblings) - 1)
            vec = pts[1] - pts[0]
            perp = np.array([-vec[1], vec[0]])
            norm = np.linalg.norm(perp)
            if norm > 1e-8:
                perp /= norm
            center = center + 0.09 * offset_k * perp
        pos_var_post[i] = center

    return pos_pre, pos_qubits, pos_var_post


# ── Visualizer & Animation Engine ────────────────────────────────────────


BG_DARK = "#0d1117"
PANEL_BG = "#161b22"
BORDER_COL = "#30363d"
TEXT_MAIN = "#f0f6fc"
TEXT_MUTED = "#8b949e"
EDGE_NEG = "#38bdf8"       # Cyan/blue: J < 0
EDGE_POS = "#fb7185"       # Coral/rose: J > 0
SPOTLIGHT_COL = "#22c55e"  # Bright neon emerald: the ONE tracked edge
HYPER_COL = "#a78bfa"      # Violet: 3+/4-body Pauli hyperedges
QUBIT_COL = "#fbbf24"      # Gold: post-PCE qubit nodes
CORR_COL = "#c084fc"       # Lavender: 2-qubit correlation variable badges

GROUP_PALETTE = [
    "#38bdf8",  # AAPL
    "#34d399",  # NVDA
    "#f472b6",  # MSFT
    "#fb923c",  # Slack
    "#a78bfa",
    "#facc15",
    "#2dd4bf",
    "#f87171",
]


def build_animation(
    tx: PCETransformation,
    n_frames: int = 180,
    fps: int = 30,
    show_hyperedges: bool = True,
    spotlight_edge: int | None = None,
    create_anim: bool = True,
):
    """Build the Matplotlib Figure and FuncAnimation for the PCE transformation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import FancyBboxPatch

    pos_pre, pos_qubits, pos_var_post = compute_layouts(tx)

    unique_groups = list(dict.fromkeys(tx.var_groups))
    group_color_map = {
        g: GROUP_PALETTE[idx % len(GROUP_PALETTE)] for idx, g in enumerate(unique_groups)
    }
    var_colors = [group_color_map[g] for g in tx.var_groups]

    # Pre-PCE edge arrays
    n_edges_pre = len(tx.edges_pre)
    pre_u = np.array([e[0] for e in tx.edges_pre], dtype=int)
    pre_v = np.array([e[1] for e in tx.edges_pre], dtype=int)
    pre_w = np.array([e[2] for e in tx.edges_pre], dtype=float)

    spot_idx = pick_spotlight_edge(tx, spotlight_edge)

    # Determine each pre-PCE edge's target endpoints in post-PCE space
    edge_target_p0 = np.zeros((n_edges_pre, 2), dtype=float)
    edge_target_p1 = np.zeros((n_edges_pre, 2), dtype=float)
    edge_target_kind = np.zeros(n_edges_pre, dtype=int)  # 1=1-body, 2=2-body, 3=higher
    for idx, sup in enumerate(tx.edge_target_supports):
        k = len(sup)
        if k == 1:
            edge_target_kind[idx] = 1
            edge_target_p0[idx] = pos_qubits[sup[0]]
            edge_target_p1[idx] = pos_qubits[sup[0]]
        elif k == 2:
            edge_target_kind[idx] = 2
            edge_target_p0[idx] = pos_qubits[sup[0]]
            edge_target_p1[idx] = pos_qubits[sup[1]]
        else:
            edge_target_kind[idx] = 3
            c = pos_qubits[list(sup)].mean(axis=0)
            edge_target_p0[idx] = c
            edge_target_p1[idx] = c

    # Post-PCE 2-body edge arrays
    post_edges = sorted(tx.post_2body.items())
    post_2_segs = np.array(
        [[pos_qubits[a], pos_qubits[b]] for (a, b), _ in post_edges], dtype=float
    )
    post_2_w = np.array([w for _, w in post_edges], dtype=float)

    # Post-PCE higher-order hyperedge polygons
    hyper_polys = []
    hyper_weights = []
    if show_hyperedges and tx.post_higher:
        sorted_higher = sorted(tx.post_higher.items(), key=lambda kv: abs(kv[1]), reverse=True)[:14]
        for sup, w in sorted_higher:
            pts = pos_qubits[list(sup)]
            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            order = np.argsort(ang)
            hyper_polys.append(pts[order])
            hyper_weights.append(abs(w))
    max_hyper_w = max(hyper_weights) if hyper_weights else 1.0

    # Node sizing (calm & balanced so all nodes read cleanly)
    q_weights = np.array([abs(tx.post_1body.get(q, 0.0)) for q in range(tx.n_qubits)], dtype=float)
    max_q_w = max(1e-9, float(np.max(q_weights))) if tx.n_qubits else 1.0
    q_sizes = 720.0 + 360.0 * (q_weights / max_q_w)

    h_abs = np.abs(tx.h_pre)
    max_h = max(1e-9, float(np.max(h_abs))) if tx.n_vars else 1.0
    base_node = 460.0 if tx.n_vars <= 10 else 240.0
    pre_var_sizes = base_node + 180.0 * (h_abs / max_h)

    is_anchor = np.array([len(s) == 1 for s in tx.var_supports], dtype=bool)
    qubit_primary_var: dict[int, int] = {}
    for i, sup in enumerate(tx.var_supports):
        if len(sup) == 1:
            qubit_primary_var[sup[0]] = i

    guide_segs_init = []
    guide_var_idx = []
    guide_q_idx = []
    for i, sup in enumerate(tx.var_supports):
        if len(sup) >= 2:
            for q in sup:
                guide_segs_init.append([pos_pre[i], pos_qubits[q]])
                guide_var_idx.append(i)
                guide_q_idx.append(q)

    # ── Figure Layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.5, 8.6), facecolor=BG_DARK)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.82, 1.0], wspace=0.04, left=0.02, right=0.98, top=0.90, bottom=0.05
    )

    ax = fig.add_subplot(gs[0, 0], facecolor=BG_DARK)
    ax_info = fig.add_subplot(gs[0, 1], facecolor=PANEL_BG)

    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.38, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    fig.suptitle(
        f"Pauli Correlation Encoding (PCE)  —  {tx.instance_desc}",
        color=TEXT_MAIN,
        fontsize=15.5,
        fontweight="bold",
        y=0.965,
    )
    subtitle_text = fig.text(
        0.5, 0.922, "", ha="center", va="center", color=TEXT_MUTED, fontsize=10.8
    )

    # ── Graph Artists (Main Canvas) ──────────────────────────────────────

    # 1. Higher-order hyperedge polygons (background)
    hyper_coll = PolyCollection(
        hyper_polys if hyper_polys else [np.zeros((3, 2))],
        facecolors=[(0.655, 0.545, 0.980, 0.0)],
        edgecolors=[(0.655, 0.545, 0.980, 0.0)],
        linewidths=1.2,
        zorder=1,
    )
    ax.add_collection(hyper_coll)

    # 2. Pre-PCE / Transforming QUBO background edges (uniform, calm styling)
    init_segs = np.stack([pos_pre[pre_u], pos_pre[pre_v]], axis=1) if n_edges_pre else np.zeros((0, 2, 2))
    pre_edge_coll = LineCollection(init_segs, linewidths=1.2, zorder=2)
    ax.add_collection(pre_edge_coll)

    # 3. Post-PCE aggregated 2-body qubit edges
    post_edge_coll = LineCollection(
        post_2_segs if len(post_2_segs) else np.zeros((0, 2, 2)),
        linewidths=2.4,
        zorder=3,
    )
    ax.add_collection(post_edge_coll)

    # 4. Dashed PCE assignment guide lines (Phase 2)
    guide_coll = LineCollection(
        guide_segs_init if guide_segs_init else np.zeros((0, 2, 2)),
        colors=CORR_COL,
        linestyles=(0, (4, 4)),
        linewidths=1.0,
        alpha=0.0,
        zorder=3,
    )
    ax.add_collection(guide_coll)

    # 5. SINGLE SPOTLIGHT EDGE (glow + bright core + floating callout)
    spot_glow_line, = ax.plot([], [], color=SPOTLIGHT_COL, linewidth=11.0, alpha=0.28, solid_capstyle="round", zorder=4)
    spot_core_line, = ax.plot([], [], color=SPOTLIGHT_COL, linewidth=4.0, alpha=0.98, solid_capstyle="round", zorder=5)
    spot_callout = ax.text(
        0.0,
        0.0,
        "",
        ha="center",
        va="center",
        color="#dcfce7",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#052e16", edgecolor=SPOTLIGHT_COL, alpha=0.92, lw=1.2),
        zorder=12,
    )

    # 6. Post-PCE Qubit Nodes (glow + core)
    qubit_glow = ax.scatter(
        pos_qubits[:, 0], pos_qubits[:, 1], s=q_sizes * 2.1, c=QUBIT_COL, alpha=0.0, edgecolors="none", zorder=5
    )
    qubit_scatter = ax.scatter(
        pos_qubits[:, 0],
        pos_qubits[:, 1],
        s=q_sizes,
        c="#1f2937",
        edgecolors=QUBIT_COL,
        linewidths=2.8,
        alpha=0.0,
        zorder=6,
    )

    # 7. Pre-PCE QUBO Variable Nodes (scalar alpha=None so per-node RGBA alphas work)
    var_scatter = ax.scatter(
        pos_pre[:, 0],
        pos_pre[:, 1],
        s=pre_var_sizes,
        c=var_colors,
        edgecolors=TEXT_MAIN,
        linewidths=1.5,
        zorder=7,
    )

    # 8. Labels for QUBO variables
    show_var_labels = tx.n_vars <= 30
    var_texts = []
    for i in range(tx.n_vars):
        txt = ax.text(
            pos_pre[i, 0] * 1.18,
            pos_pre[i, 1] * 1.18,
            tx.var_labels[i] if show_var_labels else "",
            ha="center",
            va="center",
            color=TEXT_MAIN,
            fontsize=9.6 if tx.n_vars <= 10 else 8.0,
            fontweight="bold",
            zorder=9,
        )
        var_texts.append(txt)

    # 9. Labels for Post-PCE Qubit Nodes
    qubit_texts = []
    qubit_subtexts = []
    for q in range(tx.n_qubits):
        sub_q = str(q).translate(SUB_MAP)
        t_main = ax.text(
            pos_qubits[q, 0],
            pos_qubits[q, 1],
            f"q{sub_q}",
            ha="center",
            va="center",
            color=QUBIT_COL,
            fontsize=12.5 if tx.n_qubits <= 6 else 10.0,
            fontweight="bold",
            alpha=0.0,
            zorder=10,
        )
        radial = pos_qubits[q] / max(1e-6, np.linalg.norm(pos_qubits[q]))
        bias = tx.post_1body.get(q, 0.0)
        pvar = f" ({tx.var_labels[qubit_primary_var[q]]})" if q in qubit_primary_var else ""
        t_sub = ax.text(
            pos_qubits[q, 0] + 0.25 * radial[0],
            pos_qubits[q, 1] + 0.25 * radial[1],
            f"Z{sub_q}: {bias:+.0f}{pvar}",
            ha="center",
            va="center",
            color="#fde68a",
            fontsize=9.0 if tx.n_qubits <= 6 else 8.0,
            alpha=0.0,
            visible=False,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#1f2937", edgecolor=QUBIT_COL, alpha=0.0, lw=0.9),
            zorder=10,
        )
        qubit_texts.append(t_main)
        qubit_subtexts.append(t_sub)

    # ── Right Info Dashboard ─────────────────────────────────────────────
    ax_info.add_patch(
        FancyBboxPatch(
            (0.02, 0.015),
            0.96,
            0.97,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=PANEL_BG,
            edgecolor=BORDER_COL,
            linewidth=1.2,
        )
    )

    ax_info.text(0.06, 0.945, "PCE TRANSFORMATION", color=QUBIT_COL, fontsize=11.5, fontweight="bold")
    ax_info.text(0.06, 0.912, f"Encoding: {tx.encoding_name}", color=TEXT_MAIN, fontsize=9.5)

    phase_badge = ax_info.text(
        0.06, 0.858, "Phase 1/4: Pre-PCE QUBO Graph", color="#38bdf8", fontsize=10.5, fontweight="bold"
    )
    ax_info.add_patch(
        FancyBboxPatch(
            (0.06, 0.822), 0.88, 0.017, boxstyle="round,pad=0.0,rounding_size=0.008", facecolor="#21262d", edgecolor="none"
        )
    )
    prog_bar = FancyBboxPatch(
        (0.06, 0.822), 0.01, 0.017, boxstyle="round,pad=0.0,rounding_size=0.008", facecolor="#38bdf8", edgecolor="none"
    )
    ax_info.add_patch(prog_bar)

    reduction_pct = 100.0 * (1.0 - tx.n_qubits / max(1, tx.n_vars))
    ratio = tx.n_vars / max(1, tx.n_qubits)

    ax_info.text(0.06, 0.778, "GRAPH COMPRESSION (REAL DATA)", color=TEXT_MUTED, fontsize=8.4, fontweight="bold")
    metric_qubits = ax_info.text(0.06, 0.742, "", color=TEXT_MAIN, fontsize=9.8, fontweight="bold")
    metric_edges = ax_info.text(0.06, 0.708, "", color=TEXT_MAIN, fontsize=9.2)
    ax_info.text(
        0.06,
        0.674,
        f"Compression: {tx.n_vars} vars → {tx.n_qubits} qubits ({ratio:.1f}× | -{reduction_pct:.0f}%)",
        color="#34d399",
        fontsize=9.2,
        fontweight="bold",
    )

    # Dedicated Tracked Spotlight Edge Card
    if spot_idx >= 0:
        su, sv, sw = tx.edges_pre[spot_idx]
        stgt = tx.edge_target_supports[spot_idx]
        stgt_pauli = format_z_support(stgt) if tx.is_parity_encoding else f"qubits {stgt}"
        stgt_qnames = "–".join(f"q{str(q).translate(SUB_MAP)}" for q in stgt)
        ax_info.add_patch(
            FancyBboxPatch(
                (0.055, 0.545),
                0.89,
                0.105,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor="#052e16",
                edgecolor=SPOTLIGHT_COL,
                linewidth=1.1,
                alpha=0.85,
            )
        )
        ax_info.text(0.08, 0.628, "TRACKED EDGE (FOLLOW THE GREEN LINE)", color=SPOTLIGHT_COL, fontsize=8.3, fontweight="bold")
        spot_info = (
            f"Pre-PCE:  ({tx.var_labels[su]}, {tx.var_labels[sv]})   J = {sw:+.1f}\n"
            f"Pauli:    ({tx.pauli_labels[su]}) · ({tx.pauli_labels[sv]}) = {stgt_pauli}\n"
            f"Post-PCE: maps to ({stgt_qnames})"
        )
        ax_info.text(
            0.08, 0.608, spot_info, va="top", color="#dcfce7", fontsize=8.2, family="monospace", linespacing=1.42
        )

    # Exact Hamiltonian term accounting
    ax_info.text(0.06, 0.508, "EXACT PAULI HAMILTONIAN TERMS", color=TEXT_MUTED, fontsize=8.4, fontweight="bold")
    n1 = len(tx.post_1body)
    n2 = len(tx.post_2body)
    n3 = sum(1 for k in tx.post_higher if len(k) == 3)
    n4p = sum(1 for k in tx.post_higher if len(k) >= 4)
    e1 = tx.edge_weight_counts.get(1, 0)
    e2 = tx.edge_weight_counts.get(2, 0)
    eh = sum(v for k, v in tx.edge_weight_counts.items() if k >= 3)

    if tx.is_parity_encoding:
        ham_summary = (
            f"Pre-PCE:   {tx.n_vars} linear  +  {n_edges_pre} quadratic edges\n"
            f"Edge XOR:  {e1} → 1-body | {e2} → 2-body | {eh} → 3+ body\n"
            f"Post-PCE:  {n1}×Z | {n2}×ZZ | {n3}×ZZZ" + (f" | {n4p}×Z⁴⁺" if n4p else "")
        )
    else:
        ham_summary = (
            f"Pre-PCE:   {tx.n_vars} linear  +  {n_edges_pre} quadratic edges\n"
            f"Correlators: {tx.n_vars} pair operators (X⊗X, Y⊗Y, Z⊗Z)\n"
            f"Post-PCE:  {tx.n_qubits} qubits | {n2} effective qubit pairs"
        )
    ax_info.text(
        0.06, 0.485, ham_summary, va="top", color=TEXT_MAIN, fontsize=8.2, family="monospace", linespacing=1.42
    )

    # Real variable -> Pauli string table
    ax_info.text(0.06, 0.375, "REAL VARIABLE → PAULI MAP (DIVI)", color=TEXT_MUTED, fontsize=8.4, fontweight="bold")
    max_rows = 8
    col1_lines = []
    col2_lines = []
    shown = min(tx.n_vars, max_rows * 2)
    for idx in range(shown):
        entry = f"{tx.var_labels[idx]:<8} → {tx.pauli_labels[idx]}"
        if idx < max_rows:
            col1_lines.append(entry)
        else:
            col2_lines.append(entry)
    if tx.n_vars > shown:
        col2_lines[-1] = f"... +{tx.n_vars - shown + 1} more"

    ax_info.text(
        0.06, 0.352, "\n".join(col1_lines), va="top", color="#c9d1d9", fontsize=8.4, family="monospace", linespacing=1.42
    )
    if col2_lines:
        ax_info.text(
            0.52, 0.352, "\n".join(col2_lines), va="top", color="#c9d1d9", fontsize=8.4, family="monospace", linespacing=1.42
        )

    # Visual Legend
    ax_info.text(0.06, 0.178, "LEGEND", color=TEXT_MUTED, fontsize=8.4, fontweight="bold")
    legend_items = [
        (SPOTLIGHT_COL, "━  Tracked spotlight edge (1 edge highlighted)"),
        (EDGE_NEG, "—  Other edges: J < 0 (blue) / J > 0 (coral)"),
        (QUBIT_COL, "●  Post-PCE Qubit node (1-body Z field)"),
        (CORR_COL, "●  2-qubit correlation variable (ZᵢZⱼ on edge)"),
        (HYPER_COL, "▲  3+/4-body Pauli hyperedge (XOR ≥ 3)"),
    ]
    for row_i, (col, label) in enumerate(legend_items):
        y_pos = 0.150 - row_i * 0.0235
        ax_info.text(0.06, y_pos, label, color=col, fontsize=7.9, va="center")

    ax_info.text(
        0.50,
        0.024,
        "Space: Play/Pause  |  ←/→: Step  |  R: Restart",
        ha="center",
        va="bottom",
        color=TEXT_MUTED,
        fontsize=7.6,
    )

    def hex_to_rgb(h: str) -> tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    rgb_neg = hex_to_rgb(EDGE_NEG)
    rgb_pos = hex_to_rgb(EDGE_POS)
    rgb_hyp = hex_to_rgb(HYPER_COL)

    pre_rgb = np.array([rgb_neg if w < 0 else rgb_pos for w in pre_w]) if n_edges_pre else np.zeros((0, 3))
    post_rgb = np.array([rgb_neg if w < 0 else rgb_pos for w in post_2_w]) if len(post_2_w) else np.zeros((0, 3))

    # ── Per-Frame Update Function ────────────────────────────────────────
    def update(frame_idx: int):
        u = frame_idx / max(1, n_frames - 1)
        prog_bar.set_width(max(0.01, 0.88 * u))

        p_assign = ease_in_out_cubic(clamp01((u - 0.20) / 0.16))
        p_fold = ease_in_out_cubic(clamp01((u - 0.36) / 0.40))
        p_post = ease_in_out_cubic(clamp01((u - 0.68) / 0.14))

        if u < 0.20:
            phase_badge.set_text("Phase 1/4: Pre-PCE QUBO Graph")
            phase_badge.set_color("#38bdf8")
            prog_bar.set_facecolor("#38bdf8")
            subtitle_text.set_text(
                f"Before PCE (1-to-1 QAOA): {tx.n_vars} binary variables = {tx.n_vars} qubits, "
                f"{n_edges_pre} quadratic edges (1 tracked in green)"
            )
        elif u < 0.36:
            phase_badge.set_text("Phase 2/4: Divi Pauli Assignment")
            phase_badge.set_color("#c084fc")
            prog_bar.set_facecolor("#c084fc")
            subtitle_text.set_text(
                f"Assigning each QUBO variable xᵢ to a real {tx.encoding_name} Pauli string Pᵢ "
                f"on only {tx.n_qubits} qubits"
            )
        elif u < 0.76:
            phase_badge.set_text("Phase 3/4: Folding Variables & Edges")
            phase_badge.set_color("#fbbf24")
            prog_bar.set_facecolor("#fbbf24")
            subtitle_text.set_text(
                "Watch the green tracked edge rewire via Pauli multiplication Pᵤ · Pᵥ onto its post-PCE qubit edge"
            )
        else:
            phase_badge.set_text("Phase 4/4: Post-PCE Qubit Graph")
            phase_badge.set_color("#34d399")
            prog_bar.set_facecolor("#34d399")
            subtitle_text.set_text(
                f"After PCE: {tx.n_qubits} qubits, {len(tx.post_2body)} 2-qubit ZZ edges, "
                f"{len(tx.post_higher)} higher-order term(s) ({reduction_pct:.0f}% fewer qubits)"
            )

        live_qubits = int(round(tx.n_vars + (tx.n_qubits - tx.n_vars) * p_fold))
        live_2edges = int(round(n_edges_pre + (len(tx.post_2body) - n_edges_pre) * p_fold))
        metric_qubits.set_text(f"Active Qubits:       {live_qubits:>4d}   (was {tx.n_vars})")
        metric_edges.set_text(f"2-Qubit Couplings:   {live_2edges:>4d}   (was {n_edges_pre})")

        # 1. Interpolate QUBO variable node positions
        cur_var_pos = (1.0 - p_fold) * pos_pre + p_fold * pos_var_post
        var_scatter.set_offsets(cur_var_pos)

        cur_sizes = np.empty(tx.n_vars, dtype=float)
        cur_alphas = np.empty(tx.n_vars, dtype=float)
        for i in range(tx.n_vars):
            if is_anchor[i]:
                cur_sizes[i] = pre_var_sizes[i] * (1.0 - 0.65 * p_post)
                cur_alphas[i] = max(0.0, 0.95 * (1.0 - p_post))
            else:
                cur_sizes[i] = pre_var_sizes[i] * (1.0 - 0.52 * p_fold)
                cur_alphas[i] = 0.95 - 0.10 * p_post
        var_scatter.set_sizes(cur_sizes)

        node_rgba = np.zeros((tx.n_vars, 4), dtype=float)
        edge_rgba = np.ones((tx.n_vars, 4), dtype=float)
        for i in range(tx.n_vars):
            c0 = np.array(hex_to_rgb(var_colors[i]))
            c1 = np.array(hex_to_rgb(QUBIT_COL if is_anchor[i] else CORR_COL))
            c = (1.0 - p_assign) * c0 + p_assign * c1
            node_rgba[i, :3] = c
            node_rgba[i, 3] = cur_alphas[i]
            # Highlight the two endpoints of the spotlight edge in Phase 1–2
            if spot_idx >= 0 and i in (pre_u[spot_idx], pre_v[spot_idx]) and p_fold < 0.5:
                edge_rgba[i, :3] = hex_to_rgb(SPOTLIGHT_COL)
            edge_rgba[i, 3] = cur_alphas[i]
        var_scatter.set_facecolors(node_rgba)
        var_scatter.set_edgecolors(edge_rgba)

        # 2. Variable labels
        if show_var_labels:
            for i in range(tx.n_vars):
                if p_assign < 0.45:
                    var_texts[i].set_text(tx.var_labels[i])
                else:
                    var_texts[i].set_text(
                        f"{tx.var_labels[i]}\n{tx.pauli_labels[i]}" if p_fold < 0.35 else tx.pauli_labels[i]
                    )
                radial = pos_pre[i] / max(1e-6, np.linalg.norm(pos_pre[i]))
                offset = (1.0 - p_fold) * 0.19 * radial
                var_texts[i].set_position((cur_var_pos[i, 0] + offset[0], cur_var_pos[i, 1] + offset[1]))
                if is_anchor[i]:
                    var_texts[i].set_alpha(max(0.0, 1.0 - 1.4 * p_fold))
                else:
                    var_texts[i].set_alpha(max(0.0, 0.96 - 0.12 * p_fold) if tx.n_vars <= 22 else max(0.0, 1.0 - 1.5 * p_fold))
                    var_texts[i].set_fontsize((9.4 if tx.n_vars <= 10 else 8.0) - 1.0 * p_fold)
                    var_texts[i].set_color("#f3e8ff" if p_fold > 0.5 else TEXT_MAIN)

        # 3. Guide lines (Phase 2)
        if guide_segs_init:
            guide_alpha = 0.32 * p_assign * (1.0 - p_fold)
            guide_coll.set_alpha(guide_alpha)
            if guide_alpha > 0.01:
                g_segs = np.array(
                    [[cur_var_pos[vi], pos_qubits[qi]] for vi, qi in zip(guide_var_idx, guide_q_idx)]
                )
                guide_coll.set_segments(g_segs)

        # 4. Background QUBO edges (calm & uniform — spotlight edge drawn separately)
        rewire = ease_in_out_cubic(clamp01((p_fold - 0.15) / 0.85))
        if n_edges_pre:
            p0_var = cur_var_pos[pre_u]
            p1_var = cur_var_pos[pre_v]
            seg_p0 = (1.0 - rewire) * p0_var + rewire * edge_target_p0
            seg_p1 = (1.0 - rewire) * p1_var + rewire * edge_target_p1
            pre_edge_coll.set_segments(np.stack([seg_p0, seg_p1], axis=1))

            # Calm uniform styling so no single cluster steals focus from the spotlight edge
            base_alpha = 0.30 if n_edges_pre <= 30 else (0.18 if n_edges_pre <= 200 else 0.08)
            fade_out = 1.0 - p_post
            kind_fade = np.where(edge_target_kind == 2, fade_out, np.maximum(0.0, 1.0 - 1.35 * p_fold))
            rgba = np.zeros((n_edges_pre, 4), dtype=float)
            is_h = (edge_target_kind == 3).astype(float)[:, None] * rewire
            rgba[:, :3] = (1.0 - is_h) * pre_rgb + is_h * np.array(rgb_hyp)
            rgba[:, 3] = base_alpha * kind_fade
            if spot_idx >= 0:
                rgba[spot_idx, 3] = 0.0  # Drawn cleanly by the dedicated spotlight artist
            pre_edge_coll.set_color(rgba)
            pre_edge_coll.set_linewidths(1.3 if n_edges_pre <= 30 else 0.9)

            # 5. Update the SINGLE Spotlight Edge
            if spot_idx >= 0:
                sp0 = seg_p0[spot_idx]
                sp1 = seg_p1[spot_idx]
                spot_glow_line.set_data([sp0[0], sp1[0]], [sp0[1], sp1[1]])
                spot_core_line.set_data([sp0[0], sp1[0]], [sp0[1], sp1[1]])
                pulse = 0.25 + 0.12 * math.sin(u * 8 * math.pi)
                spot_glow_line.set_alpha(pulse)

                # Floating callout offset slightly perpendicular to the spotlight edge
                mid = 0.5 * (sp0 + sp1)
                vec = sp1 - sp0
                perp = np.array([-vec[1], vec[0]])
                pnorm = np.linalg.norm(perp)
                if pnorm > 1e-6:
                    perp /= pnorm
                    # Push outward away from origin so it never covers center labels
                    if np.dot(perp, mid) < 0:
                        perp = -perp
                else:
                    perp = np.array([0.0, 1.0])
                callout_pos = mid + 0.17 * perp
                spot_callout.set_position((callout_pos[0], callout_pos[1]))

                su, sv, sw = tx.edges_pre[spot_idx]
                stgt = tx.edge_target_supports[spot_idx]
                stgt_p = format_z_support(stgt) if tx.is_parity_encoding else f"q{stgt}"
                if u < 0.20:
                    spot_callout.set_text(f"Tracked: ({tx.var_labels[su]}, {tx.var_labels[sv]})  J={sw:+.1f}")
                elif u < 0.45:
                    spot_callout.set_text(f"({tx.pauli_labels[su]}) · ({tx.pauli_labels[sv]}) = {stgt_p}")
                elif u < 0.76:
                    spot_callout.set_text(f"Rewiring → {stgt_p}")
                else:
                    stgt_q = "–".join(f"q{str(q).translate(SUB_MAP)}" for q in stgt)
                    spot_callout.set_text(f"Tracked edge → {stgt_p} ({stgt_q})")

        # 6. Post-PCE aggregated 2-body qubit edges (calm uniform widths)
        if len(post_2_w):
            p2_rgba = np.zeros((len(post_2_w), 4), dtype=float)
            p2_rgba[:, :3] = post_rgb
            p2_rgba[:, 3] = 0.68 * p_post
            post_edge_coll.set_color(p2_rgba)
            post_edge_coll.set_linewidths(2.8)

        # 7. Post-PCE higher-order hyperedge polygons
        if hyper_polys:
            h_face = [(rgb_hyp[0], rgb_hyp[1], rgb_hyp[2], 0.10 * (w / max_hyper_w) * p_post) for w in hyper_weights]
            h_edge = [(rgb_hyp[0], rgb_hyp[1], rgb_hyp[2], 0.35 * (w / max_hyper_w) * p_post) for w in hyper_weights]
            hyper_coll.set_facecolor(h_face)
            hyper_coll.set_edgecolor(h_edge)

        # 8. Post-PCE Qubit Nodes & Labels
        q_appear = ease_in_out_cubic(clamp01((p_fold - 0.10) / 0.70))
        qubit_scatter.set_alpha(q_appear)
        qubit_glow.set_alpha(0.18 * q_appear + 0.07 * math.sin(u * 4 * math.pi) * p_post)
        for q in range(tx.n_qubits):
            qubit_texts[q].set_alpha(q_appear)
            sub_vis = p_post > 0.02
            qubit_subtexts[q].set_visible(sub_vis)
            qubit_subtexts[q].set_alpha(p_post)
            if sub_vis and qubit_subtexts[q].get_bbox_patch() is not None:
                qubit_subtexts[q].get_bbox_patch().set_alpha(0.88 * p_post)

        return (
            hyper_coll,
            pre_edge_coll,
            post_edge_coll,
            guide_coll,
            spot_glow_line,
            spot_core_line,
            qubit_glow,
            qubit_scatter,
            var_scatter,
        )

    update(0)

    if not create_anim:
        return fig, None, update

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False,
        repeat=True,
    )

    state = {"paused": False, "frame": 0}

    def on_key(event):
        if event.key == " ":
            if state["paused"]:
                anim.event_source.start()
            else:
                anim.event_source.stop()
            state["paused"] = not state["paused"]
        elif event.key == "right":
            anim.event_source.stop()
            state["paused"] = True
            state["frame"] = (state["frame"] + 3) % n_frames
            update(state["frame"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            anim.event_source.stop()
            state["paused"] = True
            state["frame"] = (state["frame"] - 3) % n_frames
            update(state["frame"])
            fig.canvas.draw_idle()
        elif event.key in ("r", "R"):
            state["frame"] = 0
            update(0)
            if state["paused"]:
                anim.event_source.start()
                state["paused"] = False
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    return fig, anim, update


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Animate real QUBO -> Post-PCE Qubit Graph transformation using Divi."
    )
    ap.add_argument(
        "--instance",
        default="po_a003_t02_orig",
        help="Real QOBLIB portfolio instance name (default: po_a003_t02_orig).",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=1,
        help="Portfolio budget B (default: 1 — smallest real QUBO: 6 vars -> 3 qubits).",
    )
    ap.add_argument("--lam", default="1e-4", help="Risk aversion lambda (default: 1e-4).")
    ap.add_argument(
        "--no-budget-equality",
        action="store_true",
        help="Include budget slack bits (adds 1–2 slack variables).",
    )
    ap.add_argument(
        "--day",
        type=int,
        default=None,
        help="Optional single-day sub-problem index (useful for larger instances like po_a010_t10_orig).",
    )
    ap.add_argument(
        "--encoding",
        choices=["poly", "dense", "multibasis"],
        default="poly",
        help="Divi PCE encoding scheme (default: poly).",
    )
    ap.add_argument(
        "--spotlight-edge",
        type=int,
        default=None,
        help="Index of the single QUBO edge to highlight and track (default: auto-selects clearest 2-body edge).",
    )
    ap.add_argument("--frames", type=int, default=180, help="Total animation frames (default: 180).")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    ap.add_argument(
        "--no-hyperedges",
        action="store_true",
        help="Hide 3+/4-body Pauli hyperedge polygons in the post-PCE view.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save animation to file (.gif or .mp4) instead of opening interactive window.",
    )
    ap.add_argument(
        "--snapshots",
        type=str,
        default=None,
        help="Save keyframe PNGs (pre, assign, mid_fold, post) into this directory.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.save or args.snapshots:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    print(f"Loading real QOBLIB instance: {args.instance} (B={args.budget}, lam={args.lam})...")
    bqm, var_labels, var_groups, desc = build_real_qubo(
        args.instance,
        budget=args.budget,
        lam=args.lam,
        budget_equality=not args.no_budget_equality,
        day=args.day,
    )
    tx = compute_pce_transformation(
        bqm, var_labels, var_groups, instance_desc=desc, encoding_key=args.encoding
    )

    spot = pick_spotlight_edge(tx, args.spotlight_edge)
    print(f"  Pre-PCE QUBO:  {tx.n_vars} binary variables, {len(tx.edges_pre)} quadratic edges")
    print(f"  Divi Encoding: {tx.encoding_name}")
    print(
        f"  Post-PCE:      {tx.n_qubits} qubits "
        f"({100.0 * (1.0 - tx.n_qubits / tx.n_vars):.1f}% reduction), "
        f"{len(tx.post_2body)} 2-body ZZ edges, {len(tx.post_higher)} higher-order terms"
    )
    if spot >= 0:
        su, sv, sw = tx.edges_pre[spot]
        print(
            f"  Tracked Edge:  #{spot} ({tx.var_labels[su]} [{tx.pauli_labels[su]}], "
            f"{tx.var_labels[sv]} [{tx.pauli_labels[sv]}]) J={sw:+.1f} -> qubits {tx.edge_target_supports[spot]}"
        )

    need_anim = bool(args.save) or (not args.snapshots)
    fig, anim, update = build_animation(
        tx,
        n_frames=args.frames,
        fps=args.fps,
        show_hyperedges=not args.no_hyperedges,
        spotlight_edge=args.spotlight_edge,
        create_anim=need_anim,
    )

    if args.snapshots:
        out_dir = Path(args.snapshots)
        out_dir.mkdir(parents=True, exist_ok=True)
        keyframes = [
            ("01_pre_pce", 0),
            ("02_pauli_assign", int(0.32 * (args.frames - 1))),
            ("03_mid_fold", int(0.56 * (args.frames - 1))),
            ("04_post_pce", args.frames - 1),
        ]
        for tag, fidx in keyframes:
            update(fidx)
            out_path = out_dir / f"{tag}.png"
            fig.savefig(out_path, dpi=150, facecolor=BG_DARK)
            print(f"  Saved snapshot: {out_path}")

    if args.save:
        out_path = Path(args.save)
        print(f"Rendering animation ({args.frames} frames @ {args.fps} fps) to {out_path}...")
        writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
        anim.save(str(out_path), writer=writer, fps=args.fps, dpi=120)
        print(f"Saved animation to {out_path}")

    if not args.save and not args.snapshots:
        plt.show()


if __name__ == "__main__":
    main()
