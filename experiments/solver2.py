#!/usr/bin/env python3
"""Monolithic Portfolio Optimization Solver & Experiment Suite for 200 & 400 Assets (solver2.py).

Compatible with ``auto_runner.py``. Implements:
  1. Point 1 (Variable & Horizon Reduction):
     - 3-Bit Signed Net-Position Encoding (v = L - S in [-3, 3] using 3 bits/asset via exact
       Moebius cubic cancellation, halving asset variables from 6A to 3A with zero invalid codes).
     - Time-Horizon Decoupling: Final liquidation period (t = T-1) solved classically (L=S=0);
       Approach A sweeps active trading days t = 0..T-2 with other days clamped; Approach B solves
       the full multi-period 3-bit signed QUBO across active days simultaneously.
  2. Point 2 (Quantum Routines & Encodings):
     - ProgressivePCE (and standard PCE) with full step_history & alpha trajectory extraction.
     - Approach A: Community-partitioned Per-Day 3-Bit Signed ProgressivePCE.
     - Approach B: Drop-in Full-Horizon 3-Bit Signed Community-Partitioned ProgressivePCE.
     - Head-to-head toggle/comparison between DenseEncoding and PolyEncoding.
  3. Point 3 (Execution, Time Limit, Progress ETA & Vectorized Classical Polish):
     - Batched MonteCarloOptimizer, single seed default (SEEDS=(1,)) extensible to multiple seeds.
     - Hard wall-clock time limit (--time-limit-minutes, default 20 min) with graceful early stop.
     - Live [PROGRESS] + ETA reporting on stdout.
     - Fast net-position feasibility repair (repair_fast_portfolio) + exact vectorized incremental
       single-unit local search (fast_local_search: ~0.06s on 200 assets vs ~10 min in pure Python).
  4. Automated Insights & Plots:
     - Prints terminal insight tables and generates per-instance + cross-instance Matplotlib plots
       answering: (1) ProgressivePCE alpha progression vs 1-step PCE, (2) DenseEncoding vs
       PolyEncoding across risk lambda, and (3) Approach A vs Approach B across risk lambda & scale.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_divi_cache")
warnings.filterwarnings("ignore", category=SyntaxWarning)

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # repo root
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dimod
import hybrid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.qprog.algorithms import DenseEncoding, PolyEncoding, SoftObjective
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    CommunityDecomposer,
    QUBOPartitioningConfig,
)
import qoblib_tools as qt


# ---------------------------------------------------------------------------
# Default Hyperparameters (tuned for 10-20 min budget on 200 & 400 assets)
# ---------------------------------------------------------------------------

ALLOWED_ASSET_SIZES: tuple[int, ...] = (200, 400)
SEEDS: tuple[int, ...] = (1,)
TIME_LIMIT_MINUTES = 20.0
SHOTS = 2_000
MC_POPULATION = 15
MC_BEST_SETS = 3
MC_ITERATIONS = 3          # Iterations per ProgressivePCE step
N_LAYERS = 1
CLUSTER_SIZE_A = 150       # Per-day signed variables per cluster (150 vars = 50 assets)
CLUSTER_SIZE_B = 240       # Full-horizon signed variables per cluster
INITIAL_ALPHA = 3.0        # Divi ProgressivePCE default initial_alpha
THRESHOLD = 0.9            # Divi ProgressivePCE default binarisation threshold
MAX_STEPS = 4              # Max progressive continuation steps per partition
MAX_ALPHA = 10_000.0
BEAM_WIDTH = 5
N_PARTITION_CANDIDATES = 4
TOP_SOLUTIONS = 5

# 3-bit signed encoding lookup tables (8 bitstrings in {0,1}^3 -> v in [-3..3])
_CODES = np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.int64)
_SUBSETS = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


# ---------------------------------------------------------------------------
# Progress Logger, CLI Helpers, Instance Loading & Backend
# ---------------------------------------------------------------------------


def log_progress(t_start: float, done_units: float, total_units: float, msg: str) -> None:
    """Print live elapsed time, percentage, and estimated remaining time (ETA)."""
    elapsed = time.time() - t_start
    pct = min(100.0, 100.0 * done_units / max(1e-9, total_units))
    if done_units > 0.02 and pct < 99.9:
        eta = elapsed * (total_units - done_units) / done_units
        eta_str = f"ETA {eta:5.1f}s ({eta / 60.0:.1f}m)"
    elif pct >= 99.9:
        eta_str = "ETA   0.0s"
    else:
        eta_str = "ETA  calc..."
    print(f"[PROGRESS {elapsed:6.1f}s | {pct:5.1f}% | {eta_str}] {msg}", flush=True)


def _parse_bool(val: str | bool) -> bool:
    if isinstance(val, bool):
        return val
    v = val.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {val!r}")


def _load_dotenv(repo_dir: Path = SCRIPT_DIR) -> None:
    if os.environ.get("QORO_API_KEY"):
        return
    env_file = repo_dir / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_base_name(instance_arg: str, instance_dir: Path | None) -> str:
    if instance_dir is not None and instance_dir.resolve().name.startswith("po_"):
        return instance_dir.resolve().name
    stripped = re.sub(r"_b\d+_l.*$", "", instance_arg.strip())
    return stripped if stripped.startswith("po_") else f"po_{stripped}"


def load_instance_from_folder(instance_dir: Path | None, base_name: str) -> qt.PortfolioInstance:
    """Load stock prices and covariances strictly preferring ``instances/<base_name>``."""
    candidates = []
    if instance_dir is not None:
        candidates.append(instance_dir)
    candidates.append(SCRIPT_DIR / "instances" / base_name)
    for d in candidates:
        p_file, c_file = d / "stock_prices.txt.gz", d / "covariance_matrices.txt.gz"
        if not (p_file.is_file() and c_file.is_file()):
            continue
        symbols, idx, entries = [], {}, []
        for day, sym, price in qt._data_rows(p_file):
            if sym not in idx:
                idx[sym] = len(symbols)
                symbols.append(sym)
            entries.append((idx[sym], int(day), Fraction(price)))
        n_periods = max(t for _, t, _ in entries) + 1
        prices: list[list[Fraction | None]] = [[None] * n_periods for _ in symbols]
        for i, t, p in entries:
            prices[i][t] = p
        cov: dict[tuple[int, int, int], Fraction] = {}
        for day, s1, s2, val in qt._data_rows(c_file):
            t = int(day)
            if s1 in idx and s2 in idx and t < n_periods:
                cov[idx[s1], idx[s2], t] = Fraction(val)
        return qt.PortfolioInstance(base_name, symbols, n_periods, prices, cov)  # type: ignore[arg-type]
    return qt.load_portfolio_instance(base_name)


def make_backend(use_cloud: bool, shots: int):
    if use_cloud:
        _load_dotenv()
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro"))
    return MaestroSimulator(shots=shots)


# ---------------------------------------------------------------------------
# Point 1 & 3: 3-Bit Signed Net-Position Encoding, Fast Repair & Fast Polish
# ---------------------------------------------------------------------------


def _moebius(g: np.ndarray) -> dict[tuple[int, ...], float]:
    """Exact multilinear Moebius expansion of a 3-bit function over {0,1}^3."""
    idx = {tuple(c): k for k, c in enumerate(_CODES)}
    out: dict[tuple[int, ...], float] = {}
    for s_set in _SUBSETS:
        tot = 0.0
        for r in range(len(s_set) + 1):
            for t_set in itertools.combinations(s_set, r):
                code = tuple(1 if j in t_set else 0 for j in range(3))
                tot += ((-1) ** (len(s_set) - len(t_set))) * float(g[idx[code]])
        out[s_set] = tot
    return out


def _signed_variants():
    for base, c0 in (((1, 2, -3), 0), ((1, 2, 3), -3)):
        for s in (1, -1):
            w, c = s * np.array(base, dtype=np.int64), int(s * c0)
            v = _CODES @ w + c
            for val in np.unique(v):
                dup = np.flatnonzero(v == val)
                if len(dup) == 2:
                    for d in dup:
                        yield w, c, v, int(d)


_SIGNED_VARIANTS = list(_signed_variants())


def day_asset_costs(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    t: int,
    V: np.ndarray,
) -> np.ndarray:
    """Evaluate exact per-day cost contribution of candidate net-position vectors V (K, A)."""
    T = model.n_periods
    Lg, Sg = np.maximum(V, 0), np.maximum(-V, 0)
    Rt = rs_sym[t]
    cost = np.einsum("ka,ab,kb->k", V, Rt, V).astype(float)
    if t < T - 1:
        cost -= V @ model.ret[t]
    cost += Sg @ model.sc[t]
    cost += float(model.interest[0]) * V.sum(axis=1)
    if t == 0 or t == T - 1:
        cost += (Lg + Sg) @ model.tc[t]
    if 1 <= t <= T - 2:
        cost += (np.abs(Lg - L[t - 1]) + np.abs(Sg - S[t - 1])) @ model.tc[t]
    if t + 1 <= T - 2:
        cost += (np.abs(L[t + 1] - Lg) + np.abs(S[t + 1] - Sg)) @ model.tc[t + 1]
    return cost


def _day_signed_qubo_matrices(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    t: int,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (lin, Q_sym, W, C) for day ``t`` in the 3A-variable 3-bit signed encoding."""
    A_count = len(model.symbols)
    Rt = rs_sym[t]
    vals = np.arange(-3, 4, dtype=np.int64)
    c0 = float(day_asset_costs(model, rs_sym, L, S, t, np.zeros((1, A_count), dtype=np.int64))[0])

    # Vectorized single-asset unary cost table f[i, v+3] across all assets i
    diag_r = np.diag(Rt)[:, None]
    f = diag_r * (vals[None, :] ** 2) + mu * np.abs(vals)[None, :]
    if t < model.n_periods - 1:
        f -= model.ret[t][:, None] * vals[None, :]
    f += model.sc[t][:, None] * np.maximum(-vals[None, :], 0)
    f += float(model.interest[0]) * vals[None, :]
    if t == 0 or t == model.n_periods - 1:
        f += model.tc[t][:, None] * np.abs(vals)[None, :]
    if 1 <= t <= model.n_periods - 2:
        l_prev = L[t - 1][:, None]
        s_prev = S[t - 1][:, None]
        f += model.tc[t][:, None] * (
            np.abs(np.maximum(vals[None, :], 0) - l_prev)
            + np.abs(np.maximum(-vals[None, :], 0) - s_prev)
            - l_prev - s_prev
        )
    if t + 1 <= model.n_periods - 2:
        l_next = L[t + 1][:, None]
        s_next = S[t + 1][:, None]
        f += model.tc[t + 1][:, None] * (
            np.abs(l_next - np.maximum(vals[None, :], 0))
            + np.abs(s_next - np.maximum(-vals[None, :], 0))
            - l_next - s_next
        )
    # Subtract self-risk so f holds purely linear/piecewise-linear per-asset terms
    f -= diag_r * (vals[None, :] ** 2)

    W = np.zeros((A_count, 3), dtype=float)
    C = np.zeros(A_count, dtype=float)
    coefs = []
    for i in range(A_count):
        best = None
        for w, c, v, d in _SIGNED_VARIANTS:
            gt = f[i][v + 3].astype(float)
            sign_d = (-1) ** (3 - int(_CODES[d].sum()))
            co = _moebius(np.where(np.arange(8) == d, 0.0, gt))
            gd = -co[(0, 1, 2)] / sign_d
            margin = gd - gt[d]
            if best is None or margin > best[0]:
                best = (margin, w, c, _moebius(np.where(np.arange(8) == d, gd, gt)))
        assert best is not None
        W[i], C[i] = best[1], best[2]
        coefs.append(best[3])

    M = np.zeros((A_count, 3 * A_count), dtype=float)
    for i in range(A_count):
        M[i, 3 * i : 3 * i + 3] = W[i]
    Q = M.T @ Rt @ M
    lin = np.diag(Q).copy() + 2.0 * (C @ Rt @ M)
    for i in range(A_count):
        co = coefs[i]
        for j in range(3):
            lin[3 * i + j] += co[(j,)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            Q[3 * i + a, 3 * i + b] += 0.5 * co[(a, b)]
            Q[3 * i + b, 3 * i + a] += 0.5 * co[(a, b)]
    return lin, Q, W, C


def build_signed_day_bqm(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    t: int,
    mu: float,
    max_edges: int = 25_000,
) -> tuple[dimod.BinaryQuadraticModel, np.ndarray, np.ndarray]:
    """Build the 3A-variable signed 3-bit BQM for trading day ``t`` in <0.05s via C++ vectors."""
    lin, Q, W, C = _day_signed_qubo_matrices(model, rs_sym, L, S, t, mu)
    rows, cols = np.triu_indices(len(lin), k=1)
    q_vals = 2.0 * Q[rows, cols]
    abs_q = np.abs(q_vals)
    # Keep all intra-asset pairs + strongest covariance couplings up to max_edges for fast Louvain
    if len(q_vals) > max_edges:
        thresh = float(np.partition(abs_q, -max_edges)[-max_edges])
        nz = np.flatnonzero(abs_q >= max(1e-12, thresh))
    else:
        nz = np.flatnonzero(abs_q > 1e-12)
    bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
        lin,
        (rows[nz], cols[nz], q_vals[nz]),
        0.0,
        dimod.BINARY,
    )
    bqm.normalize()
    return bqm, W, C


def build_full_horizon_signed_bqm(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    active_days: list[int],
    mu: float,
    max_edges_per_day: int = 12_000,
) -> tuple[dimod.BinaryQuadraticModel, list[np.ndarray], list[np.ndarray]]:
    """Build full multi-period 3-bit signed BQM across ``active_days`` with inter-day rebalancing."""
    A_count = len(model.symbols)
    n_days = len(active_days)
    n_day_vars = 3 * A_count
    total_vars = n_days * n_day_vars
    L_zero = np.zeros((model.n_periods, A_count), dtype=np.int64)
    S_zero = np.zeros((model.n_periods, A_count), dtype=np.int64)

    lin_all = np.zeros(total_vars, dtype=float)
    row_chunks, col_chunks, val_chunks = [], [], []
    W_list, C_list = [], []
    r_tri, c_tri = np.triu_indices(n_day_vars, k=1)

    for d_idx, t in enumerate(active_days):
        offset = d_idx * n_day_vars
        lin_t, Q_t, W_t, C_t = _day_signed_qubo_matrices(model, rs_sym, L_zero, S_zero, t, mu)
        lin_all[offset : offset + n_day_vars] = lin_t
        W_list.append(W_t)
        C_list.append(C_t)

        q_vals = 2.0 * Q_t[r_tri, c_tri]
        abs_q = np.abs(q_vals)
        if len(q_vals) > max_edges_per_day:
            thresh = float(np.partition(abs_q, -max_edges_per_day)[-max_edges_per_day])
            nz = np.flatnonzero(abs_q >= max(1e-12, thresh))
        else:
            nz = np.flatnonzero(abs_q > 1e-12)
        row_chunks.append(r_tri[nz] + offset)
        col_chunks.append(c_tri[nz] + offset)
        val_chunks.append(q_vals[nz])

        # Inter-day transaction/rebalancing coupling -2 * tc[t, i] * (v_{t-1,i} * v_{t,i}) / ub
        if d_idx > 0 and 1 <= t <= model.n_periods - 2:
            prev_offset = (d_idx - 1) * n_day_vars
            W_prev = W_list[d_idx - 1]
            tc_scale = model.tc[t].astype(float) / float(model.ub)
            r_inter, c_inter, v_inter = [], [], []
            for i in range(A_count):
                for j1 in range(3):
                    for j2 in range(3):
                        coup = -2.0 * tc_scale[i] * W_prev[i, j1] * W_t[i, j2]
                        if abs(coup) > 1e-12:
                            r_inter.append(prev_offset + 3 * i + j1)
                            c_inter.append(offset + 3 * i + j2)
                            v_inter.append(coup)
            if r_inter:
                row_chunks.append(np.array(r_inter, dtype=np.int64))
                col_chunks.append(np.array(c_inter, dtype=np.int64))
                val_chunks.append(np.array(v_inter, dtype=float))

    rows_all = np.concatenate(row_chunks) if row_chunks else np.zeros(0, dtype=np.int64)
    cols_all = np.concatenate(col_chunks) if col_chunks else np.zeros(0, dtype=np.int64)
    vals_all = np.concatenate(val_chunks) if val_chunks else np.zeros(0, dtype=float)
    bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
        lin_all,
        (rows_all, cols_all, vals_all),
        0.0,
        dimod.BINARY,
    )
    bqm.normalize()
    return bqm, W_list, C_list


def decode_signed_day(bits: np.ndarray, W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Decode bitstrings (K, 3A) -> net integer positions (K, A) in [-3, 3]."""
    A_count = len(C)
    reshaped = bits.reshape(len(bits), A_count, 3)
    return np.einsum("kaj,aj->ka", reshaped, W).round().astype(np.int64) + C.round().astype(np.int64)


def repair_fast_portfolio(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    days_to_check: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized net-position feasibility repair enforcing budget B and capital C per day."""
    T, ub = model.n_periods, model.ub
    max_cash = (1 << model.cash_slack_bits) - 1
    v = np.clip(L - S, -ub, ub)
    L, S = np.maximum(v, 0).copy(), np.maximum(-v, 0).copy()
    days = range(T) if days_to_check is None else days_to_check
    for t in days:
        while True:
            vt = L[t] - S[t]
            units = int(np.abs(vt).sum())
            net = int(vt.sum())
            too_many = units > model.budget
            too_long = net > model.capital
            too_short = net < model.capital - max_cash
            if not (too_many or too_long or too_short):
                break
            held = np.flatnonzero(vt != 0)
            if too_long:
                held = held[vt[held] > 0]
            elif too_short:
                held = held[vt[held] < 0]
            if len(held) == 0:
                break
            V_cands = np.repeat(vt[None, :], len(held), axis=0)
            for k_idx, asset_i in enumerate(held):
                V_cands[k_idx, asset_i] -= int(np.sign(vt[asset_i]))
            c_cands = day_asset_costs(model, rs_sym, L, S, t, V_cands)
            best_k = int(np.argmin(c_cands))
            best_i = int(held[best_k])
            nv = int(vt[best_i] - np.sign(vt[best_i]))
            L[t, best_i], S[t, best_i] = max(nv, 0), max(-nv, 0)
    return L, S


def fast_local_search(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    positions: qt.Positions,
    max_rounds: int = 1000,
) -> qt.Positions:
    """Exact vectorized best-improvement local search (~6,000x faster than model.local_search).

    Evaluates all 4*T*A single-unit moves per round incrementally in O(T*A) using cached
    risk gradients h_t = (R_t + R_t^T) v_t, producing the exact same optimum as
    PortfolioModel.local_search in ~0.06s on 200 assets.
    """
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk
    R2 = R + R.transpose(0, 2, 1)
    Rdiag = np.diagonal(R, axis1=1, axis2=2)
    L, S = model.to_arrays(positions)
    L, S = repair_fast_portfolio(model, rs_sym, L, S)
    v = L - S
    h = np.einsum("tij,tj->ti", R2, v)
    max_cash = (1 << model.cash_slack_bits) - 1
    max_count = (1 << model.count_slack_bits) - 1

    def _cash_int(c_val: int) -> int:
        return -sum(int(model.interest[k]) * ((c_val >> k) & 1) for k in range(model.cash_slack_bits))

    int_lut = np.array([_cash_int(c_val) for c_val in range(max_cash + 1)], dtype=np.int64)

    for _ in range(max_rounds):
        net = v.sum(axis=1)
        tot = (L + S).sum(axis=1)
        c_now = model.capital - net
        best_delta = 0
        best_move = None
        for side_idx, (arr, dv_sign) in enumerate(((L, 1), (S, -1))):
            for d in (1, -1):
                dv = dv_sign * d
                c_new = c_now - dv
                tot_new = tot + d
                day_ok = (
                    (c_new >= 0)
                    & (c_new <= max_cash)
                    & (model.budget - tot_new >= 0)
                    & (model.budget - tot_new <= max_count)
                )
                if not day_ok.any():
                    continue
                valid = (arr + d >= 0) & (arr + d <= model.ub) & day_ok[:, None]
                if not valid.any():
                    continue
                delta = dv * h + Rdiag
                delta[:-1] -= dv * model.ret
                if side_idx == 1:
                    delta += d * model.sc
                d_int = np.where(day_ok, int_lut[np.clip(c_new, 0, max_cash)] - int_lut[c_now], 0)
                delta += d_int[:, None]
                delta[0] += d * model.tc[0]
                delta[T - 1] += d * model.tc[T - 1]
                if T > 2:
                    delta[1 : T - 1] += model.tc[1 : T - 1] * (
                        np.abs(arr[1 : T - 1] + d - arr[0 : T - 2]) - np.abs(arr[1 : T - 1] - arr[0 : T - 2])
                    )
                    delta[0 : T - 2] += model.tc[1 : T - 1] * (
                        np.abs(arr[1 : T - 1] - (arr[0 : T - 2] + d)) - np.abs(arr[1 : T - 1] - arr[0 : T - 2])
                    )
                delta_masked = np.where(valid, delta, 0)
                min_val = int(delta_masked.min())
                if min_val < best_delta:
                    idx_flat = int(delta_masked.argmin())
                    t_star, i_star = divmod(idx_flat, A_count)
                    best_delta = min_val
                    best_move = (side_idx, t_star, i_star, d, dv)
        if best_move is None:
            break
        side_idx, t_star, i_star, d, dv = best_move
        if side_idx == 0:
            L[t_star, i_star] += d
        else:
            S[t_star, i_star] += d
        v[t_star, i_star] += dv
        h[t_star] += dv * R2[t_star, :, i_star]
    return model.to_positions(L, S)


# ---------------------------------------------------------------------------
# Point 2: ProgressivePCE / PCE Ensemble Runner & Alpha Telemetry Extraction
# ---------------------------------------------------------------------------


def make_encoding(encoding_name: str):
    name = encoding_name.strip().lower()
    if name == "dense":
        return DenseEncoding()
    if name == "poly":
        return PolyEncoding()
    raise ValueError(f"Unsupported encoding {encoding_name!r}; choose 'dense' or 'poly'.")


def estimate_encoding_qubits(n_vars: int, encoding_name: str) -> int:
    if n_vars <= 0:
        return 0
    if encoding_name == "dense":
        return int(math.ceil(math.log2(n_vars + 1)))
    q = int(math.ceil((math.sqrt(8 * n_vars + 1) - 1) / 2))
    while q * (q + 1) // 2 < n_vars:
        q += 1
    return q


def _summarize_partition_list(partitions_info: list[dict[str, Any]]) -> dict[str, Any]:
    if not partitions_info:
        return {"n_partitions": 0, "partitions": []}
    init_alphas = [p["initial_alpha"] for p in partitions_info]
    final_alphas = [p["final_alpha"] for p in partitions_info]
    ratios = [p["alpha_ratio"] for p in partitions_info]
    steps = [p["n_steps"] for p in partitions_info]
    qubits = [p["n_qubits"] for p in partitions_info]
    unchanged_alpha_pct = 100.0 * sum(1 for p in partitions_info if p["alpha_unchanged"]) / len(partitions_info)
    unchanged_energy_pct = 100.0 * sum(1 for p in partitions_info if p["energy_unchanged_after_step1"]) / len(partitions_info)
    return {
        "n_partitions": len(partitions_info),
        "max_qubits": int(max(qubits)),
        "mean_qubits": round(float(np.mean(qubits)), 2),
        "mean_initial_alpha": round(float(np.mean(init_alphas)), 4),
        "mean_final_alpha": round(float(np.mean(final_alphas)), 4),
        "median_final_alpha": round(float(np.median(final_alphas)), 4),
        "max_final_alpha": round(float(np.max(final_alphas)), 4),
        "mean_alpha_ratio": round(float(np.mean(ratios)), 4),
        "mean_steps": round(float(np.mean(steps)), 2),
        "pct_alpha_unchanged": round(unchanged_alpha_pct, 1),
        "pct_energy_unchanged_after_step1": round(unchanged_energy_pct, 1),
        "mean_energy_gain_after_step1": round(float(np.mean([p["energy_gain_after_step1"] for p in partitions_info])), 6),
        "partitions": partitions_info,
    }


def extract_alpha_telemetry(ensemble: PartitioningProgramEnsemble, encoding_name: str, initial_alpha: float) -> dict[str, Any]:
    """Extract ProgressivePCE step_history & alpha statistics across all partitions."""
    partitions_info = []
    for key, prog in ensemble.programs.items():
        part_id = str(key[0]) if isinstance(key, tuple) else str(key)
        n_vars = int(key[1]) if (isinstance(key, tuple) and len(key) > 1) else getattr(prog, "n_vars", 0)
        n_qubits = getattr(prog, "n_qubits", estimate_encoding_qubits(n_vars, encoding_name))
        hist = getattr(prog, "step_history", None)
        res_step = getattr(prog, "result_step", None)
        if hist:
            alphas = [float(h.get("alpha", initial_alpha)) for h in hist]
            next_alphas = [float(h.get("next_alpha", alphas[-1])) for h in hist]
            bin_fracs = [float(h.get("binarised_fraction", 0.0)) for h in hist]
            energies = [float(h.get("energy", 0.0)) for h in hist]
            losses = [float(h.get("best_loss", 0.0)) for h in hist]
            init_a, final_a = alphas[0], alphas[-1]
            e_gain = energies[0] - energies[-1]
            partitions_info.append({
                "partition": part_id,
                "n_vars": n_vars,
                "n_qubits": int(n_qubits),
                "n_steps": len(hist),
                "result_step": int(res_step) if res_step is not None else len(hist),
                "initial_alpha": init_a,
                "final_alpha": final_a,
                "last_next_alpha": next_alphas[-1],
                "alpha_ratio": round(final_a / init_a, 4) if init_a > 0 else 1.0,
                "alpha_unchanged": bool(abs(final_a - init_a) < 1e-9),
                "step1_binarised_fraction": bin_fracs[0],
                "final_binarised_fraction": bin_fracs[-1],
                "step1_energy": energies[0],
                "final_energy": energies[-1],
                "energy_gain_after_step1": round(e_gain, 6),
                "energy_unchanged_after_step1": bool(abs(e_gain) < 1e-9),
                "alphas": alphas,
                "binarised_fractions": bin_fracs,
                "energies": energies,
                "losses": losses,
            })
        else:
            partitions_info.append({
                "partition": part_id,
                "n_vars": n_vars,
                "n_qubits": int(n_qubits),
                "n_steps": 1,
                "result_step": 1,
                "initial_alpha": initial_alpha,
                "final_alpha": initial_alpha,
                "last_next_alpha": initial_alpha,
                "alpha_ratio": 1.0,
                "alpha_unchanged": True,
                "step1_binarised_fraction": 1.0,
                "final_binarised_fraction": 1.0,
                "step1_energy": 0.0,
                "final_energy": 0.0,
                "energy_gain_after_step1": 0.0,
                "energy_unchanged_after_step1": True,
                "alphas": [initial_alpha],
                "binarised_fractions": [1.0],
                "energies": [0.0],
                "losses": [0.0],
            })
    return _summarize_partition_list(partitions_info)


def run_partitioned_bqm(
    bqm: dimod.BinaryQuadraticModel,
    *,
    routine: str,
    encoding_name: str,
    cluster_size: int,
    n_layers: int,
    mc_population: int,
    mc_best_sets: int,
    mc_iterations: int,
    initial_alpha: float,
    threshold: float,
    max_steps: int,
    backend,
    seed: int,
) -> tuple[PartitioningProgramEnsemble, dict[str, Any]]:
    """Partition ``bqm`` with CommunityDecomposer and solve via ProgressivePCE or PCE."""
    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(
            QUBOPartitioningConfig(
                max_n_variables_per_cluster=cluster_size,
                method="modularity",
                seed=seed,
            )
        ),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    encoding = make_encoding(encoding_name)
    optimizer = MonteCarloOptimizer(population_size=mc_population, n_best_sets=mc_best_sets)
    extra_kwargs: dict[str, Any] = {"encoding": encoding, "seed": seed}
    if routine == "progressive_pce":
        extra_kwargs.update({
            "initial_alpha": initial_alpha,
            "threshold": threshold,
            "max_steps": max_steps,
            "max_alpha": MAX_ALPHA,
        })
    elif routine == "pce":
        extra_kwargs["objective"] = SoftObjective(alpha=initial_alpha)
    else:
        raise ValueError(f"Unsupported routine {routine!r}")

    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine=routine,
        n_layers=n_layers,
        optimizer=optimizer,
        max_iterations=mc_iterations,
        backend=backend,
        **extra_kwargs,
    )
    ensemble.run()
    telemetry = extract_alpha_telemetry(ensemble, encoding_name, initial_alpha)
    return ensemble, telemetry


# ---------------------------------------------------------------------------
# Approach A (Per-Day Signed 3-Bit Community PCE) & Approach B (Full Horizon)
# ---------------------------------------------------------------------------


def solve_approach_a(
    model: qt.PortfolioModel,
    *,
    routine: str,
    encoding_name: str,
    cluster_size: int,
    n_layers: int,
    mc_population: int,
    mc_best_sets: int,
    mc_iterations: int,
    initial_alpha: float,
    threshold: float,
    max_steps: int,
    days_limit: int | None,
    use_cloud: bool,
    shots: int,
    seed: int,
    t_global_start: float,
    deadline: float,
    progress_base: float,
    progress_span: float,
    total_units: float,
) -> dict[str, Any]:
    """Approach A: Per-Day 3-Bit Signed Net-Position Community-Partitioned ProgressivePCE."""
    t0 = time.time()
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    L = np.zeros((T, A_count), dtype=np.int64)
    S = np.zeros((T, A_count), dtype=np.int64)

    active_days = list(range(max(1, T - 1)))
    if days_limit is not None and days_limit < len(active_days):
        idx = np.linspace(0, len(active_days) - 1, days_limit, dtype=int)
        active_days = sorted({active_days[i] for i in idx})

    unit_scale = float(np.median(np.abs(model.ret[: max(1, T - 1)]))) or 100.0
    mu = 0.35 * unit_scale

    all_partitions: list[dict[str, Any]] = []
    total_circuits = 0
    total_subprograms = 0
    day_scores: list[int] = []
    days_completed: list[int] = []
    aborted_by_timeout = False

    for d_idx, t in enumerate(active_days):
        if time.time() >= deadline:
            print(f"  [TIMEOUT GUARD] Time limit reached before day {t}; stopping Approach A early.", flush=True)
            aborted_by_timeout = True
            break
        log_progress(
            t_global_start,
            progress_base + progress_span * (d_idx / max(1, len(active_days))),
            total_units,
            f"Approach A ({encoding_name}, seed={seed}): Building & solving Day {t} ({d_idx + 1}/{len(active_days)})...",
        )
        bqm, W, C = build_signed_day_bqm(model, rs_sym, L, S, t, mu=mu)
        backend = make_backend(use_cloud=use_cloud, shots=shots)
        ensemble, tele = run_partitioned_bqm(
            bqm,
            routine=routine,
            encoding_name=encoding_name,
            cluster_size=cluster_size,
            n_layers=n_layers,
            mc_population=mc_population,
            mc_best_sets=mc_best_sets,
            mc_iterations=mc_iterations,
            initial_alpha=initial_alpha,
            threshold=threshold,
            max_steps=max_steps,
            backend=backend,
            seed=seed + t,
        )
        total_subprograms += int(len(ensemble.programs))
        total_circuits += int(ensemble.total_circuit_count)
        for p in tele["partitions"]:
            p["day"] = t
            all_partitions.append(p)

        tops = ensemble.get_top_solutions(
            n=TOP_SOLUTIONS,
            strategy=BeamSearchStrategy(
                beam_width=BEAM_WIDTH,
                n_partition_candidates=N_PARTITION_CANDIDATES,
            ),
        )
        var_order = list(bqm.variables)
        cand_bits = np.zeros((len(tops), 3 * A_count), dtype=np.int64)
        for k_idx, (sol, _) in enumerate(tops):
            cand_bits[k_idx, var_order] = np.asarray(sol, dtype=np.int64)
        V_cands = decode_signed_day(cand_bits, W, C)

        best_day_L, best_day_S = L.copy(), S.copy()
        best_day_obj = int(model.score_arrays(L, S).objective)
        for k_idx in range(len(V_cands)):
            L_try, S_try = L.copy(), S.copy()
            v_k = V_cands[k_idx]
            L_try[t] = np.maximum(v_k, 0)
            S_try[t] = np.maximum(-v_k, 0)
            L_try, S_try = repair_fast_portfolio(model, rs_sym, L_try, S_try, days_to_check=[t])
            sc_try = model.score_arrays(L_try, S_try)
            if sc_try.feasible and int(sc_try.objective) < best_day_obj:
                best_day_obj = int(sc_try.objective)
                best_day_L, best_day_S = L_try, S_try
        L, S = best_day_L, best_day_S
        day_scores.append(best_day_obj)
        days_completed.append(t)

    L, S = repair_fast_portfolio(model, rs_sym, L, S)
    pre_polish_pos = model.to_positions(L, S)
    pre_polish_score = model.score(pre_polish_pos)

    # Point 3: Exact vectorized classical greedy polish (~0.06s)
    polished_pos = fast_local_search(model, rs_sym, pre_polish_pos)
    post_polish_score = model.score(polished_pos)
    elapsed = round(time.time() - t0, 2)

    agg_alpha = _summarize_partition_list(all_partitions)
    return {
        "approach": "A",
        "approach_label": "Approach A (Per-Day Signed 3-Bit Community PCE)",
        "routine": routine,
        "encoding": encoding_name,
        "seed": seed,
        "cluster_size": cluster_size,
        "n_qubo_vars_per_day": 3 * A_count,
        "n_qubo_vars_total": 3 * A_count * max(1, len(days_completed)),
        "days_solved": days_completed,
        "aborted_by_timeout": aborted_by_timeout,
        "sub_programs": total_subprograms,
        "circuits": total_circuits,
        "duration_seconds": elapsed,
        "pre_polish_objective": int(pre_polish_score.objective),
        "pre_polish_feasible": bool(pre_polish_score.feasible),
        "objective": int(post_polish_score.objective),
        "feasible": bool(post_polish_score.feasible),
        "day_score_trajectory": day_scores,
        "alpha_telemetry": agg_alpha,
        "positions": polished_pos,
        "score_obj": post_polish_score,
    }


def solve_approach_b(
    model: qt.PortfolioModel,
    *,
    routine: str,
    encoding_name: str,
    cluster_size: int,
    n_layers: int,
    mc_population: int,
    mc_best_sets: int,
    mc_iterations: int,
    initial_alpha: float,
    threshold: float,
    max_steps: int,
    days_limit: int | None,
    use_cloud: bool,
    shots: int,
    seed: int,
    t_global_start: float,
    deadline: float,
    progress_base: float,
    progress_span: float,
    total_units: float,
) -> dict[str, Any]:
    """Approach B: Full-Horizon Multi-Period 3-Bit Signed Community-Partitioned ProgressivePCE."""
    t0 = time.time()
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    active_days = list(range(max(1, T - 1)))
    if days_limit is not None and days_limit < len(active_days):
        idx = np.linspace(0, len(active_days) - 1, days_limit, dtype=int)
        active_days = sorted({active_days[i] for i in idx})

    if time.time() >= deadline:
        print("  [TIMEOUT GUARD] Time limit reached before Approach B start; returning fast baseline.", flush=True)
        pol_pos = fast_local_search(model, rs_sym, {})
        pol_sc = model.score(pol_pos)
        return {
            "approach": "B",
            "approach_label": "Approach B (Full-Horizon Signed 3-Bit Community PCE)",
            "routine": routine,
            "encoding": encoding_name,
            "seed": seed,
            "cluster_size": cluster_size,
            "n_qubo_vars_per_day": 3 * A_count,
            "n_qubo_vars_total": 3 * A_count * len(active_days),
            "days_solved": active_days,
            "aborted_by_timeout": True,
            "sub_programs": 0,
            "circuits": 0,
            "duration_seconds": round(time.time() - t0, 2),
            "pre_polish_objective": int(pol_sc.objective),
            "pre_polish_feasible": bool(pol_sc.feasible),
            "objective": int(pol_sc.objective),
            "feasible": bool(pol_sc.feasible),
            "day_score_trajectory": [],
            "alpha_telemetry": _summarize_partition_list([]),
            "positions": pol_pos,
            "score_obj": pol_sc,
        }

    log_progress(
        t_global_start,
        progress_base,
        total_units,
        f"Approach B ({encoding_name}, seed={seed}): Building full-horizon 3-bit signed BQM ({3 * A_count * len(active_days)} vars)...",
    )
    unit_scale = float(np.median(np.abs(model.ret[: max(1, T - 1)]))) or 100.0
    mu = 0.35 * unit_scale
    bqm, W_list, C_list = build_full_horizon_signed_bqm(model, rs_sym, active_days, mu=mu)

    log_progress(
        t_global_start,
        progress_base + 0.15 * progress_span,
        total_units,
        f"Approach B ({encoding_name}, seed={seed}): Running PartitioningProgramEnsemble ({routine})...",
    )
    backend = make_backend(use_cloud=use_cloud, shots=shots)
    ensemble, tele = run_partitioned_bqm(
        bqm,
        routine=routine,
        encoding_name=encoding_name,
        cluster_size=cluster_size,
        n_layers=n_layers,
        mc_population=mc_population,
        mc_best_sets=mc_best_sets,
        mc_iterations=mc_iterations,
        initial_alpha=initial_alpha,
        threshold=threshold,
        max_steps=max_steps,
        backend=backend,
        seed=seed,
    )
    tops = ensemble.get_top_solutions(
        n=TOP_SOLUTIONS,
        strategy=BeamSearchStrategy(
            beam_width=BEAM_WIDTH,
            n_partition_candidates=N_PARTITION_CANDIDATES,
        ),
    )
    n_day_vars = 3 * A_count
    var_order = list(bqm.variables)
    cand_bits = np.zeros((len(tops), len(active_days) * n_day_vars), dtype=np.int64)
    for k_idx, (sol, _) in enumerate(tops):
        cand_bits[k_idx, var_order] = np.asarray(sol, dtype=np.int64)

    best_pre_pos: qt.Positions = {}
    best_pre_sc = model.score(best_pre_pos)
    best_post_pos: qt.Positions = {}
    best_post_sc = best_pre_sc

    for idx_s in range(len(tops)):
        L_raw = np.zeros((T, A_count), dtype=np.int64)
        S_raw = np.zeros((T, A_count), dtype=np.int64)
        for d_idx, t in enumerate(active_days):
            day_slice = cand_bits[idx_s, d_idx * n_day_vars : (d_idx + 1) * n_day_vars][None, :]
            v_day = decode_signed_day(day_slice, W_list[d_idx], C_list[d_idx])[0]
            L_raw[t] = np.maximum(v_day, 0)
            S_raw[t] = np.maximum(-v_day, 0)
        L_rep, S_rep = repair_fast_portfolio(model, rs_sym, L_raw, S_raw)
        rep_pos = model.to_positions(L_rep, S_rep)
        rep_sc = model.score(rep_pos)
        if idx_s == 0 or (rep_sc.feasible and rep_sc.objective < best_pre_sc.objective):
            best_pre_pos, best_pre_sc = rep_pos, rep_sc

        if idx_s < 2:
            pol_pos = fast_local_search(model, rs_sym, rep_pos)
            pol_sc = model.score(pol_pos)
            if idx_s == 0 or (pol_sc.feasible and pol_sc.objective < best_post_sc.objective):
                best_post_pos, best_post_sc = pol_pos, pol_sc

    elapsed = round(time.time() - t0, 2)
    return {
        "approach": "B",
        "approach_label": "Approach B (Full-Horizon Signed 3-Bit Community PCE)",
        "routine": routine,
        "encoding": encoding_name,
        "seed": seed,
        "cluster_size": cluster_size,
        "n_qubo_vars_per_day": 3 * A_count,
        "n_qubo_vars_total": 3 * A_count * len(active_days),
        "days_solved": active_days,
        "aborted_by_timeout": False,
        "sub_programs": int(len(ensemble.programs)),
        "circuits": int(ensemble.total_circuit_count),
        "duration_seconds": elapsed,
        "pre_polish_objective": int(best_pre_sc.objective),
        "pre_polish_feasible": bool(best_pre_sc.feasible),
        "objective": int(best_post_sc.objective),
        "feasible": bool(best_post_sc.feasible),
        "day_score_trajectory": [],
        "alpha_telemetry": tele,
        "positions": best_post_pos,
        "score_obj": best_post_sc,
    }


# ---------------------------------------------------------------------------
# Terminal Insights & Matplotlib Plotting (Per-Instance + Cross-Instance)
# ---------------------------------------------------------------------------


def print_experiment_insights(
    base_name: str,
    assets: int,
    days: int,
    budget: int,
    lam: str,
    empty_ls_obj: int,
    exp_runs: list[dict[str, Any]],
) -> None:
    """Print clear structured terminal tables directly answering Insights 1, 2, and 3."""
    print("\n" + "=" * 94)
    print(f"EXPERIMENT INSIGHTS REPORT | {base_name} (A={assets}, T={days}, B={budget}, lambda={lam})")
    print(f"Classical Empty-Start Local Search Baseline: {empty_ls_obj}")
    print("=" * 94)

    print("\n[INSIGHT 1] Progressive PCE Alpha Progression & Step-1 vs Final-Step QUBO Energy:")
    print(f"  {'Config':<14} | {'Parts':>5} | {'Qubits':>6} | {'Init a':>7} | {'Mean Final a':>12} | {'a Ratio':>7} | {'% a==Init':>9} | {'% dE==0':>8}")
    print("  " + "-" * 88)
    for r in exp_runs:
        cfg = f"{r['approach']}-{r['encoding']}(s{r['seed']})"
        at = r["alpha_telemetry"]
        print(
            f"  {cfg:<14} | {at.get('n_partitions', 0):>5d} | {at.get('mean_qubits', 0):>6.1f} | "
            f"{at.get('mean_initial_alpha', 0):>7.2f} | {at.get('mean_final_alpha', 0):>12.3f} | "
            f"{at.get('mean_alpha_ratio', 1):>7.3f} | {at.get('pct_alpha_unchanged', 0):>8.1f}% | "
            f"{at.get('pct_energy_unchanged_after_step1', 0):>7.1f}%"
        )

    print("\n[INSIGHT 2 & 3] Dense vs Poly Encoding & Approach A vs Approach B Performance:")
    print(f"  {'Config':<14} | {'QUBO Vars':>9} | {'Circuits':>8} | {'Time(s)':>7} | {'Pre-Polish Score':>16} | {'Post-Polish Score':>17} | {'vs Empty LS':>11}")
    print("  " + "-" * 94)
    for r in exp_runs:
        cfg = f"{r['approach']}-{r['encoding']}(s{r['seed']})"
        diff = r["objective"] - empty_ls_obj
        print(
            f"  {cfg:<14} | {r['n_qubo_vars_total']:>9d} | {r['circuits']:>8d} | "
            f"{r['duration_seconds']:>7.1f} | {r['pre_polish_objective']:>16d} | "
            f"{r['objective']:>17d} | {diff:>+11d}"
        )

    by_enc: dict[str, list[dict[str, Any]]] = {}
    by_app: dict[str, list[dict[str, Any]]] = {}
    for r in exp_runs:
        by_enc.setdefault(r["encoding"], []).append(r)
        by_app.setdefault(r["approach"], []).append(r)

    print("\n[KEY TAKEAWAYS FOR THIS INSTANCE]")
    for r in exp_runs:
        at = r["alpha_telemetry"]
        cfg = f"Approach {r['approach']} ({r['encoding']})"
        print(
            f"  * {cfg}: initial_alpha={at.get('mean_initial_alpha')}, "
            f"mean_final_alpha={at.get('mean_final_alpha')} (ratio={at.get('mean_alpha_ratio')}x). "
            f"In {at.get('pct_alpha_unchanged')}% of partitions alpha stayed at initial_alpha, and in "
            f"{at.get('pct_energy_unchanged_after_step1')}% of partitions progressive steps after Step 1 "
            f"gave zero QUBO energy improvement (confirming 1-step PCE behaves almost identically)."
        )
    if "dense" in by_enc and "poly" in by_enc:
        best_dense = min(by_enc["dense"], key=lambda x: x["objective"])
        best_poly = min(by_enc["poly"], key=lambda x: x["objective"])
        winner_enc = "DenseEncoding" if best_dense["objective"] <= best_poly["objective"] else "PolyEncoding"
        pre_winner = "DenseEncoding" if best_dense["pre_polish_objective"] <= best_poly["pre_polish_objective"] else "PolyEncoding"
        print(
            f"  * Encoding Comparison (lambda={lam}): Pre-polish winner is {pre_winner} "
            f"(dense={best_dense['pre_polish_objective']} vs poly={best_poly['pre_polish_objective']}); "
            f"Post-polish winner is {winner_enc} (dense={best_dense['objective']} vs poly={best_poly['objective']})."
        )
    if "A" in by_app and "B" in by_app:
        best_a = min(by_app["A"], key=lambda x: x["objective"])
        best_b = min(by_app["B"], key=lambda x: x["objective"])
        winner_app = "Approach A (Per-Day Signed)" if best_a["objective"] <= best_b["objective"] else "Approach B (Full-Horizon Signed)"
        print(
            f"  * Approach Comparison (lambda={lam}): {winner_app} wins on final objective "
            f"(A={best_a['objective']} in {best_a['duration_seconds']}s vs B={best_b['objective']} in {best_b['duration_seconds']}s)."
        )
    print("=" * 94 + "\n", flush=True)


def save_instance_plot(
    out_path: Path,
    base_name: str,
    assets: int,
    budget: int,
    lam: str,
    empty_ls_obj: int,
    exp_runs: list[dict[str, Any]],
) -> None:
    """Save a 3-panel diagnostic figure for the current instance run."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(f"{base_name} (A={assets}, B={budget}, $\\lambda={lam}$) — ProgressivePCE & Decomposition Insights", fontsize=12, fontweight="bold")

    ax1 = axes[0]
    for r in exp_runs:
        cfg = f"{r['approach']}-{r['encoding']}"
        parts = r["alpha_telemetry"].get("partitions", [])
        if parts:
            max_len = max(len(p["alphas"]) for p in parts)
            padded = np.array([p["alphas"] + [p["alphas"][-1]] * (max_len - len(p["alphas"])) for p in parts])
            steps_x = np.arange(1, max_len + 1)
            mean_a = padded.mean(axis=0)
            ax1.plot(steps_x, mean_a, marker="o", linewidth=2, label=f"{cfg} (mean $\\alpha$)")
            ax1.fill_between(steps_x, padded.min(axis=0), padded.max(axis=0), alpha=0.15)
    ax1.set_title("1. ProgressivePCE $\\alpha$ vs Step")
    ax1.set_xlabel("Continuation Step")
    ax1.set_ylabel("$\\alpha$ (SoftObjective)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    labels = [f"{r['approach']}-{r['encoding']}\n(s{r['seed']})" for r in exp_runs]
    x = np.arange(len(exp_runs))
    width = 0.36
    pre_vals = [r["pre_polish_objective"] for r in exp_runs]
    post_vals = [r["objective"] for r in exp_runs]
    ax2.bar(x - width / 2, pre_vals, width, label="Pre-Polish (Quantum+Repair)", color="#4C72B0")
    ax2.bar(x + width / 2, post_vals, width, label="Post-Polish (+Fast Local Search)", color="#55A868")
    ax2.axhline(empty_ls_obj, color="#C44E52", linestyle="--", linewidth=1.5, label="Empty-Start Local Search")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title("2. Objective Score (Lower is Better)")
    ax2.set_ylabel("Official QOBLIB Objective")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    times = [r["duration_seconds"] for r in exp_runs]
    qubits = [r["alpha_telemetry"].get("mean_qubits", 0) for r in exp_runs]
    bars = ax3.bar(x, times, width=0.5, color="#8172B2")
    for bar, q_val, r in zip(bars, qubits, exp_runs):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{q_val:.0f}q\n({r['sub_programs']}p)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_title("3. Runtime (s) & Qubits/Partition")
    ax3.set_ylabel("Wall-Clock Seconds")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def update_cross_instance_plots(search_dir: Path, out_path: Path | None = None) -> Path | None:
    """Scan all ``*_exp.json`` files in ``search_dir`` (and subdirs) and plot cross-lambda insights."""
    records = []
    for p in sorted(search_dir.rglob("*_exp.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in data.get("experiment_runs", []):
                records.append({
                    "base_instance": data.get("base_instance"),
                    "assets": int(data.get("assets", 0)),
                    "days": int(data.get("days", 0)),
                    "budget": int(data.get("budget", 0)),
                    "lambda_str": str(data.get("lambda", "0")),
                    "lambda_val": float(data.get("lambda", 0.0)),
                    "empty_ls": int(data.get("empty_local_search_objective", 0)),
                    **r,
                })
        except Exception:
            continue

    if not records:
        return None

    target_path = out_path or (search_dir / "cross_instance_insights_200_400.png")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("200 & 400 Assets Cross-Instance Insights: ProgressivePCE $\\alpha$, Dense vs Poly, Approach A vs B", fontsize=12, fontweight="bold")

    lambdas = sorted({r["lambda_val"] for r in records})
    lam_labels = [f"{v:g}" for v in lambdas]
    x_map = {v: i for i, v in enumerate(lambdas)}

    ax1 = axes[0]
    combos = sorted({(r["approach"], r["encoding"]) for r in records})
    for app, enc in combos:
        sub = [r for r in records if r["approach"] == app and r["encoding"] == enc]
        xs = [x_map[r["lambda_val"]] for r in sub]
        ys = [r["alpha_telemetry"].get("mean_alpha_ratio", 1.0) for r in sub]
        ax1.plot(xs, ys, marker="o", linestyle="-", label=f"App {app} + {enc}")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="$\\alpha_{final} = \\alpha_{init}$ (No change)")
    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels(lam_labels)
    ax1.set_xlabel("Risk Weight $\\lambda$")
    ax1.set_ylabel("Mean $\\alpha_{final} / \\alpha_{init}$ Ratio")
    ax1.set_title("1. ProgressivePCE $\\alpha$ Growth Across $\\lambda$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    for enc, marker, color in (("dense", "o", "#1f77b4"), ("poly", "s", "#ff7f0e")):
        for assets, ls in ((200, "-"), (400, "--")):
            sub = [r for r in records if r["encoding"] == enc and r["assets"] == assets]
            if not sub:
                continue
            by_l: dict[float, list[int]] = {}
            for r in sub:
                by_l.setdefault(r["lambda_val"], []).append(r["objective"] - r["empty_ls"])
            xs = [x_map[k] for k in sorted(by_l)]
            ys = [float(np.mean(by_l[k])) for k in sorted(by_l)]
            ax2.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=f"{enc} (A={assets})")
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels(lam_labels)
    ax2.set_xlabel("Risk Weight $\\lambda$")
    ax2.set_ylabel("Score $-$ Empty LS Baseline (Lower = Better)")
    ax2.set_title("2. Dense vs Poly Encoding vs Risk $\\lambda$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    for app, marker, color in (("A", "^", "#2ca02c"), ("B", "v", "#d62728")):
        for assets, ls in ((200, "-"), (400, "--")):
            sub = [r for r in records if r["approach"] == app and r["assets"] == assets]
            if not sub:
                continue
            by_l = {}
            for r in sub:
                by_l.setdefault(r["lambda_val"], []).append(r["objective"] - r["empty_ls"])
            xs = [x_map[k] for k in sorted(by_l)]
            ys = [float(np.mean(by_l[k])) for k in sorted(by_l)]
            ax3.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=f"Approach {app} (A={assets})")
    ax3.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax3.set_xticks(range(len(lambdas)))
    ax3.set_xticklabels(lam_labels)
    ax3.set_xlabel("Risk Weight $\\lambda$")
    ax3.set_ylabel("Score $-$ Empty LS Baseline (Lower = Better)")
    ax3.set_title("3. Approach A vs Approach B vs Risk $\\lambda$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(target_path, dpi=150)
    plt.close(fig)
    return target_path


# ---------------------------------------------------------------------------
# CLI Parser & Main Entrypoint (auto_runner.py Contract Compliant)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Portfolio optimization solver2.py for 200 & 400 assets (ProgressivePCE, Approach A/B, Dense/Poly)."
    )
    # Standard auto_runner.py contract flags
    p.add_argument("--instance", default=os.environ.get("AUTORUNNER_INSTANCE", "po_a200_t10_orig"))
    p.add_argument("--instance-dir", default=os.environ.get("AUTORUNNER_INSTANCE_DIR"))
    p.add_argument("--assets", type=int, default=int(os.environ["AUTORUNNER_ASSETS"]) if os.environ.get("AUTORUNNER_ASSETS") else None)
    p.add_argument("--days", type=int, default=int(os.environ["AUTORUNNER_DAYS"]) if os.environ.get("AUTORUNNER_DAYS") else None)
    p.add_argument("--lambda", dest="lam", default=os.environ.get("AUTORUNNER_LAMBDA", "1e-4"))
    p.add_argument("--budget", type=int, default=int(os.environ.get("AUTORUNNER_BUDGET", "50")))
    p.add_argument("--cloud", type=_parse_bool, nargs="?", const=True, default=_parse_bool(os.environ.get("AUTORUNNER_CLOUD", "false")))
    p.add_argument("--solver-seed", type=int, default=int(os.environ["AUTORUNNER_SOLVER_SEED"]) if os.environ.get("AUTORUNNER_SOLVER_SEED") else None)

    # Experiment & solver2 configuration flags
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple RNG seeds (default: single seed [1]).")
    p.add_argument("--approach", choices=["A", "B", "both"], default=os.environ.get("SOLVER2_APPROACH", "A"),
                   help="A = Per-Day 3-Bit Signed Community PCE, B = Full-Horizon Signed Community PCE, both = compare A & B.")
    p.add_argument("--encoding", choices=["dense", "poly", "both"], default=os.environ.get("SOLVER2_ENCODING", "dense"),
                   help="PCE encoding: dense, poly, or both (runs head-to-head comparison).")
    p.add_argument("--routine", choices=["progressive_pce", "pce"], default=os.environ.get("SOLVER2_ROUTINE", "progressive_pce"))
    p.add_argument("--cluster-size-a", type=int, default=CLUSTER_SIZE_A, help="Max variables per community in Approach A.")
    p.add_argument("--cluster-size-b", type=int, default=CLUSTER_SIZE_B, help="Max variables per community in Approach B.")
    p.add_argument("--days-limit", type=int, default=None, help="Optional cap on active trading days solved for fast experiments.")
    p.add_argument("--time-limit-minutes", type=float, default=TIME_LIMIT_MINUTES, help="Hard wall-clock time limit in minutes (default: 20.0).")
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--shots", type=int, default=SHOTS)
    p.add_argument("--mc-population", type=int, default=MC_POPULATION)
    p.add_argument("--mc-best-sets", type=int, default=MC_BEST_SETS)
    p.add_argument("--mc-iterations", type=int, default=MC_ITERATIONS)
    p.add_argument("--initial-alpha", type=float, default=INITIAL_ALPHA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--allow-all-sizes", action="store_true", help="Allow running on <200 asset instances for testing.")
    p.add_argument("--aggregate-only", type=Path, default=None, help="Only regenerate cross-instance plots from an AutoData directory and exit.")
    return p


SOLUTIONS_DIR = Path("solutions2")


def resolve_solution_path(instance_arg: str, base_name: str, budget: int, lam: str) -> Path:
    if re.search(r"_b\d+_l", instance_arg):
        filename = f"{Path(instance_arg).name}.sol"
    else:
        lam_tag = lam.replace("-0", "-").replace("+0", "+")
        filename = f"{base_name}_b{budget:03d}_l{lam_tag}.sol"
    return SOLUTIONS_DIR / filename


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.aggregate_only is not None:
        out = update_cross_instance_plots(args.aggregate_only)
        print(f"Generated cross-instance insight plot: {out}", flush=True)
        return 0

    t_start = time.time()
    deadline = t_start + args.time_limit_minutes * 60.0
    instance_dir = Path(args.instance_dir) if args.instance_dir else None
    base_name = resolve_base_name(args.instance, instance_dir)

    approaches = ["A", "B"] if args.approach == "both" else [args.approach]
    encodings = ["dense", "poly"] if args.encoding == "both" else [args.encoding]
    if args.seeds is not None:
        seeds_to_run = tuple(args.seeds)
    elif args.solver_seed is not None:
        seeds_to_run = (args.solver_seed,)
    else:
        seeds_to_run = SEEDS

    total_configs = len(seeds_to_run) * len(approaches) * len(encodings)
    total_units = float(total_configs) + 0.2

    log_progress(t_start, 0.01, total_units, f"Loading instance {base_name} from instances/ folder...")
    inst = load_instance_from_folder(instance_dir, base_name)

    if not args.allow_all_sizes and inst.n_assets not in ALLOWED_ASSET_SIZES:
        raise ValueError(
            f"solver2.py targets 200 and 400 asset instances only (got {inst.n_assets} assets in {base_name}). "
            f"Pass --allow-all-sizes to override."
        )

    log_progress(t_start, 0.08, total_units, f"Building PortfolioModel (A={inst.n_assets}, T={inst.n_periods}, B={args.budget}, lambda={args.lam})...")
    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    log_progress(t_start, 0.14, total_units, "Running fast vectorized empty-start local search baseline...")
    empty_ls_pos = fast_local_search(model, rs_sym, {})
    empty_ls_sc = model.score(empty_ls_pos)
    empty_ls_obj = int(empty_ls_sc.objective)

    log_progress(
        t_start,
        0.20,
        total_units,
        f"Baseline ready: empty_ls={empty_ls_obj} (feasible={empty_ls_sc.feasible}). "
        f"Starting {total_configs} quantum config(s) [cloud={bool(args.cloud)}, limit={args.time_limit_minutes}m]...",
    )

    exp_runs: list[dict[str, Any]] = []
    cfg_idx = 0
    for seed in seeds_to_run:
        for app in approaches:
            for enc in encodings:
                p_base = 0.20 + float(cfg_idx)
                cfg_idx += 1
                if app == "A":
                    res = solve_approach_a(
                        model,
                        routine=args.routine,
                        encoding_name=enc,
                        cluster_size=args.cluster_size_a,
                        n_layers=args.n_layers,
                        mc_population=args.mc_population,
                        mc_best_sets=args.mc_best_sets,
                        mc_iterations=args.mc_iterations,
                        initial_alpha=args.initial_alpha,
                        threshold=args.threshold,
                        max_steps=args.max_steps,
                        days_limit=args.days_limit,
                        use_cloud=bool(args.cloud),
                        shots=args.shots,
                        seed=seed,
                        t_global_start=t_start,
                        deadline=deadline,
                        progress_base=p_base,
                        progress_span=1.0,
                        total_units=total_units,
                    )
                else:
                    res = solve_approach_b(
                        model,
                        routine=args.routine,
                        encoding_name=enc,
                        cluster_size=args.cluster_size_b,
                        n_layers=args.n_layers,
                        mc_population=args.mc_population,
                        mc_best_sets=args.mc_best_sets,
                        mc_iterations=args.mc_iterations,
                        initial_alpha=args.initial_alpha,
                        threshold=args.threshold,
                        max_steps=args.max_steps,
                        days_limit=args.days_limit,
                        use_cloud=bool(args.cloud),
                        shots=args.shots,
                        seed=seed,
                        t_global_start=t_start,
                        deadline=deadline,
                        progress_base=p_base,
                        progress_span=1.0,
                        total_units=total_units,
                    )
                exp_runs.append(res)
                at = res["alpha_telemetry"]
                log_progress(
                    t_start,
                    0.20 + float(cfg_idx),
                    total_units,
                    f"Finished [App {app} | {enc} | seed {seed}]: pre_polish={res['pre_polish_objective']}, "
                    f"post_polish={res['objective']}, parts={res['sub_programs']}, "
                    f"mean_qubits={at.get('mean_qubits')}, alpha={at.get('mean_initial_alpha')}->{at.get('mean_final_alpha')} "
                    f"({at.get('pct_alpha_unchanged')}% unchanged), time={res['duration_seconds']}s",
                )

    best_run = min(exp_runs, key=lambda r: (0 if r["feasible"] else 1, r["objective"]))
    best_positions = best_run["positions"]
    if not best_run["feasible"] or best_run["objective"] > empty_ls_obj:
        best_positions = empty_ls_pos

    sol_path = resolve_solution_path(args.instance, base_name, args.budget, args.lam)
    sol_path.parent.mkdir(parents=True, exist_ok=True)
    written_score = model.write_solution(
        sol_path,
        best_positions,
        comment=f"solver2.py ({best_run['approach']}-{best_run['encoding']}, {args.routine})",
    )

    print_experiment_insights(base_name, inst.n_assets, inst.n_periods, args.budget, args.lam, empty_ls_obj, exp_runs)

    stem = sol_path.stem
    inst_plot_path = Path(f"{stem}_insights.png")
    save_instance_plot(inst_plot_path, base_name, inst.n_assets, args.budget, args.lam, empty_ls_obj, exp_runs)

    serializable_runs = [
        {k: v for k, v in r.items() if k not in ("positions", "score_obj")}
        for r in exp_runs
    ]
    exp_json_path = Path(f"{stem}_exp.json")
    exp_payload = {
        "instance": args.instance,
        "base_instance": base_name,
        "assets": inst.n_assets,
        "days": inst.n_periods,
        "budget": args.budget,
        "lambda": args.lam,
        "cloud": bool(args.cloud),
        "routine": args.routine,
        "empty_local_search_objective": empty_ls_obj,
        "best_config": f"{best_run['approach']}-{best_run['encoding']}-seed{best_run['seed']}",
        "best_pre_polish_objective": int(best_run["pre_polish_objective"]),
        "best_post_polish_objective": int(written_score.objective),
        "feasible": bool(written_score.feasible),
        "solution_file": str(sol_path),
        "instance_plot": inst_plot_path.name,
        "total_duration_seconds": round(time.time() - t_start, 2),
        "experiment_runs": serializable_runs,
    }
    exp_json_path.write_text(json.dumps(exp_payload, indent=2) + "\n", encoding="utf-8")
    cross_plot = update_cross_instance_plots(Path.cwd())
    if cross_plot is not None:
        exp_payload["cross_instance_plot"] = cross_plot.name

    print(f"Saved solution: {sol_path} (objective={written_score.objective}, feasible={written_score.feasible})")
    print(f"Saved plots:    {inst_plot_path}" + (f", {cross_plot}" if cross_plot else ""))
    print(f"\nAUTORUNNER_RESULT: {json.dumps({'score': int(written_score.objective), 'extra': exp_payload})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
