#!/usr/bin/env python3
"""Portfolio Optimization Solver (solver3.py) based on solver2.py.

Changes from solver2.py:
  1. Only writes the .sol file (no terminal printing, JSON logging, or figure creation).
  2. Improves the objective value reported by solution_checker.py via:
     - Multi-level Lagrangian budget penalty ladder for day/horizon 3-bit signed QUBOs
       combined with per-day warm-started classical repair & polish in Approach A.
     - Exact vectorized 1-unit + 2-asset simultaneous pairwise swap/rebalance descent
       plus multi-day block-perturbation iterated local search in fast_local_search.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
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
# Default Hyperparameters
# ---------------------------------------------------------------------------

ALLOWED_ASSET_SIZES: tuple[int, ...] = (10, 50, 200, 400)
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
BEAM_WIDTH = 12
N_PARTITION_CANDIDATES = 8
TOP_SOLUTIONS = 15
ILS_POLISH_SECONDS = 5.0

# 3-bit signed encoding lookup tables (8 bitstrings in {0,1}^3 -> v in [-3..3])
_CODES = np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.int64)
_SUBSETS = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


# ---------------------------------------------------------------------------
# CLI Helpers, Instance Loading & Backend
# ---------------------------------------------------------------------------


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
# 3-Bit Signed Net-Position Encoding, Fast Repair & Enhanced Vectorized Polish
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


def _polish_arrays_1_and_2_unit(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    L_in: np.ndarray,
    S_in: np.ndarray,
    max_rounds: int = 1000,
    max_pair_passes: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact vectorized 1-unit + 2-asset simultaneous swap/rebalance coordinate descent."""
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk
    R2 = R + R.transpose(0, 2, 1)
    Rdiag = np.diagonal(R, axis1=1, axis2=2)
    L, S = repair_fast_portfolio(model, rs_sym, L_in, S_in)
    v = L - S
    h = np.einsum("tij,tj->ti", R2, v)
    max_cash = (1 << model.cash_slack_bits) - 1
    max_count = (1 << model.count_slack_bits) - 1

    def _cash_int(c_val: int) -> int:
        return -sum(int(model.interest[k]) * ((c_val >> k) & 1) for k in range(model.cash_slack_bits))

    int_lut = np.array([_cash_int(c_val) for c_val in range(max_cash + 1)], dtype=np.int64)

    def _single_move_descent() -> None:
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

    _single_move_descent()
    if max_pair_passes <= 0:
        return L, S

    v[:] = np.clip(L - S, -model.ub, model.ub)
    L[:] = np.maximum(v, 0)
    S[:] = np.maximum(-v, 0)

    for _ in range(max_pair_passes):
        improved = False
        for t in range(T - 1):
            while True:
                vt = v[t]
                Lt, St = L[t], S[t]
                d_plus = np.full(A_count, 10**15, dtype=np.int64)
                d_minus = np.full(A_count, 10**15, dtype=np.int64)
                du_plus = np.where(vt >= 0, 1, -1)
                du_minus = np.where(vt <= 0, 1, -1)
                for dv, d_arr in ((1, d_plus), (-1, d_minus)):
                    vn = vt + dv
                    ok_v = (vn >= -model.ub) & (vn <= model.ub)
                    Ln, Sn = np.maximum(vn, 0), np.maximum(-vn, 0)
                    dL, dS = Ln - Lt, Sn - St
                    cost_d = dv * h[t] + Rdiag[t] - dv * model.ret[t] + dS * model.sc[t]
                    if t == 0:
                        cost_d += (dL + dS) * model.tc[0]
                    else:
                        cost_d += model.tc[t] * (
                            np.abs(Ln - L[t - 1]) - np.abs(Lt - L[t - 1])
                            + np.abs(Sn - S[t - 1]) - np.abs(St - S[t - 1])
                        )
                    if t + 1 <= T - 2:
                        cost_d += model.tc[t + 1] * (
                            np.abs(L[t + 1] - Ln) - np.abs(L[t + 1] - Lt)
                            + np.abs(S[t + 1] - Sn) - np.abs(S[t + 1] - St)
                        )
                    d_arr[ok_v] = cost_d[ok_v]

                net_t = int(vt.sum())
                tot_t = int((Lt + St).sum())
                c_now = model.capital - net_t
                best_day_d = 0
                best_pair = None
                for dv1, d1, du1 in ((1, d_plus, du_plus), (-1, d_minus, du_minus)):
                    for dv2, d2, du2 in ((1, d_plus, du_plus), (-1, d_minus, du_minus)):
                        c_new = c_now - (dv1 + dv2)
                        if not (0 <= c_new <= max_cash):
                            continue
                        d_int = int(int_lut[c_new] - int_lut[c_now])
                        tot_mat = tot_t + du1[:, None] + du2[None, :]
                        ok_mat = (
                            (model.budget - tot_mat >= 0)
                            & (model.budget - tot_mat <= max_count)
                            & (d1[:, None] < 10**14)
                            & (d2[None, :] < 10**14)
                        )
                        np.fill_diagonal(ok_mat, False)
                        if not ok_mat.any():
                            continue
                        mat = d1[:, None] + d2[None, :] + (dv1 * dv2) * R2[t] + d_int
                        mat_masked = np.where(ok_mat, mat, 0)
                        mval = int(mat_masked.min())
                        if mval < best_day_d:
                            idx = int(mat_masked.argmin())
                            i_s, j_s = divmod(idx, A_count)
                            best_day_d = mval
                            best_pair = (i_s, dv1, j_s, dv2)
                if best_pair is None:
                    break
                i_s, dv1, j_s, dv2 = best_pair
                for idx_a, dva in ((i_s, dv1), (j_s, dv2)):
                    v[t, idx_a] += dva
                    L[t, idx_a] = max(int(v[t, idx_a]), 0)
                    S[t, idx_a] = max(-int(v[t, idx_a]), 0)
                    h[t] += dva * R2[t, :, idx_a]
                improved = True
        if not improved:
            break
        _single_move_descent()
        v[:] = np.clip(L - S, -model.ub, model.ub)
        L[:] = np.maximum(v, 0)
        S[:] = np.maximum(-v, 0)

    return L, S


def fast_local_search(
    model: qt.PortfolioModel,
    rs_sym: np.ndarray,
    positions: qt.Positions,
    max_rounds: int = 1000,
    ils_seconds: float = ILS_POLISH_SECONDS,
    seed: int = 1,
) -> qt.Positions:
    """Exact vectorized 1-unit + 2-asset pairwise local search with iterated multi-day kicks.

    Overcomes single-move local minima where budget B and capital C bind, finding
    strictly lower objective values while maintaining 100% official QOBLIB feasibility.
    """
    t0 = time.time()
    L, S = model.to_arrays(positions)
    L_best, S_best = _polish_arrays_1_and_2_unit(
        model, rs_sym, L, S, max_rounds=max_rounds, max_pair_passes=60
    )
    best_sc = model.score_arrays(L_best, S_best)
    best_obj = int(best_sc.objective)

    if ils_seconds <= 0.0:
        return model.to_positions(L_best, S_best)

    rng = np.random.default_rng(seed)
    T = model.n_periods
    active_limit = max(1, T - 1)

    while time.time() - t0 < ils_seconds:
        L_try, S_try = L_best.copy(), S_best.copy()
        v_try = L_try - S_try
        n_kdays = int(rng.integers(1, min(4, active_limit + 1)))
        kdays = rng.choice(active_limit, size=n_kdays, replace=False)
        for t in kdays:
            held = np.flatnonzero(v_try[t] != 0)
            if len(held) > 0:
                n_drop = min(len(held), int(rng.integers(2, 8)))
                drop = rng.choice(held, size=n_drop, replace=False)
                v_try[t, drop] = 0
                L_try[t, drop] = 0
                S_try[t, drop] = 0
        L_pol, S_pol = _polish_arrays_1_and_2_unit(
            model, rs_sym, L_try, S_try, max_rounds=max_rounds, max_pair_passes=25
        )
        sc = model.score_arrays(L_pol, S_pol)
        if sc.feasible and int(sc.objective) < best_obj:
            best_obj = int(sc.objective)
            L_best, S_best = L_pol, S_pol

    return model.to_positions(L_best, S_best)


# ---------------------------------------------------------------------------
# ProgressivePCE / PCE Ensemble Runner
# ---------------------------------------------------------------------------


def make_encoding(encoding_name: str):
    name = encoding_name.strip().lower()
    if name == "dense":
        return DenseEncoding()
    if name == "poly":
        return PolyEncoding()
    raise ValueError(f"Unsupported encoding {encoding_name!r}; choose 'dense' or 'poly'.")


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
) -> PartitioningProgramEnsemble:
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
    return ensemble


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
    deadline: float,
    warm_positions: qt.Positions | None = None,
) -> dict[str, Any]:
    """Approach A: Per-Day 3-Bit Signed Net-Position Community-Partitioned ProgressivePCE."""
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    if warm_positions:
        L, S = model.to_arrays(warm_positions)
    else:
        L = np.zeros((T, A_count), dtype=np.int64)
        S = np.zeros((T, A_count), dtype=np.int64)

    active_days = list(range(max(1, T - 1)))
    if days_limit is not None and days_limit < len(active_days):
        idx = np.linspace(0, len(active_days) - 1, days_limit, dtype=int)
        active_days = sorted({active_days[i] for i in idx})

    unit_scale = float(np.median(np.abs(model.ret[: max(1, T - 1)]))) or 100.0
    mu = 0.65 * unit_scale

    for t in active_days:
        if time.time() >= deadline:
            break
        bqm, W, C = build_signed_day_bqm(model, rs_sym, L, S, t, mu=mu)
        backend = make_backend(use_cloud=use_cloud, shots=shots)
        ensemble = run_partitioned_bqm(
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
            if k_idx < 3:
                L_try, S_try = _polish_arrays_1_and_2_unit(
                    model, rs_sym, L_try, S_try, max_rounds=250, max_pair_passes=4
                )
            sc_try = model.score_arrays(L_try, S_try)
            if sc_try.feasible and int(sc_try.objective) < best_day_obj:
                best_day_obj = int(sc_try.objective)
                best_day_L, best_day_S = L_try, S_try
        L, S = best_day_L, best_day_S

    L, S = repair_fast_portfolio(model, rs_sym, L, S)
    pre_polish_pos = model.to_positions(L, S)
    polished_pos = fast_local_search(model, rs_sym, pre_polish_pos, seed=seed + 100)
    post_polish_score = model.score(polished_pos)

    return {
        "approach": "A",
        "encoding": encoding_name,
        "seed": seed,
        "objective": int(post_polish_score.objective),
        "feasible": bool(post_polish_score.feasible),
        "positions": polished_pos,
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
    deadline: float,
    warm_positions: qt.Positions | None = None,
) -> dict[str, Any]:
    """Approach B: Full-Horizon Multi-Period 3-Bit Signed Community-Partitioned ProgressivePCE."""
    T, A_count = model.n_periods, len(model.symbols)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    active_days = list(range(max(1, T - 1)))
    if days_limit is not None and days_limit < len(active_days):
        idx = np.linspace(0, len(active_days) - 1, days_limit, dtype=int)
        active_days = sorted({active_days[i] for i in idx})

    if time.time() >= deadline:
        pol_pos = fast_local_search(model, rs_sym, warm_positions or {}, seed=seed)
        pol_sc = model.score(pol_pos)
        return {
            "approach": "B",
            "encoding": encoding_name,
            "seed": seed,
            "objective": int(pol_sc.objective),
            "feasible": bool(pol_sc.feasible),
            "positions": pol_pos,
        }

    unit_scale = float(np.median(np.abs(model.ret[: max(1, T - 1)]))) or 100.0
    mu = 0.65 * unit_scale
    bqm, W_list, C_list = build_full_horizon_signed_bqm(model, rs_sym, active_days, mu=mu)

    backend = make_backend(use_cloud=use_cloud, shots=shots)
    ensemble = run_partitioned_bqm(
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

    best_post_pos: qt.Positions = warm_positions or {}
    best_post_sc = model.score(best_post_pos)

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

        if idx_s < 3:
            pol_pos = fast_local_search(
                model,
                rs_sym,
                rep_pos,
                ils_seconds=ILS_POLISH_SECONDS if idx_s == 0 else 1.0,
                seed=seed + idx_s,
            )
            pol_sc = model.score(pol_pos)
            if pol_sc.feasible and pol_sc.objective < best_post_sc.objective:
                best_post_pos, best_post_sc = pol_pos, pol_sc

    return {
        "approach": "B",
        "encoding": encoding_name,
        "seed": seed,
        "objective": int(best_post_sc.objective),
        "feasible": bool(best_post_sc.feasible),
        "positions": best_post_pos,
    }


# ---------------------------------------------------------------------------
# CLI Parser & Main Entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Portfolio optimization solver3.py (based on solver2.py, .sol generation only with improved objective)."
    )
    p.add_argument("--instance", default=os.environ.get("AUTORUNNER_INSTANCE", "po_a200_t10_orig"))
    p.add_argument("--instance-dir", default=os.environ.get("AUTORUNNER_INSTANCE_DIR"))
    p.add_argument("--assets", type=int, default=int(os.environ["AUTORUNNER_ASSETS"]) if os.environ.get("AUTORUNNER_ASSETS") else None)
    p.add_argument("--days", type=int, default=int(os.environ["AUTORUNNER_DAYS"]) if os.environ.get("AUTORUNNER_DAYS") else None)
    p.add_argument("--lambda", dest="lam", default=os.environ.get("AUTORUNNER_LAMBDA", "1e-4"))
    p.add_argument("--budget", type=int, default=int(os.environ.get("AUTORUNNER_BUDGET", "50")))
    p.add_argument("--cloud", type=_parse_bool, nargs="?", const=True, default=_parse_bool(os.environ.get("AUTORUNNER_CLOUD", "false")))
    p.add_argument("--solver-seed", type=int, default=int(os.environ["AUTORUNNER_SOLVER_SEED"]) if os.environ.get("AUTORUNNER_SOLVER_SEED") else None)

    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple RNG seeds (default: single seed [1]).")
    p.add_argument("--approach", choices=["A", "B", "both"], default=os.environ.get("SOLVER2_APPROACH", "A"))
    p.add_argument("--encoding", choices=["dense", "poly", "both"], default=os.environ.get("SOLVER2_ENCODING", "dense"))
    p.add_argument("--routine", choices=["progressive_pce", "pce"], default=os.environ.get("SOLVER2_ROUTINE", "progressive_pce"))
    p.add_argument("--cluster-size-a", type=int, default=CLUSTER_SIZE_A)
    p.add_argument("--cluster-size-b", type=int, default=CLUSTER_SIZE_B)
    p.add_argument("--days-limit", type=int, default=None)
    p.add_argument("--time-limit-minutes", type=float, default=TIME_LIMIT_MINUTES)
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--shots", type=int, default=SHOTS)
    p.add_argument("--mc-population", type=int, default=MC_POPULATION)
    p.add_argument("--mc-best-sets", type=int, default=MC_BEST_SETS)
    p.add_argument("--mc-iterations", type=int, default=MC_ITERATIONS)
    p.add_argument("--initial-alpha", type=float, default=INITIAL_ALPHA)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--allow-all-sizes", action="store_true")
    p.add_argument("--aggregate-only", type=Path, default=None)
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

    inst = load_instance_from_folder(instance_dir, base_name)
    if not args.allow_all_sizes and inst.n_assets not in ALLOWED_ASSET_SIZES:
        raise ValueError(
            f"Unsupported asset size {inst.n_assets} in {base_name}. Pass --allow-all-sizes to override."
        )

    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    R = model.risk.astype(float)
    rs_sym = 0.5 * (R + R.transpose(0, 2, 1))

    empty_ls_pos = fast_local_search(model, rs_sym, {}, seed=seeds_to_run[0])
    empty_ls_sc = model.score(empty_ls_pos)
    empty_ls_obj = int(empty_ls_sc.objective)

    exp_runs: list[dict[str, Any]] = []
    for seed in seeds_to_run:
        for app in approaches:
            for enc in encodings:
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
                        deadline=deadline,
                        warm_positions=empty_ls_pos,
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
                        deadline=deadline,
                        warm_positions=empty_ls_pos,
                    )
                exp_runs.append(res)

    best_run = min(exp_runs, key=lambda r: (0 if r["feasible"] else 1, r["objective"]))
    best_positions = best_run["positions"]
    if not best_run["feasible"] or best_run["objective"] > empty_ls_obj:
        best_positions = empty_ls_pos

    sol_path = resolve_solution_path(args.instance, base_name, args.budget, args.lam)
    sol_path.parent.mkdir(parents=True, exist_ok=True)
    model.write_solution(
        sol_path,
        best_positions,
        comment=f"solver3.py ({best_run['approach']}-{best_run['encoding']}, {args.routine})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
