"""Core helpers for the QIcing portfolio attack (QOBLIB 06).

* BKV table parsing (QOBLIB best-known values).
* Reduced-state view of the model: per (period, asset) a net position v = L - S in -ub..ub.
  Holding long AND short of the same asset is dominated (same net, extra fees and
  transaction costs), so every optimum has min(L, S) = 0.
* Exact dynamic programming over periods on the reduced state (feasible for 10 assets, B = 4).
* Solution bookkeeping: every solution we produce goes to solutions/ + results.csv.

The referee is always ``qoblib_tools.PortfolioModel.score`` (official objective).
"""
from __future__ import annotations

import csv
import datetime as dt
import itertools
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import qoblib_tools as qt  # noqa: E402

SOL_DIR = HERE / "solutions"
RESULTS_CSV = HERE / "results.csv"
BUDGET_BY_ASSETS = {10: 4, 50: 20, 200: 50, 400: 100}
LAM_TAGS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]


# ── best-known values ────────────────────────────────────────────────────


def load_bkv() -> dict:
    """{(instance, budget, lam_tag): (value, status)} from the QOBLIB table copy."""
    table = {}
    for line in (HERE / "qoblib_bkv_table.md").read_text().splitlines():
        m = re.match(r"\|\s*(a\d+_t\d+_\w+?)_b(\d+)_l([^ |]+)\s*\|\s*(-?\d+)\s*\|\s*([^|]+)\|", line)
        if not m:
            continue
        name, b, lam, val, status = m.groups()
        lam = lam.replace("e-0", "e-")  # table writes 1e-02, files 1e-2
        table[("po_" + name, int(b), lam)] = (int(val), status.strip())
    return table


def lam_value(tag: str) -> str:
    return "0" if tag == "0" else tag


def make_model(instance: str, lam_tag: str, budget: int | None = None) -> qt.PortfolioModel:
    inst = qt.load_portfolio_instance(instance)
    if budget is None:
        budget = BUDGET_BY_ASSETS[inst.n_assets]
    return qt.PortfolioModel(inst, budget=budget, lam=lam_value(lam_tag))


# ── reduced state: net positions v[t, i] ─────────────────────────────────


def v_to_positions(model: qt.PortfolioModel, v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.int64)
    return model.to_positions(np.maximum(v, 0), np.maximum(-v, 0))


def positions_to_v(model: qt.PortfolioModel, positions) -> np.ndarray:
    L, S = model.to_arrays(positions)
    return L - S


def score_v(model: qt.PortfolioModel, v: np.ndarray):
    v = np.asarray(v, dtype=np.int64)
    return model.score_arrays(np.maximum(v, 0), np.maximum(-v, 0))


def stage_tables(model: qt.PortfolioModel):
    """Per-period linear/quadratic coefficients in v (reduced state), exact integers.

    stage cost(t, v) = v R_t v + lin_t . v + absw_t . |v| + negw_t . max(-v, 0) + const_t
    transition (t-1 -> t, 1 <= t <= T-2): tc[t] . |v_t - v_{t-1}|
    """
    T, n = model.n_periods, len(model.symbols)
    lin = np.zeros((T, n), dtype=np.int64)
    absw = np.zeros((T, n), dtype=np.int64)
    negw = model.sc.astype(np.int64).copy()  # short cost per short unit
    const = np.zeros(T, dtype=np.int64)
    for t in range(T):
        if t < T - 1:
            lin[t] -= model.ret[t]
        lin[t] += 10  # interest = -10 * (C - net)  (nu * unit = 10 per unit of slack)
        const[t] = -10 * model.capital
        if t in (0, T - 1):
            absw[t] += model.tc[t]
    assert all(int(model.interest[k]) == 10 * 2**k for k in range(model.cash_slack_bits))
    return lin, absw, negw, const


def enumerate_states(n: int, ub: int, budget: int, capital: int, max_cash: int) -> np.ndarray:
    """All v in {-ub..ub}^n with sum|v| <= budget and capital - max_cash <= sum v <= capital."""
    out = []

    def rec(i, cur, used):
        if i == n:
            s = sum(cur)
            if capital - max_cash <= s <= capital:
                out.append(cur.copy())
            return
        for val in range(-ub, ub + 1):
            if used + abs(val) <= budget:
                cur.append(val)
                rec(i + 1, cur, used + abs(val))
                cur.pop()

    rec(0, [], 0)
    return np.array(out, dtype=np.int64)


from numba import njit, prange  # noqa: E402


@njit(parallel=True, cache=True)
def _dp_step(best, Va, Vb, w, out, arg):
    # best/Va: predecessor states sorted by cost (enables pruning); Vb: target states.
    K, n = Va.shape
    for b in prange(Vb.shape[0]):
        m = np.int64(1) << 60
        am = -1
        for a in range(K):
            d = best[a]
            if d >= m:
                break  # sorted: no later predecessor can win
            for i in range(n):
                x = Va[a, i] - Vb[b, i]
                d += w[i] * (x if x >= 0 else -x)
                if d >= m:
                    break
            if d < m:
                m, am = d, a
        out[b] = m
        arg[b] = am


def exact_dp(model: qt.PortfolioModel, verbose: bool = False):
    """Exact optimum by DP over periods on the reduced state. Returns (v[T, n], objective)."""
    T, n = model.n_periods, len(model.symbols)
    V = enumerate_states(n, model.ub, model.budget, model.capital, 2**model.cash_slack_bits - 1)
    K = len(V)
    lin, absw, negw, const = stage_tables(model)
    absV, negV = np.abs(V), np.maximum(-V, 0)
    stage = np.empty((T, K), dtype=np.int64)
    for t in range(T):
        risk = np.einsum("ki,ij,kj->k", V, model.risk[t], V)
        stage[t] = risk + V @ lin[t] + absV @ absw[t] + negV @ negw[t] + const[t]

    best = stage[0].copy()
    backs = []
    for t in range(1, T):
        w = model.tc[t].astype(np.int64) if t < T - 1 else np.zeros(n, dtype=np.int64)
        out = np.empty(K, dtype=np.int64)
        arg = np.empty(K, dtype=np.int64)
        if t == T - 1:  # no transition cost into the last period: free jump
            a0 = int(best.argmin())
            out[:] = best[a0]
            arg[:] = a0
        else:
            order = np.argsort(best)  # sort so the early-exit prunes well
            _dp_step(best[order], V[order], V, w, out, arg)
            arg = order[arg]
        backs.append(arg)
        best = out + stage[t]
        if verbose:
            print(f"  t={t}: states={K}, best so far {best.min()}")
    k = int(best.argmin())
    obj = int(best[k])
    chain = [k]
    for arg in reversed(backs):
        chain.append(int(arg[chain[-1]]))
    chain.reverse()
    return V[chain], obj


# ── solution bookkeeping ─────────────────────────────────────────────────


def record_solution(
    model: qt.PortfolioModel,
    positions,
    method: str,
    settings: str = "",
    n_qubo_vars: int | str = "",
    max_qubits: int | str = "",
    quantum: bool = False,
    tag: str = "",
):
    """Score with the referee, write a .sol (QOBLIB naming), append to results.csv."""
    score = model.score(positions)
    inst = model.instance.name
    lam_tag = _lam_tag(model)
    bkv = load_bkv().get((inst, model.budget, lam_tag))
    bkv_val, bkv_status = (bkv if bkv else ("", ""))
    gap = f"{(score.objective - bkv_val) / abs(bkv_val):+.4%}" if bkv and bkv_val else ""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{inst}_b{model.budget:03d}_l{lam_tag}"
    subdir = SOL_DIR / stem
    subdir.mkdir(parents=True, exist_ok=True)
    fname = subdir / f"{stem}__{tag or method}__{stamp}.sol"
    comment = (
        f"QIcing portfolio attack | method: {method}\nsettings: {settings}\n"
        f"quantum: {quantum} | qubo vars: {n_qubo_vars} | max qubits: {max_qubits}"
    )
    model.write_solution(fname, positions, comment=comment)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "instance", "budget", "lambda", "objective", "feasible", "bkv",
                        "bkv_status", "gap_vs_bkv", "method", "quantum", "qubo_vars",
                        "max_qubits", "settings", "file"])
        w.writerow([stamp, inst, model.budget, lam_tag, score.objective, score.feasible, bkv_val,
                    bkv_status, gap, method, quantum, n_qubo_vars, max_qubits, settings,
                    fname.relative_to(HERE)])
    return score, fname, bkv_val


def _lam_tag(model) -> str:
    lam = float(model.lam)
    if lam == 0:
        return "0"
    s = format(lam, "g")  # 1e-05, 5e-05, 0.0001
    mant, exp = f"{lam:.0e}".split("e")
    return f"{int(mant)}e-{int(exp[1:])}"
