"""Classical search for QOBLIB 06 (multi-period portfolio): ILS + parallel tempering.

The QOBLIB model is rewritten on integer positions v[t, i] = long - short in -ub..ub
(holding long and short units of the same asset never helps: every cost term is at
least as large and the constraints are looser for the net position).  Per day t:

    cost_t = v_t' R_t v_t + lin_t . v_t + sc_t . neg(v_t) + absc_t . |v_t| + cash[C - net_t]
    + w_t . |v_t - v_{t-1}|                         (rebalancing, t = 1..T-2)

subject to 0 <= C - net_t <= 2**cash_bits - 1 and sum_i |v_t,i| <= B.
This equals `qoblib_tools.PortfolioModel.score` (checked in `selftest`).

Moves
  * asset DP: the whole trajectory of one asset (or of a pair of assets) is optimised
    exactly by dynamic programming over the T days, with the other assets fixed.
  * PT: Metropolis on block moves (shift / set one asset or a pair over a day interval),
    replica exchange between temperatures, periodic DP polishing of the best state.
  * ILS: random kicks (set / zero / hedge blocks) followed by DP descent.

Usage
  python portfolio_search.py build --inst po_a050_t10_orig
  python portfolio_search.py run --inst po_a050_t10_orig --lam 1e-2 --minutes 5 --method hybrid
  python portfolio_search.py verify            # re-score every solutions/portfolio/*.sol
  python portfolio_search.py status [--markdown]

Solutions and log.csv go to $PORTFOLIO_SOL_DIR (default solutions/portfolio). A run warm-starts
from the .sol already there and overwrites it only when it improves. The checked hand-in set
is in deliverables/portfolio: PORTFOLIO_SOL_DIR=deliverables/portfolio python portfolio_search.py verify

Why the large-lambda instances are hard: the covariance matrices are nearly singular (about 21
of 50 eigenvalues are ~0 on a050), so good portfolios are sparse hedges of about 20 assets on single
days. No single- or pair-asset move from an empty day is profitable; `day_tabu` builds them.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numba as nb
import numpy as np

import qoblib_tools as qt

ROOT = Path(__file__).resolve().parent
CACHE = qt.DATA_DIR / "06-portfolio" / "cache"
OUT = Path(os.environ.get("PORTFOLIO_SOL_DIR", ROOT / "solutions" / "portfolio"))
LAMS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]
BUDGET = {10: 4, 50: 20, 200: 50, 400: 100}
INF = np.int64(1) << np.int64(60)


# ════════════════════════════════════════════════════════════════════════
# Instance tables (exact Zimpl rounding, cached as integers)
# ════════════════════════════════════════════════════════════════════════


def lam_key(lam: str) -> str:
    """QOBLIB README spelling of lambda: '1e-02', '5e-05', '0'."""
    x = float(Fraction(lam))
    return "0" if x == 0 else f"{x:.0e}"


def lam_tag(lam: str) -> str:
    """File-name spelling used by the notebook: '1e-2', '5e-5', '0'."""
    return lam_key(lam).replace("-0", "-")


def n_assets_of(inst: str) -> int:
    return int(re.search(r"_a(\d+)_", inst).group(1))


def n_periods_of(inst: str) -> int:
    return int(re.search(r"_t(\d+)_", inst).group(1))


def qubo_vars(n: int, T: int, ub: int = 3, cash_bits: int = 4, count_bits: int = 7) -> int:
    """Size of qoblib_tools.PortfolioQUBO: 2*ub*n position bits + slack bits, per day (710 for a010_t10)."""
    return T * (2 * ub * n + cash_bits + count_bits)


def build_tables(inst: str, unit=100_000, delta="0.001", nu="0.0001", rho="0.000025",
                 cash_bits=4) -> Path:
    """Stream the QOBLIB files once and cache every integer table for all LAMS."""
    dest = CACHE / f"{inst}.npz"
    if dest.exists():
        return dest
    t0 = time.time()
    base = f"06-portfolio/instances/{inst}"
    symbols, index, entries = [], {}, []
    for day, sym, price in qt._data_rows(qt.fetch_qoblib_file(f"{base}/stock_prices.txt.gz")):
        if sym not in index:
            index[sym] = len(symbols)
            symbols.append(sym)
        entries.append((index[sym], int(day), Fraction(price)))
    n = len(symbols)
    T = max(t for _, t, _ in entries) + 1
    prices = [[None] * T for _ in range(n)]
    for i, t, p in entries:
        prices[i][t] = p
    P = [[prices[i][t] * unit / prices[i][0] for t in range(T)] for i in range(n)]
    delta, nu, rho = Fraction(delta), Fraction(nu), Fraction(rho)
    lams = [Fraction(l) for l in LAMS]
    risk = np.zeros((len(LAMS), T, n, n), dtype=np.int64)
    seen = np.zeros((T, n, n), dtype=bool)
    for day, s1, s2, value in qt._data_rows(qt.fetch_qoblib_file(f"{base}/covariance_matrices.txt.gz")):
        t = int(day)
        if t >= T or s1 not in index or s2 not in index:
            continue
        i, j = index[s1], index[s2]
        b = Fraction(value) * P[i][t] * P[j][t]
        seen[t, i, j] = True
        for k, lam in enumerate(lams):
            if lam:
                risk[k, t, i, j] = qt._zround(lam * b)
    if not seen.all():
        raise ValueError(f"{inst}: missing covariance entries")
    ret = np.array([[qt._zround(P[i][t + 1] - P[i][t]) for i in range(n)] for t in range(T - 1)], dtype=np.int64)
    tc = np.array([[qt._zround(delta * P[i][t]) for i in range(n)] for t in range(T)], dtype=np.int64)
    sc = np.array([[qt._zround(rho * P[i][t]) for i in range(n)] for t in range(T)], dtype=np.int64)
    interest = np.array([qt._zround(nu * unit * 2**k) for k in range(cash_bits)], dtype=np.int64)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.part.npz")  # parallel builders don't collide
    np.savez_compressed(tmp, symbols=np.array(symbols), risk=risk, ret=ret, tc=tc, sc=sc,
                        interest=interest, lams=np.array(LAMS))
    tmp.replace(dest)
    print(f"built {dest.name}: n={n} T={T} in {time.time() - t0:.0f}s", flush=True)
    return dest


class Problem:
    """Integer tables of one (instance, budget, lambda) in the v-formulation."""

    def __init__(self, inst: str, lam: str, budget: int | None = None, ub=3, capital=10, cash_bits=4):
        data = np.load(build_tables(inst))
        self.inst, self.lam = inst, lam
        self.symbols = [str(s) for s in data["symbols"]]
        n = len(self.symbols)
        self.budget = int(budget if budget is not None else BUDGET[n])
        k = list(data["lams"]).index(lam)
        self.R = np.ascontiguousarray(data["risk"][k])
        T = self.R.shape[0]
        self.T, self.n = T, n
        self.RS = np.ascontiguousarray(self.R + self.R.transpose(0, 2, 1))
        self.lin = np.zeros((T, n), dtype=np.int64)
        self.lin[: T - 1] = -data["ret"]
        self.sc = data["sc"].astype(np.int64)
        tc = data["tc"].astype(np.int64)
        self.absc = np.zeros((T, n), dtype=np.int64)
        self.absc[0] += tc[0]
        self.absc[T - 1] += tc[T - 1]
        self.w = np.zeros((T, n), dtype=np.int64)
        self.w[1 : T - 1] = tc[1 : T - 1]
        self.ub, self.C = ub, capital
        self.maxcash = 2**cash_bits - 1
        interest = data["interest"]
        self.cashc = np.array(
            [-sum(int(interest[b]) for b in range(cash_bits) if (s >> b) & 1) for s in range(self.maxcash + 1)],
            dtype=np.int64,
        )
        self.args = (self.R, self.RS, self.lin, self.sc, self.absc, self.w, self.cashc,
                     self.C, self.maxcash, self.budget, self.ub)

    @property
    def qubo_vars(self) -> int:
        """Binary variables of the notebook's PortfolioQUBO (unary unit copies + cash/count slack bits)."""
        return qubo_vars(self.n, self.T, self.ub)

    @property
    def stem(self) -> str:
        return f"{self.inst}_b{self.budget:03d}_l{lam_tag(self.lam)}"

    @property
    def bkv_key(self) -> str:
        return f"{self.inst.removeprefix('po_')}_b{self.budget:03d}_l{lam_key(self.lam)}"

    # positions <-> v
    def v_from_positions(self, positions) -> np.ndarray:
        idx = {s: i for i, s in enumerate(self.symbols)}
        v = np.zeros((self.T, self.n), dtype=np.int64)
        for (t, sym), (lo, sh) in positions.items():
            v[t, idx[sym]] = lo - sh
        return v

    def positions(self, v) -> dict:
        return {(t, self.symbols[i]): (max(int(v[t, i]), 0), max(-int(v[t, i]), 0))
                for t in range(self.T) for i in range(self.n) if v[t, i]}

    def objective(self, v) -> int:
        return int(full_obj(v, *self.args))

    def feasible(self, v) -> bool:
        return bool(is_feasible(v, *self.args))


# ════════════════════════════════════════════════════════════════════════
# Numba kernels
# ════════════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@nb.njit(cache=True)
def make_state(v, RS, ub):
    T, n = v.shape
    G = np.zeros((T, n), dtype=np.int64)
    net = np.zeros(T, dtype=np.int64)
    absv = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for i in range(n):
            x = v[t, i]
            if x != 0:
                for k in range(n):
                    G[t, k] += x * RS[t, i, k]
            net[t] += x
            absv[t] += abs(x)
    return G, net, absv


@nb.njit(cache=True)
def is_feasible(v, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    T, n = v.shape
    for t in range(T):
        net = 0
        a = 0
        for i in range(n):
            if abs(v[t, i]) > ub:
                return False
            net += v[t, i]
            a += abs(v[t, i])
        if C - net < 0 or C - net > maxcash or a > B:
            return False
    return True


@nb.njit(cache=True)
def full_obj(v, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    T, n = v.shape
    obj = 0
    for t in range(T):
        net = 0
        for i in range(n):
            x = v[t, i]
            net += x
            if x != 0:
                s = 0
                for j in range(n):
                    s += R[t, i, j] * v[t, j]
                obj += x * s + lin[t, i] * x + sc[t, i] * max(-x, 0) + absc[t, i] * abs(x)
            if t > 0:
                obj += w[t, i] * abs(x - v[t - 1, i])
        slack = C - net
        if 0 <= slack <= maxcash:
            obj += cashc[slack]
    return obj


@nb.njit(cache=True)
def apply_value(t, i, x, v, G, net, absv, RS):
    d = x - v[t, i]
    if d == 0:
        return
    n = v.shape[1]
    for k in range(n):
        G[t, k] += d * RS[t, i, k]
    net[t] += d
    absv[t] += abs(x) - abs(v[t, i])
    v[t, i] = x


@nb.njit(cache=True)
def dp_single(i, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub, c, dp, arg, xout):
    """Exact best trajectory of asset i (others fixed). Returns new - old cost."""
    T = v.shape[0]
    K = 2 * ub + 1
    for t in range(T):
        vi = v[t, i]
        rii = R[t, i, i]
        coef = G[t, i] - 2 * rii * vi + lin[t, i]
        neto = net[t] - vi
        abso = absv[t] - abs(vi)
        for k in range(K):
            x = k - ub
            slack = C - neto - x
            if slack < 0 or slack > maxcash or abso + abs(x) > B:
                c[t, k] = INF
            else:
                c[t, k] = rii * x * x + coef * x + sc[t, i] * max(-x, 0) + absc[t, i] * abs(x) + cashc[slack]
    old = 0
    for t in range(T):
        old += c[t, v[t, i] + ub]
        if t > 0:
            old += w[t, i] * abs(v[t, i] - v[t - 1, i])
    for k in range(K):
        dp[0, k] = c[0, k]
    for t in range(1, T):
        wt = w[t, i]
        for k in range(K):
            if c[t, k] >= INF:
                dp[t, k] = INF
                continue
            best = INF
            ba = -1
            for kp in range(K):
                if dp[t - 1, kp] < INF:
                    val = dp[t - 1, kp] + wt * abs(k - kp)
                    if val < best:
                        best = val
                        ba = kp
            if ba < 0:
                dp[t, k] = INF
            else:
                dp[t, k] = best + c[t, k]
                arg[t, k] = ba
    best = INF
    bk = -1
    for k in range(K):
        if dp[T - 1, k] < best:
            best = dp[T - 1, k]
            bk = k
    if bk < 0:
        return np.int64(0)
    for t in range(T - 1, -1, -1):
        xout[t] = bk - ub
        if t > 0:
            bk = arg[t, bk]
    return best - old


@nb.njit(cache=True)
def dp_pair(i, j, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub,
            c, dp, argx, argy, tmp, xout, yout):
    """Exact best joint trajectory of assets i and j (others fixed). Returns new - old."""
    T = v.shape[0]
    K = 2 * ub + 1
    KK = K * K
    for t in range(T):
        vi = v[t, i]
        vj = v[t, j]
        rii = R[t, i, i]
        rjj = R[t, j, j]
        rij = RS[t, i, j]
        ci = G[t, i] - 2 * rii * vi - rij * vj + lin[t, i]
        cj = G[t, j] - 2 * rjj * vj - rij * vi + lin[t, j]
        neto = net[t] - vi - vj
        abso = absv[t] - abs(vi) - abs(vj)
        for kx in range(K):
            x = kx - ub
            fx = rii * x * x + ci * x + sc[t, i] * max(-x, 0) + absc[t, i] * abs(x)
            for ky in range(K):
                y = ky - ub
                s = kx * K + ky
                slack = C - neto - x - y
                if slack < 0 or slack > maxcash or abso + abs(x) + abs(y) > B:
                    c[t, s] = INF
                else:
                    c[t, s] = (fx + rjj * y * y + cj * y + rij * x * y + sc[t, j] * max(-y, 0)
                               + absc[t, j] * abs(y) + cashc[slack])
    old = 0
    for t in range(T):
        old += c[t, (v[t, i] + ub) * K + v[t, j] + ub]
        if t > 0:
            old += w[t, i] * abs(v[t, i] - v[t - 1, i]) + w[t, j] * abs(v[t, j] - v[t - 1, j])
    for s in range(KK):
        dp[0, s] = c[0, s]
    for t in range(1, T):
        wi = w[t, i]
        wj = w[t, j]
        # tmp[kxp, ky] = min_kyp dp[t-1, kxp, kyp] + wj |ky - kyp|
        for kxp in range(K):
            for ky in range(K):
                best = INF
                ba = -1
                for kyp in range(K):
                    d = dp[t - 1, kxp * K + kyp]
                    if d < INF:
                        val = d + wj * abs(ky - kyp)
                        if val < best:
                            best = val
                            ba = kyp
                tmp[kxp, ky] = best
                argy[t, kxp, ky] = ba
        for kx in range(K):
            for ky in range(K):
                s = kx * K + ky
                if c[t, s] >= INF:
                    dp[t, s] = INF
                    continue
                best = INF
                ba = -1
                for kxp in range(K):
                    d = tmp[kxp, ky]
                    if d < INF:
                        val = d + wi * abs(kx - kxp)
                        if val < best:
                            best = val
                            ba = kxp
                if ba < 0:
                    dp[t, s] = INF
                else:
                    dp[t, s] = best + c[t, s]
                    argx[t, s] = ba
    best = INF
    bs = -1
    for s in range(KK):
        if dp[T - 1, s] < best:
            best = dp[T - 1, s]
            bs = s
    if bs < 0:
        return np.int64(0)
    kx = bs // K
    ky = bs % K
    for t in range(T - 1, -1, -1):
        xout[t] = kx - ub
        yout[t] = ky - ub
        if t > 0:
            kxp = argx[t, kx * K + ky]
            kyp = argy[t, kxp, ky]
            kx, ky = kxp, kyp
    return best - old


@nb.njit(cache=True)
def descent(v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub, pairs, max_rounds):
    """Single-asset DP sweeps to convergence, then pair-DP sweeps over `pairs`; repeat."""
    T, n = v.shape
    K = 2 * ub + 1
    KK = K * K
    c1 = np.empty((T, K), dtype=np.int64)
    dp1 = np.empty((T, K), dtype=np.int64)
    arg1 = np.zeros((T, K), dtype=np.int64)
    x1 = np.empty(T, dtype=np.int64)
    c2 = np.empty((T, KK), dtype=np.int64)
    dp2 = np.empty((T, KK), dtype=np.int64)
    argx = np.zeros((T, KK), dtype=np.int64)
    argy = np.zeros((T, K, K), dtype=np.int64)
    tmp = np.empty((K, K), dtype=np.int64)
    x2 = np.empty(T, dtype=np.int64)
    y2 = np.empty(T, dtype=np.int64)
    total = np.int64(0)
    for _ in range(max_rounds):
        while True:
            imp = False
            perm = np.random.permutation(n)
            for i in perm:
                d = dp_single(i, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub,
                              c1, dp1, arg1, x1)
                if d < 0:
                    for t in range(T):
                        apply_value(t, i, x1[t], v, G, net, absv, RS)
                    total += d
                    imp = True
            if not imp:
                break
        imp = False
        order = np.random.permutation(pairs.shape[0])
        for q in order:
            i = pairs[q, 0]
            j = pairs[q, 1]
            d = dp_pair(i, j, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub,
                        c2, dp2, argx, argy, tmp, x2, y2)
            if d < 0:
                for t in range(T):
                    apply_value(t, i, x2[t], v, G, net, absv, RS)
                    apply_value(t, j, y2[t], v, G, net, absv, RS)
                total += d
                imp = True
        if not imp:
            break
    return total


@nb.njit(cache=True)
def block_delta(i, j, a, b, xi, xj, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    """Cost change of setting asset i (and j >= 0) to xi[t] (xj[t]) for t in [a, b]; INF if infeasible."""
    T = v.shape[0]
    delta = np.int64(0)
    for t in range(a, b + 1):
        oi = v[t, i]
        ni = xi[t]
        di = ni - oi
        oj = 0
        nj = 0
        dj = 0
        if j >= 0:
            oj = v[t, j]
            nj = xj[t]
            dj = nj - oj
        if di == 0 and dj == 0:
            continue
        if abs(ni) > ub or abs(nj) > ub:
            return INF
        nnet = net[t] + di + dj
        slack = C - nnet
        if slack < 0 or slack > maxcash:
            return INF
        if absv[t] + abs(ni) - abs(oi) + abs(nj) - abs(oj) > B:
            return INF
        dr = di * G[t, i] + di * di * R[t, i, i] + lin[t, i] * di
        dr += sc[t, i] * (max(-ni, 0) - max(-oi, 0)) + absc[t, i] * (abs(ni) - abs(oi))
        if j >= 0:
            dr += dj * G[t, j] + dj * dj * R[t, j, j] + di * dj * RS[t, i, j] + lin[t, j] * dj
            dr += sc[t, j] * (max(-nj, 0) - max(-oj, 0)) + absc[t, j] * (abs(nj) - abs(oj))
        delta += dr + cashc[slack] - cashc[C - net[t]]
    lo = max(a, 1)
    hi = min(b + 1, T - 1)
    for t in range(lo, hi + 1):
        n_t = xi[t] if t <= b else v[t, i]
        n_p = xi[t - 1] if t - 1 >= a else v[t - 1, i]
        delta += w[t, i] * (abs(n_t - n_p) - abs(v[t, i] - v[t - 1, i]))
        if j >= 0:
            n_t = xj[t] if t <= b else v[t, j]
            n_p = xj[t - 1] if t - 1 >= a else v[t - 1, j]
            delta += w[t, j] * (abs(n_t - n_p) - abs(v[t, j] - v[t - 1, j]))
    return delta


@nb.njit(cache=True)
def propose(v, ub, xi, xj):
    """Random block move. Returns (i, j, a, b) and fills xi / xj on [a, b]."""
    T, n = v.shape
    a = np.random.randint(T)
    b = np.random.randint(T)
    if a > b:
        a, b = b, a
    if np.random.random() < 0.5:  # short blocks are the common case
        b = min(b, a + np.random.randint(3))
    i = np.random.randint(n)
    r = np.random.random()
    j = -1
    if r < 0.30:  # shift one asset
        d = 1 if np.random.random() < 0.5 else -1
        for t in range(a, b + 1):
            xi[t] = v[t, i] + d
    elif r < 0.50:  # set one asset to a value (neighbour value, zero or random)
        q = np.random.random()
        if q < 0.3 and a > 0:
            val = v[a - 1, i]
        elif q < 0.6 and b < T - 1:
            val = v[b + 1, i]
        elif q < 0.8:
            val = 0
        else:
            val = np.random.randint(2 * ub + 1) - ub
        for t in range(a, b + 1):
            xi[t] = val
    else:  # pair shift (hedge / transfer / joint)
        j = np.random.randint(n - 1)
        if j >= i:
            j += 1
        di = 1 if np.random.random() < 0.5 else -1
        dj = 1 if np.random.random() < 0.5 else -1
        for t in range(a, b + 1):
            xi[t] = v[t, i] + di
            xj[t] = v[t, j] + dj
    return i, j, a, b


@nb.njit(cache=True)
def pt_chunk(V, Gs, nets, absvs, E, temps, n_sweeps, moves_per_sweep, best_v, best_e,
             R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    """Parallel tempering: n_sweeps sweeps over all replicas + neighbour exchanges."""
    nr, T, n = V.shape
    xi = np.zeros(T, dtype=np.int64)
    xj = np.zeros(T, dtype=np.int64)
    acc_swap = 0
    for _ in range(n_sweeps):
        for r in range(nr):
            v = V[r]
            G = Gs[r]
            net = nets[r]
            absv = absvs[r]
            beta = 1.0 / temps[r]
            for _m in range(moves_per_sweep):
                i, j, a, b = propose(v, ub, xi, xj)
                d = block_delta(i, j, a, b, xi, xj, v, G, net, absv, R, RS, lin, sc, absc, w, cashc,
                                C, maxcash, B, ub)
                if d >= INF:
                    continue
                if d <= 0 or np.random.random() < math.exp(-d * beta):
                    for t in range(a, b + 1):
                        apply_value(t, i, xi[t], v, G, net, absv, RS)
                        if j >= 0:
                            apply_value(t, j, xj[t], v, G, net, absv, RS)
                    E[r] += d
                    if E[r] < best_e[0]:
                        best_e[0] = E[r]
                        best_v[:, :] = v
        for r in range(nr - 1):
            x = (E[r] - E[r + 1]) * (1.0 / temps[r] - 1.0 / temps[r + 1])
            if x >= 0 or np.random.random() < math.exp(x):
                acc_swap += 1
                tv = V[r].copy()
                V[r] = V[r + 1]
                V[r + 1] = tv
                tg = Gs[r].copy()
                Gs[r] = Gs[r + 1]
                Gs[r + 1] = tg
                tn = nets[r].copy()
                nets[r] = nets[r + 1]
                nets[r + 1] = tn
                ta = absvs[r].copy()
                absvs[r] = absvs[r + 1]
                absvs[r + 1] = ta
                te = E[r]
                E[r] = E[r + 1]
                E[r + 1] = te
    return acc_swap


@nb.njit(cache=True)
def kick(v, G, net, absv, k, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub, touched):
    """Apply k random feasible block moves regardless of cost. Marks touched assets."""
    T, n = v.shape
    xi = np.zeros(T, dtype=np.int64)
    xj = np.zeros(T, dtype=np.int64)
    done = 0
    tries = 0
    while done < k and tries < 50 * k:
        tries += 1
        i, j, a, b = propose(v, ub, xi, xj)
        d = block_delta(i, j, a, b, xi, xj, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub)
        if d >= INF:
            continue
        for t in range(a, b + 1):
            apply_value(t, i, xi[t], v, G, net, absv, RS)
            if j >= 0:
                apply_value(t, j, xj[t], v, G, net, absv, RS)
        touched[i] = True
        if j >= 0:
            touched[j] = True
        done += 1
    return done


@nb.njit(cache=True)
def day_tabu(t, v, G, net, absv, iters, tenure, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    """Tabu search on the day-t vector v[t, :] (other days fixed), single and pair +-1 moves.

    Every iteration applies the best admissible move (even if worsening); a reversed move
    on a recently moved asset is tabu unless it gives a new best. The best day vector seen
    is restored at the end. Returns the (non-positive) cost change.
    """
    T, n = v.shape
    base = np.empty((n, 2), dtype=np.int64)  # cost change of v[t,i] += d (d = +1, -1), cash excluded
    okm = np.zeros((n, 2), dtype=np.bool_)
    tabu_until = np.zeros((n, 2), dtype=np.int64)  # forbids the move direction (i, d) until iteration
    best_vec = v[t].copy()
    cur = np.int64(0)
    best = np.int64(0)
    wt = w[t]
    has_next = t + 1 < T
    for it in range(1, iters + 1):
        for i in range(n):
            x = v[t, i]
            for s in range(2):
                d = 1 - 2 * s
                y = x + d
                if y > ub or y < -ub:
                    okm[i, s] = False
                    continue
                okm[i, s] = True
                c = d * G[t, i] + R[t, i, i] + lin[t, i] * d
                c += sc[t, i] * (max(-y, 0) - max(-x, 0)) + absc[t, i] * (abs(y) - abs(x))
                if t > 0:
                    c += wt[i] * (abs(y - v[t - 1, i]) - abs(x - v[t - 1, i]))
                if has_next:
                    c += w[t + 1, i] * (abs(v[t + 1, i] - y) - abs(v[t + 1, i] - x))
                base[i, s] = c
        cash0 = cashc[C - net[t]]
        bd = INF  # best admissible move
        bi = -1
        bj = -1
        bsi = 0
        bsj = 0
        fd = INF  # best move ignoring tabu (fallback when everything is tabu)
        fi = -1
        fj = -1
        fsi = 0
        fsj = 0
        for i in range(n):
            for s in range(2):
                if not okm[i, s]:
                    continue
                d = 1 - 2 * s
                da = abs(v[t, i] + d) - abs(v[t, i])
                ti = tabu_until[i, s] < it
                # single move
                slack = C - net[t] - d
                if 0 <= slack <= maxcash and absv[t] + da <= B:
                    c = base[i, s] + cashc[slack] - cash0
                    if c < fd:
                        fd = c
                        fi = i
                        fj = -1
                        fsi = s
                    if (ti or cur + c < best) and c < bd:
                        bd = c
                        bi = i
                        bj = -1
                        bsi = s
                # pair moves (i, s) + (j, s2), j > i
                for j in range(i + 1, n):
                    for s2 in range(2):
                        if not okm[j, s2]:
                            continue
                        d2 = 1 - 2 * s2
                        slack = C - net[t] - d - d2
                        if slack < 0 or slack > maxcash:
                            continue
                        if absv[t] + da + abs(v[t, j] + d2) - abs(v[t, j]) > B:
                            continue
                        c = base[i, s] + base[j, s2] + d * d2 * RS[t, i, j] + cashc[slack] - cash0
                        if c < fd:
                            fd = c
                            fi = i
                            fj = j
                            fsi = s
                            fsj = s2
                        if c < bd and ((ti and tabu_until[j, s2] < it) or cur + c < best):
                            bd = c
                            bi = i
                            bj = j
                            bsi = s
                            bsj = s2
        if bi < 0:
            if fi < 0:
                break
            bd, bi, bj, bsi, bsj = fd, fi, fj, fsi, fsj
        d = 1 - 2 * bsi
        apply_value(t, bi, v[t, bi] + d, v, G, net, absv, RS)
        tabu_until[bi, 1 - bsi] = it + tenure + np.random.randint(tenure + 1)
        if bj >= 0:
            d2 = 1 - 2 * bsj
            apply_value(t, bj, v[t, bj] + d2, v, G, net, absv, RS)
            tabu_until[bj, 1 - bsj] = it + tenure + np.random.randint(tenure + 1)
        cur += bd
        if cur < best:
            best = cur
            best_vec[:] = v[t]
    for i in range(n):
        apply_value(t, i, best_vec[i], v, G, net, absv, RS)
    return best


@nb.njit(cache=True)
def sample_deltas(v, G, net, absv, m, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub):
    T, n = v.shape
    xi = np.zeros(T, dtype=np.int64)
    xj = np.zeros(T, dtype=np.int64)
    out = np.zeros(m, dtype=np.int64)
    k = 0
    for _ in range(20 * m):
        i, j, a, b = propose(v, ub, xi, xj)
        d = block_delta(i, j, a, b, xi, xj, v, G, net, absv, R, RS, lin, sc, absc, w, cashc, C, maxcash, B, ub)
        if d < INF and d != 0:
            out[k] = abs(d)
            k += 1
            if k == m:
                break
    return out[:k]


# ════════════════════════════════════════════════════════════════════════
# Solution files, log, best-known values
# ════════════════════════════════════════════════════════════════════════


def sol_path(p: Problem) -> Path:
    return OUT / f"{p.stem}.sol"


def read_objective(path: Path) -> int | None:
    if not path.exists():
        return None
    header, _ = qt.read_portfolio_solution(path)
    return int(header["objective"]) if "objective" in header else None


def write_sol(p: Problem, v, comment: str) -> Path:
    """Write the canonical .sol (same format as PortfolioModel.write_solution) if it improves."""
    obj = p.objective(v)
    path = sol_path(p)
    old = read_objective(path)
    if old is not None and old <= obj:
        return path
    assert p.feasible(v), "refusing to write an infeasible solution"
    lines = [f"# {line}" for line in comment.splitlines()]
    lines += [f"instance {p.inst}", f"budget {p.budget}", f"lambda {qt._format_decimal(Fraction(p.lam))}",
              f"objective {obj}", "# period symbol long short"]
    for (t, sym), (lo, sh) in sorted(p.positions(v).items()):
        lines.append(f"{t} {sym} {lo} {sh}")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    return path


_BKV = None


def bkv_table() -> dict:
    global _BKV
    if _BKV is None:
        _BKV = {}
        text = qt.fetch_qoblib_file("06-portfolio/solutions/README.md").read_text()
        for m in re.finditer(r"^\| (a\d+_t\d+_\w+?_b\d+_l[\w.+-]+) \| (-?[\d.]+) \| ([\w ]+) \|", text, re.M):
            _BKV[m.group(1)] = (float(m.group(2)), m.group(3).strip())
    return _BKV


def log_result(p: Problem, obj: int, method: str):
    bkv = bkv_table().get(p.bkv_key)
    gap = "" if bkv is None or bkv[0] == 0 else f"{(obj - bkv[0]) / abs(bkv[0]) * 100:.4f}%"
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["timestamp", "instance", "budget", "lambda", "vars", "objective", "best_known", "gap", "method"])
        wr.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), p.inst, p.budget, p.lam, p.qubo_vars,
                     obj, "" if bkv is None else int(bkv[0]), gap, method])


# ════════════════════════════════════════════════════════════════════════
# Drivers
# ════════════════════════════════════════════════════════════════════════


def all_pairs(n):
    iu = np.triu_indices(n, 1)
    return np.ascontiguousarray(np.stack(iu, axis=1).astype(np.int64))


def pairs_touching(touched_idx, n):
    rows = [(i, j) for i in touched_idx for j in range(n) if j != i]
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.array(rows, dtype=np.int64)
    arr = np.sort(arr, axis=1)
    return np.unique(arr, axis=0)


class Search:
    def __init__(self, p: Problem, seed: int, label: str, deadline: float, verbose=True):
        self.p, self.seed, self.label, self.deadline, self.verbose = p, seed, label, deadline, verbose
        seed_rng(seed)
        self.rng = np.random.default_rng(seed)
        self.full_pairs = all_pairs(p.n)
        self.best_v = None
        self.best_e = None
        self.t0 = time.time()

    def offer(self, v, tag):
        e = self.p.objective(v)
        if self.best_e is None or e < self.best_e:
            self.best_e = e
            self.best_v = v.copy()
            if self.verbose:
                print(f"[{time.time() - self.t0:7.1f}s] {tag}: {e}", flush=True)
        return e

    def polish(self, v):
        """DP descent (singles + full pair sweep) in place; returns objective."""
        G, net, absv = make_state(v, self.p.RS, self.p.ub)
        descent(v, G, net, absv, *self.p.args, self.full_pairs, 1000)
        return self.p.objective(v)

    def ils(self, v, seconds, kmax=None, temp=0.0):
        p = self.p
        n = p.n
        kmax = kmax or max(3, n // 8)
        end = min(self.deadline, time.time() + seconds)
        cur = v.copy()
        G, net, absv = make_state(cur, p.RS, p.ub)
        descent(cur, G, net, absv, *p.args, self.full_pairs, 1000)
        ecur = p.objective(cur)
        self.offer(cur, "ils descent")
        it = 0
        while time.time() < end:
            it += 1
            trial = cur.copy()
            G, net, absv = make_state(trial, p.RS, p.ub)
            touched = np.zeros(n, dtype=np.bool_)
            k = int(self.rng.integers(1, kmax + 1))
            kick(trial, G, net, absv, k, *p.args, touched)
            if n <= 60 or it % 20 == 0:
                pairs = self.full_pairs
            else:
                pairs = pairs_touching(np.flatnonzero(touched), n)
            descent(trial, G, net, absv, *p.args, pairs, 1000)
            e = p.objective(trial)
            if e < ecur or (temp > 0 and self.rng.random() < math.exp(-(e - ecur) / temp)) or e == ecur:
                cur, ecur = trial, e
            if e < self.best_e:
                if n > 60:  # make sure the candidate is pair-optimal before recording it
                    G, net, absv = make_state(trial, p.RS, p.ub)
                    descent(trial, G, net, absv, *p.args, self.full_pairs, 1000)
                    e = p.objective(trial)
                    if e < ecur:
                        cur, ecur = trial.copy(), e
                self.offer(trial, f"ils it={it} k={k}")
            # occasional restart from the best
            if it % 200 == 0 and self.best_e is not None and ecur > self.best_e:
                cur, ecur = self.best_v.copy(), self.best_e
        return it

    def tabu_days(self, v, seconds, iters=None, tenure=None, reset_prob=0.3):
        """ILS over days: tabu-search one day's vector (optionally from scratch), DP descent, accept if better."""
        p = self.p
        iters = iters or 50 + 10 * p.budget
        tenure = tenure or 7 + p.n // 20
        end = min(self.deadline, time.time() + seconds)
        cur = v.copy()
        ecur = p.objective(cur)
        pairs_small = self.full_pairs if p.n <= 60 else np.zeros((0, 2), dtype=np.int64)
        tried = 0
        while time.time() < end:
            improved = False
            for t in self.rng.permutation(p.T):
                tried += 1
                trial = cur.copy()
                if self.rng.random() < reset_prob:
                    trial[t] = 0
                    if not p.feasible(trial):
                        trial[t] = cur[t]
                G, net, absv = make_state(trial, p.RS, p.ub)
                day_tabu(int(t), trial, G, net, absv, iters, tenure, *p.args)
                descent(trial, G, net, absv, *p.args, pairs_small, 1000)
                e = p.objective(trial)
                if e < ecur:
                    if p.n > 60:
                        e = self.polish(trial)
                    cur, ecur = trial, e
                    improved = True
                    self.offer(cur, f"tabu day={t} tried={tried}")
                if time.time() >= end:
                    break
            if not improved:  # perturb (ILS) and keep going from the best
                cur = self.best_v.copy()
                G, net, absv = make_state(cur, p.RS, p.ub)
                touched = np.zeros(p.n, dtype=np.bool_)
                kick(cur, G, net, absv, int(self.rng.integers(1, max(3, p.n // 8) + 1)), *p.args, touched)
                ecur = p.objective(cur)
        return tried

    def pt(self, v, seconds, n_rep=12, tmin=None, tmax=None, chunk_sweeps=None, polish_every=10.0):
        p = self.p
        end = min(self.deadline, time.time() + seconds)
        G, net, absv = make_state(v, p.RS, p.ub)
        if tmax is None or tmin is None:
            ds = sample_deltas(v, G, net, absv, 4000, *p.args)
            med = float(np.median(ds)) if len(ds) else 100.0
            tmax = tmax or max(5.0, 0.5 * med)
            tmin = tmin or max(1.0, tmax / 300.0)
        temps = np.geomspace(tmin, tmax, n_rep)
        V = np.repeat(v[None], n_rep, axis=0).copy()
        Gs = np.repeat(G[None], n_rep, axis=0).copy()
        nets = np.repeat(net[None], n_rep, axis=0).copy()
        absvs = np.repeat(absv[None], n_rep, axis=0).copy()
        e0 = p.objective(v)
        E = np.full(n_rep, e0, dtype=np.int64)
        best_v = v.copy()
        best_e = np.array([e0], dtype=np.int64)
        moves = p.T * p.n
        chunk = chunk_sweeps or max(1, int(20000 // moves))
        if self.verbose:
            print(f"  PT temps {tmin:.1f}..{tmax:.1f}, {n_rep} replicas, {moves} moves/sweep", flush=True)
        last_polish = time.time()
        sweeps = 0
        while time.time() < end:
            pt_chunk(V, Gs, nets, absvs, E, temps, chunk, moves, best_v, best_e, *p.args)
            sweeps += chunk
            if time.time() - last_polish > polish_every or time.time() >= end:
                last_polish = time.time()
                cand = best_v.copy()
                e = self.polish(cand)
                self.offer(cand, f"pt sweeps={sweeps} raw={best_e[0]}")
                if e < E[0]:  # inject the polished state into the coldest replica
                    V[0] = cand
                    g, nt, ab = make_state(cand, p.RS, p.ub)
                    Gs[0], nets[0], absvs[0] = g, nt, ab
                    E[0] = e
                    best_e[0] = min(best_e[0], e)
                    best_v[:, :] = cand
        # sanity: replica energies must match their states
        for r in range(n_rep):
            assert p.objective(V[r]) == E[r], "PT energy drift"
        return sweeps


def start_state(p: Problem, init: str):
    path = sol_path(p)
    if init == "best" and path.exists():
        _, pos = qt.read_portfolio_solution(path)
        v = p.v_from_positions(pos)
        if p.feasible(v):
            return v, "warm(best)"
    if init not in ("best", "zero") and Path(init).exists():
        _, pos = qt.read_portfolio_solution(init)
        return p.v_from_positions(pos), f"warm({Path(init).name})"
    return np.zeros((p.T, p.n), dtype=np.int64), "zero"


def cmd_run(a):
    p = Problem(a.inst, a.lam, a.budget)
    deadline = time.time() + 60 * a.minutes
    s = Search(p, a.seed, a.method, deadline)
    v, how = start_state(p, a.init)
    bkv = bkv_table().get(p.bkv_key)
    print(f"{p.stem}: n={p.n} T={p.T} B={p.budget} vars={p.n * p.T * 2}  start={how} obj={p.objective(v)}  "
          f"BKV={bkv}", flush=True)
    s.offer(v, "start")
    if a.method == "ils":
        s.ils(v, 1e9, temp=a.ils_temp)
    elif a.method == "pt":
        s.pt(v, 1e9, n_rep=a.replicas)
    elif a.method == "tabu":
        s.tabu_days(v, 1e9)
    else:  # hybrid: ILS -> day tabu -> PT (warm) -> ... alternating phases
        phase = max(10.0, 60 * a.minutes / 6)
        while time.time() < deadline:
            s.ils(s.best_v.copy(), phase / 3, temp=a.ils_temp)
            if time.time() >= deadline:
                break
            s.tabu_days(s.best_v.copy(), phase / 2)
            if time.time() >= deadline:
                break
            s.pt(s.best_v.copy(), phase / 2, n_rep=a.replicas)
    v = s.best_v
    e = p.objective(v)
    assert p.feasible(v)
    method = f"{a.method} seed={a.seed} {a.minutes}min init={how}"
    path = write_sol(p, v, f"Qoro x QOBLIB hackathon: portfolio_search.py {method}")
    log_result(p, e, method)
    gap = "" if bkv is None else f"  gap {(e - bkv[0]) / abs(bkv[0]) * 100:+.4f}% vs {bkv[0]:.0f} ({bkv[1]})"
    print(f"FINAL {p.stem}: {e}{gap}  -> {path.name} (file best {read_objective(path)})", flush=True)


def cmd_build(a):
    for inst in a.inst:
        build_tables(inst)


def cmd_selftest(a):
    """Compare the v-formulation with qoblib_tools' official scorer on random feasible states."""
    rng = np.random.default_rng(0)
    for inst in a.inst:
        inst_obj = qt.load_portfolio_instance(inst)
        for lam in LAMS:
            p = Problem(inst, lam)
            model = qt.PortfolioModel(inst_obj, budget=p.budget, lam=lam)
            assert (model.risk == p.R).all(), "risk table mismatch"
            for _ in range(30):
                v = np.zeros((p.T, p.n), dtype=np.int64)
                G, net, absv = make_state(v, p.RS, p.ub)
                touched = np.zeros(p.n, dtype=np.bool_)
                kick(v, G, net, absv, int(rng.integers(1, 40)), *p.args, touched)
                L, S = np.maximum(v, 0), np.maximum(-v, 0)
                ref = model.score_arrays(L, S)
                assert ref.feasible == p.feasible(v)
                assert ref.objective == p.objective(v), (ref.objective, p.objective(v))
                G2, net2, absv2 = make_state(v, p.RS, p.ub)
                assert (G2 == G).all() and (net2 == net).all() and (absv2 == absv).all()
            print(f"selftest ok: {inst} lam={lam}")
        # the published reference solution must score as published
        p = Problem(inst, "1e-4")
        try:
            hdr, pos = qt.read_portfolio_solution(qt.fetch_portfolio_reference(inst, p.budget, "1e-4"))
            assert int(hdr["objective"]) == p.objective(p.v_from_positions(pos))
            print(f"reference ok: {inst} 1e-4 objective {hdr['objective']}")
        except FileNotFoundError:
            pass


def cmd_verify(a):
    """Re-score every .sol in OUT with qoblib_tools.PortfolioModel (the official port)."""
    cache = {}
    for path in sorted(OUT.glob("*.sol")):
        hdr, pos = qt.read_portfolio_solution(path)
        inst, budget = hdr["instance"], int(hdr["budget"])
        if a.max_assets and n_assets_of(inst) > a.max_assets:
            continue
        if inst not in cache:
            cache.clear()
            cache[inst] = qt.load_portfolio_instance(inst)
        model = qt.PortfolioModel(cache[inst], budget=budget, lam=hdr["lambda"])
        res = model.score(pos)
        ok = res.feasible and res.objective == int(hdr["objective"])
        print(f"{'OK ' if ok else 'BAD'} {path.name}: file {hdr['objective']} official {res.objective} "
              f"feasible={res.feasible}", flush=True)


def cmd_status(a):
    rows = []
    for path in sorted(OUT.glob("*.sol")):
        hdr, _ = qt.read_portfolio_solution(path)
        inst, budget = hdr["instance"], int(hdr["budget"])
        lam = Fraction(hdr["lambda"])
        key = f"{inst.removeprefix('po_')}_b{budget:03d}_l{lam_key(str(lam))}"
        obj = int(hdr["objective"])
        bkv = bkv_table().get(key)
        gap = None if bkv is None else (obj - bkv[0]) / abs(bkv[0]) * 100
        nvars = qubo_vars(n_assets_of(inst), n_periods_of(inst))
        rows.append((path, inst, budget, lam_tag(str(lam)), nvars, obj, bkv, gap))
    if not a.markdown:
        for path, *_, obj, bkv, gap in rows:
            ref = "" if bkv is None else f"{bkv[0]:.0f} ({bkv[1]})"
            print(f"{path.stem:40s} {obj:>10d} {ref:>22s} {'' if gap is None else f'{gap:+.3f}%':>9s}")
        return
    print("| Instance | B | λ | Solution | Binary variables | Qubits | Checked objective | QOBLIB reference | Gap |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for path, inst, budget, lam, nvars, obj, bkv, gap in rows:
        ref = "*none published*" if bkv is None else f"{bkv[0]:,.0f} ({bkv[1]})"
        g = "—" if gap is None else ("0.00%" if abs(gap) < 5e-5 else f"{gap:+.2f}%".replace("-", "−"))
        if gap is not None and gap < -5e-5:
            g = f"**{g}**"
        print(f"| `{inst}` | {budget} | {lam} | [{path.name}]({path.name}) | {nvars:,} | 0 | {obj:,} | {ref} | {g} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--inst", nargs="+", required=True)
    b.set_defaults(fn=cmd_build)
    t = sub.add_parser("selftest")
    t.add_argument("--inst", nargs="+", default=["po_a010_t10_orig"])
    t.set_defaults(fn=cmd_selftest)
    r = sub.add_parser("run")
    r.add_argument("--inst", required=True)
    r.add_argument("--lam", required=True, choices=LAMS)
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--minutes", type=float, default=5)
    r.add_argument("--method", choices=["ils", "pt", "tabu", "hybrid"], default="hybrid")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--init", default="best", help="best | zero | path to a .sol")
    r.add_argument("--replicas", type=int, default=12)
    r.add_argument("--ils-temp", type=float, default=0.0)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verify")
    v.add_argument("--max-assets", type=int, default=0)
    v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("status")
    s.add_argument("--markdown", action="store_true", help="hand-in table (links relative to the .sol folder)")
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
