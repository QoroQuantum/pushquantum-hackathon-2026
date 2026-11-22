"""Large-neighbourhood search for QOBLIB 06 portfolio instances (reduced state v = L - S).

Neighbourhood: pick k assets, fix all others, re-optimise the k assets jointly over ALL
periods by exact dynamic programming (state = their net-position sub-vector, per-period
budget/capital feasibility given the fixed assets). The sub-solver is pluggable:

    solve_sub(stage_costs[T, K], trans_w[T, k], states[K, k]) -> chain (T,) of state indices

so a QUBO/QAOA solver can replace `dp_chain` later. The current assignment is always one
of the candidate chains, so every LNS step is monotone (never worse).

Score is always re-checked with the official referee (pf_core.score_v / model.score).
"""
from __future__ import annotations

import itertools
import os
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from numba import njit

import pf_core as pc

INF = np.int64(1) << np.int64(60)


@njit(cache=True)
def dp_chain(stage, tw, U):
    """Exact min-cost chain: sum_t stage[t, s_t] + sum_{t>=1} tw[t] . |U[s_t] - U[s_{t-1}]|."""
    T, K = stage.shape
    k = U.shape[1]
    inf = np.int64(1) << np.int64(60)
    best = stage[0].copy()
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        order = np.argsort(best)
        new = np.empty(K, dtype=np.int64)
        for b in range(K):
            if stage[t, b] >= inf:
                new[b] = inf
                continue
            m = inf
            am = 0
            for idx in range(K):
                a = order[idx]
                d = best[a]
                if d >= m:
                    break
                for i in range(k):
                    x = U[a, i] - U[b, i]
                    d += tw[t, i] * (x if x >= 0 else -x)
                if d < m:
                    m = d
                    am = a
            new[b] = m + stage[t, b] if m < inf else inf
            back[t, b] = am
        best = new
    chain = np.empty(T, dtype=np.int64)
    chain[T - 1] = np.argmin(best)
    for t in range(T - 1, 0, -1):
        chain[t - 1] = back[t, chain[t]]
    return chain


def solve_sub_dp(stage, tw, U):
    return dp_chain(stage, tw, U)


class FastModel:
    """Exact integer objective on reduced state v[T, n] (matches the referee)."""

    def __init__(self, model):
        self.model = model
        self.T, self.n = model.n_periods, len(model.symbols)
        self.R = model.risk.astype(np.int64)
        self.lin, self.absw, self.negw, self.const = pc.stage_tables(model)
        self.B, self.C, self.ub = model.budget, model.capital, model.ub
        self.netmin = self.C - (2**model.cash_slack_bits - 1)
        self.tw = np.zeros((self.T, self.n), dtype=np.int64)
        self.tw[1 : self.T - 1] = model.tc[1 : self.T - 1]
        self._states = {}
        # correlation-like affinity for subset selection
        diag = np.maximum(np.einsum("tii->ti", self.R).sum(0), 1)
        Rs = self.R.sum(0).astype(float)
        self.corr = Rs / np.sqrt(np.outer(diag, diag))

    def objective(self, v):
        risk = int(np.einsum("ti,tij,tj->", v, self.R, v))
        st = int((v * self.lin).sum() + (np.abs(v) * self.absw).sum()
                 + (np.maximum(-v, 0) * self.negw).sum() + self.const.sum())
        tr = int((self.tw[1:] * np.abs(np.diff(v, axis=0))).sum())
        return risk + st + tr

    def feasible(self, v):
        net, tot = v.sum(1), np.abs(v).sum(1)
        return bool((np.abs(v) <= self.ub).all() and (tot <= self.B).all()
                    and (net <= self.C).all() and (net >= self.netmin).all())

    def states(self, k):
        if k not in self._states:
            U = np.array(list(itertools.product(range(-self.ub, self.ub + 1), repeat=k)), dtype=np.int64)
            self._states[k] = (U, np.abs(U), np.maximum(-U, 0), np.abs(U).sum(1), U.sum(1))
        return self._states[k]

    def build_sub(self, v, A):
        """Stage costs [T, K] for re-optimising assets A (others fixed), + trans weights."""
        A = np.asarray(A)
        k = len(A)
        U, absU, negU, sumabs, sumU = self.states(k)
        T = self.T
        stage = np.empty((T, len(U)), dtype=np.int64)
        mask = np.ones(self.n, bool)
        mask[A] = False
        for t in range(T):
            vO = np.where(mask, v[t], 0)
            RAA = self.R[t][np.ix_(A, A)]
            cross = 2 * (self.R[t][A] @ vO)
            s = np.einsum("ki,ij,kj->k", U, RAA, U) + U @ (cross + self.lin[t, A])
            s += absU @ self.absw[t, A] + negU @ self.negw[t, A]
            used, net = int(np.abs(vO).sum()), int(vO.sum())
            bad = (sumabs > self.B - used) | (sumU > self.C - net) | (sumU < self.netmin - net)
            s[bad] = INF
            stage[t] = s
        return stage, self.tw[:, A].copy(), U


def lns(fm: FastModel, v0=None, time_limit=60.0, seed=0, solve_sub=solve_sub_dp,
        k_choices=(1, 2, 3, 3, 3, 4), stall_perturb=400, log=None, target=None):
    rng = np.random.default_rng(seed)
    T, n = fm.T, fm.n
    v = np.zeros((T, n), dtype=np.int64) if v0 is None else v0.copy()
    cur = fm.objective(v)
    best_v, best = v.copy(), cur
    t0 = time.time()
    it = stall = 0
    # initial single-asset sweeps
    for _ in range(3):
        for i in rng.permutation(n):
            stage, tw, U = fm.build_sub(v, [i])
            v[:, i] = U[solve_sub(stage, tw, U)][:, 0]
    cur = fm.objective(v)
    best_v, best = v.copy(), cur
    last_print = t0
    while time.time() - t0 < time_limit:
        it += 1
        k = int(rng.choice(k_choices))
        held = np.flatnonzero(np.abs(v).sum(0))
        r = rng.random()
        if r < 0.4 or not len(held):
            i = int(rng.choice(held)) if len(held) and rng.random() < 0.6 else int(rng.integers(n))
            cand = np.argsort(-np.abs(fm.corr[i]))
            cand = cand[cand != i][: max(8, 3 * k)]
            A = [i] + list(rng.choice(cand, size=k - 1, replace=False)) if k > 1 else [i]
        elif r < 0.7 and len(held) >= 1:
            pool = held if len(held) >= k else np.arange(n)
            A = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
            if len(A) < k:
                rest = np.setdiff1d(np.arange(n), A)
                A += list(rng.choice(rest, size=k - len(A), replace=False))
        else:
            A = list(rng.choice(n, size=k, replace=False))
        A = np.array(sorted(set(int(a) for a in A)))
        stage, tw, U = fm.build_sub(v, A)
        chain = solve_sub(stage, tw, U)
        v[:, A] = U[chain]
        new = fm.objective(v)
        if new < cur:
            stall = 0
        else:
            stall += 1
        cur = new
        if cur < best:
            best, best_v = cur, v.copy()
        if stall >= stall_perturb:  # ILS kick: drop a few held assets from a random period block
            stall = 0
            v = best_v.copy()
            held = np.flatnonzero(np.abs(v).sum(0))
            if len(held):
                drop = rng.choice(held, size=max(1, len(held) // 4), replace=False)
                t_a = int(rng.integers(0, T))
                t_b = min(T, t_a + int(rng.integers(2, T + 1)))
                v[t_a:t_b, drop] = 0
                if not fm.feasible(v):  # dropping shorts can push net above C
                    v = best_v.copy()
            cur = fm.objective(v)
        if log and time.time() - last_print > 15:
            last_print = time.time()
            log(f"    it={it} best={best} cur={cur} ({time.time() - t0:.0f}s)")
        if target is not None and best <= target - 10**9:
            break
    return best_v, best, it


# ── simulated annealing on v with single and same-period pair moves (exact deltas) ──


@njit(cache=True)
def _single_delta(v, g, R, lin, absw, negw, tw, t, i, d, T):
    x0 = v[t, i]
    x1 = x0 + d
    dl = 2 * d * g[t, i] + d * d * R[t, i, i] + d * lin[t, i]
    dl += absw[t, i] * (abs(x1) - abs(x0))
    dl += negw[t, i] * (max(-x1, 0) - max(-x0, 0))
    if t >= 1:
        p = v[t - 1, i]
        dl += tw[t, i] * (abs(x1 - p) - abs(x0 - p))
    if t + 1 < T:
        q = v[t + 1, i]
        dl += tw[t + 1, i] * (abs(q - x1) - abs(q - x0))
    return dl


@njit(cache=True)
def _apply(v, g, tot, net, R, lin, absw, negw, tw, t, i, d, T, n):
    delta = _single_delta(v, g, R, lin, absw, negw, tw, t, i, d, T)
    tot[t] += abs(v[t, i] + d) - abs(v[t, i])
    net[t] += d
    v[t, i] += d
    for a in range(n):
        g[t, a] += R[t, a, i] * d
    return delta


@njit(cache=True)
def sa_run(v, R, lin, absw, negw, tw, partners, B, C, netmin, ub, n_steps, temp0, temp1,
           pair_prob, block_prob, seed):
    """SA with compound moves: (block of periods) x (one asset or a partner pair)."""
    np.random.seed(seed)
    T, n = v.shape
    P = partners.shape[1]
    g = np.zeros((T, n), dtype=np.int64)
    tot = np.zeros(T, dtype=np.int64)
    net = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for i in range(n):
            tot[t] += abs(v[t, i])
            net[t] += v[t, i]
            for j in range(n):
                g[t, i] += R[t, i, j] * v[t, j]
    cur = np.int64(0)
    best = np.int64(0)
    best_v = v.copy()
    ratio = (temp1 / temp0) ** (1.0 / max(n_steps - 1, 1))
    temp = temp0
    for step in range(n_steps):
        temp *= ratio
        ta = np.random.randint(T - 1)  # last period stays empty
        tb = ta + 1
        if np.random.random() < block_prob:
            tb = ta + 1 + np.random.randint(T - 1 - ta)
        i = np.random.randint(n)
        d1 = 1 if np.random.random() < 0.5 else -1
        j = -1
        d2 = 0
        if np.random.random() < pair_prob:
            if np.random.random() < 0.8:
                j = partners[i, np.random.randint(P)]
            else:
                j = np.random.randint(n)
            if j == i:
                continue
            d2 = 1 if np.random.random() < 0.5 else -1
        ok = True
        for t in range(ta, tb):
            x = v[t, i] + d1
            if x > ub or x < -ub:
                ok = False
            if j >= 0:
                y = v[t, j] + d2
                if y > ub or y < -ub:
                    ok = False
        if not ok:
            continue
        delta = np.int64(0)
        for t in range(ta, tb):
            delta += _apply(v, g, tot, net, R, lin, absw, negw, tw, t, i, d1, T, n)
            if j >= 0:
                delta += _apply(v, g, tot, net, R, lin, absw, negw, tw, t, j, d2, T, n)
        for t in range(ta, tb):
            if tot[t] > B or net[t] > C or net[t] < netmin:
                ok = False
        if ok and (delta <= 0 or np.random.random() < np.exp(-delta / temp)):
            cur += delta
            if cur < best:
                best = cur
                best_v[:, :] = v
        else:
            for t in range(tb - 1, ta - 1, -1):
                if j >= 0:
                    _apply(v, g, tot, net, R, lin, absw, negw, tw, t, j, -d2, T, n)
                _apply(v, g, tot, net, R, lin, absw, negw, tw, t, i, -d1, T, n)
    return best_v, best


def sa(fm: FastModel, v0=None, n_steps=5_000_000, temp0=3000.0, temp1=5.0, pair_prob=0.5,
       block_prob=0.5, seed=0, n_partners=8):
    v = np.zeros((fm.T, fm.n), dtype=np.int64) if v0 is None else v0.copy()
    v[fm.T - 1] = 0
    c = np.abs(fm.corr.copy())
    np.fill_diagonal(c, -1)
    partners = np.argsort(-c, axis=1)[:, :n_partners].astype(np.int64)
    best_v, _ = sa_run(v, fm.R, fm.lin, fm.absw, fm.negw, fm.tw, partners, fm.B, fm.C, fm.netmin,
                       fm.ub, n_steps, temp0, temp1, pair_prob, block_prob, seed)
    return best_v, fm.objective(best_v)


# ── candidate-pool DP: per-period integer-QP candidates, exact chain over periods ──


def stage_cost(fm: FastModel, t, V):
    """Exact stage cost of candidate vectors V[M, n] in period t (no transitions)."""
    V = np.atleast_2d(V)
    return (np.einsum("ki,ij,kj->k", V, fm.R[t], V) + V @ fm.lin[t] + np.abs(V) @ fm.absw[t]
            + np.maximum(-V, 0) @ fm.negw[t] + fm.const[t])


def period_candidates(fm: FastModel, t, n_seeds=4, extra=(0.0, 1.0, 2.0), n_steps=1_500_000,
                      temp0=None, seed=0):
    """Good integer portfolios for period t alone (SA on a 1-period model)."""
    n = fm.n
    R = np.zeros((2, n, n), dtype=np.int64)
    R[0] = fm.R[t]
    lin = np.zeros((2, n), dtype=np.int64)
    lin[0] = fm.lin[t]
    negw = np.zeros((2, n), dtype=np.int64)
    negw[0] = fm.negw[t]
    tw = np.zeros((2, n), dtype=np.int64)
    c = np.abs(fm.corr.copy())
    np.fill_diagonal(c, -1)
    partners = np.argsort(-c, axis=1)[:, :8].astype(np.int64)
    scale = float(np.median(np.abs(np.diag(fm.R[t])))) + float(np.median(np.abs(fm.lin[t]))) + 1.0
    t0_ = temp0 or 2.0 * scale
    out = []
    for e in extra:  # e * tc: emulate entry/exit transaction cost per unit
        absw = np.zeros((2, n), dtype=np.int64)
        absw[0] = fm.absw[t] + np.round(e * fm.model.tc[t]).astype(np.int64)
        for s in range(n_seeds):
            v = np.zeros((2, n), dtype=np.int64)
            bv, _ = sa_run(v, R, lin, absw, negw, tw, partners, fm.B, fm.C, fm.netmin, fm.ub,
                           n_steps, t0_, 1.0, 0.5, 0.0, seed * 1000 + s + int(10 * e))
            out.append(bv[0].copy())
    return out


def chain_dp(fm: FastModel, pool):
    """Exact best assignment of one pool vector per period (last period forced empty)."""
    pool = np.unique(np.array(pool, dtype=np.int64), axis=0)
    M, T = len(pool), fm.T
    stage = np.array([stage_cost(fm, t, pool) for t in range(T)])  # (T, M)
    zero = np.flatnonzero(~pool.any(1))
    best = stage[0].copy()
    back = np.zeros((T, M), dtype=np.int64)
    for t in range(1, T):
        D = (np.abs(pool[:, None, :] - pool[None, :, :]) * fm.tw[t]).sum(-1)  # (M, M)
        tot = best[:, None] + D
        back[t] = tot.argmin(0)
        best = tot.min(0) + stage[t]
    k = int(zero[0]) if len(zero) else int(best.argmin())
    k = int(best.argmin())
    chain = [k]
    for t in range(T - 1, 0, -1):
        chain.append(int(back[t, chain[-1]]))
    chain.reverse()
    return pool[chain].copy(), int(best[k])


def pool_solve(fm: FastModel, seed=0, n_seeds=4, n_steps=1_500_000, extra=(0.0, 1.0, 2.0), log=None):
    pool = [np.zeros(fm.n, dtype=np.int64)]
    for t in range(fm.T - 1):
        pool += period_candidates(fm, t, n_seeds=n_seeds, extra=extra, n_steps=n_steps, seed=seed)
        if log:
            log(f"    candidates period {t}: pool {len(pool)}")
    v, obj = chain_dp(fm, pool)
    assert obj == fm.objective(v), (obj, fm.objective(v))
    return v, obj, pool


# ── per-period: continuous QP relaxation + rounding + exact pair-move descent ──


def period_descent(fm: FastModel, t, x, extra_abs=None, max_iter=500):
    """Best-improvement descent over single and pair unit moves in one period (exact)."""
    x = x.astype(np.int64).copy()
    R, n = fm.R[t], fm.n
    lin, negw = fm.lin[t], fm.negw[t]
    absw = fm.absw[t] + (0 if extra_abs is None else extra_abs)
    diagR = np.diag(R)
    ds = np.array([1, -1])
    for _ in range(max_iter):
        g = R @ x
        X1 = x[None, :] + ds[:, None]  # (2, n)
        s = (2 * ds[:, None] * g + diagR + ds[:, None] * lin + absw * (np.abs(X1) - np.abs(x))
             + negw * (np.maximum(-X1, 0) - np.maximum(-x, 0)))
        dabs = np.abs(X1) - np.abs(x)
        okb = np.abs(X1) <= fm.ub
        s = np.where(okb, s, 1 << 50).reshape(-1)  # index = d_idx * n + i
        da, dn = dabs.reshape(-1), np.repeat(ds, n)
        tot, net = np.abs(x).sum(), x.sum()
        # singles
        f1 = (tot + da <= fm.B) & (net + dn <= fm.C) & (net + dn >= fm.netmin)
        best1 = np.where(f1, s, 1 << 50)
        # pairs
        Rbig = np.block([[R, -R], [-R, R]])  # d1*d2*R_ij
        P = s[:, None] + s[None, :] + 2 * Rbig
        idx = np.arange(2 * n) % n
        fp = ((tot + da[:, None] + da[None, :] <= fm.B) & (net + dn[:, None] + dn[None, :] <= fm.C)
              & (net + dn[:, None] + dn[None, :] >= fm.netmin) & (idx[:, None] != idx[None, :]))
        P = np.where(fp, P, 1 << 50)
        a1 = int(best1.argmin())
        ap = np.unravel_index(int(P.argmin()), P.shape)
        if min(best1[a1], P[ap]) >= 0:
            break
        if best1[a1] <= P[ap]:
            x[a1 % n] += dn[a1]
        else:
            x[ap[0] % n] += dn[ap[0]]
            x[ap[1] % n] += dn[ap[1]]
    return x


def period_relaxation(fm: FastModel, t, extra_abs=None):
    """Continuous convex relaxation of one period (v = p - q, p, q in [0, ub])."""
    from scipy.optimize import minimize

    n = fm.n
    R = fm.R[t].astype(float)
    lin = fm.lin[t].astype(float)
    absw = (fm.absw[t] + (0 if extra_abs is None else extra_abs)).astype(float)
    negw = fm.negw[t].astype(float)

    def f(z):
        p, q = z[:n], z[n:]
        v = p - q
        Rv = R @ v
        val = v @ Rv + lin @ v + absw @ (p + q) + negw @ q
        gv = 2 * Rv + lin
        return val, np.concatenate([gv + absw, -gv + absw + negw])

    cons = [{"type": "ineq", "fun": lambda z: fm.B - z.sum(), "jac": lambda z: -np.ones(2 * n)},
            {"type": "ineq", "fun": lambda z: fm.C - (z[:n] - z[n:]).sum(),
             "jac": lambda z: np.concatenate([-np.ones(n), np.ones(n)])},
            {"type": "ineq", "fun": lambda z: (z[:n] - z[n:]).sum() - fm.netmin,
             "jac": lambda z: np.concatenate([np.ones(n), -np.ones(n)])}]
    res = minimize(f, np.full(2 * n, 0.01), jac=True, method="SLSQP", bounds=[(0, fm.ub)] * (2 * n),
                   constraints=cons, options={"maxiter": 300})
    return res.x[:n] - res.x[n:]


def _repair_period(fm, x):
    x = np.clip(x, -fm.ub, fm.ub).astype(np.int64)
    while np.abs(x).sum() > fm.B or x.sum() > fm.C or x.sum() < fm.netmin:
        if x.sum() > fm.C:
            i = int(np.flatnonzero(x > 0)[0])
        elif x.sum() < fm.netmin:
            i = int(np.flatnonzero(x < 0)[0])
        else:
            i = int(np.flatnonzero(x)[0])
        x[i] -= np.sign(x[i])
    return x


def period_candidates_qp(fm: FastModel, t, extra=(0.0, 1.0, 2.0), scales=(1, 2, 3, 5, 8, 12, 20, 40)):
    out = []
    for e in extra:
        ea = np.round(e * fm.model.tc[t]).astype(np.int64)
        xc = period_relaxation(fm, t, ea)
        seen = set()
        for sc in scales:
            xr = _repair_period(fm, np.round(sc * xc))
            key = xr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(period_descent(fm, t, xr, ea))
        out.append(period_descent(fm, t, np.zeros(fm.n, dtype=np.int64), ea))
    return out


def period_ils(fm: FastModel, t, starts, extra_abs=None, n_kicks=40, seed=0):
    """Iterated local search in one period: kick a few units, descend, keep the best few."""
    rng = np.random.default_rng(seed)
    cost = lambda x: int(stage_cost(fm, t, x)[0] + (0 if extra_abs is None else extra_abs @ np.abs(x)))
    pool = sorted(((cost(x), x.tobytes(), x) for x in starts), key=lambda r: r[0])
    best_c, _, best_x = pool[0]
    out = []
    for _ in range(n_kicks):
        x = best_x.copy()
        k = int(rng.integers(2, 7))
        idx = rng.choice(fm.n, size=k, replace=False)
        x[idx] = rng.integers(-1, 2, size=k)
        x = period_descent(fm, t, _repair_period(fm, x), extra_abs)
        c = cost(x)
        out.append(x)
        if c < best_c:
            best_c, best_x = c, x
    return out
