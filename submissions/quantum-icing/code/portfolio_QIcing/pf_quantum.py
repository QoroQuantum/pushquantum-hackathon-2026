"""Quantum side of the QIcing portfolio attack: compact QUBO + quantum large-neighbourhood search.

CompactQUBO
    Same unary copy encoding as QOBLIB's bqp_u3_c10 (x[i, m, dir, t]), but
    * no cash slack register when the capital rule cannot bind (B <= 5): the interest
      -10 (C - net) is folded into the objective as a linear term (+10 per long unit,
      -10 per short unit, constant -10 C per period);
    * budget slack register of ceil(log2(B + 1)) bits instead of 7.
    For 10 assets x 10 days: 600 position bits + 30 slack bits = 630 (official: 710).

Quantum LNS (qlns)
    Keep an incumbent; repeatedly pick a window (one period, a few assets: their 6 bits
    each + that period's budget slack bits), fix every other bit to the incumbent, and
    solve the resulting exact conditional sub-QUBO with QAOA (warm-started from the
    Characterization Service angles) or PCE. Each sampled sub-bitstring is spliced
    into the incumbent and scored with the official referee; accept if it improves.
"""
from __future__ import annotations

import math
import time

import dimod
import numpy as np

import pf_core as pc

qt = pc.qt


class CompactQUBO:
    def __init__(self, model: qt.PortfolioModel, ub: int | None = None):
        self.model = model
        self.ub = ub or model.ub
        n, T, ub = len(model.symbols), model.n_periods, self.ub
        self.n, self.T = n, T
        self.count_bits = max(1, math.ceil(math.log2(model.budget + 1)))
        self.n_x = n * ub * 2 * T
        # cash register only when the capital rule can bind (B > 5): net + cash = C, cash in 0..15
        self.cash_bits = model.cash_slack_bits if model.budget > 5 else 0
        self.n_vars = self.n_x + (self.count_bits + self.cash_bits) * T
        self.objective = self._objective()
        self.penalty = self._penalty()

    def x(self, i, m, tau, t):
        return ((i * self.ub + m) * 2 + (0 if tau == 1 else 1)) * self.T + t

    def s(self, c, t):
        return self.n_x + c * self.T + t

    def cell_vars(self, i, t):
        return [self.x(i, m, tau, t) for m in range(self.ub) for tau in (1, -1)]

    def y(self, k, t):
        return self.n_x + self.count_bits * self.T + k * self.T + t

    def slack_vars(self, t):
        return [self.s(c, t) for c in range(self.count_bits)] + [self.y(k, t) for k in range(self.cash_bits)]

    def _objective(self):
        model, T, ub, n = self.model, self.T, self.ub, self.n
        lin = np.zeros(self.n_vars)
        quad = {}
        loc = [(i, m, tau) for i in range(n) for m in range(ub) for tau in (1, -1)]
        asset = np.array([i for i, _, _ in loc])
        sign = np.array([tau for _, _, tau in loc])
        offset = 0.0
        for t in range(T):
            glob = np.array([self.x(i, m, tau, t) for i, m, tau in loc])
            M = np.outer(sign, sign) * model.risk[t][np.ix_(asset, asset)]
            lin[glob] += np.diag(M)
            up = np.triu(M + M.T, k=1)
            for a, b in zip(*np.nonzero(up)):
                quad[(int(glob[a]), int(glob[b]))] = quad.get((int(glob[a]), int(glob[b])), 0) + up[a, b]
            offset += -10 * model.capital
            for (i, m, tau), g in zip(loc, glob):
                lin[g] += 10 * tau  # folded cash interest
                if tau == -1:
                    lin[g] += model.sc[t, i]
                if t < T - 1:
                    lin[g] -= tau * model.ret[t, i]
                if t == 0 or t == T - 1:
                    lin[g] += model.tc[t, i]
                else:
                    prev = self.x(i, m, tau, t - 1)
                    lin[g] += model.tc[t, i]
                    lin[prev] += model.tc[t, i]
                    quad[(prev, int(g))] = quad.get((prev, int(g)), 0) - 2 * model.tc[t, i]
        bqm = dimod.BinaryQuadraticModel(dict(enumerate(lin)), quad, offset, dimod.BINARY)
        return bqm

    def _penalty(self):
        bqm = dimod.BinaryQuadraticModel({v: 0.0 for v in range(self.n_vars)}, {}, 0.0, dimod.BINARY)
        for t in range(self.T):
            terms = [(v, 1) for i in range(self.n) for v in self.cell_vars(i, t)]
            terms += [(self.s(c, t), 2**c) for c in range(self.count_bits)]
            bqm.add_linear_equality_constraint(terms, 1.0, -self.model.budget)
            if self.cash_bits:
                cap = [(self.x(i, m, tau, t), tau) for i in range(self.n) for m in range(self.ub) for tau in (1, -1)]
                cap += [(self.y(k, t), 2**k) for k in range(self.cash_bits)]
                bqm.add_linear_equality_constraint(cap, 1.0, -self.model.capital)
        return bqm

    def full(self, weight: float) -> dimod.BinaryQuadraticModel:
        bqm = self.objective.copy()
        pen = self.penalty.copy()
        pen.scale(weight)
        bqm.update(pen)
        return bqm

    # positions <-> bits
    def encode_v(self, v: np.ndarray) -> np.ndarray:
        bits = np.zeros(self.n_vars, dtype=np.int8)
        for t in range(self.T):
            for i in range(self.n):
                k = int(v[t, i])
                tau = 1 if k > 0 else -1
                for m in range(min(abs(k), self.ub)):
                    bits[self.x(i, m, tau, t)] = 1
            slack = self.model.budget - int(np.abs(v[t]).sum())
            for c in range(self.count_bits):
                bits[self.s(c, t)] = (slack >> c) & 1 if slack >= 0 else 0
            cash = self.model.capital - int(v[t].sum())
            for k in range(self.cash_bits):
                bits[self.y(k, t)] = (cash >> k) & 1 if cash >= 0 else 0
        return bits

    def decode_LS(self, bits) -> tuple[np.ndarray, np.ndarray]:
        bits = np.asarray(bits)
        L = np.zeros((self.T, self.n), dtype=np.int64)
        S = np.zeros_like(L)
        for i in range(self.n):
            for t in range(self.T):
                for m in range(self.ub):
                    L[t, i] += bits[self.x(i, m, 1, t)]
                    S[t, i] += bits[self.x(i, m, -1, t)]
        return L, S

    def decode_v(self, bits) -> np.ndarray:
        L, S = self.decode_LS(bits)
        return L - S  # dominance: L and S together only waste fees


def official(model, v):
    return pc.score_v(model, v)


def repair_v(model, v):
    """Referee's greedy repair on the reduced state (budget only for B <= 5)."""
    pos = model.repair(pc.v_to_positions(model, v))
    return pc.positions_to_v(model, pos)


# ── sub-QUBO windows ─────────────────────────────────────────────────────


def sub_bqm(full: dimod.BinaryQuadraticModel, incumbent_bits: np.ndarray, free: list[int]):
    """Exact conditional QUBO on `free` (others fixed to the incumbent), relabelled 0..k-1."""
    fixed = {v: int(incumbent_bits[v]) for v in full.variables if v not in set(free)}
    sub = full.copy()
    sub.fix_variables(fixed)
    for v in free:  # keep isolated free vars
        sub.add_variable(v, 0.0)
    mapping = {v: k for k, v in enumerate(free)}
    sub.relabel_variables(mapping, inplace=True)
    return sub


def correlated_partners(cq: CompactQUBO, t: int, seed_asset: int, k: int, rng) -> list[int]:
    """k assets most correlated (|corr|) with seed_asset in period t's risk matrix (hedge candidates)."""
    R = cq.model.risk[min(t, cq.T - 1)].astype(float)
    d = np.sqrt(np.maximum(np.diag(R), 1e-12))
    corr = np.abs(R[seed_asset]) / (d[seed_asset] * d)
    corr[seed_asset] = -1
    order = np.argsort(-corr + 1e-3 * rng.random(len(corr)))
    return [int(a) for a in order[:k]]


def choose_window(cq: CompactQUBO, rng, n_assets: int, v: np.ndarray, period=None, p_time: float = 0.0,
                  time_assets: int = 1, p_corr: float = 0.0):
    """Window = (one period, n_assets assets) or, with prob. p_time, (one asset, two consecutive periods)."""
    T, n = cq.T, cq.n
    if period is None and rng.random() < p_time:
        t = int(rng.integers(0, T - 2))
        held = list(np.flatnonzero(v[t] | v[t + 1]))
        assets = [int(rng.choice(held))] if held and rng.random() < 0.5 else [int(rng.integers(n))]
        if rng.random() < p_corr:
            assets += [a for a in correlated_partners(cq, t, assets[0], time_assets, rng) if a not in assets][: time_assets - 1]
        while len(assets) < time_assets:
            a = int(rng.integers(n))
            if a not in assets:
                assets.append(a)
        assets.sort()
        free = [b for tt in (t, t + 1) for a in assets for b in cq.cell_vars(a, tt)]
        free += cq.slack_vars(t) + cq.slack_vars(t + 1)
        return (t, t + 1), assets, free
    t = int(rng.integers(0, T - 1)) if period is None else period  # last period stays empty
    held = list(np.flatnonzero(v[t]))
    pool = list(range(n))
    assets = []
    if held and rng.random() < 0.7:  # usually include one currently-held asset
        assets.append(int(rng.choice(held)))
    if rng.random() < p_corr:
        if not assets:
            assets.append(int(rng.integers(n)))
        assets += [a for a in correlated_partners(cq, t, assets[0], n_assets, rng) if a not in assets][: n_assets - 1]
    while len(assets) < n_assets:
        a = int(rng.choice(pool))
        if a not in assets:
            assets.append(a)
    free = [b for a in sorted(assets) for b in cq.cell_vars(a, t)] + cq.slack_vars(t)
    return t, sorted(assets), free


# ── dense path for big instances (fast conditional sub-QUBOs) ────────────


class DenseQUBO:
    """objective + weight * penalty as a dense symmetric matrix: E(x) = x'Ux + c, U upper-triangular incl. diag."""

    def __init__(self, bqm: dimod.BinaryQuadraticModel, n_vars: int):
        U = np.zeros((n_vars, n_vars))
        for v, b in bqm.linear.items():
            U[v, v] += b
        for (a, b), w in bqm.quadratic.items():
            i, j = min(a, b), max(a, b)
            U[i, j] += w
        self.U = U
        self.S = U + U.T - np.diag(np.diag(U))  # symmetric off-diagonal couplings, diag = linear
        np.fill_diagonal(self.S, 0.0)
        self.lin = np.diag(U).copy()
        self.offset = bqm.offset

    def sub(self, bits: np.ndarray, free: list[int]) -> dimod.BinaryQuadraticModel:
        free = np.asarray(free)
        mask = np.ones(len(bits), dtype=bool)
        mask[free] = False
        fixed_idx = np.flatnonzero(mask)
        xf = bits[fixed_idx].astype(float)
        h = self.lin[free] + self.S[np.ix_(free, fixed_idx)] @ xf
        J = np.triu(self.S[np.ix_(free, free)], k=1)
        const = self.offset + self.lin[fixed_idx] @ xf + 0.5 * xf @ self.S[np.ix_(fixed_idx, fixed_idx)] @ xf
        k = len(free)
        quad = {(int(a), int(b)): float(J[a, b]) for a, b in zip(*np.nonzero(J))}
        return dimod.BinaryQuadraticModel({i: float(h[i]) for i in range(k)}, quad, float(const), dimod.BINARY)


def hedge_window(cq: CompactQUBO, rng, v: np.ndarray, n_assets: int = 2, max_len: int = 9, copies: int = 1):
    """Seed asset + its most correlated partners, over a run of periods, first `copies` unit copies
    of both directions. Use with an OBJECTIVE-ONLY sub-QUBO: feasibility is checked on splice."""
    T, n = cq.T, cq.n
    held = np.flatnonzero(np.abs(v).sum(0))
    seed = int(rng.choice(held)) if len(held) and rng.random() < 0.5 else int(rng.integers(n))
    length = int(rng.integers(2, max_len + 1))
    t0 = int(rng.integers(0, max(1, T - 1 - length + 1)))
    periods = list(range(t0, min(T - 1, t0 + length)))
    assets = sorted({seed, *correlated_partners(cq, periods[0], seed, n_assets - 1, rng)})
    free = [cq.x(a, m, tau, t) for a in assets for m in range(copies) for tau in (1, -1) for t in periods]
    return (periods[0], periods[-1]), assets, sorted(free)
