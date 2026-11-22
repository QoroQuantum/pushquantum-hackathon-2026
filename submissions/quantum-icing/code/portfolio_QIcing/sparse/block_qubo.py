"""Block-sparse compact QUBO for large portfolio instances (a200, a400).

Same variables and energy as pf_quantum.CompactQUBO (objective + penalty_weight * penalty), but never
materialised as one matrix. Every coupling is either
  * inside one day: risk between unit bits, budget and capital penalties (dense block per day), or
  * between the same unit bit (asset, copy, direction) on adjacent days: rebalancing cost.
So we store one dense float32 block per day plus the adjacent-day weights:
  memory ~ T * (6n + slack)^2 * 4 bytes  ->  a200: ~60 MB, a400: ~230 MB (dense: 2.3 GB / 9 GB).

Energy convention: E(x) = const + l.x + 1/2 x^T S x, S symmetric with zero diagonal.
`sub(bits, free)` returns the exact conditional sub-QUBO on the free bits (others fixed), up to a
constant, which is all a window solver needs (candidates are scored by the official referee).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import dimod
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import pf_core as pc  # noqa: E402


class BlockQUBO:
    def __init__(self, model, penalty_weight: float, ub: int | None = None, dtype=np.float32):
        self.model = model
        self.ub = ub or model.ub
        n, T, ub = len(model.symbols), model.n_periods, self.ub
        self.n, self.T = n, T
        self.count_bits = max(1, math.ceil(math.log2(model.budget + 1)))
        self.cash_bits = model.cash_slack_bits if model.budget > 5 else 0
        self.n_x = n * ub * 2 * T
        self.n_vars = self.n_x + (self.count_bits + self.cash_bits) * T
        self.W = float(penalty_weight)
        # local layout of one day: x bits in (asset, copy, direction) order, then budget slack, then cash
        self.loc = [(i, m, tau) for i in range(n) for m in range(ub) for tau in (1, -1)]
        self.nx_day = len(self.loc)
        self.n_day = self.nx_day + self.count_bits + self.cash_bits
        # global index of each local slot, per day
        self.glob = np.zeros((T, self.n_day), dtype=np.int64)
        for t in range(T):
            self.glob[t, : self.nx_day] = [self.x(i, m, tau, t) for i, m, tau in self.loc]
            self.glob[t, self.nx_day: self.nx_day + self.count_bits] = [self.s(c, t) for c in range(self.count_bits)]
            self.glob[t, self.nx_day + self.count_bits:] = [self.y(k, t) for k in range(self.cash_bits)]
        self.day_of = np.empty(self.n_vars, dtype=np.int64)
        self.slot_of = np.empty(self.n_vars, dtype=np.int64)
        for t in range(T):
            self.day_of[self.glob[t]] = t
            self.slot_of[self.glob[t]] = np.arange(self.n_day)
        self.lin = np.zeros(self.n_vars)
        self.blocks = np.zeros((T, self.n_day, self.n_day), dtype=dtype)
        self.link_w = np.zeros(self.n_vars)  # weight between x bit at day t and the same bit at day t-1
        self.const = 0.0
        self._build()

    # global indexing (identical to pf_quantum.CompactQUBO)
    def x(self, i, m, tau, t):
        return ((i * self.ub + m) * 2 + (0 if tau == 1 else 1)) * self.T + t

    def s(self, c, t):
        return self.n_x + c * self.T + t

    def y(self, k, t):
        return self.n_x + self.count_bits * self.T + k * self.T + t

    def cell_vars(self, i, t):
        return [self.x(i, m, tau, t) for m in range(self.ub) for tau in (1, -1)]

    def slack_vars(self, t):
        return [self.s(c, t) for c in range(self.count_bits)] + [self.y(k, t) for k in range(self.cash_bits)]

    def _build(self):
        model, T, n, W = self.model, self.T, self.n, self.W
        asset = np.array([i for i, _, _ in self.loc])
        sign = np.array([tau for _, _, tau in self.loc], dtype=float)
        nx = self.nx_day
        for t in range(T):
            g = self.glob[t]
            B = np.zeros((self.n_day, self.n_day))
            l = np.zeros(self.n_day)
            M = np.outer(sign, sign) * model.risk[t][np.ix_(asset, asset)]
            l[:nx] += np.diag(M)
            B[:nx, :nx] += M + M.T
            np.fill_diagonal(B, 0.0)
            l[:nx] += 10 * sign  # folded cash interest
            l[:nx] += np.where(sign < 0, model.sc[t, asset], 0)
            if t < T - 1:
                l[:nx] -= sign * model.ret[t, asset]
            if t == 0 or t == T - 1:
                l[:nx] += model.tc[t, asset]
            self.const += -10 * model.capital
            # penalties W (a.x - target)^2
            a = np.zeros(self.n_day)
            a[:nx] = 1
            a[nx: nx + self.count_bits] = [2**c for c in range(self.count_bits)]
            targets = [(a, model.budget)]
            if self.cash_bits:
                a2 = np.zeros(self.n_day)
                a2[:nx] = sign
                a2[nx + self.count_bits:] = [2**k for k in range(self.cash_bits)]
                targets.append((a2, model.capital))
            for vec, target in targets:
                l += W * (vec**2 - 2 * target * vec)
                O = 2 * W * np.outer(vec, vec)
                np.fill_diagonal(O, 0.0)
                B += O
                self.const += W * target**2
            self.blocks[t] = B
            self.lin[g] += l
        # rebalancing x_{t-1} XOR x_t for 1 <= t <= T-2
        for t in range(1, T - 1):
            for (i, m, tau) in self.loc:
                cur, prev = self.x(i, m, tau, t), self.x(i, m, tau, t - 1)
                w = float(model.tc[t, i])
                self.lin[cur] += w
                self.lin[prev] += w
                self.link_w[cur] = -2 * w

    # ── energies ──────────────────────────────────────────────────────────
    def energy(self, bits) -> float:
        x = np.asarray(bits, dtype=float)
        e = self.const + self.lin @ x
        for t in range(self.T):
            xt = x[self.glob[t]]
            e += 0.5 * xt @ self.blocks[t].astype(float) @ xt
        for t in range(1, self.T - 1):
            for (i, m, tau) in self.loc:
                cur = self.x(i, m, tau, t)
                e += self.link_w[cur] * x[cur] * x[self.x(i, m, tau, t - 1)]
        return float(e)

    def sub(self, bits, free) -> dimod.BinaryQuadraticModel:
        """Exact conditional QUBO on `free` (other bits fixed to `bits`), up to a constant."""
        free = np.asarray(free, dtype=np.int64)
        k = len(free)
        x = np.asarray(bits, dtype=float).copy()
        x[free] = 0.0  # free bits contribute through J, not h
        h = self.lin[free].copy()
        J = np.zeros((k, k))
        days = self.day_of[free]
        slots = self.slot_of[free]
        for t in np.unique(days):
            sel = np.flatnonzero(days == t)
            rows = self.blocks[t][slots[sel]].astype(float)  # (|sel|, n_day)
            h[sel] += rows @ x[self.glob[t]]
            J[np.ix_(sel, sel)] = rows[:, slots[sel]]
        # adjacent-day links
        pos = {int(v): p for p, v in enumerate(free)}
        for p, v in enumerate(free):
            v = int(v)
            if v >= self.n_x:
                continue
            t = self.day_of[v]
            if 1 <= t <= self.T - 2:  # link to day t-1
                prev = v - 1
                w = self.link_w[v]
                if prev in pos:
                    J[p, pos[prev]] += w
                    J[pos[prev], p] += w
                else:
                    h[p] += w * bits[prev]
            if 0 <= t <= self.T - 3:  # link to day t+1
                nxt = v + 1
                w = self.link_w[nxt]
                if nxt not in pos:
                    h[p] += w * bits[nxt]
        Ju = np.triu(J, 1)
        quad = {(int(a), int(b)): float(Ju[a, b]) for a, b in zip(*np.nonzero(Ju))}
        return dimod.BinaryQuadraticModel({i: float(h[i]) for i in range(k)}, quad, 0.0, dimod.BINARY)

    # ── positions <-> bits ───────────────────────────────────────────────
    def encode_v(self, v) -> np.ndarray:
        bits = np.zeros(self.n_vars, dtype=np.int8)
        for t in range(self.T):
            for i in np.flatnonzero(v[t]):
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

    def decode_v(self, bits) -> np.ndarray:
        xb = np.asarray(bits[: self.n_x]).reshape(self.n, self.ub, 2, self.T)  # (asset, copy, dir, day)
        L = xb[:, :, 0, :].sum(1).T
        S = xb[:, :, 1, :].sum(1).T
        return (L - S).astype(np.int64)

    def memory_mb(self) -> float:
        return (self.blocks.nbytes + self.lin.nbytes + self.link_w.nbytes + self.glob.nbytes) / 2**20
