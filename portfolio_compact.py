"""A smaller QUBO for QOBLIB 06 (portfolio) that keeps the optimum.

:class:`CompactPortfolioQUBO` is a drop-in alternative to
:class:`qoblib_tools.PortfolioQUBO` (same ``objective`` / ``penalty`` / ``encode`` /
``decode`` interface). Every reduction below is *exact*: for every feasible portfolio
the compact QUBO's energy equals the official score, so the optimum is unchanged.

1. **Redundant capital rule.** The rule is ``C - 15 <= net <= C`` per period. With at
   most ``B`` open units, ``|net| <= B``; if ``[-B, B]`` already lies inside the
   allowed range (``B <= 5`` for the official ``C = 10``), the rule can never be
   broken. Its penalty and the 4 cash-slack bits per period are dropped, and the
   cash interest (official: on the slack digits, with coefficients ``10 * 2**k``)
   becomes the equivalent linear term ``-10 * (C - net)``.
2. **Tight budget slack.** The budget slack only needs to reach ``B``, so it gets
   ``B.bit_length()`` bits instead of the official 7 (3 bits for ``B = 4``).
3. **Unit copies capped at B.** More than ``B`` copies of a position can never be used.
4. **Last period solved apart.** The final period has no return term and no
   rebalancing link to the period before it (only liquidation), so it does not
   interact with the rest of the portfolio. It is solved exactly once, classically,
   and left out of the QUBO; its cost goes into the QUBO's constant offset.

One *optional* restriction is not exact: ``budget_equality=True`` demands exactly
``B`` open units per period, which drops the budget slack entirely. It can lose the
optimum: the best-known ``po_a010_t10_orig`` solution at ``B = 4, λ = 1e-3`` holds
0-4 units per period. (At ``λ = 1e-4`` it uses the full budget every period.)

Optional *domain-wall* ordering (``domain_wall=w > 0``) is exact too. Per asset and
period it adds ``w`` times

    Σ_m  x[m+1]·(1 − x[m])   on the long copies and on the short copies (out of order)
    + x_long[0]·x_short[0]                                       (long and short at once)

to the objective, which is zero on canonical encodings. The unit copies then form a
domain-wall (thermometer) encoding of the net position ``v ∈ [-ub, ub]``: one bit
pattern per value. No state it penalises beats a valid one: sorting copies only
lowers rebalancing, and replacing ``(L, S)`` by ``(max(v, 0), max(-v, 0))`` keeps
risk, returns and interest, lowers fees, budget use and liquidation, and turns
rebalancing into ``|Δv| <= |ΔL| + |ΔS|``. So any ``w >= 0`` keeps the optimum; ``w``
only reshapes the landscape.

Unit counts stay unary (one bit per unit copy). A binary count would be smaller, but
the rebalancing cost ``tc * |L_t - L_{t-1}|`` is not quadratic in binary digits; in
unary it is a sum of XORs, which is.
"""
from __future__ import annotations

import itertools
import math
import warnings

import dimod
import numpy as np

from qoblib_tools import PortfolioModel, Positions, _as_bit_array

__all__ = ["CompactPortfolioQUBO"]


class CompactPortfolioQUBO:
    """The portfolio model as a reduced QUBO: ``objective`` + ``penalty``.

    Args:
        model: The official model (instance, budget, λ).
        drop_redundant_capital: Drop the capital rule when it cannot bind (exact).
        tight_budget_slack: Use ``B.bit_length()`` budget-slack bits (exact).
        decouple_last_period: Solve the final period apart (exact). Falls back to
            keeping it in the QUBO, with a warning, if it is too large to solve.
        budget_equality: Require exactly ``B`` open units per period and drop the
            budget slack. **Not exact**: a heuristic restriction.
        domain_wall: Weight (score units) of the domain-wall ordering terms added to
            the objective; ``0`` leaves them out (exact for any weight).
        last_period_max_configs: Size limit for enumerating the final period.

    Attributes:
        labels: Variable ``k`` is ``labels[k]``: ``("x", symbol, copy, ±1, period)``,
            ``("y", bit, period)`` (cash slack) or ``("s", bit, period)`` (budget slack).
        periods: The periods represented in the QUBO.
        last_period: ``(long, short)`` arrays fixed for the final period, or ``None``.
        reductions: Which reductions were applied, and the variable count saved.
        wall: The domain-wall terms at unit weight (zero on canonical encodings),
            built whatever ``domain_wall`` is, e.g. to count invalid patterns.
    """

    def __init__(
        self,
        model: PortfolioModel,
        *,
        drop_redundant_capital: bool = True,
        tight_budget_slack: bool = True,
        decouple_last_period: bool = True,
        budget_equality: bool = False,
        domain_wall: float = 0.0,
        last_period_max_configs: int = 200_000,
    ):
        self.model = model
        n, T, B, C = len(model.symbols), model.n_periods, model.budget, model.capital
        self.ub = min(model.ub, B)
        self.max_cash = 2**model.cash_slack_bits - 1
        self.budget_equality = budget_equality
        if budget_equality and self.ub * 2 * n < B:
            raise ValueError("budget_equality: B units per period cannot be reached")

        # 1. Capital rule: redundant iff every reachable net position satisfies it.
        net_bound = min(B, self.ub * n)
        capital_redundant = C - self.max_cash <= -net_bound and net_bound <= C
        i0 = int(model.interest[0])
        interest_linear = all(
            int(model.interest[k]) == i0 * 2**k for k in range(model.cash_slack_bits)
        )
        self.keep_capital = not (drop_redundant_capital and capital_redundant and interest_linear)
        self.cash_bits = model.cash_slack_bits if self.keep_capital else 0

        # 2. Budget slack width.
        if budget_equality:
            self.count_bits = 0
        elif tight_budget_slack:
            self.count_bits = max(1, B.bit_length())
        else:
            self.count_bits = model.count_slack_bits

        # 4. Last period.
        self.last_period = None
        self._last_cost = 0
        if decouple_last_period:
            solved = self._solve_last_period(last_period_max_configs)
            if solved is None:
                warnings.warn(
                    "final period too large to solve exactly; keeping it in the QUBO",
                    stacklevel=2,
                )
            else:
                self.last_period, self._last_cost = solved
        self.periods = list(range(T - 1 if self.last_period is not None else T))
        P = len(self.periods)

        # Variables: x (asset → copy → direction → period), then y, then s.
        self.labels: list[tuple] = []
        for i, sym in enumerate(model.symbols):
            for m in range(self.ub):
                for tau in (1, -1):
                    for t in self.periods:
                        self.labels.append(("x", sym, m, tau, t))
        self.n_x = len(self.labels)
        self.labels += [("y", k, t) for k in range(self.cash_bits) for t in self.periods]
        self.labels += [("s", c, t) for c in range(self.count_bits) for t in self.periods]
        self.n_vars = len(self.labels)

        ub = self.ub
        self._x = lambda i, m, tau, t: ((i * ub + m) * 2 + (0 if tau == 1 else 1)) * P + t
        self._y = lambda k, t: self.n_x + k * P + t
        self._s = lambda c, t: self.n_x + self.cash_bits * P + c * P + t

        if domain_wall < 0:
            raise ValueError("domain_wall must be >= 0")
        self.domain_wall = float(domain_wall)
        self.wall = self._build_wall()
        self.objective = self._build_objective()
        if self.domain_wall:
            scaled = self.wall.copy()
            scaled.scale(self.domain_wall)
            self.objective.update(scaled)
        self.penalty = self._build_penalty()

        official = (6 * n + model.cash_slack_bits + model.count_slack_bits) * T
        self.reductions = {
            "capital rule dropped": not self.keep_capital,
            "budget slack bits": f"{model.count_slack_bits} -> {self.count_bits}",
            "unit copies": f"{model.ub} -> {self.ub}",
            "last period solved apart": self.last_period is not None,
            "budget equality (heuristic)": budget_equality,
            "domain-wall weight": self.domain_wall,
            "variables": f"{official} -> {self.n_vars}",
        }

    # ── final period ─────────────────────────────────────────────────────

    def _period_cost(self, t: int, L: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Official cost of period ``t`` alone, for a batch of configurations ``(k, n)``.

        Valid for the final period, whose cost has no return or rebalancing term.
        Infeasible configurations cost ``inf``.
        """
        m = self.model
        v, total = L - S, L + S
        cash = m.capital - v.sum(axis=1)
        interest = sum(
            int(m.interest[k]) * ((np.maximum(cash, 0) >> k) & 1)
            for k in range(m.cash_slack_bits)
        )
        cost = (
            np.einsum("ki,ij,kj->k", v, m.risk[t], v)
            - interest
            + S @ m.sc[t]
            + total @ m.tc[t]
        ).astype(float)
        feasible = (cash >= 0) & (cash <= self.max_cash) & (total.sum(axis=1) <= m.budget)
        return np.where(feasible, cost, np.inf)

    def _solve_last_period(self, max_configs: int):
        """Exact best ``(L, S)`` of the final period, or ``None`` if too large."""
        m = self.model
        n, t, B, ub = len(m.symbols), m.n_periods - 1, m.budget, self.ub
        slots = 2 * n  # (asset, direction)
        n_configs = sum(math.comb(slots + k - 1, k) for k in range(B + 1))
        if n_configs <= max_configs:
            rows = []
            for k in range(B + 1):
                for combo in itertools.combinations_with_replacement(range(slots), k):
                    counts = np.bincount(combo, minlength=slots)
                    if counts.max(initial=0) <= ub:
                        rows.append(counts)
            cfg = np.array(rows).reshape(-1, n, 2)
            cost = self._period_cost(t, cfg[:, :, 0], cfg[:, :, 1])
            best = int(np.argmin(cost))
            return (cfg[best, :, 0].copy(), cfg[best, :, 1].copy()), int(cost[best])

        # Too many to enumerate: prove that holding nothing is optimal. With v = L - S,
        # |v_i| <= ub gives v_i**2 <= ub * (L_i + S_i), so risk = v'Rv >= min(0, eig_min)
        # * ub * (units held). If every unit's own cost c (liquidation, short fee,
        # interest) beats that, any holding costs more than none. Holding nothing is
        # feasible, so this also holds with the capital rule.
        R = m.risk[t].astype(float)
        i0 = int(m.interest[0])
        interest_linear = all(
            int(m.interest[k]) == i0 * 2**k for k in range(m.cash_slack_bits)
        )
        long_cost = m.tc[t] + i0  # liquidation, plus one unit less cash earning interest
        short_cost = m.tc[t] + m.sc[t] - i0  # liquidation + short fee, minus extra interest
        c_min = min(long_cost.min(), short_cost.min())
        eig_min = np.linalg.eigvalsh((R + R.T) / 2).min()
        tol = 1e-9 * max(1.0, np.abs(R).max()) * n  # floating-point slack on eig_min
        if interest_linear and c_min + min(0.0, eig_min - tol) * ub > 0:
            zero = np.zeros(n, dtype=np.int64)
            cost = self._period_cost(t, zero[None], zero[None])
            return (zero, zero.copy()), int(cost[0])
        return None

    # ── QUBO ─────────────────────────────────────────────────────────────

    def _build_objective(self) -> dimod.BinaryQuadraticModel:
        model, T, ub = self.model, self.model.n_periods, self.ub
        n, x = len(model.symbols), self._x
        linear = np.zeros(self.n_vars)
        rows, cols, vals = [], [], []
        offset = float(self._last_cost)
        i0 = int(model.interest[0])

        loc = [(i, m, tau) for i in range(n) for m in range(ub) for tau in (1, -1)]
        asset = np.array([i for i, _, _ in loc])
        sign = np.array([tau for _, _, tau in loc])
        for t in self.periods:
            glob = np.array([x(i, m, tau, t) for i, m, tau in loc])
            M = np.outer(sign, sign) * model.risk[t][np.ix_(asset, asset)]
            linear[glob] += np.diag(M)
            upper = np.triu(M + M.T, k=1)
            a, b = np.nonzero(upper)
            rows.append(glob[a]), cols.append(glob[b]), vals.append(upper[a, b])

            for (i, m, tau), g in zip(loc, glob):
                if tau == -1:
                    linear[g] += model.sc[t, i]
                if t < T - 1:
                    linear[g] -= tau * model.ret[t, i]
                if t == 0 or t == T - 1:
                    linear[g] += model.tc[t, i]
                elif t < T - 1:  # rebalancing between t-1 and t: x_{t-1} XOR x_t
                    prev = x(i, m, tau, t - 1)
                    linear[g] += model.tc[t, i]
                    linear[prev] += model.tc[t, i]
                    rows.append([prev]), cols.append([g])
                    vals.append([-2 * model.tc[t, i]])
                if not self.keep_capital:  # cash interest -i0 * (C - net), net = Σ τ x
                    linear[g] += tau * i0
            if self.keep_capital:
                for k in range(self.cash_bits):
                    linear[self._y(k, t)] -= model.interest[k]
            else:
                offset -= i0 * model.capital

        quad = (
            np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64),
            np.concatenate(cols).astype(np.int64) if cols else np.zeros(0, np.int64),
            np.concatenate(vals).astype(float) if vals else np.zeros(0),
        )
        return dimod.BinaryQuadraticModel.from_numpy_vectors(linear, quad, offset, dimod.BINARY)

    def _build_wall(self) -> dimod.BinaryQuadraticModel:
        """Domain-wall ordering terms at unit weight (see the module docstring)."""
        n, ub, x = len(self.model.symbols), self.ub, self._x
        bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
            np.zeros(self.n_vars), ([], [], []), 0.0, dimod.BINARY
        )
        for t in self.periods:
            for i in range(n):
                for tau in (1, -1):
                    for m in range(ub - 1):  # x[m+1] (1 - x[m])
                        bqm.add_linear(x(i, m + 1, tau, t), 1.0)
                        bqm.add_quadratic(x(i, m, tau, t), x(i, m + 1, tau, t), -1.0)
                bqm.add_quadratic(x(i, 0, 1, t), x(i, 0, -1, t), 1.0)
        return bqm

    def _build_penalty(self) -> dimod.BinaryQuadraticModel:
        model, n, ub = self.model, len(self.model.symbols), self.ub
        bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
            np.zeros(self.n_vars), ([], [], []), 0.0, dimod.BINARY
        )
        for t in self.periods:
            xs = [(self._x(i, m, tau, t), tau) for i in range(n) for m in range(ub) for tau in (1, -1)]
            if self.keep_capital:
                capital = xs + [(self._y(k, t), 2**k) for k in range(self.cash_bits)]
                bqm.add_linear_equality_constraint(capital, 1.0, -model.capital)
            budget = [(v, 1) for v, _ in xs]
            budget += [(self._s(c, t), 2**c) for c in range(self.count_bits)]
            bqm.add_linear_equality_constraint(budget, 1.0, -model.budget)
        return bqm

    def budget_constraint(self, t: int) -> tuple[dict, str, int]:
        """Period ``t``'s budget rule on the position bits only: ``(coeffs, op, rhs)``.

        For solvers that take constraints directly (e.g. Divi's ``LinearConstraint``),
        so the budget slack bits are not needed.
        """
        n = len(self.model.symbols)
        coeffs = {
            self._x(i, m, tau, t): 1
            for i in range(n) for m in range(self.ub) for tau in (1, -1)
        }
        return coeffs, ("==" if self.budget_equality else "<="), self.model.budget

    # ── sample <-> positions ─────────────────────────────────────────────

    def decode(self, sample) -> dict:
        """Bit assignment (dict or array in label order) → canonical positions.

        The final period is filled in from :attr:`last_period` when it was solved apart.
        """
        bits = _as_bit_array(sample, self.n_vars)
        model, T = self.model, self.model.n_periods
        L = np.zeros((T, len(model.symbols)), dtype=np.int64)
        S = np.zeros_like(L)
        for idx in np.flatnonzero(bits[: self.n_x]):
            _, sym, _, tau, t = self.labels[idx]
            i = model.symbols.index(sym)
            (L if tau == 1 else S)[t, i] += 1
        if self.last_period is not None:
            L[T - 1], S[T - 1] = self.last_period
        return model.to_positions(L, S)

    def encode(self, positions: Positions) -> np.ndarray:
        """Canonical positions → bit array (copies filled from slot 0, exact slacks).

        The final period of ``positions`` is ignored when it was solved apart.
        """
        model = self.model
        L, S = model.to_arrays(positions)
        if (L[self.periods] > self.ub).any() or (S[self.periods] > self.ub).any():
            raise ValueError(f"positions exceed this encoding's ub={self.ub}")
        bits = np.zeros(self.n_vars, dtype=np.int8)
        for t in self.periods:
            for i in range(len(model.symbols)):
                for m in range(L[t, i]):
                    bits[self._x(i, m, 1, t)] = 1
                for m in range(S[t, i]):
                    bits[self._x(i, m, -1, t)] = 1
            cash = model.capital - (L[t] - S[t]).sum()
            count = model.budget - (L[t] + S[t]).sum()
            for k in range(self.cash_bits):
                bits[self._y(k, t)] = (cash >> k) & 1 if cash >= 0 else 0
            for c in range(self.count_bits):
                bits[self._s(c, t)] = (count >> c) & 1 if count >= 0 else 0
        return bits

    def bitstring(self, positions: Positions) -> str:
        """:meth:`encode` as a ``'0101…'`` string (e.g. for ``reference_states``)."""
        return "".join(map(str, self.encode(positions)))
