"""Asset grouping for QOBLIB 06 (portfolio): cut measurement and clamp-and-iterate.

A *block* is one day × one group of assets. Each block is re-solved with the rest of
the portfolio fixed ("clamp and iterate"), and a new block is accepted only if the
official score improves. The grouping changes between sweeps, so units can move
between groups.

Blocks are solved in *net-position* space: per asset and day only ``v = L - S`` in
``[-ub, ub]`` is chosen, with ``L = max(v, 0)``, ``S = max(-v, 0)``. That loses
nothing, because holding long and short at once never beats holding the net position
(see ``check_netting`` in ``experiments/check_portfolio_compact.py``). A 5-asset group then has
``7**5 = 16807`` options, few enough to solve exactly; :func:`solve_block_exact` is
the default block solver and the reference for any other (e.g. QAOA) solver.

Block costs are computed directly from the official model's coefficient tables, so a
block's cost change equals the change of the official score (checked against
``PortfolioModel.score_arrays`` in :func:`_self_check`).

    python asset_groups.py cuts  --instance po_a050_t10_orig --budget 20 --lam 1e-3
    python asset_groups.py solve --instance po_a050_t10_orig --budget 20 --lam 1e-3
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import qoblib_tools as qt

GroupFn = Callable[[int], list[np.ndarray]]
"""``groups(sweep) -> list of asset-index arrays`` partitioning ``0..n-1``."""


# ── groupings ────────────────────────────────────────────────────────────


def asset_coupling(model: qt.PortfolioModel) -> np.ndarray:
    """Asset-level risk coupling: mean over days of ``|R_t[i, j]|`` (zero diagonal).

    Every bit pair of assets ``i, j`` in a day has QUBO coefficient ``±2 R_t[i, j]``,
    so this is proportional to the coupling weight a cut between them removes.
    """
    R = model.risk.astype(float)
    W = np.abs((R + R.transpose(0, 2, 1)) / 2).mean(axis=0)
    np.fill_diagonal(W, 0.0)
    return W


def make_groups(model: qt.PortfolioModel, size: int, method: str, seed: int = 0) -> list[np.ndarray]:
    """Partition the assets into groups of at most ``size``.

    Methods: ``contiguous`` (in file order, rotated by ``seed``), ``random``,
    ``louvain`` / ``spectral`` (Divi's size-capped community detection on
    :func:`asset_coupling`, which keeps strongly coupled assets, such as hedging
    pairs, together).
    """
    n = len(model.symbols)
    if method == "contiguous":
        order = np.roll(np.arange(n), seed % n)
        return [g for g in np.array_split(order, -(-n // size)) if len(g)]
    if method == "random":
        order = np.random.default_rng(seed).permutation(n)
        return [g for g in np.array_split(order, -(-n // size)) if len(g)]
    if method in ("louvain", "spectral"):
        from divi.qprog.problems import QUBOPartitioningConfig
        from divi.qprog.problems._qubo_partitioning_utils import partition_by_method

        W = sps.csr_matrix(asset_coupling(model))
        cfg = QUBOPartitioningConfig(
            max_n_variables_per_cluster=size,
            method="modularity" if method == "louvain" else "spectral",
            seed=seed,
        )
        return [np.sort(np.asarray(c)) for c in partition_by_method(W, cfg)]
    if method == "neighbourhood":
        # Randomised and covariance-aware: start each group at a random unassigned
        # asset and add its most strongly coupled unassigned partners.
        rng = np.random.default_rng(seed)
        W = asset_coupling(model)
        free = np.ones(n, bool)
        out = []
        for i in rng.permutation(n):
            if not free[i]:
                continue
            free[i] = False
            cand = np.flatnonzero(free)
            partners = cand[np.argsort(-W[i, cand])][: size - 1]
            free[partners] = False
            out.append(np.sort(np.concatenate([[i], partners])))
        return out
    raise ValueError(f"unknown grouping method {method!r}")


def grouping_fn(model: qt.PortfolioModel, size: int, method: str, vary: bool = True) -> GroupFn:
    """Groups per sweep; with ``vary``, a different partition each sweep."""
    cache: dict[int, list[np.ndarray]] = {}

    def groups(sweep: int) -> list[np.ndarray]:
        key = sweep if vary else 0
        if key not in cache:
            cache[key] = make_groups(model, size, method, seed=key)
        return cache[key]

    return groups


# ── cut measurement ──────────────────────────────────────────────────────


def cut_report(model: qt.PortfolioModel, groups: list[np.ndarray]) -> dict:
    """How much objective coupling each kind of cut removes, in QUBO ``|J|`` units.

    Counts the official unary QUBO's quadratic objective terms (risk within a day,
    rebalancing between consecutive days) by where they fall: inside a block,
    between groups on the same day, or between days. The budget and capital rules
    are not counted: blocks keep them exact (each block sees the rest of its day).
    """
    T, ub = model.n_periods, model.ub
    R = model.risk.astype(float)
    Rs = np.abs((R + R.transpose(0, 2, 1)) / 2)
    # Per pair of assets i != j on day t: (2 ub)^2 bit pairs, each |2 R_ij| over the
    # ordered-pair sum -> 2 * (2 ub)^2 * |R_ij| for the unordered pair.
    pair = 2 * (2 * ub) ** 2
    label = np.empty(len(model.symbols), int)
    for k, g in enumerate(groups):
        label[g] = k
    same = label[:, None] == label[None, :]
    off = ~np.eye(len(label), dtype=bool)
    # Same asset, different bits (copies/directions): C(2 ub, 2) pairs of |2 R_ii|,
    # always inside a block.
    self_pairs = 2 * (2 * ub) * (2 * ub - 1) // 2
    risk_in = risk_cut = 0.0
    for t in range(T):
        risk_in += pair * Rs[t][same & off].sum() / 2 + self_pairs * np.trace(Rs[t])
        risk_cut += pair * Rs[t][~same].sum() / 2
    # Rebalancing x_{t-1} XOR x_t: quadratic term -2 tc_t per (asset, copy, direction)
    # for t = 1..T-2 (QOBLIB charges no rebalancing into the final day).
    days_cut = sum(2 * 2 * ub * model.tc[t].sum() for t in range(1, T - 1))
    total = risk_in + risk_cut + days_cut
    return dict(
        n_groups=len(groups),
        inside_blocks=risk_in / total,
        between_groups=risk_cut / total,
        between_days=days_cut / total,
    )


# ── block costs in net-position space ────────────────────────────────────


@lru_cache(maxsize=None)
def _net_configs(g: int, ub: int) -> np.ndarray:
    """All net-position vectors of a ``g``-asset group, shape ``(7**g, g)`` for ub=3."""
    return np.array(list(itertools.product(range(-ub, ub + 1), repeat=g)), dtype=np.int64)


class BlockEvaluator:
    """Exact official-score cost of a block's candidates, the rest of the portfolio fixed."""

    def __init__(self, model: qt.PortfolioModel):
        self.model = model
        R = model.risk.astype(float)
        self.Rs = (R + R.transpose(0, 2, 1)) / 2  # v'Rv == v'Rs v
        i0 = int(model.interest[0])
        if not all(int(model.interest[k]) == i0 * 2**k for k in range(model.cash_slack_bits)):
            raise NotImplementedError("cash interest is not linear in the cash amount")
        self.i0 = i0
        self.max_cash = 2**model.cash_slack_bits - 1

    def costs(self, L, S, days, G: np.ndarray, V: np.ndarray):
        """Cost of setting assets ``G`` to net positions ``V (K, g)`` on ``days``.

        ``days`` is a day or a run of consecutive days; the same ``V`` is held on all
        of them. Returns ``(cost, feasible)``. ``cost`` differs from the official
        score of the whole portfolio by a constant (the same for every candidate).
        """
        m, T = self.model, self.model.n_periods
        days = [days] if np.isscalar(days) else list(days)
        assert days == list(range(days[0], days[-1] + 1)), "days must be consecutive"
        Lg, Sg = np.maximum(V, 0), np.maximum(-V, 0)
        tc = m.tc
        others = np.ones(len(m.symbols), bool)
        others[G] = False
        cost = np.zeros(len(V))
        feasible = np.ones(len(V), bool)
        for t in days:
            vo = (L[t] - S[t]).astype(float)
            vo[G] = 0.0
            Rt = self.Rs[t]
            cost += 2.0 * V @ (Rt[G] @ vo) + np.einsum("kg,gh,kh->k", V, Rt[np.ix_(G, G)], V)
            if t < T - 1:
                cost -= V @ m.ret[t, G]
            cost += Sg @ m.sc[t, G]
            cost += self.i0 * V.sum(axis=1)  # interest -i0 (C - net): +i0 per net unit
            if t == 0 or t == T - 1:  # buy-in / liquidation
                cost += (Lg + Sg) @ tc[t, G]
            units = np.abs(V).sum(axis=1) + (L[t, others] + S[t, others]).sum()
            net = V.sum(axis=1) + (L[t, others] - S[t, others]).sum()
            feasible &= (units <= m.budget) & (net <= m.capital) & (net >= m.capital - self.max_cash)
        # Rebalancing t-1 -> t is charged tc[t] for t = 1..T-2; only transitions with
        # an end inside the window depend on V (inside the window they cost 0).
        lo, hi = days[0], days[-1]
        if 1 <= lo <= T - 2:  # entering the window from day lo-1
            cost += (np.abs(Lg - L[lo - 1, G]) + np.abs(Sg - S[lo - 1, G])) @ tc[lo, G]
        if hi + 1 <= T - 2:  # leaving the window into day hi+1
            cost += (np.abs(L[hi + 1, G] - Lg) + np.abs(S[hi + 1, G] - Sg)) @ tc[hi + 1, G]
        return cost, feasible


@dataclass
class BlockResult:
    V: np.ndarray
    """Proposed net positions of the group, held on every day of the block."""


BlockSolver = Callable[["BlockEvaluator", np.ndarray, np.ndarray, list, np.ndarray], BlockResult]
"""``solver(evaluator, L, S, days, G) -> BlockResult``: a proposal for assets ``G`` on
``days``. The driver accepts it only if it is feasible and improves the official
score, so a solver cannot make the portfolio worse."""


def solve_block_exact(ev: BlockEvaluator, L, S, days, G: np.ndarray) -> BlockResult:
    """Best *constant* net holding of the group over ``days``, by enumeration.

    For a single day this is the block's exact optimum.
    """
    V = _net_configs(len(G), ev.model.ub)
    cost, ok = ev.costs(L, S, days, G, V)
    return BlockResult(V[int(np.argmin(np.where(ok, cost, np.inf)))].copy())


# ── clamp and iterate ────────────────────────────────────────────────────


def block_schedule(T: int, windows: tuple[int, ...]) -> list[list[int]]:
    """Day windows per sweep: for each length ``w``, overlapping runs (stride ``w // 2``)."""
    out = []
    for w in windows:
        w = min(w, T)
        starts = list(range(0, T - w + 1, max(1, w // 2)))
        if starts[-1] != T - w:
            starts.append(T - w)
        out += [list(range(s0, s0 + w)) for s0 in starts]
    return out


def clamp_and_iterate(
    model: qt.PortfolioModel,
    groups: GroupFn,
    solver: BlockSolver = solve_block_exact,
    windows: tuple[int, ...] = (1,),
    max_sweeps: int = 30,
    patience: int = 2,
    L=None,
    S=None,
    verbose: bool = True,
    days_subset=None,
    sweep_offset: int = 0,
    deadline: float | None = None,
) -> dict:
    """Re-solve every (window of days, group) block until ``patience`` sweeps improve nothing.

    A proposal is accepted only if the official score (``PortfolioModel.score_arrays``)
    improves and the portfolio stays feasible. ``days_subset`` restricts the blocks to
    windows inside those days; ``sweep_offset`` shifts which groupings are used.
    ``deadline`` (a ``time.time()`` value) stops between blocks once passed.
    """
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    L = np.zeros((T, n), np.int64) if L is None else L.copy()
    S = np.zeros((T, n), np.int64) if S is None else S.copy()
    score = model.score_arrays(L, S)
    assert score.feasible, "start must be feasible"
    best = score.objective
    schedule = block_schedule(T, windows)
    if days_subset is not None:
        keep = set(days_subset)
        schedule = [d for d in schedule if set(d) <= keep]
    history, stale, blocks, accepted, t0 = [best], 0, 0, 0, time.time()
    for sweep in range(max_sweeps):
        before = best
        for days in schedule:
            for G in groups(sweep + sweep_offset):
                if deadline is not None and time.time() > deadline:
                    break
                V = solver(ev, L, S, days, G).V
                blocks += 1
                L2, S2 = L.copy(), S.copy()
                L2[np.ix_(days, G)] = np.maximum(V, 0)
                S2[np.ix_(days, G)] = np.maximum(-V, 0)
                if np.array_equal(L2, L) and np.array_equal(S2, S):
                    continue
                new = model.score_arrays(L2, S2)
                if new.feasible and new.objective < best:
                    L, S, best = L2, S2, new.objective
                    accepted += 1
        history.append(best)
        if deadline is not None and time.time() > deadline:
            break
        if verbose:
            print(f"  sweep {sweep + 1}: {best}", flush=True)
        stale = stale + 1 if best >= before else 0
        if stale >= patience:
            break
    return dict(score=best, history=history, blocks=blocks, accepted=accepted,
                seconds=round(time.time() - t0, 1), L=L, S=S)



# ── iterated local search ────────────────────────────────────────────────


def kick(model: qt.PortfolioModel, ev: "BlockEvaluator", L, S, rng, n_blocks: int, size: int,
         one_day: bool = False):
    """Set ``n_blocks`` random (day, asset-group) blocks to random feasible holdings.

    With ``one_day`` all blocks fall on the same random day, so the re-optimisation
    after the kick only has to cover that day and its two neighbours.
    """
    T, n = model.n_periods, len(model.symbols)
    L, S = L.copy(), S.copy()
    days = []
    t_fixed = int(rng.integers(T))
    for _ in range(n_blocks):
        t = t_fixed if one_day else int(rng.integers(T))
        G = np.sort(rng.choice(n, size=min(size, n), replace=False))
        V = _net_configs(len(G), model.ub)
        _, ok = ev.costs(L, S, [t], G, V)
        k = rng.choice(np.flatnonzero(ok))  # the current holding is feasible, so ok has one
        L[t, G], S[t, G] = np.maximum(V[k], 0), np.maximum(-V[k], 0)
        days.append(t)
    return L, S, sorted(set(days))


def iterated_search(
    model: qt.PortfolioModel,
    groups: GroupFn,
    L=None,
    S=None,
    time_limit: float = 60.0,
    kick_blocks: int = 3,
    kick_size: int = 5,
    solver: BlockSolver = solve_block_exact,
    seed: int = 0,
    verbose: bool = True,
    day_solver: BlockSolver | None = None,
    day_share: float = 0.5,
<<<<<<< HEAD
    max_kicks: int = 0,
=======
    kick_one_day: bool = False,
    on_improve: Callable | None = None,
>>>>>>> 6784ddb605f5e27c6b76dfaf32d6f6393abe1352
) -> dict:
    """Clamp-and-iterate, then repeated kicks + local re-optimisation (keep if better).

    ``time_limit`` <= 0 (or None) removes the clock: the block sweeps run until they stop
    improving, then exactly ``max_kicks`` kicks / day moves follow (none by default), so the
    result depends on the seed only, not on run time or machine load.

    After a kick only the kicked days and their neighbours are re-optimised (they
    are the only blocks whose surroundings changed); after an accepted kick a full
    sweep runs as well.

    With ``day_solver`` (e.g. ``big_blocks.SABlockSolver``), whole-day blocks (all
    assets of one day) are used too: first, whole-day sweeps alternate with small-block
    sweeps until neither improves; then about ``day_share`` of the remaining time goes
    to *day moves* (re-solve one random day, then its neighbourhood with small blocks)
    instead of kicks.
    """
    rng = np.random.default_rng(seed)
    ev = BlockEvaluator(model)
    T = model.n_periods
    t0 = time.time()
    deadline = t0 + time_limit if time_limit and time_limit > 0 else None

    def more() -> bool:  # keep searching: before the deadline, or until max_kicks moves without one
        return time.time() < deadline if deadline is not None else kicks + day_moves < max_kicks

    # The deadline matters for slow block solvers (QAOA); exact sweeps finish far sooner.
    res = clamp_and_iterate(model, groups, solver, L=L, S=S, max_sweeps=200, patience=3, verbose=False,
                            deadline=deadline)
    best_L, best_S, best = res["L"], res["S"], res["score"]
    start_score = best
    trace = [(round(time.time() - t0, 1), best)]
    kicks = improvements = day_moves = day_improvements = 0
    offset = 1000
    all_assets = [np.arange(len(model.symbols))]
    whole_day = lambda sweep: all_assets  # noqa: E731
    if day_solver is not None:
        while deadline is None or time.time() < deadline:  # polish: whole days <-> small blocks
            before = best
            r = clamp_and_iterate(model, whole_day, day_solver, L=best_L, S=best_S, max_sweeps=20,
                                  patience=1, verbose=False, deadline=deadline)
            offset += 1
            r = clamp_and_iterate(model, groups, solver, L=r["L"], S=r["S"], max_sweeps=50, patience=1,
                                  verbose=False, sweep_offset=offset, deadline=deadline)
            best_L, best_S, best = r["L"], r["S"], r["score"]
            trace.append((round(time.time() - t0, 1), best))
            if best >= before:
                break
    polish_score = best
    t_day = t_kick = 0.0
    while more():
        if day_solver is not None and t_day <= day_share * (t_day + t_kick):
            t1 = time.time()
            t = int(rng.integers(T))
            r = clamp_and_iterate(model, whole_day, day_solver, L=best_L, S=best_S, max_sweeps=1,
                                  patience=1, verbose=False, days_subset=[t], deadline=deadline)
            day_moves += 1
            if r["score"] < best:
                offset += 1
                near = [d for d in (t - 1, t, t + 1) if 0 <= d < T]
                r = clamp_and_iterate(model, groups, solver, L=r["L"], S=r["S"], max_sweeps=50, patience=1,
                                      verbose=False, days_subset=near, sweep_offset=offset, deadline=deadline)
                best_L, best_S, best = r["L"], r["S"], r["score"]
                day_improvements += 1
                trace.append((round(time.time() - t0, 1), best))
                if on_improve is not None:
                    on_improve(best_L, best_S, best)
                if verbose:
                    print(f"  {time.time() - t0:6.1f}s  day move {day_moves}: {best}", flush=True)
            t_day += time.time() - t1
            continue
        t1 = time.time()
        L2, S2, kdays = kick(model, ev, best_L, best_S, rng, kick_blocks, kick_size, one_day=kick_one_day)
        if not model.score_arrays(L2, S2).feasible:
            t_kick += time.time() - t1
            continue
        near = sorted({d for t in kdays for d in (t - 1, t, t + 1) if 0 <= d < T})
        offset += 1
        r = clamp_and_iterate(model, groups, solver, L=L2, S=S2, max_sweeps=50, patience=1,
                              verbose=False, days_subset=near, sweep_offset=offset,
                              deadline=deadline)
        kicks += 1
        if r["score"] < best:
            offset += 1
            r = clamp_and_iterate(model, groups, solver, L=r["L"], S=r["S"], max_sweeps=50,
                                  patience=1, verbose=False, sweep_offset=offset,
                                  deadline=deadline)
            best_L, best_S, best = r["L"], r["S"], r["score"]
            improvements += 1
            trace.append((round(time.time() - t0, 1), best))
            if on_improve is not None:
                on_improve(best_L, best_S, best)
            if verbose:
                print(f"  {time.time() - t0:6.1f}s  kick {kicks}: {best}", flush=True)
        t_kick += time.time() - t1
    final = model.score_arrays(best_L, best_S)
    assert final.feasible and final.objective == best
    return dict(score=best, start_score=start_score, polish_score=polish_score, kicks=kicks,
                improvements=improvements, day_moves=day_moves, day_improvements=day_improvements,
                trace=trace, seconds=round(time.time() - t0, 1), L=best_L, S=best_S)


# ── classical global start: SA on the QUBO + fast repair ──────────────────


def repair_fast(model: qt.PortfolioModel, L, S) -> tuple[np.ndarray, np.ndarray]:
    """Make a portfolio feasible cheaply, in net-position space.

    Nets long against short (never worse), clips to ``ub``, then per day removes one
    unit at a time, the one whose removal costs least (exact block costs), until the
    budget and capital rules hold.
    """
    ev = BlockEvaluator(model)
    T, n, ub = model.n_periods, len(model.symbols), model.ub
    v = np.clip(L - S, -ub, ub)
    L, S = np.maximum(v, 0), np.maximum(-v, 0)
    for t in range(T):
        while True:
            units, net = (L[t] + S[t]).sum(), (L[t] - S[t]).sum()
            too_many = units > model.budget
            too_long, too_short = net > model.capital, net < model.capital - ev.max_cash
            if not (too_many or too_long or too_short):
                break
            vt = L[t] - S[t]
            held = np.flatnonzero(vt)
            if too_long:
                held = held[vt[held] > 0]
            elif too_short:
                held = held[vt[held] < 0]
            # Candidate: shrink |v_i| by one, each held asset i, scored as a 1-asset block.
            best_i, best_c = None, np.inf
            for i in held:
                G = np.array([i])
                cand = np.array([[vt[i] - np.sign(vt[i])], [vt[i]]])
                c, _ = ev.costs(L, S, [t], G, cand)
                if c[0] - c[1] < best_c:
                    best_i, best_c = i, c[0] - c[1]
            nv = vt[best_i] - np.sign(vt[best_i])
            L[t, best_i], S[t, best_i] = max(nv, 0), max(-nv, 0)
    return L, S


def sa_start(model: qt.PortfolioModel, num_reads: int = 20, num_sweeps: int = 2000,
             weight: float = 100.0, seed: int = 1, verbose: bool = True):
    """The notebook's classical baseline as a start: SA on ``objective + W·penalty``.

    Uses the compact QUBO (exact; fewer variables). Returns the best repaired sample.
    """
    from dwave.samplers import SimulatedAnnealingSampler

    from portfolio_compact import CompactPortfolioQUBO

    t0 = time.time()
    q = CompactPortfolioQUBO(model)
    bqm = q.objective.copy()
    pen = q.penalty.copy()
    pen.scale(weight)
    bqm.update(pen)
    t1 = time.time()
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    t2 = time.time()
    order = list(range(q.n_vars))
    best = None
    for row in ss.record.sample[:, [ss.variables.index(v) for v in order]]:
        L, S = model.to_arrays(q.decode(row))
        L, S = repair_fast(model, L, S)
        sc = model.score_arrays(L, S)
        assert sc.feasible
        if best is None or sc.objective < best[0]:
            best = (sc.objective, L, S)
    if verbose:
        print(f"  SA start: {q.n_vars} vars, build {t1 - t0:.0f}s, anneal {t2 - t1:.0f}s, "
              f"repair {time.time() - t2:.0f}s -> {best[0]}", flush=True)
    return best[1], best[2], dict(build=t1 - t0, anneal=t2 - t1, repair=time.time() - t2, score=best[0])


def _self_check(model: qt.PortfolioModel, trials: int = 200, seed: int = 0) -> None:
    """Block cost differences == official score differences (random blocks and windows)."""
    rng = np.random.default_rng(seed)
    ev = BlockEvaluator(model)
    T, n, ub = model.n_periods, len(model.symbols), model.ub
    checked = 0
    for _ in range(trials):
        V = rng.integers(-ub, ub + 1, (T, n)) * (rng.random((T, n)) < min(0.1, model.budget / (4 * n)))
        L, S = np.maximum(V, 0), np.maximum(-V, 0)
        w = int(rng.integers(1, 4))
        t0 = int(rng.integers(T - w + 1))
        days = list(range(t0, t0 + w))
        G = rng.choice(n, size=min(4, n), replace=False)
        cand = rng.integers(-ub, ub + 1, (4, len(G)))
        cost, ok = ev.costs(L, S, days, G, cand)
        base = model.score_arrays(L, S)
        if not base.feasible:
            continue
        checked += 1
        scores = []
        for k, c in enumerate(cand):
            L2, S2 = L.copy(), S.copy()
            L2[np.ix_(days, G)], S2[np.ix_(days, G)] = np.maximum(c, 0), np.maximum(-c, 0)
            sc = model.score_arrays(L2, S2)
            assert ok[k] == sc.feasible
            scores.append(sc.objective if sc.feasible else None)
        # Cost differences between feasible candidates == official score differences
        # (interest is linear in cash only within the cash register, i.e. if feasible).
        f = [k for k in range(len(cand)) if scores[k] is not None]
        for k in f[1:]:
            assert np.isclose(cost[k] - cost[f[0]], scores[k] - scores[f[0]]), \
                (days, cost[k] - cost[f[0]], scores[k] - scores[f[0]])
    assert checked >= trials // 10, f"only {checked} feasible random portfolios"


# ── CLI ──────────────────────────────────────────────────────────────────


def _load(instance, budget, lam):
    model = qt.PortfolioModel(qt.load_portfolio_instance(instance), budget=budget, lam=lam)
    try:
        header, _ = qt.read_portfolio_solution(
            qt.fetch_portfolio_reference(instance, budget=budget, lam_tag=lam))
        bkv = int(header["objective"])
    except Exception:  # no published solution (a200, a400)
        bkv = None
    return model, bkv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cuts", "solve", "check", "pipeline"])
    ap.add_argument("--instance", default="po_a050_t10_orig")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["contiguous", "random", "louvain", "spectral", "neighbourhood"])
    ap.add_argument("--fixed", action="store_true", help="same grouping every sweep")
    ap.add_argument("--windows", type=int, nargs="+", default=[1],
                    help="block lengths in days, e.g. 1 3 9 (9 = all days in a10/a50 t10)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-sweeps", type=int, default=100)
    ap.add_argument("--out", default="results")
    # pipeline: SA start -> blocks -> iterated search -> .sol file
    ap.add_argument("--time", type=float, default=120.0, help="iterated-search time limit (s)")
    ap.add_argument("--kick-blocks", type=int, default=6)
    ap.add_argument("--kick-one-day", action="store_true", help="put all kicked blocks on one day")
    ap.add_argument("--sa-reads", type=int, default=20)
    ap.add_argument("--sa-sweeps", type=int, default=2000)
    ap.add_argument("--no-sa", action="store_true", help="start from an empty portfolio")
    ap.add_argument("--init", default=None, help="start from this .sol file (skips SA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solutions", default="solutions")
    add_solver_args(ap)
    args = ap.parse_args()
    model, bkv = _load(args.instance, args.budget, args.lam)
    print(f"{args.instance}, B={args.budget}, λ={args.lam}, {len(model.symbols)} assets, "
          f"{model.n_periods} days; best known {bkv}")

    if args.command == "check":
        _self_check(model)
        print("block costs match official score changes")
        return
    if args.command == "pipeline":
        run_pipeline(model, bkv, args)
        return
    rows = []
    for method in args.methods:
        if args.command == "cuts":
            r = cut_report(model, make_groups(model, args.size, method))
            print(f"  {method:>10}: {r['n_groups']} groups | inside blocks {r['inside_blocks']:.1%}, "
                  f"cut between groups {r['between_groups']:.1%}, cut between days {r['between_days']:.1%}")
        else:
            print(f"\n[{method}, groups of {args.size}{', fixed' if args.fixed else ', varied per sweep'}, "
                  f"windows {args.windows}]")
            r = clamp_and_iterate(model, grouping_fn(model, args.size, method, vary=not args.fixed),
                                  windows=tuple(args.windows), max_sweeps=args.max_sweeps,
                                  patience=args.patience, verbose=False)
            r.pop("L"), r.pop("S")
            gap = None if bkv is None else (r["score"] - bkv) / abs(bkv)
            r["gap"] = gap
            print(f"  → {r['score']}" + ("" if gap is None else f" (gap {gap:+.2%})")
                  + f", {len(r['history']) - 1} sweeps, {r['blocks']} blocks, {r['seconds']}s", flush=True)
        rows.append(dict(method=method, **r))
    if args.command == "solve":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        path = Path(args.out) / (f"groups_{args.instance}_B{args.budget}_lam{args.lam}_g{args.size}"
                                f"_w{'-'.join(map(str, args.windows))}.json")
        path.write_text(json.dumps(dict(instance=args.instance, budget=args.budget, lam=args.lam,
                                        bkv=bkv, size=args.size, windows=args.windows, runs=rows), indent=2, default=float))
        print(f"\nSaved {path}")


def add_solver_args(ap: argparse.ArgumentParser) -> None:
    """Block-solver options shared by ``asset_groups.py pipeline`` and ``auto_worker.py``."""
    ap.add_argument("--solver", choices=["exact", "qaoa"], default="exact",
                    help="block solver: enumeration, or QAOA on the signed 3-bit QUBO (qaoa_signed_blocks)")
    ap.add_argument("--qaoa-layers", type=int, default=2)
    ap.add_argument("--qaoa-iters", type=int, default=30, help="COBYLA iterations per block")
    ap.add_argument("--qaoa-shots", type=int, default=10000)
    ap.add_argument("--qaoa-top", type=int, default=64, help="most probable samples decoded per block")
    ap.add_argument("--qaoa-init", choices=["tqa", "random"], default="tqa",
                    help="starting angles: TQA ramp (default) or Divi's uniform random")
    ap.add_argument("--tqa-dt", type=float, default=1.0, help="TQA time step per layer")
    ap.add_argument("--qaoa-optimizer", choices=["cobyla", "montecarlo"], default="cobyla",
                    help="cobyla, or MonteCarloOptimizer (--qaoa-population, --qaoa-iters iterations)")
    ap.add_argument("--qaoa-population", type=int, default=50, help="Monte Carlo population size")
    ap.add_argument("--tqa-dt-range", type=float, nargs=2, default=[0.05, 8.0], metavar=("LO", "HI"),
                    help="montecarlo + tqa: starting population = TQA ramps with dt in geomspace(LO, HI)")
    ap.add_argument("--tqa-dt-sampling", choices=["geom", "random", "stratified"], default="geom",
                    help="montecarlo + tqa: dt values on a log grid, uniform random draws, or one "
                         "random draw per equal slice of the range (seeded)")
    ap.add_argument("--qaoa-backend", choices=["cpu", "gpu"], default="cpu",
                    help="local Maestro simulator on CPU (default) or GPU")
    ap.add_argument("--qaoa-compare-exact", action="store_true",
                    help="diagnostic: also enumerate each block to count how often QAOA hits its optimum")
    ap.add_argument("--day-blocks", action="store_true",
                    help="also re-solve whole days (all assets) with integer SA (big_blocks.py)")
    ap.add_argument("--day-chains", type=int, default=64, help="SA chains per whole-day block")
    ap.add_argument("--day-sweeps", type=int, default=300, help="SA sweeps per whole-day block")
    ap.add_argument("--day-tscale", type=float, default=1.0,
                    help="multiply the whole-day SA start temperature (<1 = refine a good start)")
    ap.add_argument("--day-share", type=float, default=0.5,
                    help="share of the search time spent on whole-day moves instead of kicks")


def make_solver(args):
    """``(solver, describe)`` for ``args.solver``; ``describe()`` -> JSON-able stats."""
    if getattr(args, "solver", "exact") == "exact":
        return solve_block_exact, lambda: dict(name="exact")
    from qaoa_signed_blocks import QAOABlockSolver  # imports this module: keep it lazy

    backend = None
    if getattr(args, "qaoa_backend", "cpu") == "gpu":
        from divi.backends import MaestroConfig, MaestroSimulator

        backend = MaestroSimulator(shots=args.qaoa_shots,
                                   maestro_config=MaestroConfig(simulator_type="Gpu"))  # type: ignore[call-arg]
    s = QAOABlockSolver(layers=args.qaoa_layers, iters=args.qaoa_iters, shots=args.qaoa_shots,
                        top=args.qaoa_top, init=args.qaoa_init, tqa_dt=args.tqa_dt,
                        seed=args.seed, compare_exact=args.qaoa_compare_exact, backend=backend,
                        optimizer=getattr(args, "qaoa_optimizer", "cobyla"),
                        population=getattr(args, "qaoa_population", 50),
                        tqa_dt_range=getattr(args, "tqa_dt_range", (0.05, 8.0)),
                        tqa_dt_sampling=getattr(args, "tqa_dt_sampling", "geom"))
    return s, lambda: dict(name="qaoa", **s.summary())


def run_pipeline(model: qt.PortfolioModel, bkv, args) -> dict:
    """SA start (or empty) -> blocks -> iterated search; writes .sol and a JSON record."""
    gap = (lambda x: None) if bkv is None else (lambda x: (x - bkv) / abs(bkv))
    fmt = lambda x: f"{x}" + ("" if bkv is None else f" ({gap(x):+.2%})")
    t0 = time.time()
    rec = dict(instance=model.instance.name, budget=model.budget, lam=args.lam, bkv=bkv,
               size=args.size, method=args.methods[0], time_limit=args.time,
               kick_blocks=args.kick_blocks, seed=args.seed)
    if getattr(args, "init", None):
        _, pos = qt.read_portfolio_solution(args.init)
        L, S = (a.astype(np.int64) for a in model.to_arrays(pos))
        sc0 = model.score_arrays(L, S)
        assert sc0.feasible, f"{args.init} is infeasible"
        rec["start"], rec["init"] = sc0.objective, args.init
        print(f"  start ({args.init}): {fmt(sc0.objective)}", flush=True)
    elif args.no_sa:
        L = S = None
        rec["start"] = model.score_arrays(np.zeros((model.n_periods, len(model.symbols)), np.int64),
                                          np.zeros((model.n_periods, len(model.symbols)), np.int64)).objective
    else:
        L, S, info = sa_start(model, num_reads=args.sa_reads, num_sweeps=args.sa_sweeps, seed=args.seed + 1)
        rec["start"], rec["sa"] = info["score"], info
        print(f"  start (SA + fast repair): {fmt(info['score'])}", flush=True)
    groups = grouping_fn(model, args.size, args.methods[0])
    solver, describe = make_solver(args)
    day_solver = None
    if getattr(args, "day_blocks", False):
        from big_blocks import SABlockSolver  # imports this module: keep it lazy

        day_solver = SABlockSolver(chains=args.day_chains, sweeps=args.day_sweeps, seed=args.seed,
                                   t_scale=getattr(args, "day_tscale", 1.0))
    ckpt = Path(args.solutions) / f"{model.instance.name}_b{model.budget:03d}_l{args.lam}.ckpt.sol"
    last_ck = [time.time()]

    def on_improve(L_, S_, score_):
        print(f"  {time.time() - t0:7.0f}s  {score_}", flush=True)
        if time.time() - last_ck[0] > 120:
            Path(args.solutions).mkdir(parents=True, exist_ok=True)
            model.write_solution(ckpt, model.to_positions(L_, S_), comment=f"checkpoint {score_}")
            last_ck[0] = time.time()

    r = iterated_search(model, groups, L=L, S=S, time_limit=args.time, on_improve=on_improve, kick_blocks=args.kick_blocks,
                        kick_size=args.size, solver=solver, seed=args.seed, verbose=False,
                        day_solver=day_solver, day_share=getattr(args, "day_share", 0.5),
<<<<<<< HEAD
                        max_kicks=getattr(args, "kicks", 0))
=======
                        kick_one_day=getattr(args, "kick_one_day", False))
>>>>>>> 6784ddb605f5e27c6b76dfaf32d6f6393abe1352
    rec["solver"] = describe()
    if day_solver is not None:
        rec["day_blocks"] = dict(day_solver.summary(), share=args.day_share, moves=r["day_moves"],
                                 improvements=r["day_improvements"], polish_score=r["polish_score"])
    sc = model.score_arrays(r["L"], r["S"])
    assert sc.feasible and sc.objective == r["score"]
    days_note = "" if day_solver is None else (
        f" -> whole-day polish: {fmt(r['polish_score'])}")
    print(f"  blocks: {fmt(r['start_score'])}{days_note} -> iterated search: {fmt(r['score'])} "
          f"({r['kicks']} kicks, {r['improvements']} improvements"
          + ("" if day_solver is None else f"; {r['day_moves']} day moves, {r['day_improvements']} improvements")
          + f"); total {time.time() - t0:.0f}s", flush=True)
    rec.update(blocks=r["start_score"], score=r["score"], gap=gap(r["score"]), kicks=r["kicks"],
               improvements=r["improvements"], trace=r["trace"], seconds=round(time.time() - t0, 1))
    if rec["solver"]["name"] != "exact":
        print(f"  solver: {rec['solver']}", flush=True)
    Path(args.solutions).mkdir(parents=True, exist_ok=True)
    lam_tag = args.lam
    sol = Path(args.solutions) / f"{model.instance.name}_b{model.budget:03d}_l{lam_tag}.sol"
    model.write_solution(sol, model.to_positions(r["L"], r["S"]),
                         comment=f"asset groups of {args.size} ({args.methods[0]}), {rec['solver']['name']} block solver"
                                 f"{' + whole-day SA blocks' if day_solver is not None else ''}"
                                 f" + iterated search, "
                                 f"{'warm (' + Path(args.init).name + ')' if getattr(args, 'init', None) else 'empty' if args.no_sa else 'SA'} start")
    rec["solution_file"] = str(sol)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = ("" if rec["solver"]["name"] == "exact" else f"_{rec['solver']['name']}") + ("_days" if day_solver else "")
    out = Path(args.out) / f"pipeline_{model.instance.name}_B{model.budget}_lam{lam_tag}_seed{args.seed}{tag}.json"
    out.write_text(json.dumps(rec, indent=2, default=float))
    print(f"  wrote {sol} and {out}")
    return rec


if __name__ == "__main__":
    main()
