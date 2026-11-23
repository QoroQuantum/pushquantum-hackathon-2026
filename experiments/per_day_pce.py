"""Step 1 of PCE → groups → PCE: PCE on one trading day at a time.

Built on :class:`portfolio_compact.CompactPortfolioQUBO`. Day ``t``'s sub-problem
is the compact objective with every other day fixed. The rebalancing links to the
neighbouring days then become linear terms, and the sub-problem's energy equals the
official score of the whole portfolio. The day's budget is passed to Divi as a
``LinearConstraint``, so infeasible samples can be filtered out.

Two experiments (``python experiments/per_day_pce.py --help``):

* ``oracle``: each day solved with the other days fixed to the best-known solution.
  This isolates PCE's quality on a single day: its result is compared with the
  day's exact optimum, found by enumerating every portfolio of that day.
* ``sweep``: no outside information. Start from an empty portfolio and solve the
  days in turn (forward sweeps), accepting a day's new portfolio only if the
  official score improves; then local search. Run with ``--solver exact`` for the
  classical control (exact per-day solves).

When the capital rule can bind (``B > 5``, e.g. the 50-asset instances) it becomes
two more constraints per day, and cash interest is folded into a linear term. A day
is then too large to enumerate, so ``oracle`` compares against the best-known day
instead of the exact optimum, and ``--solver sa`` (simulated annealing on the same
day sub-problem, constraints as penalties) is the classical control.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import dimod
import numpy as np

import qoblib_tools as qt
from portfolio_compact import CompactPortfolioQUBO


# ── day sub-problems ─────────────────────────────────────────────────────


@dataclass
class DayProblem:
    """Day ``t`` of the compact QUBO with every other variable fixed."""

    t: int
    bqm: dimod.BinaryQuadraticModel
    """Variables ``0..k-1``; energy = official score of the whole portfolio."""
    global_index: list[int]
    """``global_index[j]``: the compact-QUBO variable behind local variable ``j``."""
    slot: list[tuple[int, int, int]]
    """``slot[j] = (asset, copy, direction)`` of local variable ``j``."""


def day_problem(q: CompactPortfolioQUBO, t: int, L: np.ndarray, S: np.ndarray) -> DayProblem:
    """Day ``t``'s sub-problem, with the other days fixed to ``(L, S)``."""
    model = q.model
    index = {s: i for i, s in enumerate(model.symbols)}
    glob, slot = [], []
    for k, lab in enumerate(q.labels):
        if lab[0] == "x" and lab[-1] == t:
            _, sym, m, tau, _ = lab
            glob.append(k)
            slot.append((index[sym], m, tau))
    bits = q.encode(model.to_positions(L, S))
    keep = set(glob)
    fixed = q.objective.copy()
    if q.keep_capital:
        # The compact QUBO pays cash interest on day t's cash-slack bits. Those bits
        # are replaced by constraints, so pay it on the positions instead: interest
        # is linear (10 * 2**k), i.e. -i0 * (C - net) with net = sum of tau * x.
        i0 = int(model.interest[0])
        assert all(int(model.interest[k]) == i0 * 2**k for k in range(q.cash_bits))
        for kb in range(q.cash_bits):
            fixed.set_linear(q._y(kb, t), 0.0)
        for g, (_, _, tau) in zip(glob, slot):
            fixed.add_linear(g, tau * i0)
        fixed.offset -= i0 * model.capital
    fixed.fix_variables({v: int(bits[v]) for v in fixed.variables if v not in keep})

    # Re-insert the day's variables as 0..k-1, in order, so that Divi's variable
    # order (and so the sample bitstrings) matches ``global_index``.
    local = {g: j for j, g in enumerate(glob)}
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    for g in glob:
        bqm.add_variable(local[g], fixed.get_linear(g))
    for (u, v), bias in fixed.quadratic.items():
        bqm.add_quadratic(local[u], local[v], bias)
    bqm.offset = fixed.offset
    return DayProblem(t, bqm, glob, slot)


def day_bits(dp: DayProblem, n_assets: int, Lt: np.ndarray, St: np.ndarray) -> np.ndarray:
    """A day's unit counts → local bits (copies filled from slot 0)."""
    bits = np.zeros(len(dp.slot), dtype=np.int8)
    for j, (i, m, tau) in enumerate(dp.slot):
        bits[j] = m < (Lt[i] if tau == 1 else St[i])
    return bits


def day_counts(dp: DayProblem, n_assets: int, bits) -> tuple[np.ndarray, np.ndarray]:
    """Local bits → the day's ``(long, short)`` unit counts."""
    Lt = np.zeros(n_assets, dtype=np.int64)
    St = np.zeros(n_assets, dtype=np.int64)
    for j, (i, _, tau) in enumerate(dp.slot):
        if int(bits[j]):
            (Lt if tau == 1 else St)[i] += 1
    return Lt, St


def day_constraints(q: CompactPortfolioQUBO, dp: DayProblem) -> list[tuple[dict, str, int]]:
    """The day's rules on its position bits: ``(coefficients, sense, bound)``."""
    k, m = len(dp.slot), q.model
    out = [({j: 1 for j in range(k)}, "==" if q.budget_equality else "<=", m.budget)]
    if q.keep_capital:  # C - max_cash <= net <= C
        net = {j: tau for j, (_, _, tau) in enumerate(dp.slot)}
        out += [(net, "<=", m.capital), (net, ">=", m.capital - q.max_cash)]
    return out


def wall_valid(dp: DayProblem, bits: np.ndarray) -> np.ndarray:
    """Which rows of local bits are in domain-wall form on every asset (copies in
    order on each direction, never long and short at once)."""
    bits = np.asarray(bits)
    col = {s: j for j, s in enumerate(dp.slot)}
    ub = 1 + max(m for _, m, _ in dp.slot)
    ok = np.ones(len(bits), dtype=bool)
    for i in sorted({i for i, _, _ in dp.slot}):
        for tau in (1, -1):
            for m in range(ub - 1):
                ok &= bits[:, col[(i, m, tau)]] >= bits[:, col[(i, m + 1, tau)]]
        ok &= ~(bits[:, col[(i, 0, 1)]].astype(bool) & bits[:, col[(i, 0, -1)]].astype(bool))
    return ok


def satisfies(constraints, bits: np.ndarray) -> np.ndarray:
    """Which rows of local bits satisfy every constraint."""
    ok = np.ones(len(bits), dtype=bool)
    for coeffs, sense, bound in constraints:
        idx = np.fromiter(coeffs.keys(), int)
        lhs = bits[:, idx] @ np.fromiter(coeffs.values(), float)
        ok &= {"<=": lhs <= bound, ">=": lhs >= bound, "==": lhs == bound}[sense]
    return ok


def bqm_energies(bqm: dimod.BinaryQuadraticModel, bits: np.ndarray) -> np.ndarray:
    """Energies of a batch of rows of local bits."""
    lin, (r, c, w), off = bqm.to_numpy_vectors(variable_order=range(bqm.num_variables))
    bits = bits.astype(float)
    return bits @ lin + (bits[:, r] * bits[:, c]) @ w + off


# ── solvers for one day ──────────────────────────────────────────────────


@dataclass
class DayResult:
    Lt: np.ndarray
    St: np.ndarray
    energy: float  # official score of the whole portfolio with this day
    feasible_found: bool = True
    n_qubits: int = 0
    circuits: int = 0
    seconds: float = 0.0
    confidence: np.ndarray | None = None
    """Per local variable: |P(x=1) − P(x=0)| over PCE's sampled distribution."""
    marginals: np.ndarray | None = None
    """Per local variable: P(x=1) over PCE's sampled distribution."""
    min_units_sampled: int = 0
    """Fewest units (bits set) in any sample: > B means no sample can be feasible."""
    valid_mass: float = math.nan
    """Probability PCE puts on samples whose every slot is in domain-wall form."""


def n_day_configs(n: int, B: int) -> int:
    """Upper bound on a day's configurations (ignoring the per-asset cap)."""
    return sum(math.comb(2 * n + k - 1, k) for k in range(B + 1))


def _day_configs(n: int, ub: int, B: int) -> np.ndarray:
    """Every ``(long, short)`` configuration of one day with at most ``B`` units."""
    rows = []
    for k in range(B + 1):
        for combo in itertools.combinations_with_replacement(range(2 * n), k):
            counts = np.bincount(combo, minlength=2 * n)
            if counts.max(initial=0) <= ub:
                rows.append(counts)
    return np.array(rows).reshape(-1, n, 2)


def solve_day_exact(dp: DayProblem, q: CompactPortfolioQUBO, configs: np.ndarray) -> DayResult:
    """The day's exact optimum, by enumerating every feasible configuration."""
    n = len(q.model.symbols)
    t0 = time.time()
    bits = np.array([day_bits(dp, n, c[:, 0], c[:, 1]) for c in configs])
    e = bqm_energies(dp.bqm, bits)
    best = int(np.argmin(e))
    return DayResult(configs[best, :, 0].copy(), configs[best, :, 1].copy(), float(e[best]),
                     seconds=time.time() - t0)


def solve_day_sa(dp: DayProblem, q: CompactPortfolioQUBO, num_reads: int = 100,
                 penalty_scale: float = 1.0, seed: int = 1) -> DayResult:
    """Classical control: simulated annealing on the day, constraints as penalties."""
    from dwave.samplers import SimulatedAnnealingSampler

    n, k = len(q.model.symbols), dp.bqm.num_variables
    cons = day_constraints(q, dp)
    bqm = dp.bqm.copy()
    W = penalty_scale * penalty_weight(dp.bqm)
    for c, (coeffs, sense, bound) in enumerate(cons):
        terms = list(coeffs.items())
        if sense == "==":
            bqm.add_linear_equality_constraint(terms, W, -bound)
        elif sense == "<=":
            bqm.add_linear_inequality_constraint(terms, W, f"c{c}", ub=bound,
                                                 lb=int(sum(min(0, a) for a in coeffs.values())))
        else:
            bqm.add_linear_inequality_constraint(terms, W, f"c{c}", lb=bound,
                                                 ub=int(sum(max(0, a) for a in coeffs.values())))
    t0 = time.time()
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=seed)
    bits = np.array([[int(r[j]) for j in range(k)] for r in ss.samples()], dtype=np.int8)
    seconds = time.time() - t0
    ok = satisfies(cons, bits)
    if not ok.any():
        return DayResult(np.zeros(n, np.int64), np.zeros(n, np.int64), math.inf,
                         feasible_found=False, seconds=seconds)
    e = np.where(ok, bqm_energies(dp.bqm, bits), np.inf)
    best = int(np.argmin(e))
    Lt, St = day_counts(dp, n, bits[best])
    return DayResult(Lt, St, float(e[best]), seconds=seconds)


@dataclass
class PCESettings:
    encoding: str = "poly"  # "poly" | "dense" | "multibasis"
    progressive: bool = False  # ProgressivePCE (iterations = per step)
    alpha: float = 2.0
    n_layers: int = 3
    population: int = 50
    iterations: int = 10
    shots: int = 10_000
    penalty_scale: float = 1.0
    top_k: int = 50
    seed: int = 1


def penalty_weight(bqm: dimod.BinaryQuadraticModel) -> float:
    """A budget-penalty weight large enough that no violation pays off.

    Breaking the budget by ``v`` units costs ``W·v²``; ``v`` extra flips gain at
    most ``v`` times the largest single-flip gain, ``|h_j| + Σ_k |J_jk|``.
    """
    gain = {v: abs(b) for v, b in bqm.linear.items()}
    for (u, v), b in bqm.quadratic.items():
        gain[u] += abs(b)
        gain[v] += abs(b)
    return max(gain.values(), default=1.0)


def solve_day_pce(dp: DayProblem, q: CompactPortfolioQUBO, settings: PCESettings,
                  backend=None) -> DayResult:
    """Solve the day with PCE; return the best feasible sample (by energy)."""
    from divi.backends import MaestroSimulator
    from divi.qprog.algorithms import (
        PCE, DenseEncoding, MultiBasisEncoding, PolyEncoding, ProgressivePCE, SoftObjective,
    )
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint

    n, k = len(q.model.symbols), dp.bqm.num_variables
    problem = BinaryOptimizationProblem(
        dp.bqm,
        constraints=[LinearConstraint(c, s_, b) for c, s_, b in day_constraints(q, dp)],
        penalty_weight=settings.penalty_scale * penalty_weight(dp.bqm),
    )
    encoding = {
        "poly": PolyEncoding, "dense": DenseEncoding, "multibasis": lambda: MultiBasisEncoding(order=2),
    }[settings.encoding]()
    common_kw = dict(
        encoding=encoding,
        n_layers=settings.n_layers,
        optimizer=MonteCarloOptimizer(population_size=settings.population),
        backend=backend or MaestroSimulator(shots=settings.shots),
        seed=settings.seed,
    )
    if settings.progressive:
        pce = ProgressivePCE(problem, max_iterations_per_step=settings.iterations, **common_kw)
    else:
        pce = PCE(problem, objective=SoftObjective(alpha=settings.alpha),
                  max_iterations=settings.iterations, **common_kw)
    t0 = time.time()
    pce.run()
    seconds = time.time() - t0

    # Per-variable marginals over the whole sampled distribution. A multi-basis
    # encoding returns one sign-rounded assignment, so its "marginals" are 0/1.
    dist = pce.get_top_solutions(n=10**9, min_prob=0.0, sort_by="prob")
    samples = np.array([[int(b) for b in s.bitstring[:k]] for s in dist], dtype=np.int8)
    probs = np.array([s.prob for s in dist])
    marg = probs @ samples / probs.sum()
    min_units = int(samples.sum(axis=1).min())
    valid_mass = float(probs @ wall_valid(dp, samples) / probs.sum())

    tops = pce.get_top_solutions(n=settings.top_k, sort_by="energy", feasibility="filter")
    common = dict(n_qubits=getattr(pce, "n_qubits", 0), circuits=pce.total_circuit_count,
                  seconds=seconds, confidence=np.abs(2 * marg - 1), marginals=marg,
                  min_units_sampled=min_units, valid_mass=valid_mass)
    if not tops:
        return DayResult(np.zeros(n, np.int64), np.zeros(n, np.int64), math.inf,
                         feasible_found=False, **common)
    bits = np.array([[int(b) for b in s.bitstring[:k]] for s in tops], dtype=np.int8)
    e = bqm_energies(dp.bqm, bits)
    # Divi's energy is the objective without penalties: it must match ours.
    assert np.allclose(e, [s.energy for s in tops], rtol=0, atol=1e-6 * max(1, abs(e).max())), \
        "variable order mismatch between the sub-problem and Divi's samples"
    best = int(np.argmin(e))
    Lt, St = day_counts(dp, n, bits[best])
    return DayResult(Lt, St, float(e[best]), **common)


# ── experiments ──────────────────────────────────────────────────────────


def load(instance: str, budget: int, lam: str, domain_wall: float = 0.0):
    model = qt.PortfolioModel(qt.load_portfolio_instance(instance), budget=budget, lam=lam)
    q = CompactPortfolioQUBO(model, domain_wall=domain_wall)
    header, ref = qt.read_portfolio_solution(
        qt.fetch_portfolio_reference(instance, budget=budget, lam_tag=lam)
    )
    return model, q, int(header["objective"]), ref


def run_oracle(model, q, bkv, ref, settings: PCESettings) -> list[dict]:
    """Each day alone, the other days fixed to the best-known solution."""
    n = len(model.symbols)
    enumerable = n_day_configs(n, model.budget) <= 200_000
    configs = _day_configs(n, q.ub, model.budget) if enumerable else None
    L, S = model.to_arrays(ref)
    rows = []
    for t in q.periods:
        dp = day_problem(q, t, L, S)
        k = dp.bqm.num_variables
        ref_bits = day_bits(dp, n, L[t], S[t])
        # Best day known: the exact optimum if enumerable, else the best-known solution's day.
        if enumerable:
            ex = solve_day_exact(dp, q, configs)
            target, target_bits = ex.energy, day_bits(dp, n, ex.Lt, ex.St)
        else:
            target, target_bits = float(bqm_energies(dp.bqm, ref_bits[None])[0]), ref_bits
        empty = float(bqm_energies(dp.bqm, np.zeros((1, k)))[0])
        sa = solve_day_sa(dp, q, penalty_scale=settings.penalty_scale, seed=settings.seed)
        pc = solve_day_pce(dp, q, settings)
        if q.domain_wall:  # report official scores, not energies with the wall terms
            for res in (sa, pc):
                if res.feasible_found:
                    L2, S2 = L.copy(), S.copy()
                    L2[t], S2[t] = res.Lt, res.St
                    res.energy = float(model.score_arrays(L2, S2).objective)
        agree = float(np.mean((pc.marginals > 0.5) == target_bits.astype(bool)))
        rows.append(dict(
            day=t, target=target, target_is_exact=enumerable, empty_day=empty,
            pce=pc.energy, gap=pc.energy - target, hit=bool(pc.energy <= target),
            feasible=pc.feasible_found, sa=sa.energy, sa_gap=sa.energy - target,
            target_units=int(target_bits.sum()), pce_units=int(pc.Lt.sum() + pc.St.sum()),
            sa_units=int(sa.Lt.sum() + sa.St.sum()), min_units_sampled=pc.min_units_sampled,
            vars=k, qubits=pc.n_qubits, circuits=pc.circuits, seconds=round(pc.seconds, 2),
            sa_seconds=round(sa.seconds, 2), marginal_agreement=round(agree, 3),
            valid_mass=round(pc.valid_mass, 4),
        ))
        print(f"  day {t}: {'exact' if enumerable else 'best-known day'} {target:.0f} "
              f"({int(target_bits.sum())} units) | empty day {empty:.0f} "
              f"| SA {sa.energy:.0f} ({sa.seconds:.1f}s) "
              f"| PCE {pc.energy:.0f} (gap {pc.energy - target:+.0f}, fewest units in a sample "
              f"{pc.min_units_sampled}, domain-wall-valid mass {pc.valid_mass:.2f}; {k} vars on {pc.n_qubits} qubits, {pc.circuits} circuits, "
              f"{pc.seconds:.1f}s)", flush=True)
    return rows


def run_sweep(model, q, bkv, settings: PCESettings, solver: str, n_sweeps: int) -> dict:
    """Forward sweeps from an empty portfolio; accept a day only if the score improves."""
    n, T = len(model.symbols), model.n_periods
    configs = _day_configs(n, q.ub, model.budget) if solver == "exact" else None
    if solver == "exact" and n_day_configs(n, model.budget) > 200_000:
        raise ValueError("days too large to enumerate: use --solver sa or pce")
    L = np.zeros((T, n), np.int64)
    S = np.zeros_like(L)
    if q.last_period is not None:
        L[-1], S[-1] = q.last_period
    score = model.score_arrays(L, S).objective
    history, circuits, t_start = [score], 0, time.time()
    for sweep in range(n_sweeps):
        improved = False
        for t in q.periods:
            dp = day_problem(q, t, L, S)
            if solver == "exact":
                res = solve_day_exact(dp, q, configs)
            elif solver == "sa":
                res = solve_day_sa(dp, q, penalty_scale=settings.penalty_scale,
                                   seed=settings.seed + 1000 * sweep + t)
            else:
                res = solve_day_pce(dp, q, PCESettings(**{**asdict(settings), "seed": settings.seed + 1000 * sweep + t}))
                circuits += res.circuits
            if not res.feasible_found:
                continue
            L2, S2 = L.copy(), S.copy()
            L2[t], S2[t] = res.Lt, res.St
            new = model.score_arrays(L2, S2)
            if new.feasible and new.objective < score:
                L, S, score, improved = L2, S2, new.objective, True
        history.append(score)
        print(f"  sweep {sweep + 1}: {score}", flush=True)
        if not improved:
            break
    polished = model.local_search(model.to_positions(L, S))
    final = model.score(polished)
    return dict(
        solver=solver, sweeps=history, before_local_search=score,
        after_local_search=final.objective, feasible=final.feasible, bkv=bkv,
        gap_before=(score - bkv) / abs(bkv), gap_after=(final.objective - bkv) / abs(bkv),
        circuits=circuits, seconds=round(time.time() - t_start, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["oracle", "sweep"])
    ap.add_argument("--instance", default="po_a010_t10_orig")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--lam", default="1e-4")
    ap.add_argument("--solver", choices=["pce", "exact", "sa"], default="pce")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--encoding", choices=["poly", "dense", "multibasis"], default="poly")
    ap.add_argument("--progressive", action="store_true", help="ProgressivePCE")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--population", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--penalty-scale", type=float, default=1.0)
    ap.add_argument("--domain-wall", type=float, default=0.0,
                    help="weight of the domain-wall ordering terms (0 = off; exact for any weight)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model, q, bkv, ref = load(args.instance, args.budget, args.lam, args.domain_wall)
    print(f"{args.instance}, B={args.budget}, λ={args.lam}: best known {bkv}; "
          f"compact QUBO {q.reductions['variables']} variables, {len(q.periods)} days in the QUBO")

    out = []
    for seed in args.seeds:
        settings = PCESettings(encoding=args.encoding, alpha=args.alpha, n_layers=args.layers,
                               population=args.population, iterations=args.iterations,
                               penalty_scale=args.penalty_scale, progressive=args.progressive,
                               seed=seed)
        print(f"\n[{args.experiment}, solver={args.solver if args.experiment == 'sweep' else 'pce'}, seed={seed}]")
        if args.experiment == "oracle":
            rows = run_oracle(model, q, bkv, ref, settings)
            hits = sum(r["hit"] for r in rows)
            feas = [r for r in rows if r["feasible"]]
            print(f"  → PCE feasible on {len(feas)}/{len(rows)} days, matched the target on "
                  f"{hits}/{len(rows)}, beat an empty day on "
                  f"{sum(r['pce'] < r['empty_day'] for r in rows)}/{len(rows)}; SA matched the target on "
                  f"{sum(r['sa'] <= r['target'] for r in rows)}/{len(rows)}")
            out.append(dict(seed=seed, settings=asdict(settings), days=rows))
        else:
            res = run_sweep(model, q, bkv, settings, args.solver, args.sweeps)
            print(f"  → {res['before_local_search']} (gap {res['gap_before']:+.2%}), after local search "
                  f"{res['after_local_search']} (gap {res['gap_after']:+.2%}); "
                  f"{res['circuits']} circuits, {res['seconds']}s")
            out.append(dict(seed=seed, settings=asdict(settings), **res))

    Path(args.out).mkdir(exist_ok=True)
    tag = f"{args.experiment}_{args.solver if args.experiment == 'sweep' else 'pce'}_{args.instance}_B{args.budget}_lam{args.lam}_{args.encoding}{'_progressive' if args.progressive else ''}{f'_dw{args.domain_wall:g}' if args.domain_wall else ''}"
    path = Path(args.out) / f"per_day_{tag}.json"
    path.write_text(json.dumps(dict(instance=args.instance, budget=args.budget, lam=args.lam,
                                    domain_wall=args.domain_wall,
                                    bkv=bkv, runs=out), indent=2, default=float))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
