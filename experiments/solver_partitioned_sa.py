#!/usr/bin/env python3
"""QOBLIB portfolio solver: clamp asset/day blocks, solve each QUBO with SA.

Run from the repository root (requires NumPy, SciPy, requests and dimod):
    python auto_runner.py experiments/solver_partitioned_sa.py --manual 0 --silent

Pass solver options after the runner's ``--`` separator, e.g. ``--reheats 2``.
Uses the optional dwave-samplers package when available; otherwise uses NumPy.
No Divi, Maestro, or credentials are needed.
"""

from __future__ import annotations

import dataclasses
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import argparse
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys
import time
from typing import Protocol
from fractions import Fraction

import dimod
import numpy as np

import qoblib_tools as qt
import day_swap
from asset_groups import (BlockEvaluator, BlockResult, clamp_and_iterate,
                          grouping_fn, iterated_search, solve_block_exact)
from auto_worker import load_instance_dir
from big_blocks import block_form, solve_sa
from qaoa_signed_blocks import CODES, block_qubo, decode

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # repo root
DEFAULT_SEEDS = (1, 2, 3)


@lru_cache(maxsize=1)
def load_dwave_sampler():
    """Import the optional compiled backend only when it is requested."""
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave.samplers.sa.sampler import default_beta_range

    return SimulatedAnnealingSampler(), default_beta_range


def numpy_anneal(bqm, starts, rng, reads, sweeps, temperature_scale, restart_all_reads):
    """Metropolis annealing of a binary BQM, vectorized over independent reads."""
    n = bqm.num_variables
    linear = np.array([bqm.linear[j] for j in range(n)], dtype=float)
    coupling = np.zeros((n, n), dtype=float)
    for (i, j), bias in bqm.quadratic.items():
        coupling[i, j] = coupling[j, i] = bias
    starts = starts[:reads]
    if restart_all_reads and len(starts):
        states = starts[np.arange(reads) % len(starts)].astype(float)
    else:
        states = rng.integers(0, 2, size=(reads, n)).astype(float)
        states[:len(starts)] = starts

    # Max absolute flip cost per bit, computed in the centered spin form.
    flip_bound = (np.abs(linear + 0.5 * coupling.sum(axis=1))
                  + 0.5 * np.abs(coupling).sum(axis=1))
    active = flip_bound[flip_bound > 0]
    if not len(active):
        return states.astype(np.int8)
    # Start hot enough to cross large barriers, then cool below a typical flip scale.
    hot = active.max() * temperature_scale / 0.5
    cold = np.median(active) / 12.0
    temperatures = np.geomspace(hot, cold, sweeps) if sweeps > 1 else (cold,)
    for temperature in temperatures:
        thresholds = rng.standard_exponential((n, reads)) * temperature
        for k, j in enumerate(rng.permutation(n)):
            delta = (1.0 - 2.0 * states[:, j]) * (linear[j] + states @ coupling[j])
            flip = delta <= thresholds[k]
            states[flip, j] = 1.0 - states[flip, j]
    return states.astype(np.int8)


class BlockStage(Protocol):
    """Return a (reads, n_bits) binary array in label order 0..n_bits-1.

    ``starts`` contains the best feasible warm start, then any promising states
    from the previous stage. A future QAOA stage can ignore the starts or use
    them to initialize a circuit, and return measured bitstrings in this order.
    """

    def __call__(
        self, bqm: dimod.BinaryQuadraticModel, starts: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class AnnealStage:
    reads: int
    sweeps: int
    start_temperature_scale: float = 1.0
    restart_all_reads: bool = False
    backend: str = "numpy"

    def __call__(
        self, bqm: dimod.BinaryQuadraticModel, starts: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.backend == "numpy":
            return numpy_anneal(
                bqm, starts, rng, self.reads, self.sweeps,
                self.start_temperature_scale, self.restart_all_reads,
            )
        sampler, default_beta_range = load_dwave_sampler()
        labels = list(range(bqm.num_variables))
        # β is inverse temperature; a scale > 1 starts hotter.
        beta_hot, beta_cold = default_beta_range(bqm)
        samples = sampler.sample(
            bqm,
            num_reads=self.reads,
            num_sweeps=self.sweeps,
            beta_range=(beta_hot / self.start_temperature_scale, beta_cold),
            initial_states=(starts[: self.reads], labels),
            initial_states_generator="tile" if self.restart_all_reads else "random",
            seed=int(rng.integers(1, 2**31)),
        )
        return samples.record.sample[:, [samples.variables.index(v) for v in labels]]


def encode_holding(holding: np.ndarray, weights: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Encode a feasible incumbent in the signed QUBO's per-asset bit order."""
    values = (CODES @ weights.T).T + offsets[:, None]
    return CODES[(values == holding[:, None]).argmax(axis=1)].reshape(-1).astype(np.int8)

class PartitionSASolver:
    def __init__(self, stages, seed, integer_polish=None, mu_ladder=(0.25, 1.0, 4.0), quantum=None):
        self.stages = stages
        self.rng = np.random.default_rng(seed)
        self.integer_polish = integer_polish
        self.mu_ladder = tuple(mu_ladder)
        self.stage_calls = 0
        self.max_qubo_vars = 0
        self.mu_raised = 0
        self.quantum = quantum            # quantum_reheat.QuantumReheat | None
        self._stalled = {}                # block key -> did not improve last time

    @staticmethod
    def _unit_scale(ev, L, S, days, G):
        g = len(G)
        V1 = np.concatenate([np.eye(g, dtype=np.int64), -np.eye(g, dtype=np.int64)])
        c1, _ = ev.costs(L, S, days, G, V1)
        c0, _ = ev.costs(L, S, days, G, np.zeros((1, g), np.int64))
        return float(np.mean(np.abs(c1 - c0[0]))) or 1.0

    def __call__(self, ev, L, S, days, G):
        if len(days) != 1 or ev.model.ub != 3:
            raise ValueError("Signed SA blocks require one day and position bounds [-3, 3]")
        current = L[days[0], G] - S[days[0], G]
        best = current
        best_cost = ev.costs(L, S, days, G, current[None])[0][0]
        cur_cost = best_cost
        unit = None
        q = self.quantum
        block_key = (tuple(days), np.asarray(G).tobytes()) if q is not None else None

        for k, step in enumerate((0.0,) + self.mu_ladder):
            if k >= 1:
                if unit is None:
                    unit = self._unit_scale(ev, L, S, days, G)
                self.mu_raised += 1
            mu = step * (unit or 0.0)
            bqm, weights, offsets, _ = block_qubo(ev, L, S, days, G, mu=mu)
            self.max_qubo_vars = max(self.max_qubo_vars, bqm.num_variables)
            labels = list(range(bqm.num_variables))
            best_bits = encode_holding(best, weights, offsets)
            starts = best_bits[None]
            low_feasible = False

            def absorb(bits):
                """Evaluate samples; update the incumbent. Returns the feasibility mask."""
                nonlocal best, best_cost, best_bits
                holdings = decode(bits, weights, offsets)
                costs, feasible = ev.costs(L, S, days, G, holdings)
                if feasible.any():
                    j = int(np.argmin(np.where(feasible, costs, np.inf)))
                    if costs[j] < best_cost - 1e-9:
                        best, best_cost, best_bits = holdings[j], costs[j], bits[j]
                return feasible

            seq, i, quantum_next = list(self.stages), 0, False
            while i < len(seq):
                stage = seq[i]
                i += 1
                was_quantum, quantum_next = quantum_next, False
                cost_before = best_cost
                bits = np.asarray(stage(bqm, starts, self.rng))
                self.stage_calls += 1
                if bits.ndim != 2 or bits.shape[1] != bqm.num_variables or not np.all(
                    (bits == 0) | (bits == 1)
                ):
                    raise ValueError("Block stages must return a (reads, n_bits) binary array")
                if not len(bits):
                    continue
                bits = bits.astype(np.int8, copy=False)
                feasible = absorb(bits)
                energies = bqm.energies((bits, labels))
                jl = int(np.argmin(energies))
                low_feasible = bool(feasible[jl])
                low_energy = bits[jl]
                starts = best_bits[None] if np.array_equal(best_bits, low_energy) else np.stack(
                    (best_bits, low_energy)
                )

                if q is None:
                    continue
                if i >= 2:                                   # a reheat stage just ran
                    improved = best_cost < cost_before - 1e-9
                    if was_quantum:
                        q.stats.reheats_after_quantum += 1
                        q.stats.improvements_after_quantum_reheat += int(improved)
                    else:
                        q.stats.reheats_after_classical += 1
                        q.stats.improvements_after_classical_reheat += int(improved)
                elif len(seq) > 1 and q.policy.mode != "off":  # stage 1 done, reheat follows
                    if q.stats.qaoa_calls >= q.policy.budget:
                        q.stats.skipped_budget += 1
                        continue
                    reason = q.policy.reason(best_cost, cur_cost, energies,
                                             self._stalled.get(block_key, False))
                    if reason is None:
                        continue
                    qbits = np.asarray(q.stage(bqm, starts, self.rng))
                    if qbits.ndim != 2 or qbits.shape[1] != bqm.num_variables:
                        raise ValueError("QAOA seed stage must return (reads, n_bits)")
                    if not len(qbits):                       # skipped (too small/large)
                        continue
                    qbits = qbits.astype(np.int8, copy=False)
                    q.stats.triggers[reason] += 1
                    before = best_cost
                    absorb(qbits)                            # QAOA samples may become incumbent
                    q.stats.qaoa_direct_improvements += int(best_cost < before - 1e-9)
                    seeds = q.stage.select_seeds(
                        qbits, bqm.energies((qbits, labels)), q.stage.n_seeds)
                    if len(seeds):
                        seeds = seeds[(seeds != best_bits).any(axis=1)]
                    starts = np.vstack([best_bits[None], seeds]).astype(np.int8)
                    if isinstance(seq[i], AnnealStage):
                        seq[i] = dataclasses.replace(seq[i], start_temperature_scale=q.reheat_scale)
                    quantum_next = True

            if low_feasible:
                break

        if q is not None:
            self._stalled[block_key] = not (best_cost < cur_cost - 1e-9)

        if self.integer_polish is not None:
            chains, sweeps = self.integer_polish
            form = block_form(ev, L, S, days, G)
            best = solve_sa(form, best, self.rng, chains=chains, sweeps=sweeps)
        return BlockResult(best)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ
    p.add_argument("--instance", default=env.get("AUTORUNNER_INSTANCE", "po_a010_t10_orig"))
    p.add_argument("--instance-dir", default=env.get("AUTORUNNER_INSTANCE_DIR"))
    p.add_argument("--assets", type=int, default=int(env["AUTORUNNER_ASSETS"]) if env.get("AUTORUNNER_ASSETS") else None)
    p.add_argument("--days", type=int, default=int(env["AUTORUNNER_DAYS"]) if env.get("AUTORUNNER_DAYS") else None)
    p.add_argument("--lambda", dest="lam", default=env.get("AUTORUNNER_LAMBDA", "1e-4"))
    p.add_argument("--budget", type=int, default=int(env.get("AUTORUNNER_BUDGET", "4")))
    p.add_argument("--cloud", type=parse_bool, nargs="?", const=True,
                   default=parse_bool(env.get("AUTORUNNER_CLOUD", "false")),
                   help="Accepted for compatibility; always runs locally")
    p.add_argument("--solver-seed", type=int,
                   default=int(env["AUTORUNNER_SOLVER_SEED"]) if env.get("AUTORUNNER_SOLVER_SEED") else None,
                   help="Single RNG seed override (default: run 1, 2, 3)")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                   help="Accepted for compatibility; this solver uses CPU")
    p.add_argument("--gpu-device", type=int, help="Accepted for compatibility; this solver uses CPU")
    p.add_argument("--group-size", type=positive_int, default=6, help="Maximum assets per block (3 bits each)")
    p.add_argument("--group-method", choices=("neighbourhood", "random", "contiguous"), default="neighbourhood")
    p.add_argument("--block-sweeps", type=positive_int, default=4, help="Maximum passes through all blocks per seed")
    p.add_argument("--patience", type=positive_int, default=2, help="Stop after this many non-improving passes")
    p.add_argument("--sa-reads", type=positive_int, default=24)
    p.add_argument("--sa-sweeps", type=positive_int, default=120, help="SA sweeps per read per stage")
    p.add_argument("--sa-backend", choices=("auto", "numpy", "dwave"), default="dwave",
                   help="Auto prefers dwave-samplers and falls back to NumPy (default: auto)")
    p.add_argument("--reheats", type=int, default=1, help="Extra warm-started stages per block; 0 disables reheats")
    p.add_argument("--reheat-temperature-scale", type=float, default=2.0, help="Initial temperature multiplier for reheats")
    p.add_argument("--integer-polish", action="store_true", help="Finish each block with constrained integer SA")
    p.add_argument("--block-solver", choices=("sa", "exact"), default="sa",
               help="exact enumerates all 7**g holdings; use for --group-size <= 5")
    p.add_argument("--mu-ladder", type=float, nargs="*", default=[0.25, 1.0, 4.0],
                   help="unit-price multipliers tried after mu=0; empty disables")
    p.add_argument("--polish-chains", type=positive_int, default=8)
    p.add_argument("--polish-sweeps", type=positive_int, default=200)
    p.add_argument("--init", default=None, help="warm-start .sol file")
    p.add_argument("--time", type=float, default=0.0, help="iterated_search seconds (0 = off)")
    p.add_argument("--kick-blocks", type=positive_int, default=3)
    p.add_argument("--kick-size", type=positive_int, default=5)
    p.add_argument("--day-swap", type=float, default=0.0,
               help="seconds of day_swap.run after block passes (0 = off)")
    p.add_argument("--quantum-reheat", choices=("off", "always", "stuck"), default="off")
    p.add_argument("--stuck-frac", type=float, default=0.5)
    p.add_argument("--quantum-min-qubits", type=positive_int, default=24)
    p.add_argument("--quantum-max-qubits", type=positive_int, default=26)
    p.add_argument("--quantum-budget", type=positive_int, default=20, help="max QAOA runs per seed")
    p.add_argument("--quantum-reheat-scale", type=float, default=1.25)
    p.add_argument("--quantum-layers", type=positive_int, default=2)
    p.add_argument("--quantum-iters", type=int, default=15, help="0 = sample TQA angles unoptimised")
    p.add_argument("--quantum-shots", type=positive_int, default=4000)
    p.add_argument("--quantum-top", type=positive_int, default=64)
    p.add_argument("--quantum-seeds", type=positive_int, default=8)
    p.add_argument("--quantum-init", choices=("tqa", "random"), default="tqa")
    p.add_argument("--quantum-tqa-dt", type=float, default=1.0)
    p.add_argument("--quantum-null-control", action="store_true",
                   help="bench only: replace QAOA samples with uniform random bitstrings")    
    return p


def resolve_solution_path(instance: str, base: str, budget: int, lam: str) -> Path:
    if re.search(r"_b\d+_l", instance):
        return Path(f"{instance}.sol")
    lam_tag = lam.replace("-0", "-").replace("+0", "+")
    return Path(f"{base}_b{budget:03d}_l{lam_tag}.sol")


def load_instance(instance: str, instance_dir: Path | None) -> qt.PortfolioInstance:
    base = re.sub(r"_b\d+_l.*$", "", instance.strip())
    if instance_dir is not None and instance_dir.resolve().name.startswith("po_"):
        base = instance_dir.resolve().name
    for directory in (instance_dir, SCRIPT_DIR / "instances" / base):
        if directory is not None and all(
            (directory / name).is_file() for name in ("stock_prices.txt.gz", "covariance_matrices.txt.gz")
        ):
            return load_instance_dir(directory, base)
    return qt.load_portfolio_instance(base)

def resolve_init_path(name: str, sol_name: str) -> Path:
    """Accept a path as given, or a bare filename found in handin/ (next to this script)."""
    p = Path(name).expanduser()
    for cand in (p, Path.cwd() / "handin" / p.name, SCRIPT_DIR / p, SCRIPT_DIR / "handin" / p.name):
        if cand.is_dir():
            cand = cand / sol_name
        if cand.is_file():
            return cand
    if p.suffix == ".sol":
        raise FileNotFoundError(f"--init {name!r} not found")
    return None

def main(argv: list[str] | None = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)
    if args.reheats < 0 or not (1 <= args.reheat_temperature_scale < float("inf")):
        p.error("--reheats must be nonnegative and --reheat-temperature-scale must be finite and at least 1")
    if args.solver_seed is not None and args.solver_seed < 0:
        p.error("--solver-seed must be nonnegative")

    started = time.perf_counter()
    backend = "numpy" if args.sa_backend == "numpy" else "dwave"
    if backend == "dwave":
        try:
            load_dwave_sampler()
        except (ImportError, OSError) as exc:
            if args.sa_backend == "dwave":
                p.error(f"D-Wave sampler unavailable ({exc}). Run "
                        f'"{sys.executable}" -m pip install -U dwave-samplers '
                        "or choose --sa-backend numpy.")
            backend = "numpy"
            print(f"D-Wave sampler unavailable ({exc}); using NumPy.", flush=True)
    print(f"SA backend: {backend} on CPU; up to {args.block_sweeps} block passes per seed, "
          f"{args.sa_reads} reads × {args.sa_sweeps} sweeps per stage, {args.reheats} reheats.",
          flush=True)
    solvers = []
    inst = load_instance(args.instance, Path(args.instance_dir) if args.instance_dir else None)
    if args.assets is not None and inst.n_assets != args.assets:
        raise ValueError(f"Asset count mismatch for {inst.name}: expected {args.assets}, got {inst.n_assets}")
    if args.days is not None and inst.n_periods != args.days:
        raise ValueError(f"Day/period count mismatch for {inst.name}: expected {args.days}, got {inst.n_periods}")
    if args.cloud:
        print("--cloud was requested; partitioned SA still runs locally on CPU.", flush=True)
    if args.quantum_reheat != "off":
        if args.block_solver != "sa" or args.reheats < 1:
            p.error("--quantum-reheat needs --block-solver sa and --reheats >= 1")
        if args.quantum_iters == 0 and args.quantum_init != "tqa":
            p.error("--quantum-iters 0 requires --quantum-init tqa")
    if args.block_solver == "exact":
        if args.group_size > 5:
            p.error("--block-solver exact is only sensible for --group-size <= 5")
        make_solver = lambda seed: solve_block_exact
    else:
        make_quantum = None
        if args.quantum_reheat != "off":
            from quantum_reheat import QuantumReheat   # imports Divi lazily, only when used
            make_quantum = lambda seed: QuantumReheat.from_args(args, seed)
        polish = (args.polish_chains, args.polish_sweeps) if args.integer_polish else None
        make_solver = lambda seed: PartitionSASolver(
            stages, seed, polish, tuple(args.mu_ladder),
            quantum=make_quantum(seed) if make_quantum else None)
    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    if not model.score({}).feasible:
        raise ValueError("The empty portfolio is infeasible for this budget; cannot initialize the block solver")
    stages = (AnnealStage(args.sa_reads, args.sa_sweeps, backend=backend),) + (
        AnnealStage(args.sa_reads, args.sa_sweeps, args.reheat_temperature_scale,
                    restart_all_reads=True, backend=backend),
    ) * args.reheats
    groups = grouping_fn(model, args.group_size, args.group_method)
    L0 = S0 = None
    if args.init:
        sol_name = resolve_solution_path(args.instance, inst.name, args.budget, args.lam).name
        init_path = resolve_init_path(args.init, sol_name)
        header, init_pos = qt.read_portfolio_solution(init_path)
        try:
            hdr_lam = Fraction(header["lambda"])
        except (KeyError, ValueError):
            hdr_lam = None
        if (header.get("instance") not in (None, inst.name)
                or int(header.get("budget", args.budget)) != args.budget
                or (hdr_lam is not None and hdr_lam != model.lam)):
            p.error(f"--init header (instance={header.get('instance')}, B={header.get('budget')}, "
            f"lambda={header.get('lambda')}) does not match "
            f"{inst.name}, B={args.budget}, lambda={args.lam}")
        L0, S0 = (a.astype(np.int64) for a in model.to_arrays(init_pos))
        init_score = model.score_arrays(L0, S0)
        if not init_score.feasible:
            raise ValueError(f"--init {args.init} is infeasible: {init_score.violations}")
        print(f"warm start from {args.init}: {init_score.objective}", flush=True)
    
    seeds = DEFAULT_SEEDS if args.solver_seed is None else (args.solver_seed,)
    seed_runs = []
    best = None
    max_vars = 0
    for seed in seeds:
        t0 = time.perf_counter()
        solver = make_solver(seed)
        solvers.append(solver)
        if args.time > 0:      # includes its own clamp_and_iterate pass
            result = iterated_search(model, groups, L=L0, S=S0, time_limit=args.time,
                                     kick_blocks=args.kick_blocks, kick_size=args.kick_size,
                                     solver=solver, seed=seed, verbose=False)
            result.setdefault("blocks", 0); result.setdefault("accepted", result["improvements"])
            result["history"] = [t[1] for t in result["trace"]]
        else:
            result = clamp_and_iterate(model, groups, solver=solver, L=L0, S=S0,
                                       max_sweeps=args.block_sweeps, patience=args.patience,
                                       sweep_offset=seed, verbose=False)
        if args.day_swap > 0:
            ds = day_swap.run(model, result["L"], result["S"], args.day_swap, seed, verbose=False)
            if ds["score"] < result["score"]:
                result.update(score=ds["score"], L=ds["L"], S=ds["S"])
        qsum = None
        if args.quantum_reheat != "off":
            from quantum_reheat import QuantumStats
            qsum = QuantumStats()
            for s_ in solvers:
                if getattr(s_, "quantum", None) is not None:
                    qsum.merge(s_.quantum.stats)
        stage_calls = getattr(solver, "stage_calls", 0)
        max_vars = max(max_vars, getattr(solver, "max_qubo_vars", 0))
        score = int(result["score"])
        seed_runs.append({
            "seed": seed, "objective": score, "feasible": True,
            "sub_programs": result["blocks"], "circuits": 0,
            "duration_seconds": round(time.perf_counter() - t0, 3),
            "stage_calls": stage_calls, "accepted_blocks": result["accepted"],
            "block_sweeps": len(result["history"]) - 1,
        })
        print(f"seed {seed}: {score} ({result['blocks']} block solves, {stage_calls} sampling stages)", flush=True)
        if best is None or score < best[0]:
            best = (score, seed, result["L"], result["S"])

    assert best is not None
    best_obj, best_seed, L, S = best
    positions = model.to_positions(L, S)
    score = model.score(positions)
    if not score.feasible or score.objective != best_obj:
        raise RuntimeError("Final portfolio did not pass official feasibility and objective checks")
    sol_path = resolve_solution_path(args.instance, inst.name, args.budget, args.lam)
    written = model.write_solution(sol_path, positions, comment="QOBLIB: partitioned simulated annealing")
    payload = {
        "score": int(written.objective),
        "extra": {
            "instance": args.instance, "base_instance": inst.name,
            "assets": inst.n_assets, "days": inst.n_periods, "budget": args.budget, "lambda": args.lam,
            "cloud": False, "cloud_requested": args.cloud, "gpu_used": False,
            "gpu_backend": None, "gpu_device": None, "device": "cpu",
            "simulator_backend": f"{backend}_simulated_annealing",
            "backend_detail": "Local CPU; no quantum circuits executed",
            "feasible": bool(written.feasible), "breakdown": written.breakdown,
            "violations": list(written.violations), "solution_file": sol_path.name,
            "routine": "partitioned_sa", "cluster_size": 3 * args.group_size,
            "n_qubo_vars": None, "max_block_qubo_vars": max_vars,  # no global QUBO built
            "best_seed": best_seed, "total_circuits": 0, "seed_runs": seed_runs,
            "total_duration_seconds": round(time.perf_counter() - started, 3),
            "sa": {
                "backend": backend, "reads": args.sa_reads, "sweeps": args.sa_sweeps, "reheats": args.reheats,
                "reheat_temperature_scale": args.reheat_temperature_scale,
                "group_method": args.group_method, "group_size_assets": args.group_size,
                "max_block_sweeps": args.block_sweeps, "patience": args.patience,
                "integer_polish": args.integer_polish,
                "quantum_reheat": args.quantum_reheat,
                **({"quantum": {**qsum.as_dict(), "reheat_scale": args.quantum_reheat_scale,
                                "layers": args.quantum_layers, "iters": args.quantum_iters,
                                "init": args.quantum_init, "budget_per_seed": args.quantum_budget}}
                   if qsum is not None else {}),
            },
        },
    }
    print(f"Wrote {sol_path}: {written}")
    print(f"AUTORUNNER_RESULT: {json.dumps(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
