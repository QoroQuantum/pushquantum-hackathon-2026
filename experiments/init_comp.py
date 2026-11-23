#!/usr/bin/env python
"""Initialisation comparison: a worker for ``auto_runner.py`` (one instance per call).

For the instance it is given, runs 5 initialisations × 3 seeds with everything else fixed:

    partitioned QAOA (p = 4):  qaoa_random, qaoa_tqa, iter_chebyshev
    PCE (whole problem):       pce_random, pce_progressive        (Divi's own PCE initialisations)

Run through the runner, then plot:

    python auto_runner.py experiments/init_comp.py --tests local_small_risk_tests
    python experiments/init_comp.py --plot AutoData/local_small_risk_tests_<HHMM>          # run dir or its JSON

Where the circuits run, decided per instance:
    ≤ 5000 variables: local (QAOA on CPU, PCE on GPU if present), or Qoro cloud with --cloud true.
    > 5000 variables (a200, a400): always Qoro cloud. Repair / local search stay local, so for these
    only the best repaired candidate is local-searched (max 20 rounds), and PCE uses 4-qubit
    correlators (≤ 28 qubits) instead of 3-qubit ones.

Environment variables (the runner cannot pass worker arguments):
    INIT_COMP_QUICK=1            smoke test: population 4, 1 iteration, p = 1, 1 seed
    INIT_COMP_WORKERS=N          run N of the 15 runs of an instance at the same time (default 1)
    INIT_COMP_GPU_MIN_QUBITS=16  local circuits use the GPU only between these qubit counts,
    INIT_COMP_GPU_MAX_QUBITS=28  CPU otherwise (12-qubit QAOA blocks: CPU; 20-qubit PCE: GPU)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent  # repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=SyntaxWarning)

import qoblib_tools as qt  # noqa: E402  (the referee; never edited)

DIVI_VERSION = "0.16.2"  # TQA / linear-ramp angles use a private Divi hook: pin the version
RESULT_MARKER = "AUTORUNNER_RESULT "

QAOA_VARIANTS = ["qaoa_random", "qaoa_tqa", "iter_chebyshev"]
PCE_VARIANTS = ["pce_random", "pce_progressive"]
VARIANTS = QAOA_VARIANTS + PCE_VARIANTS

BIG_VARS = 5000          # above this: circuits on Qoro's cloud, lighter polishing, 4-body PCE
MAX_LOCAL_QUBITS = 28    # 2^28 amplitudes = 4 GB: fits a 6 GB GPU
# Local circuits use the GPU only inside this qubit window, CPU otherwise. Measured: 12-qubit QAOA
# blocks are faster on CPU, 20-qubit PCE 18x faster on GPU; above 28 qubits a 6 GB GPU runs out.
GPU_MIN_QUBITS = int(os.environ.get("INIT_COMP_GPU_MIN_QUBITS", "16"))
GPU_MAX_QUBITS = int(os.environ.get("INIT_COMP_GPU_MAX_QUBITS", "28"))

# Everything except the initialisation is fixed (notebook 01 §1.5 / §1.7, with p = 4).
CONFIG = {
    "penalty_weight_W": 100.0,
    "normalize_bqm": True,
    "shots": 10_000,
    "population": 50,                 # MonteCarloOptimizer population = rows of initial_params
    "iterations": 10,
    "p": 4,
    "cluster_size": 12,
    "beam_width": 10,
    "n_partition_candidates": 5,
    "top_polish": 10,                 # candidates repaired (+ local-searched)
    "big_local_search_rounds": 20,    # > BIG_VARS: local search on the best repaired candidate only
    "iter_max_depth": 4,
    "iter_schedule": {1: 6, 2: 5, 3: 4, 4: 3},  # Σ depth·iters = 40 = 4 × 10: same cost as p = 4
    "pce_layers": 3,
    "pce_top": 20,
    "progressive_max_steps": 4,
    "progressive_iters_per_step": 3,
    "tqa_dt_range": [0.05, 8.0],          # geomspace over the population, β < 0
    "seeds": [1, 2, 3],
}
QUICK = {
    "population": 4, "iterations": 1, "p": 1, "iter_max_depth": 2, "iter_schedule": {1: 1, 2: 1},
    "progressive_max_steps": 1, "progressive_iters_per_step": 1, "seeds": [1], "shots": 1000,
    "top_polish": 3, "pce_top": 5, "big_local_search_rounds": 2,
}


# ── instance, QUBO, polishing ──────────────────────────────────────────────────────────


def load_instance_dir(path: Path, name: str) -> qt.PortfolioInstance:
    """Parse ``stock_prices.txt.gz`` / ``covariance_matrices.txt.gz`` like ``qt.load_portfolio_instance``."""
    symbols, index, entries = [], {}, []
    for day, sym, price in qt._data_rows(path / "stock_prices.txt.gz"):
        if sym not in index:
            index[sym] = len(symbols)
            symbols.append(sym)
        entries.append((index[sym], int(day), Fraction(price)))
    n_periods = max(t for _, t, _ in entries) + 1
    prices = [[None] * n_periods for _ in symbols]
    for i, t, p in entries:
        prices[i][t] = p
    cov = {}
    for day, s1, s2, value in qt._data_rows(path / "covariance_matrices.txt.gz"):
        if s1 in index and s2 in index and int(day) < n_periods:
            cov[index[s1], index[s2], int(day)] = Fraction(value)
    return qt.PortfolioInstance(name, symbols, n_periods, prices, cov)


def penalised(q: qt.PortfolioQUBO, weight: float, normalize: bool):
    """objective + weight·penalty as one BQM (notebook 01 §1.2)."""
    bqm = q.objective.copy()
    penalty = q.penalty.copy()
    penalty.scale(weight)
    bqm.update(penalty)
    if normalize:
        bqm.normalize()
    return bqm


def polish(model, q, samples, cfg: dict, big: bool):
    """Decode the top candidates and make them feasible and locally optimal; best (positions, score, mode)."""
    decoded = [q.decode(s) for s in list(samples)[: cfg["top_polish"]]]
    if not big:
        candidates, mode = [model.local_search(p) for p in decoded], f"local_search top {len(decoded)}"
    else:  # local search is O(moves × score) per round: ~1.8 min per round at a400
        repaired = [model.repair(p) for p in decoded]
        best = min(repaired, key=lambda p: model.score(p).objective)
        rounds = cfg["big_local_search_rounds"]
        candidates = [model.local_search(best, max_rounds=rounds)]
        mode = f"repair top {len(decoded)}, local_search best ({rounds} rounds)"
    best = min(candidates, key=lambda p: model.score(p).objective)
    return best, model.score(best), mode


def raw_best(model, q, samples) -> tuple[int | None, int]:
    """Best objective among the feasible raw samples (before repair) and how many were feasible."""
    scores = [model.score(q.decode(s)) for s in samples]
    feasible = [s.objective for s in scores if s.feasible]
    return (min(feasible) if feasible else None), len(feasible)


def pce_order(n_vars: int) -> tuple[int, int]:
    """Smallest MultiBasis correlator order k ∈ {3, 4} with ≤ MAX_LOCAL_QUBITS qubits: (k, qubits).
    MultiBasisEncoding(order=k) on n qubits holds 3·C(n, k) variables."""
    for k in (3, 4):
        n = k
        while 3 * math.comb(n, k) < n_vars:
            n += 1
        if n <= MAX_LOCAL_QUBITS or k == 4:
            return k, n
    raise AssertionError


def make_backend(cloud: bool, shots: int, seed: int, gpu: bool = False):
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.backends.runners._maestro import MaestroConfig

    if cloud:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro"))
    # GPU: 18x faster for 20-qubit PCE, but slower than CPU for the 12-qubit QAOA partitions.
    config = MaestroConfig(simulator_type="Gpu") if gpu else None  # type: ignore[call-arg]  (dynamic options)
    backend = MaestroSimulator(shots=shots, maestro_config=config) if config else MaestroSimulator(shots=shots)
    backend.set_seed(seed)  # without it the shot sampling is unseeded and runs are not reproducible
    return backend


# ── initial parameters ─────────────────────────────────────────────────────────────────


def _ramp(p: int, d_gamma: float, d_beta: float) -> np.ndarray:
    """Linear ramp, interleaved per layer as Divi expects: [γ0, β0, γ1, β1, …]."""
    s = (np.arange(p) + 0.5) / p
    row = np.empty(2 * p)
    row[0::2] = s * d_gamma
    row[1::2] = (1 - s) * d_beta
    return row


def qaoa_init_rows(variant: str, p: int, population: int, cfg: dict):
    """(rows, labels) for TQA; (None, None) = Divi's own uniform initialisation."""
    if variant == "qaoa_tqa":  # β < 0: Divi's mixer is +ΣX on |+⟩ (all 70 a010 sub-programs chose β < 0)
        dts = np.geomspace(*cfg["tqa_dt_range"], population)
        return np.array([_ramp(p, d, -d) for d in dts]), [{"dt": float(d)} for d in dts]
    return None, None


def chosen_rows(program, rows, labels) -> list:
    """Label of the starting row with the lowest loss in the first Monte Carlo iteration."""
    if rows is None:
        return []
    try:
        first = program.losses_history[0]
        evaluated = np.atleast_2d(program.param_history("all_evaluated")[0])
    except Exception:
        return []
    x = evaluated[int(np.nanargmin([first[str(i)] for i in range(len(first))]))]
    return [labels[int(np.argmin(np.linalg.norm(rows - x, axis=1)))]]


# ── one run ────────────────────────────────────────────────────────────────────────────


def run_qaoa(variant: str, bqm, seed: int, cfg: dict, backend):
    import hybrid
    from divi.qprog.algorithms import InterpolationStrategy
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    problem = BinaryOptimizationProblem(
        bqm, decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=cfg["cluster_size"])),
        composer=hybrid.SplatComposer(), local_search=True)
    kwargs = {}
    if variant.startswith("iter_"):
        routine, n_layers = "iterative_qaoa", cfg["iter_max_depth"]
        schedule = dict(cfg["iter_schedule"])
        kwargs = dict(strategy=InterpolationStrategy[variant.split("_", 1)[1].upper()],
                      max_iterations_per_depth=lambda d, s=schedule: s[d], convergence_threshold=None)
    else:
        routine, n_layers = "qaoa", cfg["p"]
    ensemble = PartitioningProgramEnsemble(
        problem=problem, quantum_routine=routine, n_layers=n_layers,
        optimizer=MonteCarloOptimizer(population_size=cfg["population"]),
        max_iterations=cfg["iterations"], backend=backend, seed=seed, **kwargs)
    rows, labels = qaoa_init_rows(variant, cfg["p"], cfg["population"], cfg)
    ensemble.create_programs()
    if rows is not None:
        # The ensemble starts every sub-program with run() and no initial_params; each VQA then asks
        # _initialize_param_sets() for its starting population, so override that per program.
        for prog in ensemble.programs.values():
            if rows.shape != tuple(prog.get_expected_param_shape()):
                raise ValueError(f"init rows {rows.shape} != expected {prog.get_expected_param_shape()}")
            prog._initialize_param_sets = (lambda r=rows: r.copy())
    ensemble.run()
    tops = ensemble.get_top_solutions(n=cfg["top_polish"], strategy=BeamSearchStrategy(
        beam_width=cfg["beam_width"], n_partition_candidates=cfg["n_partition_candidates"]))
    samples = [dict(zip(bqm.variables, solution)) for solution, _ in tops]
    chosen = [c for prog in ensemble.programs.values() for c in chosen_rows(prog, rows, labels)]
    return samples, chosen, {"circuits": ensemble.total_circuit_count, "backend_time_s": ensemble.total_run_time,
                             "n_subprograms": len(ensemble.programs), "n_qubits": cfg["cluster_size"]}


def run_pce(variant: str, bqm, seed: int, cfg: dict, backend, order: int):
    """Divi's own PCE initialisations: pce_random = uniform [0, 2π); pce_progressive = ProgressivePCE."""
    from divi.qprog.algorithms import PCE, MultiBasisEncoding, ProgressivePCE
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    common = dict(encoding=MultiBasisEncoding(order=order), n_layers=cfg["pce_layers"],
                  optimizer=MonteCarloOptimizer(population_size=cfg["population"]), backend=backend, seed=seed)
    problem = BinaryOptimizationProblem(bqm)
    if variant == "pce_progressive":
        prog = ProgressivePCE(problem, max_steps=cfg["progressive_max_steps"],
                              max_iterations_per_step=cfg["progressive_iters_per_step"], **common)
    else:  # objective=None → the encoding's default SoftObjective(alpha = 1.5·q^(k//2))
        prog = PCE(problem, max_iterations=cfg["iterations"], **common)
    prog.run()
    samples = [s.bitstring for s in prog.get_top_solutions(n=cfg["pce_top"], min_prob=0.0, sort_by="energy")]
    return samples, [], {"circuits": prog.total_circuit_count, "backend_time_s": prog.total_run_time,
                         "n_qubits": prog.n_qubits, "pce_order": order}


# ── worker ─────────────────────────────────────────────────────────────────────────────


def _credits():
    try:
        from divi.backends import QoroService

        return QoroService().get_credit_balance()
    except Exception as err:
        return {"error": str(err)}


def _summarise_chosen(chosen: list[dict]) -> str:
    if not chosen:
        return "n/a (Divi's own initialisation)"
    medians = "; ".join(f"{k}: median {np.median([c[k] for c in chosen]):.4g}" for k in chosen[0])
    common = Counter(tuple((k, round(v, 4)) for k, v in c.items()) for c in chosen).most_common(5)
    return f"{len(chosen)} sub-program(s); {medians}\n      most common: " + ", ".join(
        f"{dict(k)}×{n}" for k, n in common)


def device_for(qubits: int, ctx: dict) -> str:
    """Where a circuit of ``qubits`` qubits runs: cloud, or locally the GPU inside its useful window."""
    if ctx["cloud"]:
        return "cloud"
    return "gpu" if ctx["has_gpu"] and GPU_MIN_QUBITS <= qubits <= GPU_MAX_QUBITS else "cpu"


def run_one(ctx: dict, variant: str, seed: int) -> dict:
    """One (variant, seed) run: optimise, decode, polish, write its .sol; returns the run record."""
    model, q, bqm, cfg = ctx["model"], ctx["q"], ctx["bqm"], ctx["cfg"]
    is_qaoa = variant in QAOA_VARIANTS
    rec = {"variant": variant, "solver": "qaoa_partitioned" if is_qaoa else "pce_whole", "seed": seed,
           "objective": None, "feasible": None, "raw_best_feasible": None, "raw_n_feasible": None,
           "circuits": None, "n_qubits": None, "device": None, "polish": None, "backend_time_s": None,
           "wall_s": None, "sol": None, "chosen": [], "error": None}
    t0 = time.time()
    try:
        device = device_for(cfg["cluster_size"] if is_qaoa else ctx["pce_qubits"], ctx)
        rec["device"] = device

        def solve(on_gpu):
            backend = make_backend(ctx["cloud"], cfg["shots"], seed, on_gpu)
            if is_qaoa:
                return run_qaoa(variant, bqm, seed, cfg, backend)
            return run_pce(variant, bqm, seed, cfg, backend, ctx["pce_k"])

        try:
            samples, chosen, stats = solve(device == "gpu")
        except Exception as err:
            if device != "gpu":
                raise
            rec["device"] = f"cpu (gpu failed: {type(err).__name__})"
            samples, chosen, stats = solve(False)
        rec.update(stats, chosen=chosen)
        rec["raw_best_feasible"], rec["raw_n_feasible"] = raw_best(model, q, samples)
        positions, score, rec["polish"] = polish(model, q, samples, cfg, ctx["big"])
        sol = Path(ctx["out_dir"]) / f"{variant}__seed{seed}.sol"
        model.write_solution(sol, positions, comment=f"init_comp {variant} seed {seed}")
        rec.update(objective=int(score.objective), feasible=bool(score.feasible), sol=sol.as_posix())
    except Exception as err:
        rec["error"] = f"{type(err).__name__}: {err}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["wall_s"] = round(time.time() - t0, 2)
    status = rec["error"] or f"{rec['objective']} ({'feasible' if rec['feasible'] else 'INFEASIBLE'})"
    print(f"  {variant:16s} seed {seed}: {status}  [{rec['wall_s']}s, {rec['circuits']} circuits, "
          f"{rec['device']}]", flush=True)
    return rec


_CTX: dict | None = None


def _build_problem(spec: dict) -> dict:
    """Model, QUBO and BQM from picklable parameters (PortfolioQUBO itself cannot be pickled)."""
    model = qt.PortfolioModel(load_instance_dir(Path(spec["data_dir"]), spec["base"]),
                              budget=spec["budget"], lam=spec["lam"])
    q = qt.PortfolioQUBO(model)
    return {"model": model, "q": q, "bqm": penalised(q, spec["cfg"]["penalty_weight_W"], spec["cfg"]["normalize_bqm"])}


def _pool_init(spec: dict) -> None:
    """Each worker process rebuilds the problem once, then runs its share of the jobs."""
    global _CTX
    warnings.filterwarnings("ignore")
    _CTX = {**spec, **_build_problem(spec)}


def _run_in_pool(job: tuple[str, int]) -> dict:
    assert _CTX is not None, "pool initializer did not run"
    return run_one(_CTX, *job)


def _workers(args, n_assets: int, n_days: int, big: bool, n_jobs: int) -> int:
    """Runs in parallel inside this instance: --workers / INIT_COMP_WORKERS (default 1), capped by
    the number of runs and by free memory; big instances stay sequential (memory, cloud jobs)."""
    wanted = args.workers or int(os.environ.get("INIT_COMP_WORKERS", "1") or 1)
    if wanted <= 1 or big:
        return 1
    try:
        import psutil

        per_run_mb = 400 + 3.6e-5 * (6 * n_assets) ** 2 * n_days  # same model as the runner
        wanted = min(wanted, max(1, int(0.8 * psutil.virtual_memory().available / 2**20 // per_run_mb)))
    except ImportError:
        pass
    return max(1, min(wanted, n_jobs))


def run_worker(args) -> int:
    from importlib.metadata import version

    if (divi := version("qoro-divi")) != DIVI_VERSION:
        raise SystemExit(f"init_comp.py needs qoro-divi {DIVI_VERSION} (found {divi})")
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")  # the worker runs inside AutoData/…: load the API key explicitly
    except ImportError:
        pass

    quick = args.quick or os.environ.get("INIT_COMP_QUICK") == "1"
    cfg = {**CONFIG, **(QUICK if quick else {})}
    if args.seeds:
        cfg["seeds"] = args.seeds
    variants = args.variants or VARIANTS

    m = re.match(r"^(po_a\d+_t\d+_(?:orig|s\d\d))_b\d+_l.+$", args.instance)
    if not m:
        raise SystemExit(f"cannot parse --instance {args.instance!r}")
    spec = {"data_dir": str(Path(args.data_dir).resolve()), "base": m.group(1), "budget": args.budget,
            "lam": args.lam, "cfg": cfg}
    built = _build_problem(spec)
    model, q = built["model"], built["q"]

    big = q.n_vars > BIG_VARS
    cloud = args.cloud.lower() in ("1", "true", "yes") or big
    has_gpu = args.gpu != "off" and (args.gpu == "on" or shutil.which("nvidia-smi") is not None)
    k, pce_qubits = pce_order(q.n_vars)
    out_dir = Path(args.instance).resolve()
    out_dir.mkdir(exist_ok=True)
    spec.update(cloud=cloud, has_gpu=has_gpu, big=big, pce_k=k, pce_qubits=pce_qubits, out_dir=str(out_dir))
    ctx = {**spec, **built}
    workers = _workers(args, len(model.symbols), model.n_periods, big, len(variants) * len(cfg["seeds"]))
    routing = {
        "n_vars": q.n_vars, "big_instance": big, "runner_cloud_flag": args.cloud,
        "qaoa_device": device_for(cfg["cluster_size"], ctx), "pce_device": device_for(pce_qubits, ctx),
        "gpu_qubit_window": [GPU_MIN_QUBITS, GPU_MAX_QUBITS] if has_gpu else None,
        "pce_order": k, "pce_qubits": pce_qubits, "parallel_runs": workers,
        "reason": (f"{q.n_vars} > {BIG_VARS} variables: circuits on Qoro's cloud, lighter polishing" if big
                   else "runner --cloud true" if cloud else "local"),
    }
    credits = {"before": _credits()} if cloud else None
    print(f"{args.instance}: {q.n_vars} variables; {routing['reason']}; QAOA ({cfg['cluster_size']} qubits) on "
          f"{routing['qaoa_device']}, PCE ({pce_qubits} qubits, order {k}) on {routing['pce_device']}; "
          f"seeds {cfg['seeds']}; {workers} run(s) in parallel{' [QUICK]' if quick else ''}", flush=True)

    jobs = [(variant, seed) for variant in variants for seed in cfg["seeds"]]
    if workers <= 1:
        runs = [run_one(ctx, variant, seed) for variant, seed in jobs]
    else:  # each run in its own process; results are identical to the sequential order (seeded)
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init, initargs=(spec,)) as pool:
            runs = list(pool.map(_run_in_pool, jobs))

    lines = [f"# {args.instance}: starting row with the lowest loss in the first Monte Carlo iteration", ""]
    lines += [f"{r['variant']:16s} seed {r['seed']}: {_summarise_chosen(r['chosen'])}" for r in runs]
    (out_dir / "chosen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in runs:  # keep the result line compact: chosen labels are summarised in chosen.txt
        r["chosen"] = len(r["chosen"])
    if credits is not None:
        credits["after"] = _credits()

    done = [r for r in runs if r["objective"] is not None]
    pool = [r for r in done if r["feasible"]] or done
    best = min(pool, key=lambda r: r["objective"]) if pool else None
    if best:
        shutil.copyfile(best["sol"], f"{args.instance}.sol")
    result = {"score": best["objective"] if best else None,
              "best_sol": f"{args.instance}.sol" if best else None,
              "extra": {"best_variant": best["variant"] if best else None, "best_seed": best["seed"] if best else None,
                        "routing": routing, "config": {**cfg, "quick": quick, "variants": variants},
                        "credits": credits, "runs": runs}}
    Path(f"{args.instance}.result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"best: {result['extra']['best_variant']} seed {result['extra']['best_seed']} → {result['score']}", flush=True)
    print(RESULT_MARKER + json.dumps(result, default=str), flush=True)
    return 0 if best else 1


# ── plot: gap per instance ─────────────────────────────────────────────────────────────

TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COLORS = {v: TAB10[i] for i, v in enumerate(VARIANTS)}
LABELS = {"qaoa_random": "QAOA Random", "qaoa_tqa": "QAOA TQA", "iter_chebyshev": "Iterative QAOA CHEBYSHEV",
          "pce_random": "PCE Random", "pce_progressive": "ProgressivePCE"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POINT_FIELDS = ["instance", "n_vars", "variant", "solver", "best_objective", "feasible", "seeds",
                "reference", "reference_type", "gap_percent"]
RUN_FIELDS = ["instance", "variant", "solver", "seed", "objective", "feasible", "raw_best_feasible", "gap_percent",
              "circuits", "n_qubits", "device", "polish", "backend_time_s", "wall_s", "sol", "error"]


def _n_vars(name: str) -> int:
    m = re.search(r"_a(\d+)_t(\d+)_", name)
    return 6 * int(m.group(1)) * int(m.group(2)) + 11 * int(m.group(2)) if m else 0


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot(target: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if target.is_dir():
        jsons = sorted(target.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not jsons:
            raise SystemExit(f"no runner JSON in {target}")
        target = jsons[-1]
    report = json.loads(target.read_text(encoding="utf-8"))
    out_dir = target.parent

    def order(name):  # by size, then instance, then λ numerically
        base, _, lam = name.rpartition("_l")
        try:
            return _n_vars(name), base, float(lam)
        except ValueError:
            return _n_vars(name), base, 0.0

    instances = sorted((n for n, b in (report.get("results") or {}).items()
                        if b and (b.get("extra") or {}).get("runs")), key=order)
    if not instances:
        raise SystemExit("no init_comp results in this run")

    points, all_runs, ref_type = [], [], {}
    for name in instances:
        block = report["results"][name]
        runs = [r for r in block["extra"]["runs"] if r.get("objective") is not None]
        bkv = (block.get("best_known") or {}).get("value")
        feasible_objs = [r["objective"] for r in runs if r["feasible"]]
        if bkv is not None:
            reference, ref_type[name] = bkv, "best_known"
        else:
            reference, ref_type[name] = (min(feasible_objs) if feasible_objs else None), "best_initialisation"

        def gap(obj):
            return None if obj is None or not reference else 100.0 * (obj - reference) / abs(reference)

        for r in block["extra"]["runs"]:
            all_runs.append({"instance": name, **r, "gap_percent": gap(r.get("objective"))})
        for variant in VARIANTS:
            vr = [r for r in runs if r["variant"] == variant]
            if not vr:
                continue
            good = [r for r in vr if r["feasible"]]
            best = min(good or vr, key=lambda r: r["objective"])
            points.append({"instance": name, "n_vars": _n_vars(name), "variant": variant, "solver": best["solver"],
                           "best_objective": best["objective"], "feasible": bool(good), "seeds": len(vr),
                           "reference": reference, "reference_type": ref_type[name],
                           "gap_percent": gap(best["objective"])})

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(instances) + 4), 6), facecolor="white")
    present = [v for v in VARIANTS if any(p["variant"] == v for p in points)]
    width = 0.8 / max(len(present), 1)
    for j, variant in enumerate(present):
        offset = (j - (len(present) - 1) / 2) * width
        marker = "o" if variant in QAOA_VARIANTS else "^"
        for p in (p for p in points if p["variant"] == variant and p["gap_percent"] is not None):
            x = instances.index(p["instance"]) + offset
            ax.plot([x], [p["gap_percent"]], marker=marker, ms=9, mew=2, linestyle="none", color=COLORS[variant],
                    mfc=COLORS[variant] if p["feasible"] else "white", zorder=3)
        ax.plot([], [], marker=marker, ms=9, linestyle="none", color=COLORS[variant], label=LABELS[variant])
    ax.axhline(0, color=INK, lw=1.2, ls="--", zorder=2, label="reference (gap 0)")
    ax.set_yscale("symlog", linthresh=0.1)
    gaps = [p["gap_percent"] for p in points if p["gap_percent"] is not None]
    if gaps:  # headroom above; only show the negative side when something beats the reference
        top = max(max(gaps), 0.1) * 3
        bottom = min(gaps) * 3 if min(gaps) < 0 else -0.03
        ax.set_ylim(bottom, top)
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels([n + ("*" if ref_type[n] != "best_known" else "") for n in instances],
                       rotation=30, ha="right", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_ylabel("gap to best known (%), best of seeds — lower is better", color=INK_2)
    fig.suptitle(f"{report.get('test', '')}: gap per instance and initialisation", color=INK, x=0.01,
                 ha="left", fontsize=12)
    ax.set_title("circle = partitioned QAOA, triangle = PCE, hollow = infeasible, "
                 "* = no best known: gap vs best initialisation", color=INK_2, loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    png = out_dir / "init_comp_gap.png"
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)
    _write_csv(out_dir / "init_comp_points.csv", points, POINT_FIELDS)
    _write_csv(out_dir / "init_comp_all_runs.csv", all_runs, RUN_FIELDS)
    print(f"wrote {png}\nwrote {out_dir / 'init_comp_points.csv'}\nwrote {out_dir / 'init_comp_all_runs.csv'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", metavar="RUN_DIR_OR_JSON", help="plot a finished auto_runner run")
    p.add_argument("--instance")
    p.add_argument("--data-dir")
    p.add_argument("--assets", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--lam")
    p.add_argument("--budget", type=int)
    p.add_argument("--cloud", default="false")
    p.add_argument("--quick", action="store_true", help="tiny settings (same as INIT_COMP_QUICK=1)")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, help="subset of variants (default all 5)")
    p.add_argument("--seeds", nargs="+", type=int, help="override the seeds (default 1 2 3)")
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto",
                   help="local GPU use inside the qubit window INIT_COMP_GPU_MIN/MAX_QUBITS (default 16-28)")
    p.add_argument("--workers", type=int, help="runs in parallel inside this instance "
                   "(same as INIT_COMP_WORKERS; default 1)")
    args = p.parse_args(argv)
    if args.plot:
        return plot(Path(args.plot))
    missing = [a for a in ("instance", "data_dir", "lam", "budget") if getattr(args, a) is None]
    if missing:
        p.error(f"missing {missing}: run it through auto_runner.py, or use --plot")
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
