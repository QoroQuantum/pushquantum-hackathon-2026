"""Configuration sweep for QOBLIB 08 (network design): notebook-00 routines and the exchange mixer,
scored identically.

Every configuration produces samples -> decode arcs -> repair to a valid topology -> rank the
distinct topologies by H2 -> LP-score the top_k -> 2-exchange polish (polish_budget LP evals).
Reported per run: valid share of raw samples, best LP before/after polish, gap vs QOBLIB BKV,
wall time. A polish-only baseline with the same LP budget runs from the same random start.

Routines (knobs from notebook 00):
  qaoa_x     QAOA, X mixer, degree rules as LinearConstraints (penalty_weight), optional warm start
             from characterize_and_validate angles
  qaoa_xch   QAOA, degree-preserving exchange mixer, H2 HUBO, no penalties (network_mixer.py)
  pce        PCE on the penalised QUBO: Dense / Poly encoding, SoftObjective(alpha)
  part_qaoa  partitioned QAOA (CommunityDecomposer cluster size, modularity|spectral), BeamSearch
  part_pce   partitioned PCE (Poly encoding)

    .venv/bin/python network_sweep.py characterize --n 5          # stage 1 (cloud, cached)
    .venv/bin/python network_sweep.py screen --preset n05 --seeds 0 1 2   # stage 2 (local)
    .venv/bin/python network_sweep.py rank                         # table from runs/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import hybrid  # noqa: E402
import numpy as np  # noqa: E402

import qoblib_tools as qt  # noqa: E402
from network_mixer import exchange_mixer, h2_hubo, run_qaoa  # noqa: E402
from network_moves import bkv_gap, lp, make_surrogates, polish  # noqa: E402
from network_pipeline import BKV  # noqa: E402

RUNS = Path("runs")
CSV_PATH = RUNS / "sweep.csv"
FIELDS = ["instance", "routine", "config", "seed", "n_qubits", "valid_share", "n_topologies",
          "best_sampled", "final", "baseline", "gap_final", "seconds", "backend", "status"]

PRESETS = {
    # stage-2 screening grid on network05 (20 arc variables); every job runs locally in ~1-3 min
    "n05": [
        {"routine": "qaoa_x", "n_layers": 2, "penalty_weight": 0.5},
        {"routine": "qaoa_x", "n_layers": 2, "penalty_weight": 2.0},
        {"routine": "qaoa_x", "n_layers": 4, "penalty_weight": 0.5},
        {"routine": "qaoa_x", "n_layers": 4, "penalty_weight": 2.0},
        {"routine": "qaoa_x", "n_layers": 6, "penalty_weight": 2.0, "warm": True},
        {"routine": "qaoa_xch", "n_layers": 2},
        {"routine": "qaoa_xch", "n_layers": 3},
        {"routine": "qaoa_xch", "n_layers": 4},
        {"routine": "pce", "encoding": "dense", "alpha": 2.0, "n_layers": 4, "penalty_weight": 0.5},
        {"routine": "pce", "encoding": "dense", "alpha": 4.0, "n_layers": 4, "penalty_weight": 0.5},
        {"routine": "pce", "encoding": "poly", "alpha": 2.0, "n_layers": 4, "penalty_weight": 0.5},
        {"routine": "pce", "encoding": "poly", "alpha": 4.0, "n_layers": 4, "penalty_weight": 0.5},
        {"routine": "part_qaoa", "cluster": 8, "method": "modularity", "n_layers": 2, "penalty_weight": 0.5},
        {"routine": "part_qaoa", "cluster": 12, "method": "modularity", "n_layers": 2, "penalty_weight": 0.5},
        {"routine": "part_pce", "cluster": 12, "method": "modularity", "n_layers": 2, "penalty_weight": 0.5},
    ],
    # partition-size grid on network12 (132 arc variables), local simulator
    "n12part": [
        {"routine": "part_qaoa", "cluster": c, "method": m, "n_layers": 2, "penalty_weight": 0.5}
        for c in (8, 12, 16) for m in ("modularity", "spectral")
    ] + [
        {"routine": "pce", "encoding": e, "alpha": 2.0, "n_layers": 4, "penalty_weight": 0.5}
        for e in ("dense", "poly")
    ],
}


# ── problem builders ─────────────────────────────────────────────────────
def degree_constraints(arcs, n):
    from divi.qprog.problems import LinearConstraint

    cons = []
    for node in range(n):
        cons.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs) if i == node}, "==", 2))
        cons.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs) if j == node}, "==", 2))
    return cons


def penalised_bqm(T, penalty_weight):
    """Notebook-02 QUBO: direct-demand surrogate + penalty_weight x degree penalty, normalised."""
    surrogate, degree_penalty, arcs = qt.build_network_qubo(T)
    bqm = surrogate.copy()
    pen = degree_penalty.copy()
    pen.scale(penalty_weight)
    bqm.update(pen)
    bqm.normalize()
    return bqm, arcs


# ── routines: each returns (list of {var: bit} samples, n_qubits) ───────
def make_backend(cloud, shots=10_000):
    from divi.backends import JobConfig, MaestroSimulator, QoroService

    if cloud:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster=cloud))
    return MaestroSimulator(shots=shots)


def run_routine(cfg, T, arcs, seed, args, char_cache):
    from divi.qprog.algorithms import PCE, QAOA, DenseEncoding, PolyEncoding, SoftObjective
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    n = len(T)
    m = len(arcs)
    optimizer = MonteCarloOptimizer(population_size=args.pop)
    routine = cfg["routine"]

    def bits_to_sample(bitstring):
        return {v: int(b) for v, b in enumerate(bitstring)}

    if routine == "qaoa_x":
        surrogate, _, _ = qt.build_network_qubo(T)
        problem = BinaryOptimizationProblem(surrogate, constraints=degree_constraints(arcs, n),
                                            penalty_weight=cfg["penalty_weight"])
        qaoa = QAOA(problem, n_layers=cfg["n_layers"], optimizer=optimizer, max_iterations=args.iters,
                    backend=make_backend(args.cloud), seed=seed)
        init = None
        if cfg.get("warm", True):  # notebook-00 warm start is the default
            angles = char_cache.get(f"n{n}_angles_L{cfg['n_layers']}")
            if angles is None:
                raise RuntimeError(f"no cached angles for {cfg['n_layers']} layers: run `characterize --n {n}` first")
            base = np.array(angles).reshape(1, -1)
            init = base + np.random.default_rng(seed).normal(scale=0.05, size=(args.pop, base.shape[1]))
            init[0] = base[0]
        qaoa.run(initial_params=init)
        sols = sorted(qaoa.get_top_solutions(n=0), key=lambda s: -s.prob)
        return [bits_to_sample(s.bitstring) for s in sols], m

    if routine == "qaoa_xch":
        start = sorted(qt.random_topology(n, np.random.default_rng(1000 + seed)))
        qubit_of = {a: i for i, a in enumerate(arcs)}
        qaoa = run_qaoa(h2_hubo(T, qubit_of), m, exchange_mixer(qubit_of), [int(a in set(start)) for a in arcs],
                        cfg["n_layers"], args.cloud, args.shots, args.pop, args.iters, seed)
        sols = sorted(qaoa.get_top_solutions(n=0), key=lambda s: -s.prob)
        return [bits_to_sample(s.bitstring) for s in sols], m

    bqm, _ = penalised_bqm(T, cfg["penalty_weight"])
    if routine == "pce":
        encoding = DenseEncoding() if cfg["encoding"] == "dense" else PolyEncoding()
        pce = PCE(BinaryOptimizationProblem(bqm), encoding=encoding, objective=SoftObjective(alpha=cfg["alpha"]),
                  n_layers=cfg["n_layers"], optimizer=optimizer, max_iterations=args.iters,
                  backend=make_backend(args.cloud), seed=seed)
        pce.run()
        sols = pce.get_top_solutions(n=200, min_prob=0.0, sort_by="energy")
        return [bits_to_sample(s.bitstring) for s in sols], pce.n_qubits

    if routine in ("part_qaoa", "part_pce"):
        problem = BinaryOptimizationProblem(
            bqm,
            decomposer=CommunityDecomposer(QUBOPartitioningConfig(
                max_n_variables_per_cluster=cfg["cluster"], method=cfg["method"], seed=seed)),
            composer=hybrid.SplatComposer(),
            local_search=True,
        )
        extra = {"encoding": PolyEncoding(), "objective": SoftObjective(alpha=2.0)} if routine == "part_pce" else {}
        ensemble = PartitioningProgramEnsemble(
            problem=problem, quantum_routine="pce" if routine == "part_pce" else "qaoa",
            n_layers=cfg["n_layers"], optimizer=optimizer, max_iterations=args.iters,
            backend=make_backend(args.cloud), **extra)
        ensemble.run()
        tops = ensemble.get_top_solutions(n=100, strategy=BeamSearchStrategy(beam_width=100, n_partition_candidates=10))
        return [dict(zip(bqm.variables, sol)) for sol, _ in tops], cfg["cluster"]

    raise ValueError(routine)


# ── uniform scoring ──────────────────────────────────────────────────────
def score_samples(samples, T, arcs, top_k, polish_budget, H2, max_samples):
    """samples must be ordered best-first by the routine (probability, or energy for PCE); only the
    first max_samples are used so that repair cannot enumerate the whole space on small instances."""
    n = len(T)
    decoded = [[arcs[v] for v, b in s.items() if b] for s in samples[:max_samples]]
    valid_share = float(np.mean([qt.is_valid_topology(n, d) for d in decoded])) if decoded else 0.0
    topologies = {tuple(sorted(qt.repair_topology(n, d, T))) for d in decoded}
    ranked = sorted((list(t) for t in topologies), key=H2)
    scored = [(lp(T, t), t) for t in ranked[:top_k]]
    best_sampled, best_topology = min(scored)
    final = polish(T, best_topology, rank=H2, lp_budget=polish_budget)[1]
    return valid_share, len(topologies), best_sampled, final


def config_label(cfg):
    return ",".join(f"{k}={v}" for k, v in cfg.items() if k != "routine")


def append_row(row):
    RUNS.mkdir(exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ── commands ─────────────────────────────────────────────────────────────
def cmd_characterize(args):
    """Stage 1: notebook-00 characterize_and_validate on the penalised network QUBO, cached."""
    from divi.backends import QoroService
    from divi.backends.characterization import CharacterizationOptions, characterize_and_validate
    from divi.qprog.problems import BinaryOptimizationProblem

    n = args.n
    T = qt.load_network_demand(n)
    surrogate, _, arcs = qt.build_network_qubo(T)
    problem = BinaryOptimizationProblem(surrogate, constraints=degree_constraints(arcs, n), penalty_weight=2.0)
    if n <= 10:  # proven optimum, used only as a validation reference
        _, ref_arcs, _ = qt.read_network_solution(qt.fetch_qoblib_file(f"08-network/solutions/network{n:02d}.opt.sol"))
    else:  # QOBLIB forbids published solutions as starts: use our own best topology
        best = min((json.loads(p.read_text()) for p in (RUNS / f"network{n:02d}").glob("*.json")), key=lambda r: r["z_lp"])
        ref_arcs = best["arcs"]
    ref_bits = "".join("1" if a in {tuple(x) for x in ref_arcs} else "0" for a in arcs)
    service = QoroService()
    credits_before = float(service.get_credit_balance()["balance"])
    report = characterize_and_validate(problem, reference_states=[ref_bits], service=service,
                                       options=CharacterizationOptions(ansatz={"layers": args.layers}))
    credits_after = float(service.get_credit_balance()["balance"])
    cache_path = RUNS / "sweep_characterization.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for n_layers in range(1, 9):  # warm-start angles for every layer count we may sweep
        try:
            init = report.qaoa_initial_params(layers=n_layers)
        except Exception:  # noqa: BLE001 - the service may not provide every depth
            init = None
        if init is not None:
            cache[f"n{n}_angles_L{n_layers}"] = np.asarray(init)[0].tolist()
    cache[f"n{n}_report"] = {"recommended_layers": report.recommended_layers, "penalty": report.penalty,
                             "references": report.references, "summary": str(report)}
    RUNS.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, default=str))
    print(report)
    print(f"recommended layers {report.recommended_layers} | penalty {report.penalty} | "
          f"credits used {credits_before - credits_after:.2f} | cached to {cache_path}")


def cmd_screen(args):
    """Stage 2/3: run a preset grid over seeds; append every run to runs/sweep.csv."""
    n = {"n05": 5, "n12part": 12}[args.preset]
    T = qt.load_network_demand(n)
    arcs = qt.all_arcs(n)
    H2 = make_surrogates(T)["H2"]
    cache_path = RUNS / "sweep_characterization.json"
    char_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    budget = args.top_k + args.polish_budget
    baselines = {}
    for seed in args.seeds:  # polish-only baseline, same LP budget, same random start as qaoa_xch
        start = sorted(qt.random_topology(n, np.random.default_rng(1000 + seed)))
        baselines[seed] = polish(T, start, rank=H2, lp_budget=budget)[1]
    print(f"network{n:02d} | BKV {BKV[n]} | first {args.max_samples} samples -> top_k {args.top_k} + polish {args.polish_budget} LP evals | "
          f"backend {args.cloud or 'local'} | baselines {baselines}", flush=True)
    for cfg in PRESETS[args.preset]:
        for seed in args.seeds:
            t0 = time.time()
            row = {"instance": f"network{n:02d}", "routine": cfg["routine"], "config": config_label(cfg),
                   "seed": seed, "baseline": baselines[seed], "backend": args.cloud or "local"}
            try:
                samples, n_qubits = run_routine(cfg, T, arcs, seed, args, char_cache)
                valid_share, n_top, best_sampled, final = score_samples(
                    samples, T, arcs, args.top_k, args.polish_budget, H2, args.max_samples)
                row.update(n_qubits=n_qubits, valid_share=f"{valid_share:.3f}", n_topologies=n_top,
                           best_sampled=best_sampled, final=final,
                           gap_final=f"{(final - BKV[n]) / BKV[n]:.4f}", status="ok")
                msg = (f"valid {valid_share:6.1%} | {n_top:3d} topologies | sampled {best_sampled:.0f} -> "
                       f"polished {final:.0f} ({bkv_gap(n, final)}) | baseline {baselines[seed]:.0f}")
            except Exception as err:  # keep sweeping; record the failure
                row.update(status=f"FAILED {type(err).__name__}: {str(err)[:80]}")
                msg = row["status"]
            row["seconds"] = f"{time.time() - t0:.0f}"
            append_row(row)
            print(f"  {cfg['routine']:<9} {config_label(cfg):<55} seed {seed}: {msg} ({row['seconds']}s)", flush=True)


def cmd_rank(args):
    rows = list(csv.DictReader(CSV_PATH.open()))
    groups = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["instance"], r["routine"], r["config"]), []).append(r)
    table = []
    for (inst, routine, cfg), rs in groups.items():
        n = int(inst[-2:])
        final = [float(r["final"]) for r in rs]
        table.append((inst, routine, cfg, len(rs), sum(f == BKV[n] for f in final),
                      np.mean([float(r["valid_share"]) for r in rs]), min(final), np.mean(final),
                      np.mean([float(r["baseline"]) for r in rs]), int(np.mean([float(r["seconds"]) for r in rs]))))
    table.sort(key=lambda t: (t[0], -t[4], t[7]))
    print(f"{'instance':<10} {'routine':<9} {'config':<55} runs  hit-BKV  valid   best     mean   baseline  s")
    for t in table:
        print(f"{t[0]:<10} {t[1]:<9} {t[2]:<55} {t[3]:>4}  {t[4]:>3}/{t[3]:<3} {t[5]:6.1%} {t[6]:>8.0f} {t[7]:>8.0f} {t[8]:>8.0f} {t[9]:>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    char_p = sub.add_parser("characterize")
    char_p.add_argument("--n", type=int, default=5)
    char_p.add_argument("--layers", type=int, default=2)
    screen_p = sub.add_parser("screen")
    screen_p.add_argument("--preset", choices=sorted(PRESETS), default="n05")
    screen_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    screen_p.add_argument("--top-k", type=int, default=10)
    screen_p.add_argument("--max-samples", type=int, default=20, help="samples per run that get repaired/ranked")
    screen_p.add_argument("--polish-budget", type=int, default=20)
    screen_p.add_argument("--pop", type=int, default=20)
    screen_p.add_argument("--iters", type=int, default=5)
    screen_p.add_argument("--shots", type=int, default=5000)
    screen_p.add_argument("--cloud", default=None)
    sub.add_parser("rank")
    args = ap.parse_args()
    {"characterize": cmd_characterize, "screen": cmd_screen, "rank": cmd_rank}[args.cmd](args)
