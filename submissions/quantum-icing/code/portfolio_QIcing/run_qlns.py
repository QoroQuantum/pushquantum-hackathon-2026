"""Quantum LNS on a portfolio instance (compact QUBO, sub-QUBO windows solved by QAOA / PCE).

Example:
    python run_qlns.py --instance po_a010_t10_orig --lam 1e-4 --iters 60 --routine qaoa

Pipeline per run (follows notebooks 00/01):
  1. characterize_and_validate one representative window sub-QUBO (cloud; cached by job id)
     -> penalty guidance + QAOA angles (warm start), recommended layers.
  2. start: empty portfolio (default) or a given .sol
  3. loop: window -> exact conditional sub-QUBO -> warm-started QAOA (or PCE) on the local
     Maestro simulator -> splice every sampled sub-bitstring into the incumbent ->
     official score -> accept best improving (no classical local search inside the loop).
  4. control (not used for decisions): exact optimum of each sub-QUBO by enumeration, to
     measure how often the quantum sampler found the window optimum.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import dimod
import numpy as np

import pf_core as pc
import pf_quantum as pq

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.backends.characterization import (
    CharacterizationOptions,
    characterize_and_validate,
    get_characterization_result,
)
from divi.qprog.algorithms import PCE, QAOA, PolyEncoding, SoftObjective
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a010_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--window-assets", type=int, default=2)
ap.add_argument("--routine", default="qaoa", choices=["qaoa", "pce"])
ap.add_argument("--n-layers", type=int, default=0, help="0 = service recommendation (capped at 4)")
ap.add_argument("--penalty-weight", type=float, default=0, help="0 = from characterization")
ap.add_argument("--mc-population", type=int, default=30)
ap.add_argument("--mc-iterations", type=int, default=5)
ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--start", default="empty", help="'empty' or path to a .sol")
ap.add_argument("--char-job", default="", help="reuse a characterization job id")
ap.add_argument("--no-char", action="store_true")
ap.add_argument("--p-time", type=float, default=0.3, help="prob. of a 1-asset x 2-period window")
ap.add_argument("--time-window-assets", type=int, default=1)
ap.add_argument("--p-corr", type=float, default=0.0, help="prob. of correlation-guided partner assets")
ap.add_argument("--p-hedge", type=float, default=0.0, help="prob. of a hedge window (objective-only sub-QUBO)")
ap.add_argument("--hedge-assets", type=int, default=2)
ap.add_argument("--time-limit", type=float, default=3000)
ap.add_argument("--save-sol", default="", help="non-official λ (homotopy step): write the final portfolio here instead of recording")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
model = pc.make_model(args.instance, args.lam)
cq = pq.CompactQUBO(model)
bkv, status = pc.load_bkv().get((args.instance, model.budget, args.lam), (0, "none (homotopy step)"))
print(f"{args.instance} B={model.budget} λ={args.lam}: compact QUBO {cq.n_vars} vars; BKV {bkv} ({status})")

if args.start == "empty":
    v = np.zeros((cq.T, cq.n), dtype=np.int64)
else:
    v = pc.positions_to_v(model, qt_read := pc.qt.read_portfolio_solution(args.start)[1])
best = pq.official(model, v).objective
print("start objective", best)

backend = MaestroSimulator(shots=args.shots)


def ordered(sub: dimod.BinaryQuadraticModel) -> dimod.BinaryQuadraticModel:
    k = sub.num_variables
    return dimod.BinaryQuadraticModel({i: sub.get_linear(i) for i in range(k)},
                                      dict(sub.quadratic), sub.offset, dimod.BINARY)


# ── 1. characterization of a representative window ───────────────────────
W = args.penalty_weight or 5000.0
report = None
t_w, assets_w, free_w = pq.choose_window(cq, rng, args.window_assets, v, period=0)
if not args.no_char:
    try:
        service = QoroService()
        sub_obj = ordered(pq.sub_bqm(cq.objective, cq.encode_v(v), free_w))
        sub_pen = ordered(pq.sub_bqm(cq.penalty, cq.encode_v(v), free_w))
        if args.char_job:
            report = get_characterization_result(args.char_job, service=service)
        else:
            t0 = time.time()
            report = characterize_and_validate(
                BinaryOptimizationProblem(sub_obj, penalty=sub_pen, penalty_weight=W),
                reference_states=["0" * len(free_w)],
                service=service,
                options=CharacterizationOptions(ansatz={"layers": 2}),
            )
            print(f"characterization {time.time() - t0:.0f}s, job {report.job_id}")
        print(report)
        print("penalty:", report.penalty)
        if not args.penalty_weight and report.penalty and report.penalty.get("safe_lambda"):
            W = float(report.penalty["safe_lambda"]) * 1.2
    except Exception as err:  # noqa: BLE001
        print("characterization unavailable:", type(err).__name__, err)
        report = None
n_layers = args.n_layers or (min(int(report.recommended_layers), 4) if report is not None else 2)
init = None
if report is not None and args.routine == "qaoa":
    try:
        init = np.atleast_2d(report.qaoa_initial_params(layers=n_layers))
    except Exception as err:  # noqa: BLE001
        print("no angles at", n_layers, "layers:", err)
print(f"penalty weight {W:.0f}, n_layers {n_layers}, warm start {'yes' if init is not None else 'no'}")
full = cq.full(W)
dense = pq.DenseQUBO(full, cq.n_vars)
dense_obj = pq.DenseQUBO(cq.objective, cq.n_vars) if args.p_hedge > 0 else None


def run_quantum(sub: dimod.BinaryQuadraticModel):
    scale = max(abs(c) for c in list(sub.linear.values()) + list(sub.quadratic.values()) or [1.0])
    norm = sub.copy()
    norm.scale(1.0 / scale)
    problem = BinaryOptimizationProblem(norm)
    opt = MonteCarloOptimizer(population_size=args.mc_population)
    if args.routine == "qaoa":
        prog = QAOA(problem, n_layers=n_layers, optimizer=opt, max_iterations=args.mc_iterations,
                    backend=backend, seed=int(rng.integers(1 << 30)))
        if init is not None:
            starts = init[0] + rng.normal(scale=0.05, size=(args.mc_population, init.shape[1]))
            starts[0] = init[0]
            prog.run(initial_params=starts)
        else:
            prog.run()
        sols = prog.get_top_solutions(n=64)
    else:
        prog = PCE(problem, encoding=PolyEncoding(), objective=SoftObjective(alpha=2.0), n_layers=n_layers,
                   optimizer=opt, max_iterations=args.mc_iterations, backend=backend,
                   seed=int(rng.integers(1 << 30)))
        prog.run()
        sols = prog.get_top_solutions(n=32, min_prob=0.0, sort_by="energy")
    return [s.bitstring for s in sols], prog.n_qubits, prog.total_circuit_count


log = []
t_start = time.time()
n_hit = n_improve = 0
max_qubits = 0
circuits = 0
for it in range(args.iters):
    if time.time() - t_start > args.time_limit:
        break
    t, assets, free = pq.choose_window(cq, rng, args.window_assets, v, p_time=args.p_time,
                                         time_assets=args.time_window_assets, p_corr=args.p_corr)
    bits = cq.encode_v(v)
    sub = dense.sub(bits, free)
    if dense_obj is not None and rng.random() < args.p_hedge:
        t, assets, free = pq.hedge_window(cq, rng, v, n_assets=args.hedge_assets)
        sub = dense_obj.sub(bits, free)
    t0 = time.time()
    cands, nq, nc = run_quantum(sub)
    max_qubits, circuits = max(max_qubits, nq), circuits + nc
    # control: exact window optimum (enumeration) — measurement only
    if len(free) <= 20:
        ex = dimod.ExactSolver().sample(sub).first
        ex_bits = "".join(str(ex.sample[i]) for i in range(len(free)))
        hit = ex_bits in cands
    else:
        hit = None
    n_hit += bool(hit)
    # splice every candidate, score officially, keep best
    best_here, best_v = None, None
    for cand in cands:
        nb = bits.copy()
        nb[free] = [int(c) for c in cand]
        nv = cq.decode_v(nb)
        sc = pq.official(model, nv)
        if not sc.feasible:
            nv = pq.repair_v(model, nv)
            sc = pq.official(model, nv)
        if best_here is None or sc.objective < best_here:
            best_here, best_v = sc.objective, nv
    accepted = best_here < best
    if accepted:
        best, v = best_here, best_v
        n_improve += 1
    log.append(dict(it=it, period=t, assets=assets, qubits=nq, circuits=nc, sec=round(time.time() - t0, 1),
                    window_opt_sampled=hit, best_candidate=best_here, incumbent=best))
    print(f"it {it:3d} t={t} assets={assets} q={nq} {time.time() - t0:4.1f}s  window-opt sampled={hit!s:5}  "
          f"cand {best_here:>8}  incumbent {best:>8}  gap {((best - bkv) / abs(bkv) if bkv else 0):+.2%}", flush=True)

if not log:
    raise SystemExit(0)
score = pq.official(model, v)
print(f"\nFINAL {score}  BKV {bkv}  gap {((score.objective - bkv) / abs(bkv) if bkv else 0):+.3%}")
print(f"improvements {n_improve}, window optimum sampled {n_hit}/{len(log)}, max qubits {max_qubits}, "
      f"circuits {circuits}, {time.time() - t_start:.0f}s")
settings = (f"routine={args.routine} n_layers={n_layers} window_assets={args.window_assets} p_time={args.p_time} p_corr={args.p_corr} p_hedge={args.p_hedge} hedge_assets={args.hedge_assets} penalty_weight={W:.0f} "
            f"mc_pop={args.mc_population} mc_iter={args.mc_iterations} shots={args.shots} iters={len(log)} "
            f"seed={args.seed} start={args.start} warm_start={'char-service' if init is not None else 'none'} "
            f"char_job={report.job_id if report is not None else ''} circuits={circuits}")
if args.save_sol:
    model.write_solution(args.save_sol, pc.v_to_positions(model, v), comment=f"homotopy step λ={args.lam} (not an official instance)")
    print("saved", args.save_sol)
    raise SystemExit(0)
sc, path, _ = pc.record_solution(model, pc.v_to_positions(model, v), method=f"QLNS-{args.routine} (quantum sub-solver, Maestro)",
                                 settings=settings, n_qubo_vars=cq.n_vars, max_qubits=max_qubits, quantum=True,
                                 tag=f"qlns-{args.routine}")
print("recorded", path)
stem = f"qlns_{args.instance}_l{args.lam}_{args.routine}_s{args.seed}"
(pc.HERE / "logs" / f"{stem}.json").write_text(json.dumps(dict(args=vars(args), settings=settings, log=log,
                                                                final=score.objective, bkv=bkv), indent=1))
