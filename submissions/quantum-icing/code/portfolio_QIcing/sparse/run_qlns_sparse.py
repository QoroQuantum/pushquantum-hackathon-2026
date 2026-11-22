"""PCE quantum LNS on large instances (a200, a400) using the block-sparse QUBO.

Same method as ../run_qlns.py (window -> exact conditional sub-QUBO -> PCE on the local Maestro
simulator -> splice every sample -> official referee -> accept if better), but the QUBO is a
BlockQUBO (hundreds of MB instead of GB). Infeasible splices are repaired cheaply per day.

Example: python run_qlns_sparse.py --instance po_a200_t10_orig --lam 1e-4 --time-limit 3600
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from block_qubo import BlockQUBO, HERE, pc
import pf_quantum as pq

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import PCE, PolyEncoding, SoftObjective
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a200_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--iters", type=int, default=100000)
ap.add_argument("--time-limit", type=float, default=3600)
ap.add_argument("--window-assets", type=int, default=5)
ap.add_argument("--time-window-assets", type=int, default=2)
ap.add_argument("--p-time", type=float, default=0.4)
ap.add_argument("--p-corr", type=float, default=0.5)
ap.add_argument("--penalty-weight", type=float, default=5000.0)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--mc-population", type=int, default=40)
ap.add_argument("--mc-iterations", type=int, default=8)
ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--start", default="empty")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

t_load = time.time()
model = pc.make_model(args.instance, args.lam)
bq = BlockQUBO(model, args.penalty_weight)
bkv, status = pc.load_bkv().get((args.instance, model.budget, args.lam), (None, "none published"))
T, n = bq.T, bq.n
C, netmin, B = model.capital, model.capital - (2**model.cash_slack_bits - 1), model.budget
print(f"{args.instance} λ={args.lam} B={B}: {bq.n_vars} QUBO vars, blocks {bq.memory_mb():.0f} MB, "
      f"load {time.time() - t_load:.0f}s; BKV {bkv} ({status})", flush=True)

if args.start == "empty":
    v = np.zeros((T, n), dtype=np.int64)
else:
    v = pc.positions_to_v(model, pc.qt.read_portfolio_solution(args.start)[1])
best = pc.score_v(model, v).objective
start_obj = best
backend = MaestroSimulator(shots=args.shots)


def repair(nv):
    """Cheap per-day repair: drop units until budget and capital hold."""
    nv = nv.copy()
    for t in range(T):
        k = 0
        while np.abs(nv[t]).sum() > B or nv[t].sum() > C or nv[t].sum() < netmin:
            if nv[t].sum() > C:
                cand = np.flatnonzero(nv[t] > 0)
            elif nv[t].sum() < netmin:
                cand = np.flatnonzero(nv[t] < 0)
            else:
                cand = np.flatnonzero(nv[t])
            i = int(cand[k % len(cand)])
            nv[t, i] -= np.sign(nv[t, i])
            k += 1
    return nv


def gap(obj):
    return f"{(obj - bkv) / abs(bkv):+.2%}" if bkv else "no BKV"


log, t_start, circuits, max_q = [], time.time(), 0, 0
for it in range(args.iters):
    if time.time() - t_start > args.time_limit:
        break
    t, assets, free = pq.choose_window(bq, rng, args.window_assets, v, p_time=args.p_time,
                                       time_assets=args.time_window_assets, p_corr=args.p_corr)
    bits = bq.encode_v(v)
    sub = bq.sub(bits, free)
    scale = max(abs(c) for c in list(sub.linear.values()) + list(sub.quadratic.values()) or [1.0])
    sub.scale(1.0 / scale)
    t0 = time.time()
    prog = PCE(BinaryOptimizationProblem(sub), encoding=PolyEncoding(), objective=SoftObjective(alpha=2.0),
               n_layers=args.n_layers, optimizer=MonteCarloOptimizer(population_size=args.mc_population),
               max_iterations=args.mc_iterations, backend=backend, seed=int(rng.integers(1 << 30)))
    prog.run()
    cands = [s.bitstring for s in prog.get_top_solutions(n=32, min_prob=0.0, sort_by="energy")]
    circuits += prog.total_circuit_count
    max_q = max(max_q, prog.n_qubits)
    best_here, best_v = None, None
    for c in cands:
        nb = bits.copy()
        nb[free] = [int(ch) for ch in c]
        nv = bq.decode_v(nb)
        sc = pc.score_v(model, nv)
        if not sc.feasible:
            nv = repair(nv)
            sc = pc.score_v(model, nv)
        if best_here is None or sc.objective < best_here:
            best_here, best_v = sc.objective, nv
    if best_here < best:
        best, v = best_here, best_v
    log.append(dict(it=it, window=str(t), assets=assets, qubits=prog.n_qubits, cand=best_here, incumbent=best,
                    sec=round(time.time() - t0, 2)))
    if it % 10 == 0:
        print(f"it {it:5d} t={t} q={prog.n_qubits} {time.time() - t0:4.1f}s cand {best_here:>10} incumbent "
              f"{best:>10} gap {gap(best)} units/day {np.abs(v).sum(1).tolist()}", flush=True)

score = pc.score_v(model, v)
held = int((np.abs(v).sum(0) > 0).sum())
print(f"\nFINAL {score}  BKV {bkv}  gap {gap(score.objective)}  windows {len(log)}  circuits {circuits}  "
      f"max qubits {max_q}  assets held {held}  {time.time() - t_start:.0f}s", flush=True)
settings = (f"block-sparse QUBO ({bq.n_vars} vars, {bq.memory_mb():.0f} MB), PCE PolyEncoding n_layers={args.n_layers} "
            f"mc={args.mc_population}x{args.mc_iterations} shots={args.shots} window_assets={args.window_assets} "
            f"time_window_assets={args.time_window_assets} p_time={args.p_time} p_corr={args.p_corr} "
            f"penalty_weight={args.penalty_weight} windows={len(log)} seed={args.seed} start={args.start} circuits={circuits}")
if score.objective < start_obj:
    _, path, _ = pc.record_solution(model, pc.v_to_positions(model, v),
                                    method="QLNS-pce sparse (quantum sub-solver, Maestro)", settings=settings,
                                    n_qubo_vars=bq.n_vars, max_qubits=max_q, quantum=True, tag="qlns-pce-sparse")
    print("recorded", path)
(HERE / "logs").mkdir(exist_ok=True)
(HERE / "logs" / f"qlns_sparse_{args.instance}_l{args.lam}_s{args.seed}.json").write_text(
    json.dumps(dict(args=vars(args), log=log, final=score.objective, bkv=bkv), indent=1))
