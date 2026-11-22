"""Q-EXCHANGE: XY-mixer QAOA that starts IN the incumbent and moves units around.

The quantum analogue of the network 2-exchange polish:
  * window = a few assets x 1-2 days; per (asset, day, direction) it holds the top "on" unit bit
    and the next "off" unit bit of the unary encoding (so a unit can leave one asset and enter another);
  * initial state = incumbent bits (CustomPerQubitState), so QAOA at zero angles = incumbent;
  * mixer = XY on edges between bits of the SAME day (transfer a unit between assets/directions:
    Hamming weight per day = units per day is conserved -> budget rule holds by construction)
    plus, optionally, edges between the same (asset, direction) on adjacent days (shift a holding in time);
  * cost = exact objective-only conditional sub-QUBO (penalties unnecessary: the mixer keeps the
    unit count; capital is checked by the referee on splice).
Every sampled bitstring is spliced into the incumbent and scored by the official referee.

Backends: local Maestro for < --cloud-from qubits, Qoro cloud cluster qoro_cpu otherwise.

Example:
  python q_exchange.py --instance po_a050_t10_orig --lam 1e-2 --start best --iters 40
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import qf_common as qf
import pf_quantum as pq

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.hamiltonians import xy_mixer
from divi.qprog.algorithms import QAOA
from divi.qprog.initial_states import CustomPerQubitState
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a050_t10_orig")
ap.add_argument("--lam", default="1e-2")
ap.add_argument("--start", default="best", help="'best' (our best .sol), 'classical', or a .sol path")
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--assets", type=int, default=4)
ap.add_argument("--days", type=int, default=1)
ap.add_argument("--time-edges", action="store_true")
ap.add_argument("--max-qubits", type=int, default=18)
ap.add_argument("--n-layers", type=int, default=2)
ap.add_argument("--mc-population", type=int, default=20)
ap.add_argument("--mc-iterations", type=int, default=4)
ap.add_argument("--shots", type=int, default=2000)
ap.add_argument("--cloud-from", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--time-limit", type=float, default=1800)
ap.add_argument("--tag", default="")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

model, fm, (bkv, status) = qf.load(args.instance, args.lam)
cq = pq.CompactQUBO(model)
D = pq.DenseQUBO(cq.objective, cq.n_vars)
if args.start in ("best", "classical"):
    found = qf.best_known_sol(args.instance, model.budget, args.lam,
                              method_filter="classical" if args.start == "classical" else None)
    start_file = found[1]
    v = qf.pc.positions_to_v(model, found[2])
else:
    start_file = args.start
    v = qf.pc.positions_to_v(model, qf.qt.read_portfolio_solution(args.start)[1])
best = qf.pc.score_v(model, v).objective
print(f"{args.instance} λ={args.lam}: start {best} from {start_file}; BKV {bkv} gap {qf.gap(best, bkv)}", flush=True)

local = MaestroSimulator(shots=args.shots)
cloud = None


def backend_for(nq):
    global cloud
    if nq < args.cloud_from:
        return local, "local"
    if cloud is None:
        cloud = QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster="qoro_cpu"))
    return cloud, "qoro_cpu"


def window(v):
    """Pick days + assets (held ones + most correlated partners); return bit list and XY edges."""
    T, n = cq.T, cq.n
    t0 = int(rng.integers(0, T - args.days))
    days = list(range(t0, t0 + args.days))
    held = np.flatnonzero(np.abs(v[days]).sum(0))
    seed = int(rng.choice(held)) if len(held) else int(rng.integers(n))
    assets = [seed]
    partners = qf.np.argsort(-np.abs(fm.corr[seed]))
    for a in partners:
        if len(assets) >= args.assets:
            break
        if a != seed and (rng.random() < 0.7 or a in held):
            assets.append(int(a))
    bits, where = [], []
    for t in days:
        for a in sorted(assets):
            L, S = max(v[t, a], 0), max(-v[t, a], 0)
            for tau, cnt in ((1, L), (-1, S)):
                if cnt > 0:
                    bits.append(cq.x(a, cnt - 1, tau, t)); where.append((t, a, tau))
                if cnt < cq.ub:
                    bits.append(cq.x(a, cnt, tau, t)); where.append((t, a, tau))
    bits, where = bits[: args.max_qubits], where[: args.max_qubits]
    edges = []
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            same_day = where[i][0] == where[j][0]
            time_pair = (args.time_edges and abs(where[i][0] - where[j][0]) == 1
                         and where[i][1:] == where[j][1:])
            if same_day or time_pair:
                edges.append((i, j))
    return days, sorted(assets), bits, edges


log, circuits, max_q, t_start = [], 0, 0, time.time()
for it in range(args.iters):
    if time.time() - t_start > args.time_limit:
        break
    days, assets, free, edges = window(v)
    if not edges:
        continue
    xbits = cq.encode_v(v)
    sub = D.sub(xbits, free)
    init = "".join(str(int(xbits[b])) for b in free)
    scale = max(abs(c) for c in list(sub.linear.values()) + list(sub.quadratic.values()) or [1.0])
    norm = sub.copy(); norm.scale(1.0 / scale)
    nq = len(free)
    be, where_run = backend_for(nq)
    t0 = time.time()
    try:
        prog = QAOA(BinaryOptimizationProblem(norm), initial_state=CustomPerQubitState(init),
                    n_layers=args.n_layers, optimizer=MonteCarloOptimizer(population_size=args.mc_population),
                    max_iterations=args.mc_iterations, backend=be, seed=int(rng.integers(1 << 30)))
        prog.mixer_hamiltonian = xy_mixer(edges, n_qubits=nq)
        prog.run()
        cands = [s.bitstring for s in prog.get_top_solutions(n=64)]
        circuits += prog.total_circuit_count
    except Exception as err:  # noqa: BLE001 (cloud hiccups)
        print(f"it {it}: {where_run} failed: {type(err).__name__} {err}", flush=True)
        continue
    max_q = max(max_q, nq)
    best_here, best_v, same_hw = None, None, 0
    for c in cands:
        same_hw += c.count("1") == init.count("1")
        nb = xbits.copy(); nb[free] = [int(ch) for ch in c]
        nv = cq.decode_v(nb)
        sc = qf.pc.score_v(model, nv)
        if sc.feasible and (best_here is None or sc.objective < best_here):
            best_here, best_v = sc.objective, nv
    acc = best_here is not None and best_here < best
    if acc:
        best, v = best_here, best_v
    log.append(dict(it=it, days=days, assets=assets, qubits=nq, edges=len(edges), backend=where_run,
                    sec=round(time.time() - t0, 1), weight_kept=same_hw, n_cands=len(cands),
                    cand=best_here, incumbent=best))
    print(f"it {it:3d} days={days} assets={assets} q={nq} {where_run} {time.time() - t0:5.1f}s "
          f"HW-kept {same_hw}/{len(cands)} cand {best_here} incumbent {best} gap {qf.gap(best, bkv)}", flush=True)

score = qf.pc.score_v(model, v)
print(f"\nFINAL {score}  BKV {bkv}  gap {qf.gap(score.objective, bkv)}  circuits {circuits}  max qubits {max_q}")
settings = (f"XY-exchange QAOA from {start_file}; assets={args.assets} days={args.days} time_edges={args.time_edges} "
            f"n_layers={args.n_layers} mc={args.mc_population}x{args.mc_iterations} shots={args.shots} "
            f"iters={len(log)} seed={args.seed} cloud_from={args.cloud_from} circuits={circuits}")
start_obj = int(open(start_file).read().split("objective ")[1].split()[0]) if args.start != "empty" else None
if score.objective < (start_obj if start_obj is not None else 10**12):
    _, path, _ = qf.pc.record_solution(model, qf.pc.v_to_positions(model, v),
                                       method="Q-EXCHANGE XY-mixer QAOA polish (quantum moves on a precomputed start)",
                                       settings=settings, n_qubo_vars=cq.n_vars, max_qubits=max_q, quantum=True,
                                       tag=f"qexchange{args.tag}")
    print("recorded", path)
else:
    print("no improvement over the start: nothing recorded")
(qf.HERE / "logs" / f"qexchange_{args.instance}_l{args.lam}_s{args.seed}{args.tag}.json").write_text(
    json.dumps(dict(args=vars(args), start=str(start_file), log=log, final=score.objective, bkv=bkv), indent=1))
