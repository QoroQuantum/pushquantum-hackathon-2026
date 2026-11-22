"""Q-ROUND: relaxation-warm-started quantum rounding + exact DP over days.

Per day t (and entry-cost emulation e, scale s):
  1. classical convex relaxation w* of the one-day problem (lns.period_relaxation);
  2. QUBO over rounding bits b_i: v_i = floor(s * w*_i) + b_i (|v| stays linear in b, so the
     one-day cost is EXACTLY quadratic in b), + slack-encoded budget / capital penalties;
  3. PCE (Divi, PolyEncoding) samples the rounding QUBO -> top bitstrings by energy;
  4. decoded day portfolios (repaired if infeasible) go into a candidate pool.
Then lns.chain_dp picks one pool vector per day with the exact transaction costs.

Controls (same pool pipeline, different rounding sampler): 'naive' (b = round), 'sa'
(simulated annealing on the same QUBO), 'random' (uniform random bits).

Example: python q_round.py --instance po_a050_t10_s00 --lam 1e-2 --sampler pce --workers 6
"""
from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import qf_common as qf

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a050_t10_s00")
ap.add_argument("--lam", default="1e-2")
ap.add_argument("--sampler", default="pce", choices=["pce", "sa", "naive", "random"])
ap.add_argument("--extras", default="0,1,2")
ap.add_argument("--scales", default="1,2,3,5,8,12,20")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--mc-population", type=int, default=40)
ap.add_argument("--mc-iterations", type=int, default=10)
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--penalty-scale", type=float, default=1.0)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-record", action="store_true")
args = ap.parse_args()

model, fm, (bkv, status) = qf.load(args.instance, args.lam)
n, T = fm.n, fm.T
B_BITS = max(1, math.ceil(math.log2(fm.B + 1)))
C_BITS = 4


def rounding_qubo(t, e, scale, w):
    """Exact one-day cost in rounding bits + penalties. Returns (bqm, base, active idx)."""
    target = np.clip(scale * w, -fm.ub, fm.ub)
    base = np.floor(target).astype(np.int64)
    base = np.clip(base, -fm.ub, fm.ub - 1)
    act = np.flatnonzero(np.abs(target) > 1e-6)  # assets the relaxation uses
    k = len(act)
    R = fm.R[t].astype(float)
    lin = fm.lin[t].astype(float)
    absw = fm.absw[t] + np.round(e * fm.model.tc[t])
    negw = fm.negw[t].astype(float)
    v0 = np.zeros(n)
    v0[act] = base[act]
    # v = v0 + E b, E = unit vectors on act
    # |v_i| = |base| + sgn_i b, sgn = +1 if base >= 0 else -1 ; max(-v,0) = max(-base,0) - [base<0] b
    sgn = np.where(base[act] >= 0, 1.0, -1.0)
    neg_slope = np.where(base[act] < 0, -1.0, 0.0)
    Ra = R[np.ix_(act, act)]
    h = Ra.diagonal() + 2 * (R[act] @ v0) + lin[act] + absw[act] * sgn + negw[act] * neg_slope
    J = 2 * np.triu(Ra, 1)
    const = (v0 @ R @ v0 + lin @ v0 + absw @ np.abs(v0) + negw @ np.maximum(-v0, 0) + fm.const[t])
    N = k + B_BITS + C_BITS
    lin_q = np.zeros(N)
    lin_q[:k] = h
    Q = np.zeros((N, N))
    Q[:k, :k] = J
    pen = args.penalty_scale * (float(np.median(np.abs(np.diag(R)))) + 2 * float(np.abs(lin).max()) + 1.0)
    # budget: sum|v| + slack = B  ->  (sum|v0| + sgn.b + sum 2^c s_c - B)^2
    a = np.zeros(N)
    a[:k] = sgn
    a[k:k + B_BITS] = [2**c for c in range(B_BITS)]
    c0 = np.abs(v0).sum() - fm.B
    # capital: net + cash = C -> (sum v0 + b.1 + sum 2^j y_j - C)^2
    a2 = np.zeros(N)
    a2[:k] = 1.0
    a2[k + B_BITS:] = [2**j for j in range(C_BITS)]
    c2 = v0.sum() - fm.C
    for vec, cc in ((a, c0), (a2, c2)):
        lin_q += pen * (vec**2 + 2 * cc * vec)
        Q += pen * 2 * np.triu(np.outer(vec, vec), 1)
        const += pen * cc**2
    quad = {(int(i), int(j)): float(Q[i, j]) for i, j in zip(*np.nonzero(Q))}
    bqm = qf.dimod.BinaryQuadraticModel({i: float(lin_q[i]) for i in range(N)}, quad, float(const), qf.dimod.BINARY)
    return bqm, base, act


def decode(bits, base, act):
    v = np.zeros(n, dtype=np.int64)
    v[act] = base[act] + np.asarray([int(c) for c in bits[: len(act)]])
    return qf.repair_period(fm, v)


def job(task):
    t, e, scale, seed = task
    w = qf.lns.period_relaxation(fm, t, np.round(e * fm.model.tc[t]).astype(np.int64))
    bqm, base, act = rounding_qubo(t, e, scale, w)
    N = bqm.num_variables
    out, nq, nc = [], 0, 0
    t0 = time.time()
    if args.sampler == "naive":
        target = np.clip(scale * w, -fm.ub, fm.ub)
        out.append(qf.repair_period(fm, np.round(target).astype(np.int64)))
    elif args.sampler == "random":
        rng = np.random.default_rng(seed)
        for _ in range(args.top):
            out.append(decode(rng.integers(0, 2, N), base, act))
    elif args.sampler == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=args.top, num_sweeps=1000, seed=seed)
        for s in ss.samples():
            out.append(decode([s[i] for i in range(N)], base, act))
    else:
        from divi.backends import MaestroSimulator
        from divi.qprog.algorithms import PCE, PolyEncoding, SoftObjective
        from divi.qprog.optimizers import MonteCarloOptimizer
        from divi.qprog.problems import BinaryOptimizationProblem
        norm = bqm.copy()
        norm.normalize()
        prog = PCE(BinaryOptimizationProblem(norm), encoding=PolyEncoding(),
                   objective=SoftObjective(alpha=args.alpha), n_layers=args.n_layers,
                   optimizer=MonteCarloOptimizer(population_size=args.mc_population),
                   max_iterations=args.mc_iterations, backend=MaestroSimulator(shots=4000), seed=seed)
        prog.run()
        for s in prog.get_top_solutions(n=args.top, min_prob=0.0, sort_by="energy"):
            out.append(decode(s.bitstring, base, act))
        nq, nc = prog.n_qubits, prog.total_circuit_count
    best_stage = min(int(qf.lns.stage_cost(fm, t, v)[0]) for v in out)
    return t, e, scale, out, nq, nc, N, best_stage, time.time() - t0


if __name__ == "__main__":
    extras = [float(x) for x in args.extras.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    tasks = [(t, e, s, args.seed * 100000 + 1000 * t + int(10 * e) * 50 + int(s)) for t in range(T - 1)
             for e in extras for s in scales]
    print(f"{args.instance} λ={args.lam} B={fm.B}: {len(tasks)} rounding problems, sampler={args.sampler}; BKV {bkv}",
          flush=True)
    t0 = time.time()
    pool = [np.zeros(n, dtype=np.int64)]
    max_q = n_circ = max_vars = 0
    with Pool(args.workers) as p:
        for t, e, s, out, nq, nc, N, bst, dt in p.imap_unordered(job, tasks):
            pool += out
            max_q, n_circ, max_vars = max(max_q, nq), n_circ + nc, max(max_vars, N)
            print(f"  day {t} e={e} scale={s:>4}: {N} vars, {nq} qubits, best day cost {bst} ({dt:.1f}s)", flush=True)
    v, obj = qf.lns.chain_dp(fm, pool)
    score = qf.pc.score_v(model, v)
    assert score.feasible and score.objective == obj
    units = np.abs(v).sum(1)
    print(f"\nRESULT sampler={args.sampler}: {obj}  BKV {bkv}  gap {qf.gap(obj, bkv)}  pool {len(pool)}  "
          f"units/day {units.tolist()}  max qubits {max_q}  circuits {n_circ}  {time.time() - t0:.0f}s", flush=True)
    if not args.no_record:
        settings = (f"sampler={args.sampler} extras={args.extras} scales={args.scales} top={args.top} n_layers={args.n_layers} "
                    f"mc={args.mc_population}x{args.mc_iterations} alpha={args.alpha} penalty_scale={args.penalty_scale} "
                    f"seed={args.seed} pool={len(pool)} circuits={n_circ}")
        q = args.sampler == "pce"
        _, path, _ = qf.pc.record_solution(
            model, qf.pc.v_to_positions(model, v),
            method=f"Q-ROUND relaxation + {args.sampler} rounding + chain DP" + (" (quantum rounding)" if q else " (classical control)"),
            settings=settings, n_qubo_vars=max_vars, max_qubits=max_q, quantum=q, tag=f"qround-{args.sampler}")
        print("recorded", path)
        np.save(qf.HERE / f"pool_{args.instance}_l{args.lam}_{args.sampler}_s{args.seed}.npy", np.array(pool))
