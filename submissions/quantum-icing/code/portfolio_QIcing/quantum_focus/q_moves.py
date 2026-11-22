"""Q-MOVES: move-selection QUBO (the portfolio analogue of the network 2-exchange polish).

Given an incumbent, generate M candidate moves. Each move flips a set of bits of the compact
QUBO and keeps the unary copies "sorted" (filled from copy 0), so the QUBO energy equals the
official objective:
  transfer  a held unit-run of asset a (direction tau, days t1..t2) to a correlated asset b
  flip      a held unit-run of asset a: long <-> short (optionally onto partner b)
  shift     move a unit-run one day earlier / later
  extend / shrink a unit-run by one day at either end
  drop      remove a unit-run
  add       open a hedged run: +1 long a, +1 short b on days t1..t2
Binary y_m = apply move m. With delta_m the +-1 bit-change vector of move m (disjoint supports):
    E(x + D^T y) = E(x) + y . D (lin + S x) + 1/2 y^T (D S D^T) y        (exact)
Moves touching the same (asset, direction, day) cell conflict -> penalty on y_m y_m'.
The move QUBO (M variables) is solved by PCE (Divi), QAOA, or classical controls; selected move
sets are applied, scored by the official referee, and the best improvement is accepted.

Example: python q_moves.py --instance po_a050_t10_s00 --lam 1e-2 --start <sol> --moves 60 --solver pce
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import qf_common as qf
import pf_quantum as pq

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a050_t10_s00")
ap.add_argument("--lam", default="1e-2")
ap.add_argument("--start", default="best")
ap.add_argument("--moves", type=int, default=60)
ap.add_argument("--solver", default="pce", choices=["pce", "qaoa", "sa", "greedy"])
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--mc-population", type=int, default=40)
ap.add_argument("--mc-iterations", type=int, default=8)
ap.add_argument("--cloud", action="store_true", help="QAOA on Qoro qoro_cpu")
ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--max-run", type=int, default=9)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--time-limit", type=float, default=1800)
ap.add_argument("--tag", default="")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

model, fm, (bkv, status) = qf.load(args.instance, args.lam)
cq = pq.CompactQUBO(model)
D = pq.DenseQUBO(cq.objective, cq.n_vars)
T, n, ub = cq.T, cq.n, cq.ub
if args.start == "best":
    found = qf.best_known_sol(args.instance, model.budget, args.lam)
    start_file, v = found[1], qf.pc.positions_to_v(model, found[2])
elif args.start == "empty":
    start_file, v = "empty", np.zeros((T, n), dtype=np.int64)
else:
    start_file = args.start
    v = qf.pc.positions_to_v(model, qf.qt.read_portfolio_solution(args.start)[1])
best = qf.pc.score_v(model, v).objective
start_obj = best
print(f"{args.instance} λ={args.lam}: start {best} ({start_file}); BKV {bkv} gap {qf.gap(best, bkv)}", flush=True)
corr = np.abs(fm.corr.copy())
np.fill_diagonal(corr, -1)
partners = np.argsort(-corr, axis=1)[:, :10]


def runs(v):
    """Unit runs: (asset, tau, copy m, t1, t2) maximal day ranges where copy m of direction tau is on."""
    out = []
    for a in range(n):
        for tau in (1, -1):
            cnt = np.maximum(tau * v[:, a], 0)
            for m in range(ub):
                on = cnt > m
                t = 0
                while t < T:
                    if on[t]:
                        t1 = t
                        while t < T and on[t]:
                            t += 1
                        out.append((a, tau, m, t1, t - 1))
                    else:
                        t += 1
    return out


def cnt_of(v, a, tau, t):
    return max(tau * int(v[t, a]), 0)


def gen_moves(v):
    """Candidate moves as dicts: changes {(asset, tau, day): +1/-1 units}."""
    moves = []
    R = runs(v)
    for a, tau, m, t1, t2 in R:
        if m != max(cnt_of(v, a, tau, t) for t in range(t1, t2 + 1)) - 1:
            pass  # any copy works as long as we always remove the TOP unit of each day
        b = int(rng.choice(partners[a]))
        moves.append(("transfer", {**{(a, tau, t): -1 for t in range(t1, t2 + 1)},
                                   **{(b, tau, t): +1 for t in range(t1, t2 + 1)}}))
        b2 = int(rng.choice(partners[a]))
        moves.append(("flip", {**{(a, tau, t): -1 for t in range(t1, t2 + 1)},
                               **{(b2, -tau, t): +1 for t in range(t1, t2 + 1)}}))
        moves.append(("drop", {(a, tau, t): -1 for t in range(t1, t2 + 1)}))
        if t1 > 0:
            moves.append(("extend-left", {(a, tau, t1 - 1): +1}))
        if t2 < T - 2:
            moves.append(("extend-right", {(a, tau, t2 + 1): +1}))
        if t2 > t1:
            moves.append(("shrink-left", {(a, tau, t1): -1}))
            moves.append(("shrink-right", {(a, tau, t2): -1}))
        if t2 < T - 2:
            moves.append(("shift-right", {(a, tau, t1): -1, (a, tau, t2 + 1): +1}))
        if t1 > 0:
            moves.append(("shift-left", {(a, tau, t2): -1, (a, tau, t1 - 1): +1}))
    for _ in range(max(10, args.moves // 3)):  # new hedged runs
        a = int(rng.integers(n))
        b = int(rng.choice(partners[a]))
        L = int(rng.integers(1, args.max_run + 1))
        t1 = int(rng.integers(0, max(1, T - 1 - L + 1)))
        days = range(t1, min(T - 1, t1 + L))
        moves.append(("add-hedge", {**{(a, 1, t): +1 for t in days}, **{(b, -1, t): +1 for t in days}}))
    rng.shuffle(moves)
    # merge per-cell totals must stay within 0..ub individually
    ok = []
    for kind, ch in moves:
        if len({(a, t) for (a, tau, t) in ch}) == len(ch) and all(
                0 <= cnt_of(v, a, tau, t) + d <= ub for (a, tau, t), d in ch.items()) and all(
                not (d > 0 and cnt_of(v, a, -tau, t) > 0) for (a, tau, t), d in ch.items()):
            ok.append((kind, ch))
    # keep a CONFLICT-FREE set (disjoint (asset, day) cells): no penalty terms needed, the move
    # QUBO then contains only the true objective couplings between moves
    chosen, used = [], set()
    for kind, ch in ok:
        cells = {(a, t) for (a, tau, t) in ch}
        if cells & used:
            continue
        chosen.append((kind, ch))
        used |= cells
        if len(chosen) >= args.moves:
            break
    return chosen


def delta_vector(v, ch):
    """Bit-change vector for a move: remove/add the TOP unit copy of each touched cell."""
    d = np.zeros(cq.n_vars)
    for (a, tau, t), dd in ch.items():
        c = cnt_of(v, a, tau, t)
        if dd < 0:
            d[cq.x(a, c - 1, tau, t)] = -1
        else:
            d[cq.x(a, c, tau, t)] = +1
    return d


def move_qubo(v, moves):
    x = cq.encode_v(v).astype(float)
    Dm = np.array([delta_vector(v, ch) for _, ch in moves])  # (M, N)
    g = Dm @ (D.lin + D.S @ x)
    G = Dm @ D.S @ Dm.T
    h = g + 0.5 * np.diag(G)
    J = np.triu(G, 1)
    M = len(moves)
    cells = [{(a, t) for (a, tau, t) in ch} for _, ch in moves]
    pen = 10 * (np.abs(h).max() + np.abs(J).max() + 1)
    for i in range(M):
        for j in range(i + 1, M):
            if cells[i] & cells[j]:
                J[i, j] += pen
    quad = {(int(i), int(j)): float(J[i, j]) for i, j in zip(*np.nonzero(J))}
    return qf.dimod.BinaryQuadraticModel({i: float(h[i]) for i in range(M)}, quad, 0.0, qf.dimod.BINARY)


def apply(v, moves, sel, gains=None, strict=False):
    """Apply selected moves. strict: None on any conflict; else keep the best-gain non-conflicting
    subset (ordered by single-move gain) and repair days that break budget/capital."""
    nv = v.copy()
    touched = set()
    order = sorted(sel, key=lambda k: gains[k]) if gains is not None else list(sel)
    for k in order:
        cells = {(a, t) for (a, tau, t) in moves[k][1]}
        if cells & touched:
            if strict:
                return None
            continue
        touched |= cells
        for (a, tau, t), dd in moves[k][1].items():
            nv[t, a] += tau * dd
    if not strict:
        for t in range(T):
            if not (np.abs(nv[t]).sum() <= fm.B and fm.netmin <= nv[t].sum() <= fm.C):
                nv[t] = qf.repair_period(fm, nv[t])
    return nv


def solve(bqm):
    M = bqm.num_variables
    if args.solver == "greedy":
        return [[int(i) for i in np.flatnonzero([bqm.get_linear(i) < 0 for i in range(M)])][:1]], 0, 0
    if args.solver == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=32, num_sweeps=500, seed=int(rng.integers(1 << 30)))
        return [[i for i in range(M) if s[i]] for s in ss.samples()], 0, 0
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.algorithms import PCE, QAOA, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem
    norm = bqm.copy()
    norm.normalize()
    be = (QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster="qoro_cpu")) if args.cloud
          else MaestroSimulator(shots=args.shots))
    opt = MonteCarloOptimizer(population_size=args.mc_population)
    if args.solver == "pce":
        prog = PCE(BinaryOptimizationProblem(norm), encoding=PolyEncoding(), objective=SoftObjective(alpha=2.0),
                   n_layers=args.n_layers, optimizer=opt, max_iterations=args.mc_iterations, backend=be,
                   seed=int(rng.integers(1 << 30)))
        prog.run()
        sols = prog.get_top_solutions(n=32, min_prob=0.0, sort_by="energy")
    else:
        prog = QAOA(BinaryOptimizationProblem(norm), n_layers=args.n_layers, optimizer=opt,
                    max_iterations=args.mc_iterations, backend=be, seed=int(rng.integers(1 << 30)))
        prog.run()
        sols = prog.get_top_solutions(n=64)
    return [[i for i, c in enumerate(s.bitstring) if c == "1"] for s in sols], prog.n_qubits, prog.total_circuit_count


log, circuits, max_q, t_start = [], 0, 0, time.time()
for it in range(args.iters):
    if time.time() - t_start > args.time_limit:
        break
    moves = gen_moves(v)
    if not moves:
        break
    bqm = move_qubo(v, moves)
    t0 = time.time()
    try:
        sels, nq, nc = solve(bqm)
    except Exception as err:  # noqa: BLE001
        print(f"it {it}: solver failed {type(err).__name__}: {err}", flush=True)
        continue
    circuits += nc
    max_q = max(max_q, nq)
    best_here, best_v, best_sel = None, None, None
    gains = [bqm.get_linear(i) for i in range(bqm.num_variables)]
    for sel in sels:
        nv = apply(v, moves, sel, gains)
        if nv is None:
            continue
        sc = qf.pc.score_v(model, nv)
        if sc.feasible and (best_here is None or sc.objective < best_here):
            best_here, best_v, best_sel = sc.objective, nv, sel
    if best_here is not None and best_here < best:
        best, v = best_here, best_v
    kinds = sorted({moves[k][0] for k in (best_sel or [])})
    log.append(dict(it=it, n_moves=len(moves), qubits=nq, n_selected=len(best_sel or []), kinds=kinds,
                    cand=best_here, incumbent=best, sec=round(time.time() - t0, 1)))
    print(f"it {it:3d} moves={len(moves)} q={nq} sel={len(best_sel or [])} {kinds} {time.time() - t0:4.1f}s "
          f"cand {best_here} incumbent {best} gap {qf.gap(best, bkv)}", flush=True)

score = qf.pc.score_v(model, v)
units = np.abs(v).sum(1).tolist()
print(f"\nFINAL {score}  BKV {bkv}  gap {qf.gap(score.objective, bkv)}  units/day {units}  circuits {circuits}  "
      f"max qubits {max_q}  {time.time() - t_start:.0f}s")
quantum = args.solver in ("pce", "qaoa")
settings = (f"move-selection QUBO, solver={args.solver} moves={args.moves} n_layers={args.n_layers} "
            f"mc={args.mc_population}x{args.mc_iterations} cloud={args.cloud} iters={len(log)} seed={args.seed} "
            f"start={start_file} (start objective {start_obj}) circuits={circuits}")
if score.objective < start_obj:
    _, path, _ = qf.pc.record_solution(
        model, qf.pc.v_to_positions(model, v),
        method=f"Q-MOVES move-selection QUBO ({args.solver})" + ("" if quantum else " (classical control)"),
        settings=settings, n_qubo_vars=args.moves, max_qubits=max_q, quantum=quantum, tag=f"qmoves-{args.solver}{args.tag}")
    print("recorded", path)
(qf.HERE / "logs" / f"qmoves_{args.instance}_l{args.lam}_{args.solver}_s{args.seed}{args.tag}.json").write_text(
    json.dumps(dict(args=vars(args), start=str(start_file), start_obj=start_obj, log=log, final=score.objective,
                    bkv=bkv), indent=1))
