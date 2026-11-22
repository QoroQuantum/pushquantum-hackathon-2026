"""Divi proposes batches of topology changes, the exact routing LP decides which are kept.

    NETWORK_SOL_DIR=solutions/agent_divi taskset -c 11-15 .venv/bin/python network_divi.py \
        --minutes 25 --driver divi --seed 0

One step of the search, around the current topology T (network24, out/in-degree 2):

1. Score every 2-exchange that touches one of the most congested arcs of T (largest LP duals),
   plus some dual-biased random ones, with the LP:  Δ_i = LP(T ∘ move_i) − LP(T).
2. Keep a pool of M moves (default 16) with the best Δ_i, at most a few per arc.
3. Score every compatible pair with the LP:  J_ij = Δ_ij − Δ_i − Δ_j  (exact pairwise interaction).
   Pairs that share an arc, or whose union is not a valid topology, get a penalty.
4. QUBO over "apply move i":  E(x) = Σ Δ_i x_i + Σ J_ij x_i x_j  (normalised).
5. Sample it with Divi QAOA on the local MaestroSimulator (M qubits), decode the top bitstrings into
   batches of compatible moves, score every batch with the LP.  The same QUBO is also sampled by
   dwave's simulated annealing, by exact enumeration (2^M) and by random batches, so every step
   gives a paired comparison of the samplers on identical QUBOs.
6. Move to the best batch of the *driver* sampler (default: Divi) with SA-style acceptance; bank
   every strictly better topology through ``network_pipeline.bank`` (checker-verified .sol).

Uses the LOCAL simulator only. ``qoblib_tools.py`` and ``network_pipeline.py`` are not modified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

import qoblib_tools as qt
from network_design import pipeline as npl

N = 24
Arc = tuple[int, int]
Move = tuple[Arc, Arc]  # ((a,b),(c,d)) → removes both, adds (a,d),(c,b)


# ── Moves ──────────────────────────────────────────────────────────────────────────────

def added(m: Move) -> tuple[Arc, Arc]:
    (a, b), (c, d) = m
    return (a, d), (c, b)


def apply_moves(cur: frozenset, moves: list[Move]) -> list[Arc] | None:
    """Apply several 2-exchanges at once; None unless the result is a valid topology."""
    rem = [a for m in moves for a in m]
    add = [a for m in moves for a in added(m)]
    if len(set(rem)) != len(rem) or len(set(add)) != len(add):
        return None
    if any(a not in cur for a in rem) or any(i == j for i, j in add):
        return None
    new = (cur - set(rem)) | set(add)
    if len(new) != len(cur) or not qt.is_valid_topology(N, new):
        return None
    return sorted(new)


def lp_value(arcs: list[Arc]) -> float:
    return npl.score_topology(arcs)[0]


class Scorer:
    """LP scoring through a small process pool, with a cache keyed by the arc set."""

    def __init__(self, pool: ProcessPoolExecutor, workers: int):
        self.pool, self.workers, self.cache, self.n_lp = pool, workers, {}, 0

    def score(self, topologies: list[list[Arc]]) -> list[float]:
        keys = [tuple(t) for t in topologies]
        todo = list({k for k in keys if k not in self.cache})
        if todo:
            vals = self.pool.map(lp_value, todo, chunksize=max(1, len(todo) // (4 * self.workers)))
            self.cache.update(zip(todo, vals))
            self.n_lp += len(todo)
        return [self.cache[k] for k in keys]


# ── QUBO construction ──────────────────────────────────────────────────────────────────

def candidate_moves(cur: list[Arc], duals: np.ndarray, rng, n_hot: int, n_random: int) -> list[Move]:
    cur_set = set(cur)
    order = np.argsort(-duals)
    hot = [cur[i] for i in order[:n_hot]]
    w = np.clip(duals, 0, None) + 1e-9
    w /= w.sum()
    seen, out = set(), []

    def push(p: Arc, q: Arc):
        (a, b), (c, d) = p, q
        if a == c or b == d or a == d or c == b or (a, d) in cur_set or (c, b) in cur_set:
            return
        key = frozenset((p, q))
        if key not in seen:
            seen.add(key)
            out.append((p, q))

    for p in hot:
        for q in cur:
            if q != p:
                push(p, q)
    for _ in range(n_random):
        p = cur[rng.choice(len(cur), p=w)] if rng.random() < 0.5 else cur[rng.integers(len(cur))]
        push(p, cur[rng.integers(len(cur))])
    return out


def pick_pool(moves: list[Move], deltas: list[float], M: int, per_arc: int, rng,
              explore: float) -> list[int]:
    """Best Δ first (with a little noise for diversity), at most ``per_arc`` moves per arc."""
    noise = rng.exponential(explore, len(moves)) if explore > 0 else np.zeros(len(moves))
    order = np.argsort(np.asarray(deltas) + noise)
    use: dict[Arc, int] = {}
    pool = []
    for i in order:
        m = moves[i]
        if any(use.get(a, 0) >= per_arc for a in m):
            continue
        pool.append(int(i))
        for a in m:
            use[a] = use.get(a, 0) + 1
        if len(pool) == M:
            break
    return pool


def build_qubo(cur_set, cur_z, pool_moves, d1, scorer, k_batch: int):
    """Linear Δ_i, exact pairwise J_ij from LP on every compatible pair, penalty on conflicts."""
    M = len(pool_moves)
    pairs, tops = [], []
    conflict = []
    for i, j in itertools.combinations(range(M), 2):
        t = apply_moves(cur_set, [pool_moves[i], pool_moves[j]])
        if t is None:
            conflict.append((i, j))
        else:
            pairs.append((i, j))
            tops.append(t)
    zs = scorer.score(tops)
    J = {}
    d2 = {}
    for (i, j), z in zip(pairs, zs):
        d2[i, j] = z - cur_z
        J[i, j] = d2[i, j] - d1[i] - d1[j]
    scale = max(1.0, max(abs(v) for v in list(d1) + list(J.values())))
    pen = 3.0 * scale  # one conflicting pair costs more than any single gain
    lin = {i: d1[i] / scale for i in range(M)}
    quad = {k: v / scale for k, v in J.items()}
    for k in conflict:
        quad[k] = pen / scale
    bqm = dimod.BinaryQuadraticModel(lin, quad, 0.0, dimod.BINARY)
    # Batch size: in a local optimum every single move and every pair is worse (all scored exactly
    # above), and the min-max objective makes J_ij < 0 for redundant moves, so the unconstrained
    # QUBO ground state is either empty or a huge batch. Ask for exactly k_batch moves instead.
    bqm_free = bqm.copy()
    if k_batch > 0:
        bqm.add_linear_equality_constraint([(i, 1.0) for i in range(M)], 3.0, -k_batch)
    return bqm, bqm_free, conflict, d2, scale


# ── Samplers: each returns a list of move-index subsets (best-first) ──────────────────

def energies_all(bqm: dimod.BinaryQuadraticModel, M: int) -> tuple[np.ndarray, np.ndarray]:
    X = ((np.arange(2 ** M)[:, None] >> np.arange(M)[None, :]) & 1).astype(np.float64)
    h = np.array([bqm.linear[i] for i in range(M)])
    Q = np.zeros((M, M))
    for (i, j), v in bqm.quadratic.items():
        Q[i, j] = v
    E = X @ h + np.einsum("ni,ij,nj->n", X, Q, X)
    return X.astype(np.int8), E


def sample_exact(X, E, K):
    idx = np.argsort(E)[:K]
    return [tuple(np.flatnonzero(X[i])) for i in idx]


def sample_sa(bqm, K, seed):
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=4 * K, seed=seed).aggregate()
    out = []
    for rec in sorted(ss.data(["sample", "energy"]), key=lambda r: r.energy):
        out.append(tuple(v for v in sorted(rec.sample) if rec.sample[v]))
        if len(out) == K:
            break
    return out


def sample_random(M, K, k, conflict_set, rng):
    """K random conflict-free k-batches (the 'no model' baseline)."""
    out = []
    for _ in range(50 * K):
        s = tuple(sorted(rng.choice(M, size=min(k, M), replace=False).tolist()))
        if s not in out and not any(p in conflict_set for p in itertools.combinations(s, 2)):
            out.append(s)
        if len(out) == K:
            break
    return out


def _fixed_weight_problem(bqm, k):
    """QUBO restricted to batches of exactly k moves: Dicke-state start + ring XY mixer.

    The XY mixer conserves Hamming weight, so every QAOA sample is a k-batch (no cardinality
    penalty in the cost Hamiltonian, feasible by construction)."""
    from divi.hamiltonians import xy_mixer
    from divi.qprog.initial_states import DickeState
    from divi.qprog.problems import BinaryOptimizationProblem

    M = bqm.num_variables

    class FixedWeight(BinaryOptimizationProblem):
        @property
        def mixer_hamiltonian(self):
            return xy_mixer([(i, (i + 1) % M) for i in range(M)], n_qubits=M)

        @property
        def recommended_initial_state(self):
            return DickeState(k, n_qubits=M)

    return FixedWeight(bqm)


def sample_divi(bqm, K, seed, args, stats, bqm_free=None):
    from divi.backends import MaestroSimulator
    from divi.qprog.algorithms import PCE, QAOA, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    if args.quantum == "qaoa" and args.xy and bqm_free is not None and args.k_batch > 0:
        problem = _fixed_weight_problem(bqm_free, args.k_batch)
    else:
        problem = BinaryOptimizationProblem(bqm)
    backend = MaestroSimulator(shots=args.shots)
    opt = MonteCarloOptimizer(population_size=args.mc_pop)
    if args.quantum == "qaoa":
        prog = QAOA(problem, n_layers=args.layers, optimizer=opt, max_iterations=args.mc_iter,
                    backend=backend, seed=seed)
        prog.run()
        top = prog.get_top_solutions(n=K)
    else:
        prog = PCE(problem, encoding=PolyEncoding(), objective=SoftObjective(alpha=2.0),
                   n_layers=args.layers, optimizer=opt, max_iterations=args.mc_iter,
                   backend=backend, seed=seed)
        prog.run()
        top = prog.get_top_solutions(n=K, min_prob=0.0, sort_by="energy")
    stats["qubits"] = int(getattr(prog, "n_qubits", None) or problem.cost_hamiltonian.num_qubits)
    stats["circuits"] = int(prog.total_circuit_count)
    var_order = list(bqm.variables)
    out = []
    for s in top:
        bits = s.bitstring
        out.append(tuple(sorted(var_order[k] for k, b in enumerate(bits) if b == "1")))
    stats["probs"] = [float(getattr(s, "prob", 0.0) or 0.0) for s in top]
    return out


def repair(subset, conflict_set, d1):
    """Drop conflicting moves greedily (worst Δ first) so the batch is applicable."""
    s = sorted(subset, key=lambda i: d1[i])
    keep = []
    for i in s:
        if all((min(i, j), max(i, j)) not in conflict_set for j in keep):
            keep.append(i)
    return tuple(sorted(keep))


# ── Search ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--driver", choices=["divi", "sa", "exact", "random"], default="divi",
                    help="whose batches the search moves to (all samplers are always logged)")
    ap.add_argument("--quantum", choices=["qaoa", "pce"], default="qaoa")
    ap.add_argument("--M", type=int, default=16, help="moves in the pool = QUBO variables")
    ap.add_argument("--K", type=int, default=16, help="batches per sampler per step")
    ap.add_argument("--k-batch", type=int, default=3, help="moves per batch (constraint in the QUBO)")
    ap.add_argument("--oracle", type=int, default=600,
                    help="score every conflict-free k-batch when there are at most this many (0: off)")
    ap.add_argument("--hot", type=int, default=8, help="congested arcs whose 2-exchanges are scored")
    ap.add_argument("--random-moves", type=int, default=120)
    ap.add_argument("--per-arc", type=int, default=3)
    ap.add_argument("--explore", type=float, default=300.0, help="noise (LP units) in the pool choice")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--xy", type=int, default=1, help="QAOA with Dicke start + XY mixer (exact k-batches)")
    ap.add_argument("--take-construction", type=int, default=1,
                    help="also move to an improving single/pair found while building the QUBO")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--mc-pop", type=int, default=30)
    ap.add_argument("--mc-iter", type=int, default=8)
    ap.add_argument("--t0", type=float, default=1500.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4, help="LP worker processes (+1 main process)")
    ap.add_argument("--start", default=None, help=".sol to start from (default: our banked file)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    demand = qt.load_network_demand(N)
    rng = np.random.default_rng(args.seed)
    start = Path(args.start) if args.start else npl.banked_path(N)
    _, cur, _ = qt.read_network_solution(start)
    cur = sorted(cur)
    assert qt.is_valid_topology(N, cur)
    banked = npl.banked_score(N, demand)
    log_path = Path(args.log) if args.log else npl.SOL_DIR / f"divi_run_{args.driver}_s{args.seed}.jsonl"
    log_f = log_path.open("a")
    samplers = ["divi", "sa", "exact", "random"]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=npl._init_worker,
                             initargs=(demand,)) as pool:
        scorer = Scorer(pool, args.workers)
        cur_z, cur_d = npl.score_topology(cur, demand)
        best_z, best = cur_z, cur
        print(f"start {start.name}: LP {cur_z:.1f} (ceil {math.ceil(cur_z - 1e-6)}), banked {banked}, "
              f"driver={args.driver} quantum={args.quantum} M={args.M}", flush=True)
        t_start = time.time()
        deadline = t_start + 60 * args.minutes
        step, since_best = 0, 0
        while time.time() < deadline:
            t_it = time.time()
            frac = (t_it - t_start) / (deadline - t_start)
            temp = args.t0 * (args.t1 / args.t0) ** frac
            cur_set = frozenset(cur)
            # 1. single moves
            moves = candidate_moves(cur, cur_d, rng, args.hot, args.random_moves)
            singles = [apply_moves(cur_set, [m]) for m in moves]
            keep = [i for i, t in enumerate(singles) if t is not None]
            moves = [moves[i] for i in keep]
            zs = scorer.score([singles[i] for i in keep])
            deltas = [z - cur_z for z in zs]
            # 2. pool + 3. pairs + 4. QUBO
            pool_idx = pick_pool(moves, deltas, args.M, args.per_arc, rng, args.explore)
            pm = [moves[i] for i in pool_idx]
            d1 = [deltas[i] for i in pool_idx]
            M = len(pm)
            bqm, bqm_free, conflict, d2, scale = build_qubo(cur_set, cur_z, pm, d1, scorer, args.k_batch)
            conflict_set = set(conflict)
            t_build = time.time() - t_it
            # 5. samplers on the same QUBO
            X, E = energies_all(bqm, M)
            e_min = float(E.min())
            props, info, timing = {}, {}, {}
            t = time.time()
            dstats = {}
            props["divi"] = sample_divi(bqm, args.K, args.seed * 100003 + step, args, dstats, bqm_free)
            timing["divi"] = time.time() - t
            gs = tuple(np.flatnonzero(X[int(np.argmin(E))]))
            divi_gs_prob = sum(p for s, p in zip(props["divi"], dstats["probs"]) if s == gs)
            t = time.time()
            props["sa"] = sample_sa(bqm, args.K, args.seed * 100003 + step)
            timing["sa"] = time.time() - t
            props["exact"] = sample_exact(X, E, args.K)
            props["random"] = sample_random(M, args.K, args.k_batch, conflict_set, rng)
            batches = {}
            for name in samplers:
                lst = []
                for s in props[name]:
                    r = repair(s, conflict_set, d1)
                    if r and r not in lst:
                        lst.append(r)
                props[name] = lst
                for r in lst:
                    if r not in batches:
                        batches[r] = apply_moves(cur_set, [pm[i] for i in r])
            batches = {k: v for k, v in batches.items() if v is not None}
            keys = list(batches)
            bz = dict(zip(keys, scorer.score([batches[k] for k in keys])))
            # per-sampler best real LP result
            for name in samplers:
                cand = [(bz[r], r) for r in props[name] if r in bz]
                if cand:
                    z, r = min(cand)
                    e = float(sum(bqm.linear[i] for i in r) + sum(bqm.quadratic.get((i, j), 0.0)
                                                                for i, j in itertools.combinations(r, 2)))
                    info[name] = {"best_dz": z - cur_z, "size": len(r), "n_batches": len(cand),
                                  "qubo_e": e, "hit_ground": bool(abs(e - e_min) < 1e-9)}
                else:
                    info[name] = {"best_dz": math.inf, "size": 0, "n_batches": 0}
            # oracle: every conflict-free k-batch, to rank what the samplers proposed
            oracle = None
            all_k = [c for c in itertools.combinations(range(M), args.k_batch)
                     if not any(p in conflict_set for p in itertools.combinations(c, 2))]
            if args.oracle and len(all_k) <= args.oracle:
                tops = {c: apply_moves(cur_set, [pm[i] for i in c]) for c in all_k}
                tops = {c: t for c, t in tops.items() if t is not None}
                ozs = dict(zip(tops, scorer.score(list(tops.values()))))
                vals = np.sort(np.array(list(ozs.values())))
                oracle = {"n": len(vals), "best_dz": float(vals[0] - cur_z)}
                for name in samplers:
                    if info[name]["n_batches"]:
                        z = info[name]["best_dz"] + cur_z
                        info[name]["rank"] = int((vals < z - 1e-6).sum()) + 1
            # 6. move with the driver's best batch
            drv = props[args.driver]
            cand = [(bz[r], args.driver, batches[r]) for r in drv if r in bz]
            if args.take_construction:
                if deltas:
                    i = int(np.argmin(deltas))
                    cand.append((cur_z + deltas[i], "single", apply_moves(cur_set, [moves[i]])))
                if d2:
                    (i, j), dz = min(d2.items(), key=lambda kv: kv[1])
                    cand.append((cur_z + dz, "pair", apply_moves(cur_set, [pm[i], pm[j]])))
            cand.sort(key=lambda c: c[0])
            accepted, source = False, None
            if cand:
                z, source, top = cand[0]
                if z <= cur_z - 1e-9 or rng.random() < math.exp(-(z - cur_z) / temp):
                    cur, cur_z = top, z
                    cur_d = npl.score_topology(cur, demand)[1]
                    accepted = True
            step += 1
            since_best += 1
            msg = ""
            if cur_z < best_z - 1e-6:
                best_z, best, since_best = cur_z, cur, 0
                if banked is None or math.ceil(best_z - 1e-6) < banked:
                    ok, score, msg = npl.bank(N, demand, best, f"network_divi src={source} driver={args.driver} "
                                              f"{args.quantum} M={M} seed={args.seed} step={step}")
                    if ok:
                        banked = score
            if since_best >= args.patience:
                cur, cur_z = best, best_z
                cur_d = npl.score_topology(cur, demand)[1]
                since_best = 0
                msg += " (restart from best)"
            best_single = min(deltas) if deltas else math.inf
            best_pair = min(d2.values()) if d2 else math.inf
            rec = {"step": step, "t": round(time.time() - t_start, 1), "T": round(temp, 1),
                   "cur_z": cur_z, "best_z": best_z, "banked": banked, "accepted": accepted, "source": source,
                   "M": M, "n_single": len(moves), "n_pairs": len(d2), "n_conflict": len(conflict),
                   "best_single_dz": best_single, "best_pair_dz": best_pair,
                   "qubits": dstats.get("qubits"), "circuits": dstats.get("circuits"),
                   "divi_p_ground": divi_gs_prob, "gs_size": len(gs),
                   "oracle": oracle, "k_batch": args.k_batch,
                   "t_build": round(t_build, 2), "t_divi": round(timing["divi"], 2),
                   "t_sa": round(timing["sa"], 3), "samplers": info, "n_lp": scorer.n_lp}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"step {step:3d} t={rec['t']:6.0f}s T={temp:6.0f} cur {cur_z:9.1f} best {best_z:9.1f} | "
                  f"1-move {best_single:+7.0f} pair {best_pair:+7.0f} | "
                  + (f"all{args.k_batch}: {oracle['best_dz']:+7.0f} " if oracle else "")
                  + " ".join(f"{s}:{info[s]['best_dz']:+7.0f}/r{info[s].get('rank', '-')}" for s in samplers)
                  + f" | {'acc ' + source if accepted else 'rej'} {msg}", flush=True)
    print(f"done: {step} steps, best LP {best_z:.1f} (ceil {math.ceil(best_z - 1e-6)}), banked {banked} "
          f"(gap {npl.gap(N, banked):+.3%}), {scorer.n_lp} LPs", flush=True)


if __name__ == "__main__":
    main()
