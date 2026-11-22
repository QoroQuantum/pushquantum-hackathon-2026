"""Quantum-proposer LNS where a variable is a head swap between two arcs (a 2-exchange).

The cell encoding of goat_lns.py (arc f takes slot s, r*r bits) carries permutation
constraints that a PCE parity encoding knows nothing about. Here the variables are the moves
themselves: y_e = 1 if the two arcs of pair e = (f, g) swap heads, (t_f -> h_f), (t_g -> h_g)
=> (t_f -> h_g), (t_g -> h_f). Any set of DISJOINT swaps keeps every node at 2 in / 2 out, so
the only constraint is "each arc in at most one swap", and a sample that breaks it is repaired by
dropping swaps. The objective is the demand-weighted hop count S (Spearman 0.85-0.89 against the
exact LP on network24 blocks, see block_fidelity.py), expanded EXACTLY to second order around
the current topology:

    S(y) ~ S0 + sum_e h_e y_e + sum_{e,e' disjoint} J_ee' y_e y_e'
    h_e   = S(swap e) - S0
    J_ee' = S(swap e, e') - S(swap e) - S(swap e') + S0

so the QUBO is right on every single swap and every pair of swaps, and only misses third and
higher-order interactions. Overlapping pairs get a penalty. With the poly encoding, a block of
r arcs has up to r(r-1)/2 swaps on about r qubits: 190 swaps on 19 qubits at r = 20.

Proposers on the same block, same LP budget (top K by hop count -> exact LP):
  quantum  PCE on the QUBO: sampled bitstrings + the per-swap expectation values as soft scores;
  sa       simulated annealing on the same QUBO;
  greedy   take improving swaps by h_e, best first, skipping conflicts (classical, no pairs);
  sampled  random disjoint swap sets, best by hop count;
  random   random disjoint swap sets.

    venv/bin/python TheGOAT_Testing/goat_swaps.py --tag sw_r20 --r 20 --steps 40
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import csr_array  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.sa_search import lp_eval  # noqa: E402
from network_lns import load_start, pick_free  # noqa: E402

OUT = ROOT / "runs" / "goat" / "swaps"


def hop_cost(demand, arcs):
    """Demand-weighted hop count sum_ij d_ij hops(i, j); inf if some demand has no path."""
    n = len(demand)
    t, h = np.array(arcs).T
    D = shortest_path(csr_array((np.ones(len(arcs)), (t, h)), shape=(n, n)), unweighted=True, directed=True)
    return float(np.where(demand > 0, demand * D, 0.0).sum())


class SwapBlock:
    """The disjoint head swaps among r freed arcs of a topology."""

    def __init__(self, arcs, free, n):
        self.arcs, self.n = list(arcs), n
        current = set(arcs)
        self.swaps = []                                    # (f, g) arc indices, f < g
        for i, f in enumerate(free):
            for g in free[i + 1:]:
                (tf, hf), (tg, hg) = arcs[f], arcs[g]
                if tf == tg or hf == hg or tf == hg or tg == hf:
                    continue                               # no change, or a self-loop
                if (tf, hg) in current or (tg, hf) in current:
                    continue                               # would duplicate an existing arc
                self.swaps.append((f, g))
        self.m = len(self.swaps)
        arcs_of = [set(e) for e in self.swaps]
        self.conflict = np.array([[i != j and bool(arcs_of[i] & arcs_of[j]) for j in range(self.m)]
                                  for i in range(self.m)], dtype=bool)

    def apply(self, chosen):
        new = list(self.arcs)
        for k in chosen:
            f, g = self.swaps[k]
            (tf, hf), (tg, hg) = self.arcs[f], self.arcs[g]
            new[f], new[g] = (tf, hg), (tg, hf)
        return new

    def repair(self, order):
        """Swaps taken in the given order, skipping any that shares an arc with one taken."""
        taken, used = [], set()
        for k in order:
            f, g = self.swaps[k]
            if f not in used and g not in used:
                taken.append(k)
                used |= {f, g}
        return tuple(sorted(taken))

    def topology(self, chosen):
        new = self.apply(chosen)
        if len(set(new)) != len(new) or not qt.is_valid_topology(self.n, new):
            return None
        return new

    def unique_valid(self, sets):
        out, seen = [], set()
        for c in sets:
            if not c or c in seen:
                continue
            seen.add(c)
            t = self.topology(c)
            if t is not None:
                out.append((c, t))
        return out

    def random_sets(self, k, rng, density=0.3):
        sets = []
        for _ in range(4 * k):
            order = rng.permutation(self.m)
            sets.append(self.repair(order[: max(1, int(density * self.m * rng.random()) + 1)]))
        return self.unique_valid(sets)[:k]


def second_order_qubo(demand, block):
    """Exact h and J of the hop count around the current topology (J only on disjoint pairs)."""
    s0 = hop_cost(demand, block.arcs)
    single = np.array([hop_cost(demand, block.apply([k])) for k in range(block.m)])
    h = single - s0
    J = np.zeros((block.m, block.m))
    for i in range(block.m):
        for j in range(i + 1, block.m):
            if not block.conflict[i, j]:
                J[i, j] = hop_cost(demand, block.apply([i, j])) - single[i] - single[j] + s0
    return h, J, s0


def qubo_matrix(h, J, block, penalty, trust=0.0):
    """Upper-triangular QUBO. ``trust`` charges every swap trust * std(h): a trust region in the
    number of swaps, keeping the optimum where the second-order expansion is still accurate
    (measured: without it the QUBO optimum stacks many swaps and LP-scores worse than random)."""
    h = h + trust * (np.std(h) + 1e-12)
    scale = max(np.abs(h).max(), np.abs(J).max(), 1e-9)
    Q = (J + np.diag(h)) / scale
    P = penalty * max(np.abs(Q).max(), 1e-9)
    iu = np.triu_indices(block.m, 1)
    Q[iu] += P * block.conflict[iu]
    return Q


def qubo_energy(Q, chosen):
    idx = list(chosen)
    return float(Q[np.ix_(idx, idx)].sum()) if idx else 0.0     # Q is upper triangular


def propose_sa(block, Q, k, rng, sweeps=300, restarts=6):
    """Bit-flip SA on the QUBO (conflict penalty included); visited states repaired and kept."""
    m = block.m
    Qs = Q + Q.T - np.diag(np.diag(Q))
    found = {}
    for _ in range(restarts):
        y = np.zeros(m, dtype=bool)
        e = 0.0
        T0 = np.abs(np.diag(Q)).max() + 1e-9
        for t in range(sweeps * m):
            T = T0 * (1 - t / (sweeps * m)) + 1e-6
            i = int(rng.integers(m))
            delta = (1 - 2 * y[i]) * (Q[i, i] + Qs[i][y].sum() - (Qs[i, i] if y[i] else 0.0))
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                y[i] = not y[i]
                e += delta
                if y.any():
                    found[tuple(np.flatnonzero(y))] = e
    ranked = sorted(found, key=found.get)
    h = np.diag(Q)
    return block.unique_valid(block.repair(sorted(c, key=lambda i: h[i])) for c in ranked)[:k]


def propose_greedy(block, h, k, rng):
    """Improving swaps best first, conflicts skipped; then Gumbel-perturbed variants."""
    scale = 0.3 * (np.abs(h).std() + 1e-12)
    sets = [block.repair([i for i in np.argsort(h) if h[i] < 0])]
    for _ in range(4 * k):
        hp = h + scale * rng.gumbel(size=block.m)
        sets.append(block.repair([i for i in np.argsort(hp) if hp[i] < 0]))
    return block.unique_valid(sets)[:k]


def propose_quantum(block, Q, h, k, args, seed, rng):
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.algorithms import PCE, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    backend = (QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster=args.cloud, tag="goat-swaps"))
               if args.cloud else MaestroSimulator(shots=args.shots))
    optimizer = (MonteCarloOptimizer(population_size=args.pop) if args.optimizer == "mc"
                 else ScipyOptimizer(method={"cobyla": ScipyMethod.COBYLA, "lbfgs": ScipyMethod.L_BFGS_B}[args.optimizer]))
    prog = PCE(BinaryOptimizationProblem(Q), encoding=PolyEncoding(), objective=SoftObjective(alpha=args.alpha),
               n_layers=args.layers, optimizer=optimizer, max_iterations=args.iters, backend=backend, seed=seed)
    prog.run()
    sols = prog.get_top_solutions(n=args.top, sort_by="energy")
    sets = []
    # soft read-out: x_e = 1 where <P_e> = +1, so <P_e> ranks the swaps; perturbed for variety
    expect = prog._variable_expectations(np.atleast_2d(prog.best_params))[0]
    scale = 0.3 * (expect.std() + 1e-12)
    for j in range(k):
        s = expect + (scale * rng.gumbel(size=block.m) if j else 0.0)
        sets.append(block.repair([i for i in np.argsort(-s) if s[i] > 0]))
    for sol in sols:
        bits = np.array([int(c) for c in sol.bitstring])
        sets.append(block.repair(sorted(np.flatnonzero(bits), key=lambda i: h[i])))
    raw_ok = float(np.mean([not any(block.conflict[i, j] for i in np.flatnonzero(b) for j in np.flatnonzero(b))
                            for b in (np.array([int(c) for c in s.bitstring]) for s in sols)])) if sols else 0.0
    info = {"qubits": int(prog.n_qubits), "circuits": int(prog.total_circuit_count), "raw_disjoint_share": raw_ok}
    return block.unique_valid(sets)[:k], info


def lp_top(demand, cands, K):
    if not cands:
        return math.inf, None, 0
    hc = np.array([hop_cost(demand, t) for _, t in cands])
    best = (math.inf, None)
    order = np.argsort(hc)[:K]
    for i in order:
        z, _ = lp_eval(demand, cands[i][1])
        if z < best[0]:
            best = (z, cands[i][1])
    return best[0], best[1], len(order)


def step(demand, cur, hot, args, rng, seed):
    t0 = time.time()
    free = pick_free(cur, hot, args.r, rng)
    block = SwapBlock(cur, free, args.n)
    h, J, s0 = second_order_qubo(demand, block)
    Q = qubo_matrix(h, J, block, args.penalty, args.trust)
    rec = {"free": free, "n_swaps": block.m, "n_improving_singles": int((h < 0).sum()),
           "t_qubo": round(time.time() - t0, 1)}
    props = {}
    t = time.time()
    props["quantum"], info = propose_quantum(block, Q, h, args.cands, args, seed, rng)
    rec |= info
    rec["t_quantum"] = round(time.time() - t, 1)
    props["sa"] = propose_sa(block, Q, args.cands, rng)
    props["greedy"] = propose_greedy(block, h, args.cands, rng)
    pool = block.random_sets(max(args.cands, args.samples), rng)
    hc = np.array([hop_cost(demand, tp) for _, tp in pool]) if pool else np.array([])
    props["sampled"] = [pool[i] for i in np.argsort(hc)[: args.cands]]
    props["random"] = pool[: args.cands]
    best = {}
    for name, cands in props.items():
        z, arcs, n_lp = lp_top(demand, cands, args.K)
        best[name] = (z, arcs)
        rec[name] = {"z": z, "distinct": len(cands), "lp": n_lp}
    rec["t_step"] = round(time.time() - t0, 1)
    return rec, best


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--start", default="solutions", choices=["solutions", "runs", "published", "greedy"])
    p.add_argument("--r", type=int, default=20)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--cands", type=int, default=60)
    p.add_argument("--samples", type=int, default=600, help="random swap sets for the 'sampled' control")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--penalty", type=float, default=2.0)
    p.add_argument("--trust", type=float, default=1.0, help="cost per swap, in units of std(h)")
    p.add_argument("--follow", default="quantum", choices=["quantum", "sa", "greedy", "sampled", "random", "best"])
    p.add_argument("--optimizer", default="cobyla", choices=["mc", "cobyla", "lbfgs"])
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--shots", type=int, default=5000)
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--cloud", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    run = OUT / a.tag
    run.mkdir(parents=True, exist_ok=True)
    state_path, log_path = run / "state.json", run / "steps.jsonl"
    demand = qt.load_network_demand(a.n)
    if state_path.exists():
        st = json.loads(state_path.read_text())
        cur, done, best_z = [tuple(x) for x in st["arcs"]], st["step"], st["best_z"]
        print(f"resuming {a.tag} at step {done}", flush=True)
    else:
        cur, done, best_z = load_start(a.n, a.start, 0), 0, None
    z_cur, hot = lp_eval(demand, cur)
    best_z = z_cur if best_z is None else best_z
    (run / "args.json").write_text(json.dumps(vars(a)))
    service = None
    if a.cloud:
        from divi.backends import QoroService
        service = QoroService()
        print(f"credits at start: {service.get_credit_balance()['balance']}", flush=True)
    print(f"start {math.ceil(z_cur - 1e-6)}  r={a.r}  cloud={a.cloud}", flush=True)

    names = ("quantum", "sa", "greedy", "sampled", "random")
    for k in range(done + 1, a.steps + 1):
        rng = np.random.default_rng([a.seed, k])
        rec, best = step(demand, cur, hot, a, rng, a.seed * 1000 + k)
        lead = min(best, key=lambda m: best[m][0]) if a.follow == "best" else a.follow
        z_new, arcs_new = best[lead]
        moved = ""
        if arcs_new is not None and z_new <= z_cur + 1e-6:
            moved = "down" if z_new < z_cur - 1e-6 else "sideways"
            cur = arcs_new
            z_cur, hot = lp_eval(demand, cur)
            if z_cur < best_z - 1e-6:
                best_z = z_cur
                moved = "NEW BEST " + npl.bank(a.n, demand, cur, f"goat_swaps:{a.tag}")[2]
        rec |= {"step": k, "z_cur": z_cur, "moved": moved, "lead": lead}
        if service is not None:
            rec["credits"] = float(service.get_credit_balance()["balance"])
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        state_path.write_text(json.dumps({"arcs": cur, "step": k, "best_z": best_z}))
        print(f"step {k:3d} {rec['qubits']}q {rec['n_swaps']} swaps ({rec['n_improving_singles']} improving) "
              f"{rec['t_step']:5.0f}s | " + "  ".join(f"{m} {rec[m]['z']:9.0f}" for m in names)
              + f" | cur {math.ceil(z_cur - 1e-6)} {moved}"
              + (f" | credits {rec['credits']:.1f}" if "credits" in rec else ""), flush=True)


if __name__ == "__main__":
    main()
