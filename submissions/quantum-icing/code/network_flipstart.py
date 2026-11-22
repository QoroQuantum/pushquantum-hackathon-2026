"""QAOA on network05 started from the bit-flipped best-known topology.

Compares three QAOA initial states on the notebook-02 surrogate QUBO (degree rules as
LinearConstraints, penalty 2.0, standard X mixer):
  plus     |+>^n           (Divi default)
  opt      |x*>            (published optimum as a basis state)
  flipped  |NOT x*>        (every bit flipped; still a valid topology for n=5)

Every feasible sample is scored by the exact LP.

    .venv/bin/python network_flipstart.py --layers 1 2 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import qoblib_tools as qt  # noqa: E402
from divi.backends import MaestroSimulator  # noqa: E402
from divi.qprog.algorithms import QAOA  # noqa: E402
from divi.qprog.initial_states import CustomPerQubitState, SuperpositionState  # noqa: E402
from divi.qprog.optimizers import MonteCarloOptimizer  # noqa: E402
from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint  # noqa: E402

N = 5
demand = qt.load_network_demand(N)
surrogate, _, arcs = qt.build_network_qubo(demand)
constraints = []
for node in range(N):
    constraints.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs) if i == node}, "==", 2))
    constraints.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs) if j == node}, "==", 2))
problem = BinaryOptimizationProblem(surrogate, constraints=constraints, penalty_weight=2.0)

# ground truth: all valid topologies
rows = [list(itertools.combinations([j for j in range(N) if j != i], 2)) for i in range(N)]
all_scores = {}
for choice in itertools.product(*rows):
    a = tuple(sorted((i, j) for i, pair in enumerate(choice) for j in pair))
    if qt.is_valid_topology(N, a):
        all_scores[a] = qt.route_min_congestion(demand, a).objective
OPT = min(all_scores.values())
ranked = np.array(sorted(all_scores.values()))

_, ref_arcs, _ = qt.read_network_solution(qt.fetch_qoblib_file(f"08-network/solutions/network{N:02d}.opt.sol"))
ref = {tuple(a) for a in ref_arcs}
opt_bits = "".join("1" if a in ref else "0" for a in arcs)
flip_bits = "".join("1" if b == "0" else "0" for b in opt_bits)
flip_arcs = tuple(sorted(qt.decode_topology(flip_bits, arcs)))
assert qt.is_valid_topology(N, flip_arcs)
flip_score = all_scores[flip_arcs]
print(f"opt     {opt_bits}  score {OPT:.0f}  energy {problem.evaluate(opt_bits) if hasattr(problem, 'evaluate') else ''}")
print(f"flipped {flip_bits}  score {flip_score:.0f}  rank {(ranked < flip_score).sum() + 1}/{len(ranked)}")
print(f"Hamming distance {sum(a != b for a, b in zip(opt_bits, flip_bits))}\n")

STARTS = {
    "plus": SuperpositionState(),
    "opt": CustomPerQubitState(opt_bits),
    "flipped": CustomPerQubitState(flip_bits),
}
# controls: random valid topology (no info about the optimum) and its bit-flip
_rand = qt.random_topology(N, np.random.default_rng(7))
rand_bits = "".join("1" if a in set(_rand) else "0" for a in arcs)
rflip_bits = "".join("1" if b == "0" else "0" for b in rand_bits)
for name, b in [("rand", rand_bits), ("randflip", rflip_bits)]:
    t = tuple(sorted(qt.decode_topology(b, arcs)))
    sc = all_scores.get(t, float("nan"))
    print(f"{name:<8}{b}  score {sc:.0f}  rank {(ranked < sc).sum() + 1}/{len(ranked)}")
    STARTS[name] = CustomPerQubitState(b)
print()


def run(start, n_layers, seed, shots, pop, iters):
    t0 = time.time()
    q = QAOA(
        problem,
        n_layers=n_layers,
        initial_state=STARTS[start],
        optimizer=MonteCarloOptimizer(population_size=pop),
        max_iterations=iters,
        backend=MaestroSimulator(shots=shots),
        seed=seed,
    )
    q.run()
    sols = q.get_top_solutions(n=0, feasibility="filter")
    scores = {s.bitstring: all_scores[tuple(sorted(qt.decode_topology(s.bitstring, arcs)))] for s in sols}
    p_valid = sum(s.prob for s in sols)
    p_opt = sum(s.prob for s in sols if scores[s.bitstring] == OPT)
    p_flip = sum(s.prob for s in sols if s.bitstring == flip_bits)
    best = min(scores.values()) if scores else float("inf")
    params = getattr(q, "best_params", None)
    return dict(start=start, n_layers=n_layers, seed=seed, p_valid=p_valid, p_opt=p_opt, p_flip=p_flip,
                best=best, rank=int((ranked < best).sum() + 1), secs=time.time() - t0,
                params=None if params is None else np.round(np.asarray(params).ravel(), 3).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    print(f"{'start':<8} layers seed  P(valid)  P(opt)   P(start=flip)  best   rank  secs  params")
    for n_layers in args.layers:
        for start in STARTS:
            for seed in args.seeds:
                r = run(start, n_layers, seed, args.shots, args.pop, args.iters)
                print(f"{r['start']:<8} {r['n_layers']:>6} {r['seed']:>4}  {r['p_valid']:7.2%}  {r['p_opt']:7.3%}  {r['p_flip']:9.3%}"
                      f"  {r['best']:6.0f} {r['rank']:>4}  {r['secs']:4.0f}  {r['params']}", flush=True)
