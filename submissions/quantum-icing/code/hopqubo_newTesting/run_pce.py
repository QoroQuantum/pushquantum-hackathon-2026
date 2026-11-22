"""Full-problem PCE on the whole topology (n(n-1) arc variables), comparing encodings.

Problem: H2 hop-surrogate HUBO (network_mixer.h2_hubo) plus the degree rules (out-degree 2 and
in-degree 2 per node) as Divi LinearConstraints (squared penalties).

Decoding. Divi's default PCE read-out rounds the sign of each variable's correlator on sampled
bitstrings. With a parity encoding (Poly / Dense) every variable is a parity of q qubits, so the
decoded bitstrings span at most a 2^q-element GF(2) subspace of {0,1}^{n(n-1)}; that subspace
almost never contains a valid 2-out/2-in topology. We instead take the trained soft values
(1 + tanh(α⟨P_v⟩))/2 and decode them with a degree-2 b-matching (plus Gumbel-perturbed copies),
so every decoded topology is valid by construction. The node-pair encoding puts one qubit per
node (arc i→j is X_i X_j for i<j, Y_i Y_j for i>j), so node v's degree constraint lives on the
correlators of qubit v. Divi's own decode is logged alongside for comparison. A classical
same-budget baseline (random start + 2-exchange polish) is run once. Gap vs QOBLIB best known is
always reported.

    .venv/bin/python hopqubo_newTesting/run_pce.py --n 12 --encodings nodepair poly
    .venv/bin/python hopqubo_newTesting/run_pce.py --n 24 --encodings nodepair --cloud qoro_cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DIVI_DISABLE_PROGRESS", "1")

import numpy as np  # noqa: E402

import common  # noqa: E402
import divi_ext  # noqa: E402
from common import RunLog, gap_str, load  # noqa: E402


def build_problem(T, n, arcs_index, penalty_weight):
    from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint
    from network_mixer import h2_hubo

    h = {k: v for k, v in h2_hubo(T, var_of={a: v for v, a in enumerate(arcs_index)}, fixed=None).items() if k}
    cons = []
    for node in range(n):
        cons.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs_index) if i == node}, "==", 2))
        cons.append(LinearConstraint({v: 1 for v, (i, j) in enumerate(arcs_index) if j == node}, "==", 2))
    return BinaryOptimizationProblem(h, constraints=cons, penalty_weight=penalty_weight,
                                     hamiltonian_builder="native")


def make_encoding(name, n, arcs_index):
    from divi.qprog.algorithms import DenseEncoding, PolyEncoding

    if name == "nodepair":
        return divi_ext.node_pair_layout(n, arcs_index)
    if name == "poly":
        return PolyEncoding()
    if name == "dense":
        return DenseEncoding()
    raise ValueError(name)


def run_encoding(args, T, arcs_index, problem, enc_name, log):
    from divi.qprog.algorithms import SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer

    n = args.n
    a = args.alpha if args.alpha is not None else (1.5 * n if enc_name == "nodepair" else 3.0)
    t0 = time.time()
    pce = divi_ext.WarmPCE(problem, encoding=make_encoding(enc_name, n, arcs_index),
                           objective=SoftObjective(alpha=a), n_layers=args.layers,
                           optimizer=MonteCarloOptimizer(population_size=args.pop),
                           max_iterations=args.iters,
                           backend=common.make_backend(args.cloud, shots=args.shots), seed=args.seed)
    pce.run()
    secs = time.time() - t0
    log(f"[{enc_name}] {pce.n_qubits} qubits, alpha {a:g}, {pce.total_circuit_count} circuits, {secs:.0f}s")

    # decode B: Divi default (sign rounding), for comparison
    soft = divi_ext.pce_soft_values(pce, a)
    if enc_name == "nodepair":
        sign_arcs = [arcs_index[v] for v in np.flatnonzero(soft > 0.5)]
        divi_valid = common.is_valid(n, sign_arcs)
        log(f"  sign read (soft>0.5): {len(sign_arcs)} arcs, valid {divi_valid}")
    else:
        sols = pce.get_top_solutions(n=100, min_prob=0.0, sort_by="energy")
        divi_valid = sum(common.is_valid(n, [arcs_index[v] for v, c in enumerate(s.bitstring) if c == "1"])
                         for s in sols)
        log(f"  divi decode: {divi_valid}/{len(sols)} valid")

    # decode A: soft values → degree-2 b-matching (valid by construction)
    tops = common.soft_to_topologies(n, arcs_index, soft, k=args.k_decode, rng=np.random.default_rng(args.seed))
    scored = common.rank_and_score(T, tops, top_k=args.lp_topk)
    if not scored:
        log(f"[{enc_name}] no valid topology decoded")
        log.data.setdefault("final", {})[enc_name] = {"n_qubits": pce.n_qubits, "circuits": pce.total_circuit_count,
                                                      "seconds": secs, "divi_valid": divi_valid, "n_decoded": 0}
        log.save()
        return None
    pre, best_arcs = scored[0]
    arcs, z, used = common.polish(T, best_arcs, lp_budget=args.polish_budget)
    log(f"[{enc_name}] {len(tops)} distinct valid decoded, best LP {pre:.0f} ({gap_str(n, pre)}) → "
        f"polish {z:.0f} ({gap_str(n, z)}) | LP calls {len(scored) + used}")
    log.event(encoding=enc_name, n_qubits=pce.n_qubits, circuits=pce.total_circuit_count, seconds=secs,
              alpha=a, divi_valid=divi_valid, n_decoded=len(tops), pre_polish=pre, z=z)
    log.data.setdefault("final", {})[enc_name] = {
        "n_qubits": pce.n_qubits, "circuits": pce.total_circuit_count, "seconds": secs, "alpha": a,
        "divi_valid": divi_valid, "n_decoded": len(tops), "pre_polish": pre, "pre_gap": gap_str(n, pre),
        "z": z, "gap": gap_str(n, z), "lp_calls": len(scored) + used, "arcs": arcs}
    log.save()
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--encodings", nargs="+", default=["nodepair", "poly"], choices=["nodepair", "poly", "dense"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2, help="PCE n_layers")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="default: nodepair 1.5*n, poly/dense 3.0")
    ap.add_argument("--penalty-weight", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--cloud", default=None, help="qoro_cpu | qoro_gpu (default: local Maestro)")
    ap.add_argument("--k-decode", type=int, default=30, help="topologies decoded from soft values")
    ap.add_argument("--lp-topk", type=int, default=10)
    ap.add_argument("--polish-budget", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    n = args.n
    T = load(n)
    arcs_index = [(i, j) for i in range(n) for j in range(n) if i != j]
    log = RunLog(f"pce_n{n:02d}_s{args.seed}{('_' + args.tag) if args.tag else ''}")
    log.data["args"] = vars(args)
    log(f"network{n:02d}: {len(arcs_index)} arc variables, encodings {args.encodings}")

    start = common.start_topology(n, T, "random", args.seed)
    _, zc, usedc = common.polish(T, start, lp_budget=args.lp_topk + args.polish_budget)
    log(f"[classical] random + polish {zc:.0f} ({gap_str(n, zc)}) | LP calls {usedc}")
    log.data.setdefault("final", {})["classical"] = {"z": zc, "gap": gap_str(n, zc), "lp_calls": usedc}
    log.save()

    problem = build_problem(T, n, arcs_index, args.penalty_weight)
    results = {e: run_encoding(args, T, arcs_index, problem, e, log) for e in args.encodings}
    log("summary: classical " + f"{zc:.0f} ({gap_str(n, zc)}) | "
        + " | ".join(f"{e} {z:.0f} ({gap_str(n, z)})" if z is not None else f"{e} none"
                     for e, z in results.items()))


if __name__ == "__main__":
    main()
