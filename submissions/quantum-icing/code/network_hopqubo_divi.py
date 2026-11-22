"""Run the path-variable QUBOs (network_hopqubo.py) through Divi: partitioned QAOA, partitioned PCE
and whole-problem PCE. Every sample: decode arcs -> repair -> exact LP.

    .venv/bin/python network_hopqubo_divi.py --n 12 --qubo direct soft strict --method pqaoa ppce pce_dense
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")
import hybrid  # noqa: E402
import numpy as np  # noqa: E402

import qoblib_tools as qt  # noqa: E402
from divi.backends import JobConfig, MaestroSimulator, QoroService  # noqa: E402
from divi.qprog.algorithms import PCE, DenseEncoding, PolyEncoding, SoftObjective  # noqa: E402
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble  # noqa: E402
from divi.qprog.optimizers import MonteCarloOptimizer  # noqa: E402
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig  # noqa: E402
from network_hopqubo import aux_assignment, build_path_qubo, build_soft_qubo  # noqa: E402

POP, ITERS = 50, 10
CLUSTER = None  # set from --cloud: Qoro simulator cluster name


def make_backend(shots=10_000):
    if CLUSTER:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster=CLUSTER))
    return MaestroSimulator(shots=shots)


def get_qubo(kind, T, penalty_weight):
    n = len(T)
    if kind == "direct":
        sur, pen, arcs = qt.build_network_qubo(T)
        bqm = sur.copy()
        bqm.update(pen)
        return bqm.relabel_variables({v: ("x", *arcs[v]) for v in bqm.variables}, inplace=False)
    if kind == "soft":
        return build_soft_qubo(T, penalty_weight=penalty_weight, c=2.0)
    return build_path_qubo(T, hops=2, penalty_weight=max(penalty_weight, 4.0))


def run(method, bqm_int, cluster, n_layers, seed):
    backend = make_backend()
    if method in ("pqaoa", "ppce", "pqaoa_spec"):
        prob = BinaryOptimizationProblem(
            bqm_int,
            decomposer=CommunityDecomposer(QUBOPartitioningConfig(
                max_n_variables_per_cluster=cluster,
                method="spectral" if method.endswith("spec") else "modularity", seed=seed)),
            composer=hybrid.SplatComposer(),
            local_search=True,
        )
        kw = dict(encoding=PolyEncoding(), objective=SoftObjective(alpha=2.0)) if method == "ppce" else {}
        ens = PartitioningProgramEnsemble(
            problem=prob, quantum_routine="pce" if method == "ppce" else "qaoa", n_layers=n_layers,
            optimizer=MonteCarloOptimizer(population_size=POP), max_iterations=ITERS, backend=backend, **kw)
        ens.run()
        tops = ens.get_top_solutions(n=100, strategy=BeamSearchStrategy(beam_width=100, n_partition_candidates=10))
        return [dict(zip(bqm_int.variables, sol)) for sol, _ in tops]
    enc = DenseEncoding() if method == "pce_dense" else PolyEncoding()
    pce = PCE(BinaryOptimizationProblem(bqm_int), encoding=enc, objective=SoftObjective(alpha=2.0), n_layers=n_layers,
              optimizer=MonteCarloOptimizer(population_size=POP), max_iterations=ITERS, backend=backend, seed=seed)
    pce.run()
    tops = pce.get_top_solutions(n=100, min_prob=0.0, sort_by="energy")
    out = []
    for s in tops:
        b = s.bitstring
        out.append({v: int(b[i]) for i, v in enumerate(bqm_int.variables)} if isinstance(b, str) else dict(zip(bqm_int.variables, b)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--qubo", nargs="+", default=["direct", "soft", "strict"])
    ap.add_argument("--method", nargs="+", default=["pqaoa", "pqaoa_spec", "ppce", "pce_dense"])
    ap.add_argument("--cluster", type=int, default=12)
    ap.add_argument("--pce-cluster", type=int, default=40)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--penalty-weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ref", default="runs/network12/seed200.json")
    ap.add_argument("--cloud", default=None, help="Qoro cluster: qoro_gpu | qoro_cpu | qoro_maestro")
    a = ap.parse_args()
    CLUSTER = a.cloud
    svc = QoroService() if CLUSTER else None
    bal = lambda: float(svc.get_credit_balance()["balance"]) if svc else 0.0  # noqa: E731
    print(f"backend: {CLUSTER or 'local MaestroSimulator'}" + (f", credits {bal():.2f}" if svc else ""), flush=True)
    T = qt.load_network_demand(a.n)
    ref = [tuple(map(int, x)) for x in json.load(open(a.ref))["arcs"]] if a.ref else None
    for kind in a.qubo:
        bqm = get_qubo(kind, T, a.penalty_weight)
        bqm.normalize()
        labels = list(bqm.variables)
        bqm_int = bqm.relabel_variables({v: i for i, v in enumerate(labels)}, inplace=False)
        e_ref = None
        if ref is not None:
            smp = aux_assignment(T, ref, bqm) if kind != "direct" else {v: int((v[1], v[2]) in set(ref)) for v in labels}
            e_ref = bqm.energy(smp)
        print(f"\nQUBO={kind}: {len(labels)} vars, {bqm.num_interactions} couplings, E(ref topology, LP "
              f"{qt.route_min_congestion(T, ref).objective:.0f}) = {e_ref:.3f}", flush=True)
        for m in a.method:
            t0 = time.time()
            c0 = bal()
            try:
                samples = run(m, bqm_int, a.pce_cluster if m == "ppce" else a.cluster, a.layers, a.seed)
            except KeyboardInterrupt:
                raise
            except Exception as err:  # keep sweeping
                print(f"  {m:<11} FAILED: {type(err).__name__}: {str(err)[:150]}", flush=True)
                continue
            energies = [bqm_int.energy(s) for s in samples]
            decoded = [[(labels[i][1], labels[i][2]) for i, b in s.items() if b and labels[i][0] == "x"] for s in samples]
            raw_valid = sum(qt.is_valid_topology(a.n, d) for d in decoded)
            tops = {tuple(sorted(qt.repair_topology(a.n, d, T))) for d in decoded}
            zs = [qt.route_min_congestion(T, t).objective for t in tops]
            print(f"  {m:<11} best E {min(energies):8.3f} | raw-valid {raw_valid}/{len(samples)} | {len(tops)} topologies "
                  f"| LP best {min(zs):.0f} median {np.median(zs):.0f} ({time.time() - t0:.0f}s"
                  + (f", {c0 - bal():.2f} credits)" if svc else ")"), flush=True)
