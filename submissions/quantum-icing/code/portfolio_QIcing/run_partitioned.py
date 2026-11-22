"""Notebook-01 route B on the compact QUBO: partition -> QAOA/PCE per part -> beam stitch.

Records two solutions: stitched + referee repair only ("raw"), and + referee local search.
Example: python run_partitioned.py --instance po_a010_t10_orig --lam 1e-4 --routine qaoa
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import hybrid
import numpy as np

import pf_core as pc
import pf_quantum as pq
from divi.backends import MaestroSimulator
from divi.qprog.algorithms import PolyEncoding, SoftObjective
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a010_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--routine", default="qaoa")
ap.add_argument("--cluster-size", type=int, default=12)
ap.add_argument("--n-layers", type=int, default=2)
ap.add_argument("--penalty-weight", type=float, default=100.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mc-population", type=int, default=50)
ap.add_argument("--mc-iterations", type=int, default=10)
args = ap.parse_args()

model = pc.make_model(args.instance, args.lam)
cq = pq.CompactQUBO(model)
bkv, status = pc.load_bkv()[(args.instance, model.budget, args.lam)]
bqm = cq.full(args.penalty_weight)
bqm.normalize()
t0 = time.time()
problem = BinaryOptimizationProblem(
    bqm,
    decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=args.cluster_size)),
    composer=hybrid.SplatComposer(),
    local_search=True,  # Divi's greedy bit-flip on the full QUBO (as in notebook 01)
)
kw = {"encoding": PolyEncoding(), "objective": SoftObjective(alpha=2.0)} if args.routine == "pce" else {}
ens = PartitioningProgramEnsemble(problem=problem, quantum_routine=args.routine, n_layers=args.n_layers,
                                  optimizer=MonteCarloOptimizer(population_size=args.mc_population),
                                  max_iterations=args.mc_iterations, backend=MaestroSimulator(shots=10_000),
                                  seed=args.seed, **kw)
ens.run()
tops = ens.get_top_solutions(n=10, strategy=BeamSearchStrategy(beam_width=10, n_partition_candidates=5))
max_q = max(p.n_qubits for p in ens.programs.values())
raw = []
for sol, _ in tops:
    bits = np.array([int(sol[k]) for k in range(len(sol))])
    sample = dict(zip(bqm.variables, bits))
    vbits = np.array([sample[i] for i in range(cq.n_vars)])
    v = pq.repair_v(model, cq.decode_v(vbits))
    raw.append((pq.official(model, v).objective, v))
raw.sort(key=lambda r: r[0])
obj_raw, v_raw = raw[0]
polished = [pc.positions_to_v(model, model.local_search(pc.v_to_positions(model, v))) for _, v in raw]
v_ls = min(polished, key=lambda v: pq.official(model, v).objective)
obj_ls = pq.official(model, v_ls).objective
el = time.time() - t0
print(f"{args.instance} λ={args.lam}: {len(ens.programs)} sub-programs (≤{max_q} qubits), {ens.total_circuit_count} circuits, {el:.0f}s")
print(f"  stitched+repair: {obj_raw} (gap {(obj_raw - bkv) / abs(bkv):+.2%}) | +local search: {obj_ls} (gap {(obj_ls - bkv) / abs(bkv):+.2%}) | BKV {bkv}")
st = (f"routine={args.routine} cluster={args.cluster_size} n_layers={args.n_layers} penalty_weight={args.penalty_weight} "
      f"mc_pop={args.mc_population} mc_iter={args.mc_iterations} seed={args.seed} circuits={ens.total_circuit_count}")
_, p1, _ = pc.record_solution(model, pc.v_to_positions(model, v_raw), f"partitioned-{args.routine} + repair", st,
                              cq.n_vars, max_q, True, tag=f"part-{args.routine}-raw-s{args.seed}")
_, p2, _ = pc.record_solution(model, pc.v_to_positions(model, v_ls), f"partitioned-{args.routine} + repair + local search", st,
                              cq.n_vars, max_q, True, tag=f"part-{args.routine}-ls-s{args.seed}")
print("RAW_SOL", p1)
print("LS_SOL", p2)
