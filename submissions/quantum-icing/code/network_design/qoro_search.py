"""Run Divi's partitioned QAOA pipeline on a QOBLIB network topology QUBO."""

import argparse
from collections import Counter
import math
from pathlib import Path
import random

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.backends.characterization import CharacterizationOptions, characterize_and_validate
from divi.qprog.ensembles import BatchConfig, BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig
import hybrid
import networkx as nx

import qoblib_tools as qt
from network_design import baseline, block_search, search
from network_design.solutions import BEST_KNOWN, checked_score, save_checked_solution


OUTPUT_DIR = Path("runs/network-qoro")


def build_qubo(demand, incumbent, free_links, seed):
    objective, penalty, arcs = qt.build_network_qubo(demand)
    penalty.scale(2.0)
    objective.update(penalty)
    removed = set(random.Random(seed).sample(sorted(incumbent), free_links)) if free_links else incumbent
    fixed = incumbent - removed
    sources = {i for i, _ in removed}
    destinations = {j for _, j in removed}
    # A replacement can only use an outgoing and an incoming slot freed by the block.
    objective.fix_variables({
        v: int((i, j) in fixed)
        for v, (i, j) in enumerate(arcs)
        if (i, j) in fixed or i not in sources or j not in destinations
    })
    objective.normalize()
    return objective, arcs, fixed


def repair_samples(samples, demand, incumbent, fixed, allowed):
    """Project stitched samples onto the remaining degree constraints."""
    outgoing = Counter(i for i, _ in incumbent - fixed)
    incoming = Counter(j for _, j in incumbent - fixed)
    repair = nx.DiGraph()
    repair.add_nodes_from((("out", i), {"demand": -count}) for i, count in sorted(outgoing.items()))
    repair.add_nodes_from((("in", j), {"demand": count}) for j, count in sorted(incoming.items()))
    # Keeping one sampled link outweighs all of the direct-demand tie breakers combined.
    keep_weight = (len(incumbent) - len(fixed)) * int(demand.max()) + 1
    for i, j in allowed:
        repair.add_edge(("out", i), ("in", j), capacity=1)

    topologies = set()
    for selected in samples:
        proposed = fixed | selected
        if qt.is_valid_topology(len(demand), proposed):
            topologies.add(tuple(sorted(proposed)))
            continue
        for i, j in allowed:
            repair[("out", i)][("in", j)]["weight"] = -int(demand[i, j]) - keep_weight * ((i, j) in selected)
        flow = nx.min_cost_flow(repair)
        repaired = fixed | {(i, j) for i, j in allowed if flow[("out", i)][("in", j)]}
        topologies.add(tuple(sorted(repaired)))
    return topologies


def run(args, demand, start, backend, service):
    incumbent_score = checked_score(start, demand)
    _, incumbent_arcs, _ = qt.read_network_solution(start)
    incumbent = set(incumbent_arcs)
    qubo, arcs, fixed = build_qubo(demand, incumbent, args.free_links, args.seed)
    problem = BinaryOptimizationProblem(
        qubo,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(
            max_n_variables_per_cluster=args.cluster_size, seed=args.seed,
        )),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    order = problem.canonical_problem.variable_order
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=args.layers,
        optimizer=MonteCarloOptimizer(population_size=args.population),
        max_iterations=args.iterations,
        backend=backend,
        seed=args.seed,
        reporting_level="off",
    )
    ensemble.create_programs()
    widths = [program.n_qubits for program in ensemble.programs.values()]
    if not widths:
        raise RuntimeError("Divi found no coupled partitions. Try a larger --free-links or --cluster-size.")
    print(f"network{args.nodes:02d}: incumbent {incumbent_score:,} from {start}", flush=True)
    print(f"{type(backend).__name__}: {len(arcs)} possible links, {len(order)} QUBO variables, "
          f"{len(widths)} quantum partitions, largest {max(widths)} qubits", flush=True)
    print(f"QAOA: {args.layers} layers, {args.population} population, {args.iterations} iterations, "
          f"{args.shots} shots, seed {args.seed}", flush=True)

    if args.characterize:
        reference = "".join("1" if arcs[v] in incumbent else "0" for v in order)
        report = characterize_and_validate(
            problem, reference_states=[reference], service=service,
            options=CharacterizationOptions(ansatz={"layers": args.layers}),
        )
        print(f"Characterization job {report.job_id}: {report.summary}", flush=True)

    ensemble.run(batch_config=BatchConfig(max_concurrent_programs=-1))
    stitched = ensemble.get_top_solutions(
        n=args.candidates,
        strategy=BeamSearchStrategy(beam_width=max(32, args.candidates), n_partition_candidates=8),
    )
    samples = [
        {arcs[v] for v, bit in zip(order, bits, strict=True) if bit}
        for bits, _ in stitched
    ]
    topologies = repair_samples(samples, demand, incumbent, fixed, [arcs[v] for v in order])
    topologies.discard(tuple(sorted(incumbent)))
    scored = sorted((qt.route_min_congestion(demand, topology).objective, topology) for topology in topologies)
    routable = [(score, topology) for score, topology in scored if math.isfinite(score)]
    print(f"Divi ran {ensemble.total_circuit_count:,} circuits and stitched {len(stitched)} samples. "
          f"{len(routable)} distinct new topologies passed routing.", flush=True)
    if not routable:
        print("No new routable candidate from this run.")
        return

    best_lp, best_topology = routable[0]
    routing = qt.route_integral(demand, best_topology)
    backend_dir = "cloud" if args.cloud else "local"
    output = OUTPUT_DIR / backend_dir / start.name
    saved = save_checked_solution(output, demand, routing)
    score = int(routing.objective)
    print(f"Best new Divi candidate: {score:,} checked (LP {best_lp:,.0f}), "
          f"{score - incumbent_score:+,} vs incumbent, {score - BEST_KNOWN[args.nodes]:+,} vs QOBLIB best")
    print(f"Best saved {backend_dir} Divi solution: {saved:,}, {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=BEST_KNOWN, default=11)
    parser.add_argument("--free-links", type=int, default=8, help="incumbent links freed (0 for the full QUBO)")
    parser.add_argument("--cluster-size", type=int, default=10, help="maximum variables per partition")
    parser.add_argument("--layers", type=int, default=1, help="QAOA layers")
    parser.add_argument("--population", type=int, default=8, help="Monte Carlo population")
    parser.add_argument("--iterations", type=int, default=4, help="optimizer iterations per partition")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=64, help="stitched candidates to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cloud", action="store_true", help="run circuits on Qoro's cloud Maestro simulator")
    parser.add_argument("--characterize", action="store_true", help="run Qoro cloud characterization before QAOA")
    args = parser.parse_args()
    if args.free_links != 0 and not 2 <= args.free_links <= 2 * args.nodes:
        parser.error(f"--free-links must be zero or between 2 and {2 * args.nodes}")
    if args.cluster_size < 2 or min(args.layers, args.population, args.iterations, args.shots, args.candidates) < 1:
        parser.error("--cluster-size must be at least 2 and circuit settings must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")

    service = None
    if args.cloud or args.characterize:
        service = QoroService(
            job_config=JobConfig(shots=args.shots, simulator_cluster="qoro_maestro"),
            use_saved_defaults=False,
        )
    backend = service if args.cloud else MaestroSimulator(shots=args.shots)

    demand = qt.load_network_demand(args.nodes)
    filename = f"network{args.nodes:02d}.sol"
    starts = [directory / filename for directory in (baseline.OUTPUT_DIR, search.OUTPUT_DIR, block_search.OUTPUT_DIR)]
    available = [path for path in starts if path.is_file()]
    if not available:
        baseline.build_baseline(starts[0], demand)
        available.append(starts[0])
    start = min(available, key=lambda path: checked_score(path, demand))
    run(args, demand, start, backend, service)


if __name__ == "__main__":
    main()
