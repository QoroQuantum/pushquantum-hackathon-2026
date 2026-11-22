"""Verify the network checker and produce a feasible starting topology."""

import argparse
from pathlib import Path

import numpy as np

import qoblib_tools as qt
from network_design.solutions import BEST_KNOWN, save_checked_solution


OUTPUT_DIR = Path("runs/network-baseline")
NETWORK05_OPTIMUM = 65_500
SEED = 42
RANDOM_TOPOLOGIES = 16


def verify_network05(path):
    demand = qt.load_network_demand(5)
    reference = qt.fetch_qoblib_file("08-network/solutions/network05.opt.sol")
    _, reference_arcs, _ = qt.read_network_solution(reference)
    routing = qt.route_integral(demand, reference_arcs)
    score = save_checked_solution(path, demand, routing, expected_score=NETWORK05_OPTIMUM)
    print(f"network05: checked score {score:,} (proven optimum), wrote {path}", flush=True)


def build_baseline(path, demand):
    n_nodes = len(demand)
    rng = np.random.default_rng(SEED)
    # The step-1 cycle guarantees that this seed can route all demands.
    second_step = 3 if n_nodes == 11 else 5
    cycle_topology = [(i, (i + step) % n_nodes) for i in range(n_nodes) for step in (1, second_step)]
    topologies = [cycle_topology]
    topologies.extend(qt.random_topology(n_nodes, rng) for _ in range(RANDOM_TOPOLOGIES))
    best = min(
        (qt.route_min_congestion(demand, arcs) for arcs in topologies),
        key=lambda routing: routing.objective,
    )
    print(f"{path.stem}: best LP score {best.objective:,.0f} from {len(topologies)} topologies", flush=True)

    routing = qt.route_integral(demand, best.arcs)
    score = save_checked_solution(path, demand, routing)
    target = BEST_KNOWN[n_nodes]
    gap = (score - target) / target
    print(f"{path.stem}: checked baseline {score:,}, gap {gap:+.2%}, solution {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=BEST_KNOWN, default=24, help="instance size")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="solution directory")
    args = parser.parse_args()

    verify_network05(args.output_dir / "network05.sol")
    build_baseline(args.output_dir / f"network{args.nodes:02d}.sol", qt.load_network_demand(args.nodes))


if __name__ == "__main__":
    main()
