"""Search larger degree-preserving neighborhoods of the network11 incumbent."""

import argparse
from itertools import permutations
from math import comb
from pathlib import Path
import random
from time import monotonic

import qoblib_tools as qt
from network_design.search import OUTPUT_DIR as ANNEALING_DIR
from network_design.solutions import BEST_KNOWN, checked_score, save_checked_solution


N_NODES = 11
FILENAME = f"network{N_NODES:02d}.sol"
OUTPUT_DIR = Path("runs/network-block-search")


def rewire_block(topology, removed):
    """Yield all valid completions of a block of removed links."""
    fixed = topology.difference(removed)
    sources, destinations = zip(*removed)
    for permuted in sorted(set(permutations(destinations))):
        # Keeping both endpoint multisets preserves every node's two degrees.
        replacement = frozenset(zip(sources, permuted))
        if len(replacement) != len(removed) or replacement & fixed:
            continue
        if any(source == destination for source, destination in replacement):
            continue
        yield fixed | replacement


def search(demand, start_path, output, seconds, block_size, seed):
    best_path = start_path
    best_score = checked_score(start_path, demand)
    _, arcs, _ = qt.read_network_solution(start_path)
    incumbent = frozenset(arcs)
    seen = {incumbent}
    tried_blocks = set()
    block_count = comb(len(incumbent), block_size)
    rng = random.Random(seed)
    started = monotonic()
    deadline = started + seconds
    next_report = started + 15
    evaluated = 0
    completed_blocks = 0
    print(f"Starting from {start_path} ({best_score:,}), freeing {block_size} links, seed {seed}", flush=True)

    try:
        while monotonic() < deadline and len(tried_blocks) < block_count:
            block_topology = incumbent
            removed = tuple(sorted(rng.sample(sorted(block_topology), block_size)))
            if removed in tried_blocks:
                continue
            tried_blocks.add(removed)
            for candidate in rewire_block(block_topology, removed):
                if monotonic() >= deadline:
                    break
                if candidate in seen:
                    continue
                seen.add(candidate)
                evaluated += 1
                lp_score = qt.route_min_congestion(demand, candidate).objective
                if lp_score < best_score:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        routing = qt.route_integral(demand, candidate, time_limit=min(3.0, remaining))
                        if routing.objective < best_score:
                            best_score = save_checked_solution(output, demand, routing)
                            best_path = output
                            incumbent = candidate
                            print(f"New checked best {best_score:,} after {evaluated:,} topologies", flush=True)
                    except RuntimeError as error:
                        seen.remove(candidate)
                        print(f"Could not verify LP score {lp_score:,.0f}: {error}", flush=True)
                now = monotonic()
                if now >= next_report:
                    print(f"{now - started:.0f}s, {evaluated:,} topologies, best checked {best_score:,}", flush=True)
                    next_report = now + 15
            else:
                completed_blocks += 1
            if incumbent != block_topology:
                tried_blocks.clear()
    except KeyboardInterrupt:
        print("Stopping search. The best checked solution is on disk.", flush=True)

    target = BEST_KNOWN[N_NODES]
    gap = (best_score - target) / target
    print(f"network11: {best_score:,} checked, gap {gap:+.2%} to {target:,}, "
          f"{completed_blocks:,} blocks completed, {evaluated:,} topologies, {best_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=180, help="search time in seconds")
    parser.add_argument("--block-size", type=int, choices=range(4, 9), default=6, help="links freed per block")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--start", type=Path, default=ANNEALING_DIR / FILENAME, help="checked starting solution")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="improved solution directory")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    demand = qt.load_network_demand(N_NODES)
    output = args.output_dir / FILENAME
    starting_solutions = [path for path in (args.start, output) if path.is_file()]
    if not starting_solutions:
        parser.error("run python network_search.py --nodes 11 first or pass --start with a checked solution")
    start_path = min(starting_solutions, key=lambda path: checked_score(path, demand))
    search(demand, start_path, output, args.seconds, args.block_size, args.seed)


if __name__ == "__main__":
    main()
