"""Improve a checked QOBLIB network baseline with simulated annealing."""

import argparse
import math
from pathlib import Path
import random
from time import monotonic

import qoblib_tools as qt
from network_design.baseline import OUTPUT_DIR as BASELINE_DIR, build_baseline
from network_design.solutions import BEST_KNOWN, checked_score, save_checked_solution


OUTPUT_DIR = Path("runs/network-search")
COOLING_STEPS = 400
START_TEMPERATURE = 6_000
END_TEMPERATURE = 400


def switch_links(topology, rng):
    """Swap two link destinations while preserving all in- and out-degrees."""
    links = sorted(topology)
    for _ in range(12):
        first, second = rng.sample(links, 2)
        a, b = first
        c, d = second
        if a == d or c == b or (a, d) in topology or (c, b) in topology:
            continue
        return (topology - {first, second}) | {(a, d), (c, b)}
    return topology


def search(demand, start_path, output, seconds, seed):
    rng = random.Random(seed)
    saved_score = checked_score(start_path, demand)
    _, arcs, _ = qt.read_network_solution(start_path)
    best_topology = frozenset(arcs)
    best_lp = qt.route_min_congestion(demand, best_topology).objective
    lp_scores = {best_topology: best_lp}
    current, current_lp = best_topology, best_lp
    started = monotonic()
    next_report = started + 15
    step = 0
    print(f"Starting from {start_path} (checked score {saved_score:,}), seed {seed}", flush=True)

    try:
        while monotonic() - started < seconds:
            phase = step % COOLING_STEPS
            if phase == 0:
                # Reheat from the best topology so a new chain can escape its neighborhood.
                current, current_lp = best_topology, best_lp
            temperature = START_TEMPERATURE * (END_TEMPERATURE / START_TEMPERATURE) ** (
                phase / (COOLING_STEPS - 1)
            )
            step += 1
            candidate = switch_links(current, rng)
            if rng.random() < 0.15:
                candidate = switch_links(candidate, rng)
            if candidate == current:
                continue

            if candidate not in lp_scores:
                lp_scores[candidate] = qt.route_min_congestion(demand, candidate).objective
            score = lp_scores[candidate]
            delta = score - current_lp
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current, current_lp = candidate, score
            if score < best_lp:
                best_topology, best_lp = candidate, score

            if score < saved_score:
                try:
                    routing = qt.route_integral(demand, candidate, time_limit=3.0)
                    if routing.objective < saved_score:
                        saved_score = save_checked_solution(output, demand, routing)
                        start_path = output
                        print(f"New checked best {saved_score:,} after {len(lp_scores):,} topologies", flush=True)
                except RuntimeError as error:
                    print(f"Could not export LP score {score:,.0f}: {error}", flush=True)

            now = monotonic()
            if now >= next_report:
                print(f"{now - started:.0f}s, {len(lp_scores):,} topologies, best checked {saved_score:,}", flush=True)
                next_report = now + 15
    except KeyboardInterrupt:
        print("Stopping search. The best checked solution is on disk.", flush=True)

    target = BEST_KNOWN.get(len(demand))
    gap = "" if target is None else f", gap {(saved_score - target) / target:+.2%} to {target:,}"
    print(f"{output.stem}: {saved_score:,} checked{gap}, {len(lp_scores):,} topologies, {start_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=BEST_KNOWN, default=24, help="instance size")
    parser.add_argument("--seconds", type=int, default=180, help="search time in seconds")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="solution directory")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    demand = qt.load_network_demand(args.nodes)
    filename = f"network{args.nodes:02d}.sol"
    baseline = BASELINE_DIR / filename
    output = args.output_dir / filename
    if not baseline.is_file():
        build_baseline(baseline, demand)
    start_path = baseline
    if output.is_file() and checked_score(output, demand) <= checked_score(baseline, demand):
        start_path = output
    search(demand, start_path, output, args.seconds, args.seed)


if __name__ == "__main__":
    main()
