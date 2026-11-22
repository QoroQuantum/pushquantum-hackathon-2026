"""Search degree-preserving network rewirings with a simulated Grover mixer.

The classical front end builds a pool using head permutations and estimates
congestion with weighted shortest paths. Three selectors get the same number
of routing LP evaluations on that pool. Only the quantum selector drives the
walk. The simulator applies the phase and mixer to the feasible statevector.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import json
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

import qoblib_tools as qt
from network_design.iterated_local_search import WarmLP
from network_design.pipeline import BKV
from network_design.solutions import checked_score, save_checked_solution

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("quantum", "greedy", "uniform")
# Objective weights for the first, unweighted shortest-path screen.
SCREEN_WEIGHTS = np.array([[1., 0., 0.], [1., .5, 0.], [1., 1.5, 0.], [1., 0., .5]])


@njit(cache=True)
def shortest_path_screen(heads, demand, cap):
    """Hop lower bound and load imbalance with equal splitting at predecessor ties."""
    n = len(demand)
    distance = np.empty(n, np.int32)
    order = np.empty(n, np.int32)
    pending = np.empty(n)
    incoming = np.empty((n, 2), np.int32)
    counts = np.empty(n, np.int32)
    loads = np.zeros(2 * n)
    total = 0.
    for source in range(n):
        distance[:] = -1
        distance[source] = 0
        order[0] = source
        end = 1
        for step in range(n):
            if step == end:
                return np.inf, np.inf, np.inf
            u = order[step]
            for edge in range(2 * u, 2 * u + 2):
                v = heads[edge]
                if distance[v] < 0:
                    distance[v] = distance[u] + 1
                    order[end] = v
                    end += 1
        counts[:] = 0
        for v in range(n):
            pending[v] = demand[source, v]
            total += pending[v] * distance[v]
        for edge, v in enumerate(heads):
            if distance[edge // 2] + 1 == distance[v]:
                incoming[v, counts[v]] = edge
                counts[v] += 1
        for step in range(n - 1, 0, -1):
            v = order[step]
            flow = pending[v] / counts[v]
            for p in range(counts[v]):
                edge = incoming[v, p]
                loads[edge] += flow
                pending[edge // 2] += flow
    hop = total / len(heads)
    return hop, np.maximum(loads - cap, 0.).mean(), np.sqrt(((loads - hop) ** 2).mean())


@njit(cache=True)
def screen_rewirings(start, demand, subsets, orders, cap):
    """Score all valid derangements of each selected set of slots."""
    features = np.full((len(subsets), len(orders), 3), np.inf)
    heads = start.copy()
    for index, slots in enumerate(subsets):
        for choice, order in enumerate(orders):
            for i, edge in enumerate(slots):
                head = start[slots[order[i]]]
                if head == edge // 2 or head == start[edge] or head == start[edge ^ 1]:
                    break
                heads[edge] = head
            else:
                valid = True
                for edge in slots:
                    if heads[edge] == heads[edge ^ 1]:
                        valid = False
                        break
                if valid:
                    features[index, choice] = shortest_path_screen(heads, demand, cap)
            heads[slots] = start[slots]
    return features


@njit(cache=True)
def priced_shortest_paths(heads, demand, weights, loads):
    """Return the multicommodity weak-duality bound for this set of arc tolls."""
    n = len(demand)
    distances = np.empty(n)
    parent = np.empty(n, np.int32)
    done = np.empty(n, np.bool_)
    order = np.empty(n, np.int32)
    pending = np.empty(n)
    total = 0.
    loads[:] = 0.
    for source in range(n):
        distances[:] = np.inf
        distances[source] = 0.
        done[:] = False
        for step in range(n):
            u = -1
            best = np.inf
            for v in range(n):
                if not done[v] and distances[v] < best:
                    best = distances[v]
                    u = v
            done[u] = True
            order[step] = u
            for edge in range(2 * u, 2 * u + 2):
                v = heads[edge]
                trial = best + weights[edge]
                if trial < distances[v]:
                    distances[v] = trial
                    parent[v] = edge
        for v in range(n):
            pending[v] = demand[source, v]
            total += pending[v] * distances[v]
        for step in range(n - 1, 0, -1):
            v = order[step]
            edge = parent[v]
            loads[edge] += pending[v]
            pending[edge // 2] += pending[v]
    return total / weights.sum()


@njit(cache=True)
def congestion_energy(pool, coarse, demand, duals, cap, passes):
    """Combine cheap lower bounds and the excess of averaged feasible routings."""
    estimate = np.empty(len(pool))
    m = 2 * len(demand)
    loads = np.empty(m)
    for i, heads in enumerate(pool):
        cumulative = np.zeros(m)
        weights = .55 + .45 * duals
        lower = coarse[i, 0]
        for step in range(1, passes + 1):
            lower = max(lower, priced_shortest_paths(heads, demand, weights, loads))
            cumulative += loads / cap
            average = cumulative / step
            strength = min(12., 2.5 * np.sqrt(step))
            weights = np.exp(strength * (average - average.max()))
        excess = np.maximum(average - 1., 0.).mean() * cap
        estimate[i] = .75 * lower + .25 * coarse[i, 0] + excess + .25 * coarse[i, 1]
    return estimate


def topology(heads):
    return [(i // 2, int(j)) for i, j in enumerate(heads)]


def ordered_heads(arcs):
    return np.asarray([j for _, j in sorted(arcs)], dtype=np.int32)


def propose_pool(heads, demand, cap, orders, subsets_per_radius, shortlist, rng, priority):
    """Give each radius and each screening objective its own shortlist quota."""
    pool = {}
    m = len(heads)
    for radius, order in orders.items():
        trials = subsets_per_radius * 9 // len(order) + 1
        # Sample a mix of uniform and elite-biased subsets without replacement.
        keys = -np.log(np.maximum(rng.random((trials, m)), np.finfo(float).tiny))
        keys[trials // 3:] /= priority
        subsets = np.sort(np.argpartition(keys, radius - 1, axis=1)[:, :radius], axis=1)
        subsets = np.unique(subsets, axis=0)
        features = screen_rewirings(heads, demand, subsets, order, cap)
        flat = features.reshape(-1, 3)
        valid = np.flatnonzero(np.isfinite(flat[:, 0]))
        coarse = flat[valid]
        count = min(shortlist // (len(orders) * len(SCREEN_WEIGHTS)), len(valid))
        if not count:
            continue
        ranks = coarse @ SCREEN_WEIGHTS.T
        for indices in np.argpartition(ranks, count - 1, axis=0)[:count].T:
            for pos in indices:
                row, choice = divmod(int(valid[pos]), len(order))
                removed = subsets[row]
                proposed = heads.copy()
                proposed[removed] = heads[removed[order[choice]]]
                # Duplicate index assignments can differ only in the order of two
                # heads leaving the same node. Use a canonical key for the pool.
                key = tuple(np.sort(proposed.reshape(-1, 2), axis=1).ravel())
                pool.setdefault(key, (proposed, coarse[pos]))
    return pool


def grover_probabilities(energy, mark_fraction):
    """Simulate the threshold phase oracle followed by diffusion on valid states."""
    marked_count = max(1, round(len(energy) * mark_fraction))
    threshold = np.partition(energy, marked_count - 1)[marked_count - 1]
    marked = energy <= threshold
    fraction = float(marked.mean())
    angle = math.asin(math.sqrt(fraction))
    layers = max(0, round(math.pi / (4 * angle) - .5))
    state = np.full(len(energy), 1 / math.sqrt(len(energy)))
    for _ in range(layers):
        state[marked] *= -1.
        state = 2 * state.mean() - state
    probs = state ** 2
    return probs / probs.sum(), layers, int(marked.sum())


def selections(energy, budget, shots, fraction, rng, methods):
    count = min(budget, len(energy))
    selected = {}
    layers = marked = 0
    if "quantum" in methods:
        probability, layers, marked = grover_probabilities(energy, fraction)
        drawn = rng.choice(len(energy), shots, replace=True, p=probability)
        # Keep first measurement of each index. Every selected candidate has
        # a quantum measurement as provenance, including when repeats dominate.
        distinct = list(dict.fromkeys(drawn))
        count = min(count, len(distinct))
        selected["quantum"] = distinct[:count]
    if "greedy" in methods:
        selected["greedy"] = np.argsort(energy, kind="stable")[:count].tolist()
    if "uniform" in methods:
        selected["uniform"] = rng.choice(len(energy), count, replace=False).tolist()
    return selected, layers, marked


def init_routing_worker(demand, heads):
    global routing_model, routing_basis
    routing_model = WarmLP(demand, topology(heads))
    routing_model.h.resetGlobalScheduler(True)
    routing_model.solve(topology(heads))
    routing_basis = routing_model.h.getBasis()


def score_candidate(task):
    heads, cutoff = task
    z, _ = routing_model.solve(topology(heads), cutoff=cutoff, basis=routing_basis)
    return math.inf if z is None else z


def run(args):
    n = args.nodes
    name = f"network{n:02d}"
    start = args.start or ROOT / "deliverables" / name / f"{name}.sol"
    demand = qt.load_network_demand(n)
    scaled = demand.astype(float) * qt.INTSCALE
    record = checked_score(start, demand)
    initial = ordered_heads(qt.read_network_solution(start)[1])
    best, current = initial, initial
    initial_score = record
    rng = np.random.default_rng(args.seed)
    priority = np.ones(len(initial))
    dest = args.output_dir / f"{name}.sol"
    logfile = args.output_dir / f"{name}.jsonl"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    orders = {
        r: np.asarray([p for p in permutations(range(r)) if all(p[i] != i for i in range(r))])
        for r in args.radii
    }
    started = perf_counter()
    rounds = []
    print(f"{name} from {record:,} ({(record / BKV[n] - 1):+.3%} vs QOBLIB)", flush=True)
    stall = 0
    current_score = float(record)
    seen = set()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_routing_worker,
                             initargs=(demand, initial),
                             mp_context=multiprocessing.get_context("fork")) as executor:
        context = WarmLP(demand, topology(initial))
        context.h.resetGlobalScheduler(True)
        for step in range(args.rounds):
            if args.restart and stall >= args.restart:
                current, current_score = best, float(record)
                priority.fill(1.)
                stall = 0
            _, dual = context.solve(topology(current))
            duals = np.ones(len(current)) if dual is None else np.maximum(dual, 0.) * len(current)
            pool = propose_pool(current, scaled, record, orders, args.subsets, args.shortlist, rng, priority)
            keys = [key for key in pool if key not in seen]
            if not keys:
                continue
            heads, coarse = map(np.asarray, zip(*(pool[key] for key in keys)))
            energy = congestion_energy(heads, coarse, scaled, duals, record, args.passes)
            selected, layers, marked = selections(energy, args.budget, args.shots, args.mark_fraction,
                                                  rng, args.methods)
            if not selected["quantum"]:
                continue
            ids = sorted({i for choices in selected.values() for i in choices})
            cutoff = max(record, current_score) + args.slack
            scores = list(executor.map(score_candidate, ((heads[i], cutoff) for i in ids), chunksize=4))
            lp = dict(zip(ids, scores))
            best_ids = {method: min(choices, key=lp.__getitem__) for method, choices in selected.items()}
            metrics = {
                method: {
                    "lp_best": lp[i] if math.isfinite(lp[i]) else None,
                    "lp_budget": len(selected[method]),
                    "hits": sum(lp[j] <= record - 1 + 1e-6 for j in selected[method]),
                }
                for method, i in best_ids.items()
            }
            index = best_ids["quantum"]
            score = lp[index]
            pre_record = record
            if score <= record - 1 + 1e-6:
                for candidate in sorted(selected["quantum"], key=lp.__getitem__):
                    if lp[candidate] > record - 1 + 1e-6:
                        break
                    try:
                        routing = qt.route_integral(demand, topology(heads[candidate]))
                    except RuntimeError as error:
                        print(f"Integer routing at round {step + 1}: {error}", flush=True)
                        continue
                    if routing.objective < record:
                        record = int(routing.objective)
                        best = heads[candidate]
                        save_checked_solution(dest, demand, routing, expected_score=record)
                        current, current_score = best, lp[candidate]
                        print(f"Checked quantum improvement {record:,} at round {step + 1}", flush=True)
                        break
            if record < pre_record:
                stall = 0
            else:
                stall += 1
                # A bounded uphill move lets the quantum branch test a nearby basin.
                if score < record + args.slack and (
                    score < current_score or rng.random() < args.uphill
                ):
                    current, current_score = heads[index], score
            weights = np.ones(len(initial))
            for i in sorted(selected["quantum"], key=lp.__getitem__)[:8]:
                if math.isfinite(lp[i]):
                    weights[heads[i] != current] += 1.
            priority = .5 * priority + .5 * weights
            seen.update(keys[i] for i in selected["quantum"])
            log = {
                "round": step + 1, "seed": args.seed, "pool": len(heads), "marked": marked,
                "index_qubits": (len(heads) - 1).bit_length(), "grover_layers": layers,
                "shots": args.shots, "physical_lp_calls": len(ids), "checked_best": record,
                "current_lp": current_score, "methods": metrics,
                "seconds": round(perf_counter() - started, 3),
            }
            rounds.append(log)
            with logfile.open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            scores_text = " ".join(f"{m}={values['lp_best']}" for m, values in metrics.items())
            print(f"{step + 1:3d} {log['seconds']:6.1f}s pool={len(heads)} {scores_text} "
                  f"checked={record:,} current={current_score:,.0f}", flush=True)
    summary = {
        "start": initial_score, "checked_best": record, "qoblib": BKV[n],
        "rounds": len(rounds), "max_index_qubits": max((r["index_qubits"] for r in rounds), default=0),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    logfile.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=BKV, default=24)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--rounds", type=int, default=52)
    parser.add_argument("--radii", type=int, nargs="+", choices=range(2, 8), default=[2, 3, 4, 5, 6])
    parser.add_argument("--subsets", type=int, default=950, help="approximate subsets at radius four")
    parser.add_argument("--shortlist", type=int, default=3250)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--budget", type=int, default=120, help="routing LP evaluations per selector")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--mark-fraction", type=float, default=.12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--restart", type=int, default=19)
    parser.add_argument("--slack", type=float, default=18000.)
    parser.add_argument("--uphill", type=float, default=.35)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    args = parser.parse_args()
    if "quantum" not in args.methods:
        parser.error("--methods must include quantum")
    if min(args.rounds, args.subsets, args.passes, args.budget, args.shots, args.workers) < 1:
        parser.error("run and budget settings must be positive")
    if args.shortlist < 4 * len(set(args.radii)):
        parser.error("--shortlist must allow a candidate per radius and screen")
    if not 0 < args.mark_fraction < 1 or not 0 <= args.uphill <= 1 or args.slack <= 0 or args.restart < 0:
        parser.error("invalid marking, acceptance, or restart setting")
    name = f"network{args.nodes:02d}"
    if args.start is None and not (ROOT / "deliverables" / name / f"{name}.sol").is_file():
        parser.error(f"--start is required for {name}")
    run(args)


if __name__ == "__main__":
    main()
