"""Run a fixed-volume XY quantum search over independent asset additions/removals.

Quantum registers hold one bit per eligible asset. An XY exchange can pick up
a unit from any donor and put it into any receiver on the same side.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import time

import numba as nb
import numpy as np

import qlns_core as qc
from paired_trajectory_qaoa import margin
from trajectory_guide import trajectory_guide


@lru_cache(maxsize=48)
def exchange_basis(groups, long_donors, short_donors):
    """Enumerate fixed-weight states and transitions within each sign register."""
    basis = np.array([0], dtype=np.int64)
    edges = []
    shift = 0
    for width, donors in zip(groups, (long_donors, short_donors)):
        if width == 0:
            continue
        states = np.array([
            sum(1 << i for i in selected) << shift
            for selected in itertools.combinations(range(width), donors)
        ], dtype=np.int64)
        basis = (states[:, None] + basis).ravel()
        edges.extend((shift + i, shift + i + 1) for i in range(width - 1))
        if width > 2:
            edges.append((shift, shift + width - 1))
        shift += width
    basis.sort()
    index = np.full(1 << shift, -1, dtype=np.int32)
    index[basis] = np.arange(len(basis))
    left = []
    right = []
    for i, j in edges:
        from_state = ((basis >> i) & 1 != 0) & ((basis >> j) & 1 == 0)
        positions = np.flatnonzero(from_state).astype(np.int32)
        left.append(positions)
        right.append(index[basis[positions] ^ (1 << i) ^ (1 << j)])
    offsets = np.r_[0, np.cumsum([len(edge) for edge in left])]
    return basis, np.concatenate(left), np.concatenate(right), offsets


@nb.njit(cache=True)
def energy_values(a, b, assignments):
    result = np.empty(len(assignments), dtype=np.int64)
    positions = np.empty(len(a), dtype=np.int64)
    for index, mask in enumerate(assignments):
        count = 0
        cost = 0
        for j in range(len(a)):
            if mask >> j & 1:
                cost += a[j]
                for k in range(count):
                    cost += b[positions[k], j]
                positions[count] = j
                count += 1
        result[index] = cost
    return result


@nb.njit(cache=True)
def evolve_xy(energies, initial, left, right, offsets, gammas, betas):
    state = initial.astype(np.complex128)
    for gamma, beta in zip(gammas, betas):
        for i in range(len(state)):
            state[i] *= np.exp(-1j * gamma * energies[i])
        c, s = np.cos(beta), 1j * np.sin(beta)
        for edge in range(len(offsets) - 1):
            for position in range(offsets[edge], offsets[edge + 1]):
                i, j = left[position], right[position]
                first, second = state[i], state[j]
                state[i] = c * first + s * second
                state[j] = c * second + s * first
    return np.abs(state)**2


def candidates(state, t, rng, side_widths, noise, guide):
    p = state.p
    v = state.v[t]
    assigned = set()
    groups = []
    donor_counts = []
    original = 0
    for sign, width in zip((1, -1), side_widths):
        start_index = sum(map(len, groups))
        if width == 0:
            groups.append([])
            donor_counts.append(0)
            continue
        available = np.ones(p.n, dtype=bool)
        available[list(assigned)] = False
        signed = sign * v
        goals = sign * guide[t]
        direction = rng.normal(size=p.n) @ p.RS[t]
        tilt = direction * rng.choice([0.0, 140.0, 375.0, 1025.0, 1800.0]) / direction.std()
        remove_cost = margin(state, t, -sign) - sign * tilt
        add_cost = margin(state, t, sign) + sign * tilt
        near_guide = rng.choice([0, 0, 100, 700, 1800])
        remove_cost += near_guide * (goals - signed)
        add_cost += near_guide * (signed - goals)
        donors = np.flatnonzero((signed > 0) & available)
        count = min(len(donors), width // 2)
        donors = donors[np.argsort(remove_cost[donors] + rng.gumbel(0, noise, len(donors)))[:count]]
        available[donors] = False
        receivers = np.flatnonzero((signed >= 0) & (signed < p.ub) & available)
        # Account for the risk profile of the selected donors when scoring adds.
        if len(donors):
            coupling = p.RS[t][np.ix_(donors, receivers)]
            add_cost[receivers] -= rng.choice([0, 0, 0.22, 0.5, 0.8]) * np.max(coupling, axis=0)
        receivers = receivers[np.argsort(add_cost[receivers] + rng.gumbel(0, noise, len(receivers)))[:width - count]]
        if count == 0 or len(receivers) == 0:
            donor_counts.append(0)
            groups.append([])
            continue
        moves = []
        for donor, receiver in itertools.zip_longest(donors, receivers):
            if donor is not None:
                original |= 1 << (start_index + len(moves))
                moves.append((t, int(donor), -sign))
            if receiver is not None:
                moves.append((t, int(receiver), sign))
        assigned.update(asset for _, asset, _ in moves)
        groups.append(moves)
        donor_counts.append(count)
    return groups, tuple(donor_counts), original


def choose_sector(state, day, groups, counts, guide, rng):
    """Select a feasible pair of long/short volumes for a circuit."""
    p = state.p
    target = np.array([np.maximum(sign * guide[day], 0).sum() for sign in (1, -1)])
    gross, net = state.absv[day], state.net[day]
    current = np.array([(gross + net) // 2, (gross - net) // 2])
    options = []
    for dl, ds in itertools.product(range(-2, 3), repeat=2):
        if (dl == 0 and ds == 0) or gross + dl + ds > p.budget:
            continue
        if not p.C - p.maxcash <= net + dl - ds <= p.C:
            continue
        weights = (counts[0] + dl, counts[1] + ds)
        if all(0 <= weight <= len(group) for group, weight in zip(groups, weights)):
            distance = ((current + [dl, ds] - target)**2).sum()
            options.append((weights, distance))
    if not options:
        return counts
    distance = np.array([adjustment for _, adjustment in options])
    probabilities = np.exp((distance.min() - distance) / rng.choice([0.7, 2.0, 4.0]))
    return options[rng.choice(len(options), p=probabilities / probabilities.sum())][0]


def sample_window(state, day, rng, guide, widths, noise, shots, perturb, mode, change_sector):
    groups, counts, incumbent_mask = candidates(state, day, rng, widths, noise, guide)
    moves = list(itertools.chain.from_iterable(groups))
    if not moves:
        return state, None
    if change_sector:
        counts = choose_sector(state, day, groups, counts, guide, rng)
    basis, left, right, offsets = exchange_basis(tuple(map(len, groups)), *counts)
    assignments = basis ^ incumbent_mask
    a, b, _ = qc.build_qubo(state, moves)
    energies = energy_values(a, b, assignments)
    oracle = int(energies.min())
    if oracle >= 0 and not perturb:
        return state, None
    if mode == "oracle" and not perturb:
        selected = np.array([int(np.argmin(energies))])
    elif mode == "uniform":
        selected = rng.integers(len(basis), size=shots)
    else:
        warm = rng.choice([0.0, 0.0, 0.14, 0.33]) if not perturb else 0
        distance = np.fromiter((int(mask).bit_count() for mask in assignments), dtype=float, count=len(basis))
        initial = np.exp(-warm * distance)
        initial /= np.linalg.norm(initial)
        normalized = (energies - energies.mean()) / max(1, energies.std())
        gammas = np.array([0.58, 1.17, 1.95])
        betas = np.array([0.70, 0.45, 0.23])
        probabilities = evolve_xy(
            normalized, initial, left, right, offsets, gammas, betas
        )
        selected = rng.choice(len(basis), size=shots, p=probabilities / probabilities.sum())
    index = int(selected[np.argmin(energies[selected])])
    if perturb:
        target = int(rng.choice([800, 1700, 3000, 5300]))
        nearby = selected[(energies[selected] > target * 0.45) & (energies[selected] <= target * 1.6)]
        if len(nearby):
            index = int(rng.choice(nearby))
    elif energies[index] >= 0:
        return state, (len(moves), oracle, int(energies[index]))
    v = state.v.copy()
    for i, (t, asset, change) in enumerate(moves):
        if (assignments[index] >> i) & 1:
            v[t, asset] += change
    if not state.p.feasible(v) or state.p.objective(v) != state.obj + energies[index]:
        raise RuntimeError("Incorrect fixed-weight circuit decoding")
    return qc.State(state.p, v), (len(moves), oracle, int(energies[index]))


def search(p, start, guide, rounds, seed, shots, mode, sectors=False, checkpoint=None):
    rng = np.random.default_rng(seed)
    state = qc.State(p, start)
    log = []
    sampled = 0
    improving_windows = 0
    optimum_hits = 0
    width = 0
    t0 = time.monotonic()
    for r in range(rounds):
        last = state.obj
        for t in rng.permutation(p.T - 1):
            trial = state
            kick = r % 5 > 1 and rng.random() < 0.45
            for attempt in range(3 if kick else 1):
                perturb = kick and attempt == 0
                sizes = ((9, 9), (8, 10), (10, 8), (16, 0), (0, 16), (6, 12), (12, 6))
                long_short = sizes[rng.integers(len(sizes))]
                noise = int(rng.choice([55, 160, 420, 840, 1600, 2500]))
                change_sector = sectors and rng.random() < (0.70 if perturb else 0.40)
                trial, info = sample_window(
                    trial, t, rng, guide, long_short, noise, shots, perturb, mode, change_sector
                )
                if info is not None:
                    q, oracle, found = info
                    width = max(q, width)
                    sampled += 1
                    improving_windows += oracle < 0
                    optimum_hits += oracle < 0 and found == oracle
            if trial.obj < state.obj:
                log.append(dict(round=r, day=int(t), score=trial.obj, gain=state.obj - trial.obj))
                state = trial
                if checkpoint is not None:
                    np.save(checkpoint, state.v)
        if state.obj < last or r % 5 == 0:
            print(f"Round {r:3d}   {state.obj:9d}   sampled {sampled:4d}   {time.monotonic() - t0:.1f} s",
                  flush=True)
    return state.v, dict(start=p.objective(start), final=state.obj, mode=mode, seed=seed, shots=shots,
                         sectors=sectors,
                         rounds=rounds, sampled=sampled, max_qubits=width, exact_hits=optimum_hits,
                         improving_windows=improving_windows, accepted=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=75)
    parser.add_argument("--shots", type=int, default=480)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--mode", choices=("xy", "uniform", "oracle"), default="xy")
    parser.add_argument("--sectors", action="store_true", help="Also sample feasible changes in net and gross volume")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.rounds < 0 or args.shots < 1:
        parser.error("Rounds must be nonnegative and shots must be positive")
    p = qc.ps.Problem("po_a400_t15_orig", "1e-3")
    start = np.load(args.start) if args.start.suffix == ".npy" else qc.load_v(p, args.start)
    if start.shape != (p.T, p.n) or not p.feasible(start):
        parser.error("The start portfolio is infeasible")
    guide = trajectory_guide(p, start, steps=450)
    v, summary = search(p, start, guide, args.rounds, args.seed, args.shots, args.mode,
                        args.sectors, args.checkpoint)
    model, score = qc.check_with_referee(p, v)
    if not score.feasible or score.objective != summary["final"]:
        raise RuntimeError(f"Scorer mismatch: {score}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.write_solution(args.output, p.positions(v), comment=f"Fixed-volume exchange search ({args.mode}).")
    args.output.with_suffix(".log.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(score, flush=True)


if __name__ == "__main__":
    main()
