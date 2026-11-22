"""Categorical quantum neighborhoods for the QOBLIB portfolio problem.

Each binary register indexes several valid allocations of one asset block.
Blocks have disjoint assets and preserve each day's long and short volumes,
except for at most one block that can change volumes within the day's slack.
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
from paired_trajectory_qaoa import SEARCH_ANGLES, evolve, pack_wide


@lru_cache
def magnitude_grid(size, limit):
    return np.array(list(itertools.product(range(limit + 1), repeat=size)), dtype=np.int64)


def atom_groups(state, days, rng, blocks, noise):
    """Give each block two to three compatible transfer proposals for its asset support."""
    candidates = []
    p = state.p
    for t in days:
        direction = rng.normal(size=p.n) @ p.RS[t]
        scale = rng.choice([0.0, 120.0, 420.0, 850.0])
        proposals = pack_wide(state, t, rng, blocks * 4, noise, direction * (scale / direction.std()))
        candidates.extend(group for group in proposals if len(group) == 2)
    if len(days) > 1:
        # Interleave the two days by candidate rank before perturbing the order.
        candidates.sort(key=lambda g: g[0][0])
        left = [g for g in candidates if g[0][0] == days[0]]
        right = [g for g in candidates if g[0][0] == days[1]]
        candidates = [g for row in itertools.zip_longest(left, right) for g in row if g is not None]
    rank = np.arange(len(candidates)) + rng.gumbel(0.0, 3.0, len(candidates))
    candidates = [candidates[i] for i in np.argsort(rank)]
    used = set()
    while candidates and blocks > 0:
        anchor = candidates.pop(0)
        group = [anchor]
        members = {asset for _, asset, _ in anchor}
        if members & used:
            continue
        size = int(rng.choice([2, 2, 3]))
        while candidates and len(group) < size:
            shortlist = []
            for k, atom in enumerate(candidates[:18]):
                assets = {asset for _, asset, _ in atom}
                if assets & used:
                    continue
                # Shared assets provide a direct turnover coupling across days.
                shared = len(assets & members)
                cross = 0
                for prior in group:
                    for t, i, di in prior:
                        for u, j, dj in atom:
                            if t == u and i != j:
                                cross += di * dj * p.RS[t, i, j]
                priority = 25 * k + cross - 300 * shared + rng.gumbel(0, 185)
                shortlist.append((priority, k))
            if not shortlist:
                break
            _, k = min(shortlist)
            group.append(candidates.pop(k))
            members.update(asset for _, asset, _ in group[-1])
        blocks -= 1
        used.update(members)
        candidates = [
            atom for atom in candidates if not any(asset in members for _, asset, _ in atom)
        ]
        yield np.array(sorted(members)), group


def day_options(state, t, assets, atoms, free_volumes, rng, noise):
    p = state.p
    old = state.v[t, assets]
    side = np.sign(old)
    for atom in atoms:
        for day, asset, delta in atom:
            i = np.searchsorted(assets, asset)
            if day == t and side[i] == 0:
                side[i] = np.sign(delta)
    undecided = side == 0
    side[undecided] = np.where(
        p.lin[t, assets[undecided]] + state.G[t, assets[undecided]] <= 0, 1, -1
    )
    grid = magnitude_grid(len(assets), p.ub)
    longs = grid[:, side > 0].sum(axis=1)
    shorts = grid[:, side < 0].sum(axis=1)
    old_long = np.maximum(old, 0).sum()
    old_short = np.maximum(-old, 0).sum()
    if free_volumes:
        gross = state.absv[t] + longs + shorts - old_long - old_short
        net = state.net[t] + longs - shorts - old_long + old_short
        valid = (gross <= p.budget) & (net >= p.C - p.maxcash) & (net <= p.C)
    else:
        valid = (longs == old_long) & (shorts == old_short)
    delta = grid[valid] * side - old
    delta = delta[np.any(delta, axis=1)]
    delta = np.vstack((np.zeros(len(assets), dtype=np.int64), delta))
    new = old + delta
    coefficients = state.G[t, assets] + p.lin[t, assets] + 10
    risk = p.R[t][np.ix_(assets, assets)]
    energy = delta @ coefficients + np.sum((delta @ risk) * delta, axis=1)
    energy += (np.maximum(-new, 0) - np.maximum(-old, 0)) @ p.sc[t, assets]
    energy += (np.abs(new) - np.abs(old)) @ p.absc[t, assets]
    if t:
        prev = state.v[t - 1, assets]
        energy += (np.abs(new - prev) - np.abs(old - prev)) @ p.w[t, assets]
    if t + 1 < p.T:
        after = state.v[t + 1, assets]
        energy += (np.abs(after - new) - np.abs(after - old)) @ p.w[t + 1, assets]
    priority = energy[1:] + rng.gumbel(0, noise, len(energy) - 1)
    index = np.r_[0, 1 + np.argsort(priority)[:13]]
    return delta[index], energy[index]


def register_options(state, days, assets, atoms, free_volumes, rng, noise, max_levels):
    left, energy = day_options(state, days[0], assets, atoms, free_volumes, rng, noise)
    if len(days) == 1:
        moves = left[:, None, :]
    else:
        t, u = days
        right, right_energy = day_options(state, u, assets, atoms, free_volumes, rng, noise)
        old = state.v[u, assets] - state.v[t, assets]
        first, second = left[:, None, :], right[None, :, :]
        turnover = np.abs(old + second - first) - np.abs(old + second) - np.abs(old - first) + np.abs(old)
        energy = (energy[:, None] + right_energy + turnover @ state.p.w[u, assets]).ravel()
        x, y = np.indices((len(left), len(right)))
        moves = np.stack((left[x.ravel()], right[y.ravel()]), axis=1)
    levels = min(max_levels, 1 << (len(energy).bit_length() - 1))
    if levels == 1:
        return None
    priority = energy[1:] + rng.gumbel(0, noise, len(energy) - 1)
    index = np.r_[0, 1 + np.argsort(priority)[:levels - 1]]
    return moves[index], energy[index]


def build_window(state, days, rng, registers, max_levels, noise):
    p = state.p
    options = []
    linear = []
    for i, (assets, atoms) in enumerate(atom_groups(state, days, rng, registers, noise)):
        option = register_options(state, days, assets, atoms, i == 0, rng, noise, max_levels)
        if option is not None:
            moves, energy = option
            options.append((assets, moves))
            linear.append(energy)
    if not options:
        return None
    sizes = np.array([len(energy) for energy in linear], dtype=np.int64)
    width = int(np.prod(sizes)).bit_length() - 1
    if width > 18:
        raise ValueError("Window exceeds the statevector limit")
    fields = np.zeros((len(sizes), max(sizes)), dtype=np.int64)
    cross = np.zeros((len(sizes), len(sizes), max(sizes), max(sizes)), dtype=np.int64)
    for i, (assets, moves) in enumerate(options):
        fields[i, :sizes[i]] = linear[i]
        for j in range(i + 1, len(options)):
            other, alternatives = options[j]
            for k, t in enumerate(days):
                risk = p.RS[t][np.ix_(assets, other)]
                cross[i, j, :sizes[i], :sizes[j]] += (moves[:, k] @ risk) @ alternatives[:, k].T
    return options, sizes, energy_table(fields, cross, sizes), width


@nb.njit(cache=True)
def energy_table(fields, cross, sizes):
    values = np.zeros(np.prod(sizes), dtype=np.int64)
    stride = 1
    for j, size in enumerate(sizes):
        for choice in range(1, size):
            for prefix in range(stride):
                energy = fields[j, choice]
                remaining = prefix
                for i in range(j):
                    energy += cross[i, j, remaining % sizes[i], choice]
                    remaining //= sizes[i]
                values[choice * stride + prefix] = values[prefix] + energy
        stride *= size
    return values


def decode(v, options, days, index):
    candidate = v.copy()
    for assets, moves in options:
        index, digit = divmod(index, len(moves))
        candidate[np.ix_(days, assets)] += moves[digit]
    return candidate


def grover_samples(energies, rng, shots):
    """Simulate Grover reflections with a sampled incumbent as the phase threshold."""
    samples = [int(rng.integers(len(energies)))]
    n = len(energies)
    probability = np.full(n, 1.0 / n)
    incumbent = energies[samples[0]]
    # Geometrically widen the randomized search depth after an unsuccessful
    # threshold test. Only the ideal simulator uses the marked count.
    depth = 1.0
    for batch in np.array_split(np.arange(shots - 1), min(24, shots - 1)):
        iterations = rng.integers(max(1, int(np.ceil(depth))), size=len(batch))
        good = energies < incumbent
        count = int(good.sum())
        if 0 < count < n:
            angle = np.arcsin(np.sqrt(count / n))
            # Averaging the probabilities corresponds to an independently
            # randomized number of Grover reflections per shot.
            weight = np.mean(np.sin((2 * iterations + 1) * angle)**2)
            probability[good] = weight / count
            probability[~good] = (1 - weight) / (n - count)
        else:
            probability.fill(1.0 / n)
        measurements = rng.choice(n, size=len(batch), p=probability)
        samples.extend(measurements)
        candidate = energies[measurements].min()
        depth = min(np.sqrt(n), depth * 1.6) if candidate >= incumbent else max(depth / 1.2, 1.0)
        incumbent = min(incumbent, candidate)
    return np.asarray(samples, dtype=np.int64)


def sample_state(energies, width, rng, shots, sampler, perturb):
    if sampler == "grover" and not perturb:
        return grover_samples(energies, rng, shots)
    if sampler == "uniform":
        return rng.integers(len(energies), size=shots)
    if sampler == "oracle" and not perturb:
        return np.array([int(np.argmin(energies))])
    normalized = (energies - energies.mean()) / max(1, energies.std())
    probabilities = evolve(normalized, width, *SEARCH_ANGLES)
    return rng.choice(len(energies), size=shots, p=probabilities / probabilities.sum())


def search(p, start, rounds, seed, shots, span, sampler, checkpoint=None):
    p_start = p.objective(start)
    state = qc.State(p, start)
    generator = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    control = np.random.default_rng(seed + 2)
    log = []
    stats = dict(windows=0, sampled=0, viable=0, q_better=0, uniform_better=0, exact=0, max_qubits=0)
    timestamp = time.monotonic()
    for epoch in range(rounds):
        previous = state.obj
        for t in generator.permutation(p.T - 1):
            if span == "two" or (span == "mixed" and generator.random() < 0.45):
                days = (min(t, p.T - 3), min(t, p.T - 3) + 1)
            else:
                days = (t,)
            trial = state
            kick = epoch % 5 >= 2 and generator.random() < 0.50
            steps = 3 if kick else 1
            for step in range(steps):
                perturb = kick and step == 0
                noise = generator.choice([55, 150, 420, 840, 1650]) if not perturb else 950
                levels = int(generator.choice([4, 4, 8]))
                registers = 7 if levels == 4 else 5
                window = build_window(trial, days, generator, registers, levels, noise)
                if window is None:
                    continue
                options, sizes, energies, width = window
                stats["windows"] += 1
                stats["max_qubits"] = max(stats["max_qubits"], width)
                if energies.min() >= 0 and not perturb:
                    continue
                stats["sampled"] += 1
                samples = sample_state(energies, width, rng, shots, sampler, perturb)
                index = int(samples[np.argmin(energies[samples])])
                if not perturb:
                    random_energy = energies[control.integers(len(energies), size=shots)].min()
                    stats["viable"] += 1
                    stats["q_better"] += int(energies[index] < random_energy)
                    stats["uniform_better"] += int(energies[index] > random_energy)
                    stats["exact"] += int(energies[index] == energies.min())
                if perturb:
                    target = generator.choice([500, 1300, 2700, 4200])
                    nearby = samples[(energies[samples] >= target / 2) & (energies[samples] <= 2 * target)]
                    if len(nearby):
                        index = int(rng.choice(nearby))
                elif energies[index] >= 0:
                    continue
                candidate = decode(trial.v, options, days, index)
                exact = p.objective(candidate)
                if not p.feasible(candidate) or exact != trial.obj + energies[index]:
                    raise RuntimeError("Allocation or energy mismatch in the decoded state")
                trial = qc.State(p, candidate)
            if trial.obj < state.obj:
                log.append(dict(epoch=epoch, days=[int(t) for t in days], gain=state.obj - trial.obj,
                                score=trial.obj))
                state = trial
                if checkpoint is not None:
                    np.save(checkpoint, state.v)
        if state.obj < previous or epoch % 5 == 0:
            print(f"Round {epoch:3d}   {state.obj:9d}   sampled {stats['sampled']:4d}   "
                  f"{time.monotonic() - timestamp:.1f} s", flush=True)
    summary = dict(start=p_start, final=state.obj, span=span, sampler=sampler, seed=seed, shots=shots,
                   stats=stats, accepted=log)
    return state.v, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_start = qc.ROOT / "deliverables/portfolio_quantum/po_a400_t15_orig_b100_l1e-3__paired-trajectory-qaoa.sol"
    parser.add_argument("--start", type=Path, default=default_start)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=55)
    parser.add_argument("--shots", type=int, default=480)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--span", choices=("one", "two", "mixed"), default="mixed")
    parser.add_argument("--sampler", choices=("qaoa", "grover", "uniform", "oracle"), default="qaoa")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.rounds < 0 or args.shots < 6:
        parser.error("Rounds must be nonnegative and shots must be at least six")
    p = qc.ps.Problem("po_a400_t15_orig", "1e-3")
    start = np.load(args.start) if args.start.suffix == ".npy" else qc.load_v(p, args.start)
    if start.shape != (p.T, p.n) or not p.feasible(start):
        parser.error("The start portfolio is infeasible")
    v, report = search(p, start, args.rounds, args.seed, args.shots, args.span, args.sampler, args.checkpoint)
    model, score = qc.check_with_referee(p, v)
    if not score.feasible or score.objective != report["final"]:
        raise RuntimeError(f"Scorer mismatch: {score}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.write_solution(args.output, p.positions(v), comment=f"Categorical portfolio neighborhoods ({args.sampler}).")
    args.output.with_suffix(".log.json").write_text(json.dumps(report, indent=2) + "\n")
    print(score, flush=True)
    print(report["stats"], flush=True)


if __name__ == "__main__":
    main()
