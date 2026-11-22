"""Quantum fusion of two daily portfolios through balanced asset transfers.

A perturbed tabu search proposes alternate portfolios. A disjoint decomposition
of their differences defines binary switches for the quantum sampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import qlns_core as qc
from paired_trajectory_qaoa import all_energies, evolve, SEARCH_ANGLES
from block_trajectory_qaoa import grover_samples


def flow_components(reference, alternate, rng):
    """Partition changed assets into components balanced in long and short volume."""
    assets = np.flatnonzero(reference != alternate)
    # One extra node joins all unmatched units in a single feasible component.
    dummy = len(assets)
    parents = list(range(dummy + 1))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for sign in (1, -1):
        delta = np.maximum(sign * alternate[assets], 0) - np.maximum(sign * reference[assets], 0)
        received = rng.permutation(np.repeat(np.arange(dummy), np.maximum(delta, 0))).tolist()
        sent = rng.permutation(np.repeat(np.arange(dummy), np.maximum(-delta, 0))).tolist()
        length = max(len(sent), len(received))
        sent.extend([dummy] * (length - len(sent)))
        received.extend([dummy] * (length - len(received)))
        for donor, receiver in zip(sent, received):
            parents[root(donor)] = root(receiver)
    components = {}
    for i, asset in enumerate(assets):
        components.setdefault(root(i), []).append(asset)
    return [np.array(group, dtype=np.int64) for group in components.values()]


def fusion_qubo(state, t, alternate, components):
    p = state.p
    old = state.v[t]
    diff = alternate - old
    changes = np.zeros((len(components), p.n), dtype=np.int64)
    for i, assets in enumerate(components):
        changes[i, assets] = diff[assets]
    unary = diff * (state.G[t] + p.lin[t] + 10)
    unary += (np.maximum(-alternate, 0) - np.maximum(-old, 0)) * p.sc[t]
    unary += (np.abs(alternate) - np.abs(old)) * p.absc[t]
    if t:
        prev = state.v[t - 1]
        unary += (np.abs(alternate - prev) - np.abs(old - prev)) * p.w[t]
    if t + 1 < p.T:
        after = state.v[t + 1]
        unary += (np.abs(after - alternate) - np.abs(after - old)) * p.w[t + 1]
    risk = (changes @ p.RS[t]) @ changes.T
    a = np.array([unary[assets].sum() for assets in components]) + risk.diagonal() // 2
    return a, np.triu(risk, k=1)


def alternative_day(state, t, rng, steps, phase, pool):
    p = state.p
    v = state.v.copy()
    if pool and rng.random() < 0.55:
        v[t] = pool[rng.integers(len(pool))]

    premium = rng.choice([0, 40, 90, 180, 400, 720, 1200, 2400])
    if phase == "factor":
        tilt = rng.normal(size=p.n) @ p.RS[t]
        tilt /= tilt.std()
    else:
        tilt = rng.standard_normal(p.n)
    lin = p.lin.copy()
    lin[t] += np.rint(premium * tilt).astype(np.int64)
    trial = qc.State(p, v)
    tenure = int(rng.choice([5, 9, 15, 31, 51]))
    qc.ps.day_tabu(t, trial.v, trial.G, trial.net, trial.absv, steps, tenure, *p.args[:2], lin, *p.args[3:])
    if premium and rng.random() < 0.76:
        qc.ps.day_tabu(t, trial.v, trial.G, trial.net, trial.absv, max(75, steps // 3), tenure, *p.args)
    return trial.v[t].copy()


def fusion_window(state, alternatives, rng):
    """Build the exact QUBO, including turnover when two dates share assets."""
    p = state.p
    days = sorted(alternatives)
    groups = [flow_components(state.v[t], alternatives[t], rng) for t in days]
    offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
    components = [(t, assets) for t, group in zip(days, groups) for assets in group]
    a = np.zeros(len(components), dtype=np.int64)
    b = np.zeros((len(components), len(components)), dtype=np.int64)
    for k, (t, group) in enumerate(zip(days, groups)):
        part = slice(offsets[k], offsets[k + 1])
        a[part], b[part, part] = fusion_qubo(state, t, alternatives[t], group)
    for k, (t, u) in enumerate(zip(days, days[1:])):
        if u != t + 1:
            continue
        for i, left in enumerate(groups[k], start=offsets[k]):
            for j, right in enumerate(groups[k + 1], start=offsets[k + 1]):
                assets = np.intersect1d(left, right, assume_unique=True)
                old = state.v[u, assets] - state.v[t, assets]
                first = alternatives[t][assets] - state.v[t, assets]
                second = alternatives[u][assets] - state.v[u, assets]
                b[i, j] = p.w[u, assets] @ (
                    np.abs(old + second - first) - np.abs(old + second)
                    - np.abs(old - first) + np.abs(old)
                )
    if len(components) > 17:
        kept = rng.choice(len(components), size=17, replace=False)
        # Preserve the upper triangle when removing components.
        kept.sort()
        a, b = a[kept], b[np.ix_(kept, kept)]
        components = [components[j] for j in kept]
    return a, b, components


def fuse(state, alternatives, rng, shots, sampler, tries, stats):
    p = state.p
    best = state
    alternatives = {t: alt for t, alt in alternatives.items() if not np.array_equal(state.v[t], alt)}
    if not alternatives:
        return best
    parent = state.v.copy()
    for t, alt in alternatives.items():
        parent[t] = alt
    parent_score = p.objective(parent)
    stats["best_alternate_score"] = min(stats["best_alternate_score"], parent_score)
    for _ in range(tries):
        a, b, components = fusion_window(state, alternatives, rng)
        energies = all_energies(a, b)
        stats["windows"] += 1
        stats["max_qubits"] = max(stats["max_qubits"], len(components))
        if energies.min() >= 0:
            continue
        stats["sampled"] += 1
        if sampler == "grover":
            candidates = grover_samples(energies, rng, shots)
        else:
            normalized = (energies - energies.mean()) / energies.std()
            prob = evolve(normalized, len(components), *SEARCH_ANGLES)
            candidates = rng.choice(len(energies), size=shots, p=prob / prob.sum())
        chosen = int(candidates[np.argmin(energies[candidates])])
        if state.obj + energies[chosen] >= best.obj:
            continue
        candidate = state.v.copy()
        for i, (t, assets) in enumerate(components):
            if chosen >> i & 1:
                candidate[t, assets] = alternatives[t][assets]
        if not p.feasible(candidate) or p.objective(candidate) != state.obj + energies[chosen]:
            raise RuntimeError("Fusion energy or feasibility mismatch")
        best = qc.State(p, candidate)
    if best.obj < min(state.obj, parent_score):
        stats["strict_fusion_gains"] += 1
    return best


def search(p, start, parents, rounds, steps, seed, shots, sampler, span, checkpoint=None):
    rng = np.random.default_rng(seed)
    qc.ps.seed_rng(seed)
    elite = qc.State(p, start)
    pool = [[v[t].copy() for v in parents if np.any(v[t] != start[t])] for t in range(p.T)]
    stats = dict(windows=0, sampled=0, max_qubits=0, strict_fusion_gains=0,
                 best_alternate_score=elite.obj)
    log = []
    t0 = time.monotonic()
    for epoch in range(rounds):
        initial = elite.obj
        for t in rng.permutation(p.T - 1):
            days = (t, t + 1) if span == "mixed" and t < p.T - 2 and rng.random() < 0.65 else (t,)
            alternatives = {}
            guide = elite
            for day in days:
                alt = alternative_day(
                    guide, day, rng, steps, "factor" if epoch % 3 else "independent", pool[day]
                )
                alternatives[day] = alt
                if day != days[-1]:
                    next_guide = guide.v.copy()
                    next_guide[day] = alt
                    guide = qc.State(p, next_guide)
            previous = elite.obj
            result = fuse(elite, alternatives, rng, shots, sampler, 4, stats)
            if result.obj < previous:
                log.append(dict(round=epoch, days=[int(day) for day in days], score=result.obj,
                                gain=previous - result.obj))
                for day in days:
                    pool[day].append(elite.v[day].copy())
                elite = result
                if checkpoint is not None:
                    np.save(checkpoint, elite.v)
            for day, alt in alternatives.items():
                if not np.array_equal(alt, elite.v[day]) and not any(np.array_equal(alt, prior) for prior in pool[day]):
                    # Keep varied local basins. The selection is refreshed after each turn.
                    if len(pool[day]) >= 10:
                        pool[day].pop(int(rng.integers(len(pool[day]))))
                    pool[day].append(alt)
        if elite.obj < initial or epoch % 5 == 0:
            print(f"Round {epoch:3d}   {elite.obj:9d}   sampled {stats['sampled']:4d}   "
                  f"{time.monotonic() - t0:.1f} s", flush=True)
    return elite.v, dict(start=p.objective(start), final=elite.obj, seed=seed, shots=shots, sampler=sampler, span=span,
                         rounds=rounds, steps=steps, stats=stats, accepted=log)


def combine_pool(p, start, parents, rounds, seed, shots, sampler):
    """Fuse nearby days from different recorded parents, including mixed pairs."""
    rng = np.random.default_rng(seed)
    elite = qc.State(p, start)
    stats = dict(windows=0, sampled=0, max_qubits=0, strict_fusion_gains=0,
                 best_alternate_score=elite.obj)
    log = []
    choices = [
        np.unique(np.stack([parent[t] for parent in parents]), axis=0)
        for t in range(p.T)
    ]
    for epoch in range(rounds):
        initial = elite.obj
        for t in rng.permutation(p.T - 1):
            proposals = [{t: row} for row in choices[t]]
            if t < p.T - 2:
                proposals.extend(
                    {t: left, t + 1: right}
                    for left in choices[t] for right in choices[t + 1]
                )
            rng.shuffle(proposals)
            for proposal in proposals:
                previous = elite.obj
                elite = fuse(elite, proposal, rng, shots, sampler, 3, stats)
                if elite.obj < previous:
                    log.append(dict(round=epoch, days=list(map(int, proposal)),
                                    score=elite.obj, gain=previous - elite.obj))
        print(f"Population round {epoch}   {elite.obj}   sampled {stats['sampled']}", flush=True)
        if elite.obj == initial:
            break
    return elite.v, dict(start=p.objective(start), final=elite.obj, seed=seed, shots=shots, sampler=sampler,
                         mode="population", rounds=rounds, stats=stats, accepted=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True,
                        help="Canonical .sol, NumPy checkpoint, or replay .npz with start and parents")
    parser.add_argument("--parents", type=Path)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--shots", type=int, default=480)
    parser.add_argument("--sampler", choices=("qaoa", "grover"), default="qaoa")
    parser.add_argument("--span", choices=("one", "mixed"), default="one")
    parser.add_argument("--population-only", action="store_true", help="Fuse saved parents directly")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 6 or args.rounds < 0 or args.steps < 1:
        parser.error("Use at least six shots and positive tabu steps")
    p = qc.ps.Problem("po_a400_t15_orig", "1e-3")
    parents = []
    if args.start.suffix == ".npz":
        with np.load(args.start, allow_pickle=False) as data:
            start = data["start"].astype(np.int64)
            parents = data["parents"].astype(np.int64)
    elif args.start.suffix == ".npy":
        start = np.load(args.start)
    else:
        start = qc.load_v(p, args.start)
    if args.parents:
        with np.load(args.parents, allow_pickle=False) as data:
            parents = data["parents"].astype(np.int64)
    if start.shape != (p.T, p.n) or not p.feasible(start) or any(
        v.shape != start.shape or not p.feasible(v) for v in parents
    ):
        parser.error("Start and parents must be feasible portfolios")
    if args.population_only:
        if not len(parents):
            parser.error("Population fusion requires --parents")
        v, summary = combine_pool(p, start, parents, args.rounds, args.seed, args.shots, args.sampler)
    else:
        v, summary = search(p, start, parents, args.rounds, args.steps, args.seed, args.shots,
                            args.sampler, args.span, args.checkpoint)
    model, score = qc.check_with_referee(p, v)
    if not score.feasible or score.objective != summary["final"]:
        raise RuntimeError(f"Scorer mismatch: {score}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.write_solution(args.output, p.positions(v), comment=f"Balanced-transfer fusion ({args.sampler}).")
    args.output.with_suffix(".log.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(score, flush=True)
    print(summary["stats"], flush=True)


if __name__ == "__main__":
    main()
