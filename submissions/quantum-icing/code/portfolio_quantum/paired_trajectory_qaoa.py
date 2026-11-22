"""Experimental portfolio QAOA with one qubit per feasible transfer.

Run from the repository root with the project Python. This implementation
simulates diagonal cost evolution and single-qubit RX mixers with a statevector.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numba as nb
import numpy as np
from scipy.optimize import linear_sum_assignment

import qlns_core as qc
from trajectory_guide import trajectory_guide


TARGET = -3_162_386
SEARCH_ANGLES = (np.array([0.48, 0.95, 1.6]), np.array([0.61, 0.42, 0.23]))
PERTURB_ANGLES = (np.array([0.24, 0.47, 0.79]), np.array([0.28, 0.23, 0.12]))


@nb.njit(cache=True)
def all_energies(a, b):
    """Build the energy diagonal for the local simulator, in little-endian order."""
    n = len(a)
    energies = np.zeros(1 << n, np.int64)
    for j in range(n):
        m = 1 << j
        for i in range(m):
            e = a[j]
            for k in range(j):
                if (i >> k) & 1:
                    e += b[k, j]
            energies[m + i] = energies[i] + e
    return energies


@nb.njit(cache=True)
def evolve(energies, n, gammas, betas):
    """Simulate alternating cost phases and RX(-2 beta) gates from the plus state."""
    size = len(energies)
    state = np.full(size, 1 / np.sqrt(size), np.complex128)
    for gamma, beta in zip(gammas, betas):
        for i in range(size):
            state[i] *= np.exp(-1j * gamma * energies[i])
        c, s = np.cos(beta), 1j * np.sin(beta)
        for k in range(n):
            span = 1 << k
            for base in range(0, size, 2 * span):
                for i in range(base, base + span):
                    e, f = state[i], state[i + span]
                    state[i] = c * e + s * f
                    state[i + span] = s * e + c * f
    return np.abs(state)**2


def flow_round(p, guide):
    """Round toward the guide while choosing feasible long and short totals."""
    base = np.trunc(guide+np.sign(guide)*.0001).astype(np.int64)
    upper = np.clip(np.rint(guide+np.sign(guide)*.5).astype(np.int64), -p.ub, p.ub)
    v = base.copy()
    for t in range(p.T):
        desired = [np.maximum(0, f*guide[t]).sum() for f in (1, -1)]
        counts = [np.maximum(0, f*base[t]).sum() for f in (1, -1)]
        score, best = float('inf'), None
        groups = []
        for f in (1, -1):
            idx = np.flatnonzero((f*(upper[t]-base[t]) == 1) & (f*guide[t] > 0) & (abs(base[t]) < p.ub))
            groups.append(idx[np.argsort(-abs(guide[t, idx]-base[t, idx]))])
        for l in range(counts[0], counts[0]+len(groups[0])+1):
            for s in range(counts[1], counts[1]+len(groups[1])+1):
                cost = (l-desired[0])**2+(s-desired[1])**2
                if l+s <= p.budget and p.C-p.maxcash <= l-s <= p.C and cost < score:
                    score, best = cost, (l, s)
        for q, f in enumerate((1, -1)):
            v[t, groups[q][:best[q]-counts[q]]] += f
    return v, base, upper


def move_coeffs(state, t):
    return np.stack((margin(state, t, 1), margin(state, t, -1)))


def pack(state, t, base, upper, rng, width, noise, strict, risk_bias):
    v = state.v[t]
    p = state.p
    c = move_coeffs(state, t)
    result = []
    for s, f in enumerate((1, -1)):
        donors = np.flatnonzero(f*v > 0)
        rec = np.flatnonzero((f*v >= 0) & (f*v < p.ub))
        if strict:
            donors = donors[(v[donors] == upper[donors]) & (f*(upper[donors]-base[donors]) == 1)]
            rec = rec[(v[rec] == base[rec]) & (f*(upper[rec]-base[rec]) == 1)]
        else:
            # Limit candidates to likely improving trades before bipartite matching.
            donors = donors[np.argsort(c[1-s, donors])[:22]]
            kscore = c[s, rec]+rng.gumbel(0, noise, size=len(rec))
            rec = rec[np.argsort(kscore)[:30]]
        delta = (c[1-s, donors, None]+c[s, rec]-p.RS[t][np.ix_(donors, rec)]).astype(float)
        if not delta.size:
            continue
        norm = p.R[t].diagonal()[donors, None]+p.R[t].diagonal()[rec]-p.RS[t][np.ix_(donors, rec)]
        weights = delta+risk_bias*norm+rng.gumbel(0, noise, size=delta.shape)
        weights[donors[:, None] == rec] = 1e12
        ii, jj = linear_sum_assignment(weights)
        for i, j in zip(ii, jj):
            if weights[i, j] < 1e11:
                result.append((weights[i, j], int(donors[i]), int(rec[j]), f))
    result.sort()
    used = set()
    trades = []
    for _, i, j, f in result:
        if i not in used and j not in used:
            trades.append([(t, i, -f), (t, j, f)])
            used.update((i, j))
            if len(trades) == width:
                break
    return trades


def collapse(state, trades):
    """Contract all the bits of each disjoint trade onto its single logical qubit."""
    moves = [move for turn in trades for move in turn]
    qa, qb, _ = qc.build_qubo(state, moves)
    owners = np.repeat(np.arange(len(trades)), [len(q) for q in trades])
    compressed = np.zeros((len(trades), len(trades)), dtype=np.int64)
    np.add.at(compressed, (owners[:, None], owners[None, :]), qb)
    np.add.at(compressed, (owners, owners), qa)
    return compressed.diagonal().copy(), np.triu(compressed, k=1)


def rounding_search(start, guide, path, scheme, rounds=32, seed=1, shots=448):
    R = np.random.default_rng(seed)
    reference = np.random.default_rng(seed+100)
    p = qc.ps.Problem('po_a400_t15_orig', '1e-3')
    rnd, base, upper = flow_round(p, guide)
    state = qc.State(p, rnd if start is None else start)
    assert p.feasible(state.v)
    initial = state.obj
    print('seed', initial, 'rounded guide', p.objective(rnd), 'gross', state.absv, 'net', state.net, flush=True)
    t0 = time.time()
    scorelog = []
    benchmarking = []
    for k in range(rounds):
        wins = 0
        for t in R.permutation(p.T-1):
            strict = (k < 6 and start is None) or (start is None and k%3 == 2)
            width = 16 if strict else 17
            noise = R.choice([80, 200, 500, 1000, 2000])
            risk_bias = R.choice([0, .2, .7, 1.4]) if strict else R.choice([0, 0, .1, .35])
            trades = pack(state, t, base[t], upper[t], R, width, noise, strict, risk_bias)
            n = len(trades)
            if n < 1:
                continue
            a, b = collapse(state, trades)
            energies = all_energies(a, b)
            if energies.min() >= 0:
                continue
            # The simulator represents the exact diagonal phase and product of RX gates.
            if scheme == 'oracle':
                mask = int(np.argmin(energies))
            elif scheme == 'uniform':
                samples = reference.integers(len(energies), size=shots)
                mask = int(samples[np.argmin(energies[samples])])
            else:
                scale = energies.std()
                prob = evolve((energies - energies.mean()) / scale, n, *SEARCH_ANGLES)
                samples = reference.choice(len(energies), size=shots, p=prob/prob.sum())
                mask = int(samples[np.argmin(energies[samples])])
                uniform = R.integers(len(energies), size=shots)
                benchmarking.append((energies[mask], int(energies[uniform].min()), int(energies.min()), n))
            if energies[mask] < 0:
                v = state.v.copy()
                for j, trade in enumerate(trades):
                    if mask>>j & 1:
                        for day, i, d in trade:
                            v[day, i] += d
                delta = state.obj
                state = qc.State(p, v)
                assert state.obj == delta+energies[mask] and p.feasible(v)
                scorelog.append((k, int(t), n, mask, state.obj, int(energies[mask])))
                wins += 1
        print(k, state.obj, wins, round(time.time()-t0, 2), flush=True)
        if state.obj <= TARGET:
            break
    if benchmarking:
        a = np.array(benchmarking)
        print('same-window', len(a), 'q better', sum(a[:, 0] < a[:, 1]), 'u better', sum(a[:, 0] > a[:, 1]), 'q opt', sum(a[:, 0] == a[:, 2]), 'u opt', sum(a[:, 1] == a[:, 2]), 'maxq', max(a[:, 3]), flush=True)
    Path(str(path)+'.json').write_text(json.dumps(dict(start=initial, rows=scorelog, benchmarking=benchmarking), default=int))
    np.save(path, state.v)
    print('final', state.obj)


def quantum(state, trades, shots, rng, scheme):
    if not trades:
        return state, False
    a, b = collapse(state, trades)
    energies = all_energies(a, b)
    if energies.min() >= 0:
        return state, False
    if scheme == 'oracle':
        k = int(np.argmin(energies))
    elif scheme == 'uniform':
        qs = rng.integers(len(energies), size=shots)
        k = int(qs[np.argmin(energies[qs])])
    else:
        n = len(trades)
        normalized = (energies-energies.mean())/energies.std()
        p = evolve(normalized, n, *SEARCH_ANGLES)
        qs = rng.choice(len(energies), size=shots, p=p/p.sum())
        k = int(qs[np.argmin(energies[qs])])
    if energies[k] >= 0:
        return state, True
    v = state.v.copy()
    for j, trade in enumerate(trades):
        if (k>>j) & 1:
            for t, i, d in trade:
                v[t, i] += d
    new = qc.State(state.p, v)
    assert new.obj == state.obj+energies[k] and state.p.feasible(v)
    return new, True


def reround(p, t, center, base, upper, rng, prev):
    noise = rng.choice([.13, .30, .53, 1.])
    F = [1, -1]
    desired = np.array([np.maximum(0, f*center).sum() for f in F])
    counts = np.array([np.maximum(0, f*base).sum() for f in F])
    groups = []
    for f in F:
        idx = np.flatnonzero(f*(upper-base) == 1)
        preference = abs(center[idx]-base[idx])+rng.gumbel(0, noise, len(idx))
        groups.append(idx[np.argsort(-preference)])
    options = []
    for l in range(max(counts[0], int(desired[0])-3), min(counts[0]+len(groups[0]), int(desired[0])+4)+1):
        for s in range(max(counts[1], int(desired[1])-3), min(counts[1]+len(groups[1]), int(desired[1])+4)+1):
            if l+s <= p.budget and p.C-p.maxcash <= l-s <= p.C:
                penalty = 1.8*(l+s-p.budget)**2 + .7*((l-desired[0])**2+(s-desired[1])**2)
                options.append((l, s, penalty))
    if not options:
        return prev
    options = np.array(options)
    likelihood = np.exp(-(options[:, 2]-min(options[:, 2]))/1.7)
    new = options[rng.choice(len(options), p=likelihood/likelihood.sum()), :2].astype(int)
    v = base.copy()
    for i, f in enumerate(F):
        v[groups[i][:new[i]-counts[i]]] += f
    return v


def augment(state, t, trades, rng):
    p = state.p
    v = state.v[t]
    used = [i for group in trades for _, i, d in group]
    allowed = np.ones(p.n, bool)
    allowed[used] = False
    cost = move_coeffs(state, t)
    gross, net = state.absv[t], state.net[t]
    best = []
    for q, f in enumerate((1, -1)):
        for typ in (1, -1):
            legal = allowed & (abs(v+f) <= p.ub) & (abs(v+f)-abs(v) == typ)
            idx = np.flatnonzero(legal)
            if len(idx):
                vals = cost[q, idx]+rng.gumbel(0, 450, len(idx))
                ii = np.argsort(vals)[:8]
                best += [(vals[k], int(idx[k]), f, typ) for k in ii]
    candidates = []
    for delta, i, f, typ in best:
        if gross+typ <= p.budget and p.C-p.maxcash <= net+f <= p.C:
            candidates.append((delta, [(t, i, f)]))
        for delta2, j, g, typ2 in best:
            if i < j and typ+typ2 <= 0 and gross+typ+typ2 <= p.budget and p.C-p.maxcash <= net+f+g <= p.C and (typ+typ2 != 0 or f+g != 0):
                candidates.append((delta+delta2+f*g*p.RS[t, i, j], [(t, i, f), (t, j, g)]))
    if candidates:
        trades.append(min(candidates, key=lambda q:q[0])[1])
    return trades


def restart_search(start, guide, out, rounds=60, seed=21, scheme='qaoa'):
    p = qc.ps.Problem('po_a400_t15_orig', '1e-3')
    rng = np.random.default_rng(seed)
    rnd, base, upper = flow_round(p, guide)
    s = qc.State(p, start)
    log = []
    tm = time.time()
    n = 0
    for k in range(rounds):
        wins = 0
        for t in rng.permutation(p.T-1):
            trial = s.v.copy()
            method = k%6
            if method in [0, 3, 4]:
                trial[t] = reround(p, t, guide[t], base[t], upper[t], rng, s.v[t])
                b = qc.State(p, trial)
                assert p.feasible(trial)
            else:
                b = s
            proposed = 0
            for j in range(6 if method in [0, 3, 4] else 3):
                use_strict = j < 2 and method in [0, 3, 4]
                noise = rng.choice([55, 160, 360, 800, 2000])
                width = int(rng.choice([12, 13, 14, 15]))
                alpha = rng.choice([0, 0, .2, .6, 1.4]) if use_strict else rng.choice([0, 0, .13, .35])
                trades = pack(b, t, base[t], upper[t], rng, width, noise, use_strict, alpha)
                if not use_strict:
                    trades = augment(b, t, trades, rng)
                q, executed = quantum(b, trades, 600, rng, scheme)
                n += executed
                proposed += int(b is not q)
                b = q
            if b.obj < s.obj and proposed:
                log.append((k, int(t), method, b.obj, s.obj-b.obj, proposed))
                wins += 1
                s = b
        if wins or k%5 == 0:
            print(k, s.obj, wins, n, round(time.time()-tm, 1), flush=True)
        if (wins and k%4 == 0) or s.obj <= TARGET:
            np.save(out, s.v)
        if s.obj <= TARGET:
            break
    np.save(out, s.v)
    Path(out+'.json').write_text(json.dumps(dict(final=s.obj, counts=n, steps=log), default=int))
    print('final', s.obj, 'shots per sampler', 600, 'circuits', n, flush=True)


def margin(state, t, d):
    p = state.p
    v = state.v[t]
    w = v+d
    cost = d*(state.G[t]+p.lin[t]+10)+d*d*p.R[t].diagonal()
    cost += p.sc[t]*(np.maximum(-w, 0)-np.maximum(-v, 0))
    cost += p.absc[t]*(abs(w)-abs(v))
    if t:
        cost += p.w[t]*(abs(w-state.v[t-1])-abs(v-state.v[t-1]))
    if t+1 < p.T:
        cost += p.w[t+1]*(abs(w-state.v[t+1])-abs(v-state.v[t+1]))
    return cost


def pack_wide(s, t, rng, width, noise, tilt):
    p = s.p
    v = s.v[t]
    risk = p.RS[t]
    allowed = np.ones(p.n, bool)
    candidates = []
    for f in (1, -1):
        for unit in (1, 2, 3):
            d = f*unit
            donors = np.flatnonzero(f*v >= unit)
            rec = np.flatnonzero((f*v >= 0) & (f*v+unit <= p.ub))
            if len(donors) == 0 or len(rec) == 0:
                continue
            cov = risk[np.ix_(donors, rec)]
            a = margin(s, t, -d)[donors, None]+margin(s, t, d)[rec]-unit**2*cov
            z = (a+d*(tilt[rec]-tilt[donors, None])).astype(float)
            z += rng.gumbel(0, noise*(1.25 if unit == 1 else 1), size=z.shape)
            z[donors[:, None] == rec] = np.inf
            idxs = np.argpartition(z.ravel(), min(1099, z.size-1))[:1100]
            i, j = np.unravel_index(idxs, z.shape)
            candidates.append(np.c_[z[i, j], donors[i], rec[j], np.full(len(i), d)])
    if not candidates:
        return []
    c = np.concatenate(candidates)
    c = c[np.argsort(c[:, 0])]
    groups = []
    for _, i, j, d in c:
        i, j, d = int(i), int(j), int(d)
        if (allowed[i] and allowed[j]):
            groups.append([(t, i, -d), (t, j, d)])
            allowed[i] = allowed[j] = False
            if len(groups) == width:
                break
    return augment(s, t, groups, rng)


def coefficients(s, t, groups):
    """Build the exact QUBO of disjoint transfers, including multi-unit transfers."""
    q = len(groups)
    p = s.p
    D = np.zeros((q, p.n), np.int64)
    aa = np.zeros(q, np.int64)
    tables = {d:margin(s, t, d) for d in {d for group in groups for _, i, d in group}}
    for k, group in enumerate(groups):
        for _, i, d in group:
            D[k, i] = d
            aa[k] += tables[d][i]
        if len(group) == 2:
            (_, i, d), (_, j, e) = group
            aa[k] += d*e*p.RS[t, i, j]
    return aa, np.triu((D @ p.RS[t]) @ D.T, k=1), D


def choose_step(s, t, groups, rng, shots=448, perturb=0):
    if not groups:
        return s, False
    a, b, D = coefficients(s, t, groups)
    energies = all_energies(a, b)
    if energies.min() >= 0 and not perturb:
        return s, False
    normalized = (energies-energies.mean())/max(1., energies.std())
    angles = PERTURB_ANGLES if perturb else SEARCH_ANGLES
    prob = evolve(normalized, len(a), *angles)
    candidates = rng.choice(len(energies), size=shots, p=prob / prob.sum())
    imin = int(candidates[np.argmin(energies[candidates])])
    if perturb and energies[imin] < perturb:
        near = candidates[(energies[candidates] > perturb*.3) & (energies[candidates] < perturb*2)]
        if len(near):
            imin = int(rng.choice(near))
    if energies[imin] >= 0 and not perturb:
        return s, True
    v = s.v.copy()
    bit = np.array([(imin>>k) & 1 for k in range(len(groups))], np.int64)
    v[t] += bit @ D
    out = qc.State(s.p, v)
    assert out.obj == s.obj+energies[imin] and s.p.feasible(v)
    return out, True


def transfer_search(v, out, steps, seed):
    p = qc.ps.Problem('po_a400_t15_orig', '1e-3')
    rng = np.random.default_rng(seed)
    elite = qc.State(p, v)
    t0 = time.time()
    total = 0
    log = []
    init = elite.obj
    for k in range(steps):
        wins = 0
        for t in rng.permutation(p.T-1):
            s = elite
            for j in range(4):
                pert = 0 if (k%5 < 2 or j) else rng.choice([750, 1800, 3000, 7000])
                noise = rng.choice([70, 220, 540, 1100, 2300]) if not pert else np.sqrt(pert)*14
                vector = rng.normal(size=p.n)
                tilt = vector @ p.RS[t]
                tilt *= rng.choice([0, 0, 200, 550, 1000, 1750])/tilt.std()
                width = int(rng.choice([12, 13, 14, 15, 16]))
                groups = pack_wide(s, t, rng, width, noise, tilt)
                s, ex = choose_step(s, t, groups, rng, perturb=pert)
                total += ex
                if not ex and j >= 2:
                    break
            if s.obj < elite.obj:
                log.append((k, int(t), s.obj, s.obj-elite.obj, total))
                elite = s
                wins += 1
        if wins or k%5 == 0:
            print(k, elite.obj, wins, total, round(time.time()-t0, 1), flush=True)
        if wins or k%10 == 0:
            np.save(out, elite.v)
        if elite.obj <= TARGET:
            break
    np.save(out, elite.v)
    Path(out+'.json').write_text(json.dumps(dict(start=init, final=elite.obj, circuits=total, steps=log), default=int))
    print('final', elite.obj, total, flush=True)


def merge(p, current, alternative, rng):
    """Sample choices between two feasible daily schedules on a chain QUBO."""
    E = p.objective(current)
    days = np.flatnonzero(np.any(current != alternative, axis=1))
    if not len(days):
        return current.copy()
    a = np.empty(len(days), np.int64)
    b = np.zeros((len(days), len(days)), np.int64)
    for k, t in enumerate(days):
        probe = current.copy()
        probe[t] = alternative[t]
        a[k] = p.objective(probe)-E
        if k+1 < len(days) and days[k+1] == t+1:
            weight = p.w[t + 1]
            b[k, k + 1] = weight @ (
                abs(alternative[t + 1] - alternative[t])
                - abs(alternative[t + 1] - current[t])
                - abs(current[t + 1] - alternative[t])
                + abs(current[t + 1] - current[t])
            )
    energies = all_energies(a, b)
    if energies.min() >= 0:
        return current.copy()
    shifted = (energies-energies.mean())/energies.std()
    prob = evolve(shifted, len(days), *SEARCH_ANGLES)
    samples = rng.choice(len(energies), size=960, p=prob/prob.sum())
    k = int(samples[np.argmin(energies[samples])])
    out = current.copy()
    used = days[np.array([bool((k>>i) & 1) for i in range(len(days))])]
    out[used] = alternative[used]
    assert p.objective(out) == E+energies[k] and p.feasible(out)
    print('mosaic', E, '->', p.objective(out), 'oracle', E+energies.min(), 'days', used, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/paired_trajectory_a400.sol"))
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--resume", type=Path, help="Run only the transfer search from a canonical .sol")
    modes.add_argument("--replay", type=Path, help="Recombine the saved parents of the reported result")
    ap.add_argument("--seed", type=int, default=215, help="Seed of the final transfer search")
    ap.add_argument("--rounds", type=int, default=110, help="Rounds per randomized restart search")
    ap.add_argument("--transfer-rounds", type=int, default=85)
    args = ap.parse_args()
    if args.rounds < 0 or args.transfer_rounds < 0:
        ap.error("Round counts must be nonnegative")

    p = qc.ps.Problem("po_a400_t15_orig", "1e-3")
    with TemporaryDirectory(prefix="paired-trajectory-") as directory:
        work = Path(directory)
        final = work / "final.npy"
        if args.replay:
            with np.load(args.replay, allow_pickle=False) as replay:
                parents = replay["parents"].astype(np.int64)
                rng = np.random.default_rng(int(replay["seed"]))
            if parents.ndim != 3 or len(parents) < 2 or parents.shape[1:] != (p.T, p.n):
                ap.error("Unexpected replay parent dimensions")
            if any(not p.feasible(parent) for parent in parents):
                ap.error("A replay parent is infeasible")
            current = parents[0]
            for alternate in parents[1:]:
                candidate = merge(p, current, alternate, rng)
                if p.objective(candidate) < p.objective(current):
                    current = candidate
        elif args.resume:
            current = qc.load_v(p, args.resume)
            if not p.feasible(current):
                ap.error("The resume portfolio is infeasible")
        else:
            warm_file = qc.ROOT / "deliverables" / "portfolio_quantum" / f"{p.stem}.sol"
            guide = trajectory_guide(p, qc.load_v(p, warm_file))
            pool = []
            for index, (round_seed, restart_seed, transfer_seed) in enumerate(
                ((1, 21, 46), (118, 119, 830))
            ):
                rounded = work / f"rounded_{index}.npy"
                restarted = work / f"restarted_{index}.npy"
                transferred = work / f"transferred_{index}.npy"
                rounding_search(None, guide, rounded, "qaoa", seed=round_seed)
                restart_search(np.load(rounded), guide, str(restarted), args.rounds, seed=restart_seed)
                transfer_search(np.load(restarted), str(transferred), args.transfer_rounds, transfer_seed)
                pool.extend(np.load(path) for path in (restarted, transferred))
            pool.sort(key=p.objective)
            current = pool[0]
            rng = np.random.default_rng(args.seed)
            for alternate in pool[1:]:
                candidate = merge(p, current, alternate, rng)
                if p.objective(candidate) < p.objective(current):
                    current = candidate
        if not args.replay and p.objective(current) > TARGET:
            transfer_search(current, str(final), args.transfer_rounds, args.seed)
            current = np.load(final)

    model, score = qc.check_with_referee(p, current)
    if not score.feasible or score.objective != p.objective(current):
        raise RuntimeError(f"Referee disagrees with search: {score}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.write_solution(
        args.output, p.positions(current),
        comment="Paired trajectory QAOA, local statevector simulation.",
    )
    print(f"Referee score {score}. Saved {args.output}")


if __name__ == "__main__":
    main()
