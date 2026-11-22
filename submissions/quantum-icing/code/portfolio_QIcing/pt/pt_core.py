"""Parallel tempering (replica exchange) on the reduced state v[t, i] = L - S for QOBLIB 06.

Every move is a list of (t, i, d) integer changes. Deltas are exact int64: each change updates
g[t] = R_t v_t incrementally (O(n) per change), so the energy of every replica is always the
exact referee objective (minus nothing: we start from fm.objective(v0) and add deltas).

Move kinds (one per step, sampled by `mix`):
  0 single     : v[t, i] += d on a block of consecutive days [ta, tb)
  1 transfer   : one unit of a held (t, i) moves to a correlated partner j, block of days
  2 hedge      : +d on a and -sign(corr_ab) d on a correlated b, block of days
  3 time-asset : one asset: copy a neighbour day's value (extend / shrink a run), or clear a run
  4 time-day   : whole portfolio: copy day t +- 1 into day t (extend / shrink a run in time)
  5 shift      : shift one asset's (or the whole portfolio's) holding window by one day
States stay feasible: a move that breaks |v| <= ub, sum|v| <= B or netmin <= sum v <= C on any
touched day is rejected (reverted). Day T-1 is never touched (holding there is pure cost; the
exit transition into it is free).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

MAXCH = 4096


@njit(cache=True, inline="always")
def _delta1(v, g, R, lin, absw, negw, tw, t, i, d, T):
    x0 = v[t, i]
    x1 = x0 + d
    dl = 2 * d * g[t, i] + d * d * R[t, i, i] + d * lin[t, i]
    dl += absw[t, i] * (abs(x1) - abs(x0))
    dl += negw[t, i] * (max(-x1, 0) - max(-x0, 0))
    if t >= 1:
        p = v[t - 1, i]
        dl += tw[t, i] * (abs(x1 - p) - abs(x0 - p))
    if t + 1 < T:
        q = v[t + 1, i]
        dl += tw[t + 1, i] * (abs(q - x1) - abs(q - x0))
    return dl


@njit(cache=True, inline="always")
def _apply1(v, g, tot, net, R, lin, absw, negw, tw, t, i, d, T, n):
    dl = _delta1(v, g, R, lin, absw, negw, tw, t, i, d, T)
    tot[t] += abs(v[t, i] + d) - abs(v[t, i])
    net[t] += d
    v[t, i] += d
    for a in range(n):
        g[t, a] += R[t, a, i] * d
    return dl


@njit(cache=True)
def init_aux(v, R):
    T, n = v.shape
    g = np.zeros((T, n), dtype=np.int64)
    tot = np.zeros(T, dtype=np.int64)
    net = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for i in range(n):
            tot[t] += abs(v[t, i])
            net[t] += v[t, i]
            for j in range(n):
                g[t, i] += R[t, i, j] * v[t, j]
    return g, tot, net


@njit(cache=True)
def _block(T, block_prob):
    ta = np.random.randint(T - 1)
    tb = ta + 1
    if np.random.random() < block_prob:
        tb = ta + 1 + np.random.randint(T - 1 - ta)
    return ta, tb


@njit(cache=True)
def _pick_held(v, t, n):
    for _ in range(4 * n):
        i = np.random.randint(n)
        if v[t, i] != 0:
            return i
    return -1


@njit(cache=True)
def _propose(v, partners, psign, mix, block_prob, ub, ct, ci, cd):
    """Fill change buffers; return count (0 = no valid proposal), move kind."""
    T, n = v.shape
    P = partners.shape[1]
    u = np.random.random()
    kind = 0
    acc = mix[0]
    while u > acc and kind < len(mix) - 1:
        kind += 1
        acc += mix[kind]
    m = 0
    if kind == 0:  # single unit, block of days
        ta, tb = _block(T, block_prob)
        i = np.random.randint(n)
        d = 1 if np.random.random() < 0.5 else -1
        for t in range(ta, tb):
            ct[m], ci[m], cd[m] = t, i, d
            m += 1
    elif kind == 1:  # transfer one unit from held (t, i) to partner j (same direction)
        ta, tb = _block(T, block_prob)
        i = _pick_held(v, ta, n)
        if i < 0:
            return 0, kind
        s = 1 if v[ta, i] > 0 else -1
        k = np.random.randint(P)
        j = partners[i, k]
        sj = s * psign[i, k]  # same factor exposure: positive corr -> same sign
        for t in range(ta, tb):
            ct[m], ci[m], cd[m] = t, i, -s
            m += 1
            ct[m], ci[m], cd[m] = t, j, sj
            m += 1
    elif kind == 2:  # hedge pair: +d on a, -sign(corr) d on b
        ta, tb = _block(T, block_prob)
        a = np.random.randint(n)
        k = np.random.randint(P)
        b = partners[a, k]
        d = 1 if np.random.random() < 0.5 else -1
        for t in range(ta, tb):
            ct[m], ci[m], cd[m] = t, a, d
            m += 1
            ct[m], ci[m], cd[m] = t, b, -psign[a, k] * d
            m += 1
    elif kind == 3:  # one asset: extend / shrink run (copy neighbour day) or clear a run
        i = np.random.randint(n)
        if np.random.random() < 0.25:
            ta, tb = _block(T, block_prob)
            for t in range(ta, tb):
                if v[t, i] != 0:
                    ct[m], ci[m], cd[m] = t, i, -v[t, i]
                    m += 1
        else:
            t = np.random.randint(T - 1)
            src = t - 1 if np.random.random() < 0.5 else t + 1
            if src < 0:
                src = 1
            d = v[src, i] - v[t, i]
            if d != 0:
                ct[m], ci[m], cd[m] = t, i, d
                m += 1
    elif kind == 4:  # whole portfolio: day t := day t +- 1
        t = np.random.randint(T - 1)
        src = t - 1 if np.random.random() < 0.5 else t + 1
        if src < 0:
            src = 1
        for i in range(n):
            d = v[src, i] - v[t, i]
            if d != 0:
                ct[m], ci[m], cd[m] = t, i, d
                m += 1
    else:  # shift a holding window by one day (one asset or whole portfolio)
        ta = np.random.randint(T - 1)
        tb = ta + 2 + np.random.randint(T - ta)
        if tb > T - 1:
            tb = T - 1
        if tb - ta < 2:
            return 0, kind
        right = np.random.random() < 0.5
        whole = np.random.random() < 0.4
        i0 = 0 if whole else np.random.randint(n)
        i1 = n if whole else i0 + 1
        for i in range(i0, i1):
            for t in range(ta, tb):
                if right:
                    nv = v[t - 1, i] if t > ta else v[ta, i] * 0
                else:
                    nv = v[t + 1, i] if t + 1 < tb else 0
                d = nv - v[t, i]
                if d != 0 and m < MAXCH:
                    ct[m], ci[m], cd[m] = t, i, d
                    m += 1
    # bound check (cells are unique within a move)
    for k in range(m):
        x = v[ct[k], ci[k]] + cd[k]
        if x > ub or x < -ub:
            return 0, kind
    return m, kind


@njit(cache=True)
def replica_steps(v, g, tot, net, E, temp, n_steps, R, lin, absw, negw, tw, partners, psign,
                  mix, block_prob, B, C, netmin, ub, best_v, best_E, stats):
    """Metropolis steps on one replica. stats[kind, 0/1] = proposed / accepted."""
    T, n = v.shape
    ct = np.empty(MAXCH, dtype=np.int64)
    ci = np.empty(MAXCH, dtype=np.int64)
    cd = np.empty(MAXCH, dtype=np.int64)
    for _ in range(n_steps):
        m, kind = _propose(v, partners, psign, mix, block_prob, ub, ct, ci, cd)
        if m == 0:
            continue
        stats[kind, 0] += 1
        delta = np.int64(0)
        tlo, thi = T, -1
        for k in range(m):
            delta += _apply1(v, g, tot, net, R, lin, absw, negw, tw, ct[k], ci[k], cd[k], T, n)
            if ct[k] < tlo:
                tlo = ct[k]
            if ct[k] > thi:
                thi = ct[k]
        ok = True
        for t in range(tlo, thi + 1):
            if tot[t] > B or net[t] > C or net[t] < netmin:
                ok = False
                break
        if ok and (delta <= 0 or np.random.random() < np.exp(-delta / temp)):
            E += delta
            stats[kind, 1] += 1
            if E < best_E:
                best_E = E
                best_v[:, :] = v
        else:
            for k in range(m - 1, -1, -1):
                _apply1(v, g, tot, net, R, lin, absw, negw, tw, ct[k], ci[k], -cd[k], T, n)
    return E, best_E


@njit(cache=True, parallel=True)
def sweep_all(V, G, TOT, NET, E, temps, lev, n_steps, RL, lin, absw, negw, tw, partners,
              psign, mix, block_prob, B, C, netmin, ub, BV, BE, STATS, seed):
    K = V.shape[0]
    for r in prange(K):
        np.random.seed(seed + 7919 * r)
        E[r], BE[r] = replica_steps(V[r], G[r], TOT[r], NET[r], E[r], temps[lev[r]], n_steps,
                                    RL[r], lin, absw, negw, tw, partners, psign, mix, block_prob,
                                    B, C, netmin, ub, BV[r], BE[r], STATS[r])


def partner_table(fm, n_partners=8):
    c = fm.corr.copy()
    np.fill_diagonal(c, 0)
    partners = np.argsort(-np.abs(c), axis=1)[:, :n_partners].astype(np.int64)
    psign = np.sign(np.take_along_axis(c, partners, 1)).astype(np.int64)
    psign[psign == 0] = 1
    return partners, psign


@njit(cache=True)
def risk_of(v, R):
    T, n = v.shape
    s = np.int64(0)
    for t in range(T):
        for i in range(n):
            if v[t, i] != 0:
                acc = np.int64(0)
                for j in range(n):
                    acc += R[t, i, j] * v[t, j]
                s += v[t, i] * acc
    return s


@njit(cache=True)
def swap_pass(V, G, TOT, NET, E, RL, temps, parity, u, acc_out):
    """Exchange states between neighbour levels (k, k+1), k = parity, parity+2, ...
    Level k has Hamiltonian E_k(x) = base(x) + x R_k x at temperature temps[k]."""
    K = V.shape[0]
    for k in range(parity, K - 1, 2):
        l = k + 1
        rk_k, rk_l = risk_of(V[k], RL[k]), risk_of(V[l], RL[l])
        base_k, base_l = E[k] - rk_k, E[l] - rk_l
        rl_k, rk_lstate = risk_of(V[k], RL[l]), risk_of(V[l], RL[k])  # x_k under H_l, x_l under H_k
        new_k = base_l + rk_lstate  # level k receives x_l
        new_l = base_k + rl_k
        x = (E[k] - new_k) / temps[k] + (E[l] - new_l) / temps[l]
        acc_out[k] = 0
        if x >= 0 or u[k] < np.exp(x):
            acc_out[k] = 1
            tmp = V[k].copy()
            V[k][:, :] = V[l]
            V[l][:, :] = tmp
            E[k], E[l] = new_k, new_l
            G[k], TOT[k], NET[k] = init_aux(V[k], RL[k])
            G[l], TOT[l], NET[l] = init_aux(V[l], RL[l])


def parallel_tempering(fm, starts, n_replicas=24, t_min=20.0, t_max=20000.0, s_min=1.0,
                       time_limit=120.0, steps_per_round=400,
                       mix=(0.15, 0.2, 0.3, 0.15, 0.1, 0.1), block_prob=0.6, adapt_frac=0.4,
                       target_acc=0.3, reseed_every=0, seed=0, log=None, n_partners=8):
    """Replica exchange. Level k sits at ladder position x_k in [0, 1] (x_0 = 0, x_{K-1} = 1):
    temperature T_k = t_min (t_max / t_min)^x_k and risk weight s_k = s_min^x_k (s_min = 1: plain
    temperature PT; s_min < 1: Hamiltonian / lambda-tempering, s = 0.1 ~ the lambda = 1e-3
    landscape). Endpoints are fixed; interior spacing adapts to target swap acceptance during the
    first adapt_frac of the run. Only s = 1 levels (exact objective) feed the best-so-far."""
    import time

    rng = np.random.default_rng(seed)
    T, n = fm.T, fm.n
    K = n_replicas
    xs = np.linspace(0, 1, K)

    def ladder(xs):
        temps = t_min * (t_max / t_min) ** xs
        ss = s_min ** xs
        RL = np.stack([fm.R if s == 1.0 else np.round(s * fm.R).astype(np.int64) for s in ss])
        return temps, ss, RL

    temps, ss, RL = ladder(xs)
    V = np.stack([starts[r % len(starts)].astype(np.int64).copy() for r in range(K)])
    G = np.empty((K, T, n), dtype=np.int64)
    TOT = np.empty((K, T), dtype=np.int64)
    NET = np.empty((K, T), dtype=np.int64)

    def rebuild(k):
        G[k], TOT[k], NET[k] = init_aux(V[k], RL[k])
        return fm.objective(V[k]) - risk_of(V[k], fm.R) + risk_of(V[k], RL[k])

    for k in range(K):
        assert fm.feasible(V[k])
    E = np.array([rebuild(k) for k in range(K)], dtype=np.int64)
    NEG = np.int64(-(1 << 62))
    exact = ss == 1.0
    BV, BE = V.copy(), np.where(exact, E, NEG).astype(np.int64)
    best_v, best_E = V[int(np.argmin(np.where(exact, E, -NEG)))].copy(), int(np.min(np.where(exact, E, -NEG)))
    for k in range(K):  # also consider the start itself
        e = fm.objective(V[k])
        if e < best_E:
            best_E, best_v = e, V[k].copy()
    STATS = np.zeros((K, 6, 2), dtype=np.int64)
    sw_try = np.zeros(K - 1)
    sw_acc = np.zeros(K - 1)
    win_try = np.zeros(K - 1)
    win_acc = np.zeros(K - 1)
    acc_out = np.zeros(K - 1, dtype=np.int64)
    partners, psign = partner_table(fm, n_partners)
    mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
    lev = np.arange(K)
    sweep_all(V, G, TOT, NET, E, temps, lev, 0, RL, fm.lin, fm.absw, fm.negw, fm.tw,
              partners, psign, mix, block_prob, fm.B, fm.C, fm.netmin, fm.ub, BV, BE, STATS, 0)
    swap_pass(V, G, TOT, NET, E, RL, temps, 0, np.ones(K), acc_out * 0)  # compile only (u=1 > all)
    t0 = time.time()  # after the compile-triggering warm-up calls
    rnd = 0
    last_log = t0
    trace = []
    adapting = adapt_frac > 0
    while time.time() - t0 < time_limit:
        BE[~exact] = NEG  # non-exact levels never record a best
        sweep_all(V, G, TOT, NET, E, temps, lev, steps_per_round, RL, fm.lin, fm.absw,
                  fm.negw, fm.tw, partners, psign, mix, block_prob, fm.B, fm.C, fm.netmin, fm.ub,
                  BV, BE, STATS, int(seed * 1_000_003 + rnd * 101))
        rnd += 1
        par = rnd % 2
        swap_pass(V, G, TOT, NET, E, RL, temps, par, rng.random(K), acc_out)
        idx = np.arange(par, K - 1, 2)
        sw_try[idx] += 1
        win_try[idx] += 1
        sw_acc[idx] += acc_out[idx]
        win_acc[idx] += acc_out[idx]
        for k in np.flatnonzero(exact):
            if E[k] < BE[k]:
                BE[k], BV[k] = E[k], V[k].copy()
        k = int(np.argmin(np.where(exact, BE, -NEG)))
        if exact[k] and BE[k] < best_E:
            best_E, best_v = int(BE[k]), BV[k].copy()
        elapsed = time.time() - t0
        if adapting and rnd % 20 == 0:
            if elapsed < 0.1 * time_limit:  # burn-in: identical starts swap trivially
                pass
            elif elapsed < adapt_frac * time_limit:  # move interior points; endpoints fixed
                acc = (win_acc + 0.5) / (win_try + 1.0)
                gaps = np.diff(xs) * np.exp(0.7 * (acc - target_acc))
                xs = np.concatenate([[0.0], np.cumsum(gaps) / gaps.sum()])
                old_ss = ss
                temps, ss, RL = ladder(xs)
                exact = ss == 1.0
                if s_min != 1.0:
                    for kk in range(K):
                        if ss[kk] != old_ss[kk]:
                            E[kk] = rebuild(kk)
                win_try[:] = 0
                win_acc[:] = 0
                sw_try[:] = 0  # report post-adaptation rates only
                sw_acc[:] = 0
            else:
                adapting = False
        if reseed_every and rnd % reseed_every == 0 and fm.objective(V[0]) > best_E:
            V[0] = best_v  # coldest replica <- global best
            E[0] = rebuild(0)
        if time.time() - last_log > 15:
            last_log = time.time()
            trace.append((round(elapsed, 1), best_E))
            if log:
                r = sw_acc / np.maximum(sw_try, 1)
                log(f"  [{elapsed:5.0f}s] rnd {rnd} best {best_E} cold E {E[0]} hot E {E[-1]} "
                    f"T=[{temps[0]:.0f}..{temps[-1]:.0f}] s=[{ss[0]:.2f}..{ss[-1]:.2f}] "
                    f"swap {r.min():.2f}-{r.max():.2f}")
    assert fm.objective(best_v) == best_E, (fm.objective(best_v), best_E)
    for k in range(K):
        assert fm.objective(V[k]) - risk_of(V[k], fm.R) + risk_of(V[k], RL[k]) == E[k]
    return dict(v=best_v, E=int(best_E), temps=temps, ss=ss, swap_rate=sw_acc / np.maximum(sw_try, 1),
                move_stats=STATS.sum(0), rounds=rnd, trace=trace,
                steps=int(rnd * steps_per_round * K))
