"""Do bigger blocks help? Classical big-block solver and tests (step 1 before cloud QAOA).

A block (one day or a run of days, assets ``G``, the same net holding ``v`` on each day,
the rest of the portfolio fixed) has the form

    cost(v) - cost(0) = v A v + sum_i f_i(v_i)
    subject to  sum_i |v_i| <= U  and  lo <= sum_i v_i <= hi

(:func:`block_form`; checked against ``BlockEvaluator.costs`` in :func:`check`). That is
the problem a QAOA block solver gets, without the QUBO encoding or the price ``mu``.
:func:`solve_sa` is a vectorised integer SA on it (many chains, one- and two-asset
moves that keep the rules), so blocks of any size can be solved, up to a whole day
(50 assets = 150 qubits in the signed encoding). :func:`solve_exact` enumerates
(``7**g``, up to 7 assets) as a reference.

Tests, all from the verified hand-in solutions (local optima for groups of 3):

    python big_blocks.py check                      # form == costs, SA vs exact
    python big_blocks.py oracle                     # gain of pasting in a best-known day
    python big_blocks.py sweep --sizes 3 6 8 12 50  # what blocks of each size still find
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import qoblib_tools as qt
from asset_groups import BlockEvaluator, _net_configs

VALS = np.arange(-3, 4)
A050 = [("po_a050_t10_orig", 20), ("po_a050_t10_s00", 20), ("po_a050_t10_s01", 20), ("po_a050_t10_s02", 20)]


# ── block form ───────────────────────────────────────────────────────────


def block_form(ev: BlockEvaluator, L, S, days, G: np.ndarray) -> dict:
    """``A (g, g)``, ``f (g, 7)`` (column ``v + 3``), unit cap ``U``, net range ``lo..hi``."""
    m = ev.model
    days = [days] if np.isscalar(days) else list(days)
    G = np.asarray(G)
    g = len(G)
    A = sum(ev.Rs[t][np.ix_(G, G)] for t in days)
    c0 = ev.costs(L, S, days, G, np.zeros((1, g), np.int64))[0][0]
    V = np.zeros((7 * g, g), np.int64)
    for i in range(g):
        V[7 * i:7 * i + 7, i] = VALS
    c = ev.costs(L, S, days, G, V)[0].reshape(g, 7)
    f = c - c0 - np.diag(A)[:, None] * VALS[None] ** 2
    others = np.ones(len(m.symbols), bool)
    others[G] = False
    ou = np.array([(L[t, others] + S[t, others]).sum() for t in days])
    on = np.array([(L[t, others] - S[t, others]).sum() for t in days])
    return dict(A=A, f=f, U=int(m.budget - ou.max()), lo=int((m.capital - ev.max_cash - on).max()),
                hi=int((m.capital - on).min()), c0=float(c0))


def form_cost(F: dict, V: np.ndarray):
    """(cost - cost(0), feasible) for ``V (K, g)``."""
    q = np.einsum("kg,gh,kh->k", V, F["A"], V)
    sep = F["f"][np.arange(V.shape[1])[None], V + 3].sum(axis=1)
    units, net = np.abs(V).sum(axis=1), V.sum(axis=1)
    return q + sep, (units <= F["U"]) & (net >= F["lo"]) & (net <= F["hi"])


# ── solvers ──────────────────────────────────────────────────────────────


def solve_exact(F: dict, chunk: int = 200_000) -> np.ndarray:
    g = F["A"].shape[0]
    Vall = _net_configs(g, 3)
    best, bv = np.inf, None
    for s0 in range(0, len(Vall), chunk):
        V = Vall[s0:s0 + chunk]
        c, ok = form_cost(F, V)
        c = np.where(ok, c, np.inf)
        k = int(np.argmin(c))
        if c[k] < best:
            best, bv = c[k], V[k].copy()
    return bv


def solve_sa(F: dict, v0: np.ndarray, rng, chains: int = 64, sweeps: int = 300,
             t_hi: float | None = None, t_lo: float | None = None, t_scale: float = 1.0) -> np.ndarray:
    """Integer SA from the feasible ``v0``; every move keeps the rules, so chains stay feasible.

    Moves: set one asset to a random other value, or shift two assets by ±1 each (needed
    when the unit cap binds). Temperature falls geometrically from ``t_hi`` (default:
    median one-asset cost spread) to ``t_lo = t_hi / 1000``.
    """
    A, f, U, lo, hi = F["A"], F["f"], F["U"], F["lo"], F["hi"]
    g = A.shape[0]
    K = chains
    V = np.repeat(v0[None].astype(np.int64), K, axis=0)
    AV = V @ A
    cost = form_cost(F, V)[0]
    best_c, best_v = cost.min(), V[int(np.argmin(cost))].copy()
    if t_hi is None:
        spread = np.ptp(f + np.diag(A)[:, None] * VALS ** 2, axis=1)
        t_hi = (float(np.median(spread)) or 1.0) * t_scale
    t_lo = t_hi / 1000 if t_lo is None else t_lo
    steps = sweeps * g
    temps = t_hi * (t_lo / t_hi) ** (np.arange(steps) / max(1, steps - 1))
    ar = np.arange(K)
    Adiag = np.diag(A)
    for s in range(steps):
        T = temps[s]
        i = rng.integers(0, g, K)
        vi = V[ar, i]
        if g > 1 and rng.random() < 0.5:  # two-asset move
            j = (i + rng.integers(1, g, K)) % g
            di, dj = rng.choice([-1, 1], K), rng.choice([-1, 1], K)
            vj = V[ar, j]
            ni, nj = vi + di, vj + dj
            okr = (np.abs(ni) <= 3) & (np.abs(nj) <= 3)
            ni, nj = np.clip(ni, -3, 3), np.clip(nj, -3, 3)
            di, dj = ni - vi, nj - vj
            d = (2 * di * AV[ar, i] + Adiag[i] * di ** 2 + 2 * dj * AV[ar, j] + Adiag[j] * dj ** 2
                 + 2 * di * dj * A[i, j] + f[i, ni + 3] - f[i, vi + 3] + f[j, nj + 3] - f[j, vj + 3])
            units = np.abs(V).sum(1) - np.abs(vi) - np.abs(vj) + np.abs(ni) + np.abs(nj)
            net = V.sum(1) + di + dj
        else:  # one asset to a random other value
            ni = (vi + 3 + rng.integers(1, 7, K)) % 7 - 3
            di = ni - vi
            j, dj, okr = None, 0, np.ones(K, bool)
            d = 2 * di * AV[ar, i] + Adiag[i] * di ** 2 + f[i, ni + 3] - f[i, vi + 3]
            units = np.abs(V).sum(1) - np.abs(vi) + np.abs(ni)
            net = V.sum(1) + di
        ok = okr & (units <= U) & (net >= lo) & (net <= hi)
        acc = ok & ((d <= 0) | (rng.random(K) < np.exp(-np.maximum(d, 0) / T)))
        if not acc.any():
            continue
        a = ar[acc]
        V[a, i[acc]] = ni[acc]
        AV[a] += di[acc, None] * A[i[acc]]
        if j is not None:
            V[a, j[acc]] = nj[acc]
            AV[a] += dj[acc, None] * A[j[acc]]
        cost[a] += d[acc]
        k = int(np.argmin(cost))
        if cost[k] < best_c - 1e-9:
            best_c, best_v = cost[k], V[k].copy()
    return best_v


class SABlockSolver:
    """``asset_groups.BlockSolver`` using :func:`solve_sa` on :func:`block_form` (any block size).

    Meant for whole-day blocks (all assets); each call uses fresh randomness, so solving
    the same block again can find a different (better) holding.
    """

    def __init__(self, chains: int = 64, sweeps: int = 300, seed: int = 0, t_scale: float = 1.0):
        self.chains, self.sweeps, self.t_scale = chains, sweeps, t_scale
        self.rng = np.random.default_rng(seed)
        self.calls, self.seconds = 0, 0.0

    def __call__(self, ev: BlockEvaluator, L, S, days, G):
        from asset_groups import BlockResult

        t0 = time.time()
        days = [days] if np.isscalar(days) else list(days)
        G = np.asarray(G)
        F = block_form(ev, L, S, days, G)
        v0 = (L[days[0], G] - S[days[0], G]).astype(np.int64)
        if not form_cost(F, v0[None])[1][0]:  # holding varies over the window
            v0 = np.zeros(len(G), np.int64)
            if not form_cost(F, v0[None])[1][0]:
                return BlockResult((L[days[0], G] - S[days[0], G]).astype(np.int64))
        V = solve_sa(F, v0, self.rng, chains=self.chains, sweeps=self.sweeps, t_scale=self.t_scale)
        self.calls += 1
        self.seconds += time.time() - t0
        return BlockResult(V)

    def summary(self) -> dict:
        return dict(chains=self.chains, sweeps=self.sweeps, t_scale=self.t_scale, calls=self.calls, seconds=round(self.seconds, 1))


# ── data ─────────────────────────────────────────────────────────────────


def load(inst: str, budget: int, lam: str, start: str = "handin"):
    """model, (L, S) of the start ('handin' file or 'empty'), (L, S) of the best known, bkv."""
    model = qt.PortfolioModel(qt.load_portfolio_instance(inst), budget=budget, lam=lam)
    hb, pb = qt.read_portfolio_solution(qt.fetch_portfolio_reference(inst, budget=budget, lam_tag=lam))
    Lb, Sb = model.to_arrays(pb)
    if start == "empty":
        T, n = model.n_periods, len(model.symbols)
        L, S = np.zeros((T, n), np.int64), np.zeros((T, n), np.int64)
    else:
        _, p = qt.read_portfolio_solution(f"handin/{inst}_b{budget:03d}_l{lam}.sol")
        L, S = model.to_arrays(p)
    return model, (L.astype(np.int64), S.astype(np.int64)), (Lb.astype(np.int64), Sb.astype(np.int64)), int(hb["objective"])


# ── tests ────────────────────────────────────────────────────────────────


def check() -> None:
    rng = np.random.default_rng(0)
    model, (L, S), _, _ = load("po_a050_t10_s00", 20, "1e-3")
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    # 1. form == BlockEvaluator.costs (differences and feasibility), days and windows
    for _ in range(30):
        w = int(rng.integers(1, 4))
        t0 = int(rng.integers(0, T - w + 1))
        days = list(range(t0, t0 + w))
        G = np.sort(rng.choice(n, int(rng.integers(2, 12)), replace=False))
        F = block_form(ev, L, S, days, G)
        V = rng.integers(-3, 4, (200, len(G))) * (rng.random((200, len(G))) < 0.4)
        c1, ok1 = form_cost(F, V)
        c2, ok2 = ev.costs(L, S, days, G, V)
        assert np.array_equal(ok1, ok2), "feasibility differs"
        assert np.allclose(c1 + F["c0"], c2), "cost differs"
    print("block form == BlockEvaluator.costs on 30 random blocks/windows")
    # 2. SA vs exact enumeration, 6- and 7-asset blocks around the hand-in solution
    for g in (6, 7):
        hits, tot, gaps, t_sa = 0, 0, [], 0.0
        for _ in range(15):
            t = int(rng.integers(0, T))
            G = np.sort(rng.choice(n, g, replace=False))
            F = block_form(ev, L, S, [t], G)
            v0 = L[t, G] - S[t, G]
            ve = solve_exact(F)
            t1 = time.time()
            vs = solve_sa(F, v0, rng)
            t_sa += time.time() - t1
            ce, cs = form_cost(F, np.stack([ve, vs]))[0]
            assert form_cost(F, vs[None])[1][0]
            hits += int(np.isclose(cs, ce)); tot += 1; gaps.append(cs - ce)
        print(f"SA vs exact, {g} assets: optimum {hits}/{tot}, worst gap {max(gaps):.0f}, "
              f"SA {t_sa / tot:.2f}s/block")


def oracle(lam: str = "1e-3") -> list[dict]:
    """Paste one best-known day (all assets) into our solution: score change per day."""
    rows = []
    for inst, B in A050:
        model, (L, S), (Lb, Sb), bkv = load(inst, B, lam)
        base = model.score_arrays(L, S).objective
        gains = []
        for t in range(model.n_periods):
            L2, S2 = L.copy(), S.copy()
            L2[t], S2[t] = Lb[t], Sb[t]
            sc = model.score_arrays(L2, S2)
            gains.append(int(sc.objective - base) if sc.feasible else None)
        diff = [int(((L[t] != Lb[t]) | (S[t] != Sb[t])).sum()) for t in range(model.n_periods)]
        better = [g for g in gains if g is not None and g < 0]
        r = dict(instance=inst, score=base, bkv=bkv, gap=(base - bkv) / abs(bkv), day_gain=gains,
                 assets_differing=diff, n_improving_days=len(better), best_day_gain=min(better, default=0))
        print(f"{inst}: {base} vs bkv {bkv} (gap {r['gap']:+.2%}); pasting one bkv day changes the score by "
              f"{gains}; assets differing per day {diff}", flush=True)
        rows.append(r)
    return rows


def sweep(model, L, S, size: int, rng, time_limit: float, windows=(1,), sa_kw=None) -> dict:
    """Clamp-and-iterate with random groups of ``size`` (``size >= n``: whole days), SA blocks.

    Accepts a block only if the official score improves. Stops after a sweep without
    improvement or at ``time_limit``.
    """
    sa_kw = sa_kw or {}
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    best = model.score_arrays(L, S).objective
    t0, sweeps, blocks, acc, hist = time.time(), 0, 0, 0, [(0.0, best)]
    exact = size <= 5
    while time.time() - t0 < time_limit:
        before = best
        perm = rng.permutation(n)
        groups = [np.sort(g) for g in np.array_split(perm, -(-n // min(size, n)))]
        sched = [list(range(s0, s0 + w)) for w in windows for s0 in range(0, T - w + 1)]
        for days in [sched[k] for k in rng.permutation(len(sched))]:
            for G in groups:
                if time.time() - t0 > time_limit:
                    break
                F = block_form(ev, L, S, days, G)
                v0 = L[days[0], G] - S[days[0], G]
                if not form_cost(F, v0[None])[1][0]:  # current holding varies over the window
                    v0 = np.zeros(len(G), np.int64)
                    if not form_cost(F, v0[None])[1][0]:
                        continue
                V = solve_exact(F) if exact else solve_sa(F, v0, rng, **sa_kw)
                blocks += 1
                L2, S2 = L.copy(), S.copy()
                L2[np.ix_(days, G)], S2[np.ix_(days, G)] = np.maximum(V, 0), np.maximum(-V, 0)
                new = model.score_arrays(L2, S2)
                if new.feasible and new.objective < best:
                    L, S, best = L2, S2, new.objective
                    acc += 1
        sweeps += 1
        hist.append((round(time.time() - t0, 1), best))
        if best >= before:
            break
    return dict(score=best, sweeps=sweeps, blocks=blocks, accepted=acc, seconds=round(time.time() - t0, 1),
                history=hist, L=L, S=S)


def run_sweeps(args) -> None:
    out = Path("results/big_blocks")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    insts = A050 if args.instance == "all" else [(args.instance, args.budget)]
    for inst, B in insts:
        model, (L, S), _, bkv = load(inst, B, args.lam, args.start)
        base = model.score_arrays(L, S).objective
        print(f"\n{inst} λ{args.lam} start {args.start}: {base} (bkv {bkv}, gap {(base - bkv) / abs(bkv):+.2%})", flush=True)
        for size in args.sizes:
            for w in args.windows:
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    r = sweep(model, L, S, size, rng, args.time, windows=(w,),
                              sa_kw=dict(chains=args.chains, sweeps=args.sa_sweeps))
                    sc = model.score_arrays(r["L"], r["S"])
                    assert sc.feasible and sc.objective == r["score"]
                    r.pop("L"); r.pop("S")
                    r.update(instance=inst, lam=args.lam, start=args.start, size=size, window=w, seed=seed,
                             base=base, bkv=bkv, gain=r["score"] - base, gap=(r["score"] - bkv) / abs(bkv))
                    rows.append(r)
                    print(f"  size {size:>2} window {w} seed {seed}: {base} -> {r['score']} "
                          f"(gain {r['gain']}, gap {r['gap']:+.2%}) | {r['sweeps']} sweeps, {r['blocks']} blocks, "
                          f"{r['accepted']} accepted, {r['seconds']}s", flush=True)
    tag = f"{args.instance}_lam{args.lam}_{args.start}_s{'-'.join(map(str, args.sizes))}_w{'-'.join(map(str, args.windows))}"
    path = out / f"sweep_{tag}.json"
    path.write_text(json.dumps(dict(vars(args), rows=rows), indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "oracle", "sweep"])
    ap.add_argument("--instance", default="all")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--start", choices=["handin", "empty"], default="handin")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 6, 8, 12, 50])
    ap.add_argument("--windows", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time", type=float, default=120.0, help="per (size, window, seed) run")
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--sa-sweeps", type=int, default=300)
    a = ap.parse_args()
    if a.command == "check":
        check()
    elif a.command == "oracle":
        rows = oracle(a.lam)
        Path("results/big_blocks").mkdir(parents=True, exist_ok=True)
        Path(f"results/big_blocks/oracle_lam{a.lam}.json").write_text(json.dumps(rows, indent=1))
    else:
        run_sweeps(a)
