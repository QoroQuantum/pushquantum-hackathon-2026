"""Whole-day pair-move local search for the big portfolio instances (a200 / a400).

At 400 assets the unit budget binds on every day, so an improving move usually has to
take a unit from one asset and give it to another. Random groups of 3 almost never
contain the right pair; here every (asset, ±1/±2) x (asset, ±1/±2) pair and every
single-asset value change of a whole day is scored at once (``big_blocks.block_form``
gives the day's exact cost given its neighbours), and steepest descent applies the
best one. Iterated: random feasible moves on one day (kick), descend, keep if better.

    python day_swap.py --instance po_a400_t15_orig --budget 100 --lam 1e-3 \
        --init handin/po_a400_t15_orig_b100_l1e-3.sol --time 3600
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

import qoblib_tools as qt
from asset_groups import BlockEvaluator
from big_blocks import block_form, form_cost

VALS = np.arange(-3, 4)
_STOP = [False]  # set by SIGTERM: finish the current move, then write the best solution


def _on_term(signum, frame):  # noqa: ARG001
    _STOP[0] = True

try:
    from numba import njit
except ImportError:  # pragma: no cover - numpy fallback below
    njit = None

if njit is not None:

    @njit(cache=True)
    def _descend_nb(A, Ad, f, U, lo, hi, v, max_moves):
        """Steepest descent over all single value changes and all ±1 x ±1 pair moves."""
        g = v.shape[0]
        AV = A @ v.astype(np.float64)
        units = 0
        net = 0
        for i in range(g):
            units += abs(v[i])
            net += v[i]
        dc = np.empty((g, 2))
        du = np.empty((g, 2), np.int64)
        moves = 0
        while moves < max_moves:
            best = -1e-9
            bi, bdi, bj, bdj = -1, 0, -1, 0
            for i in range(g):
                vi = v[i]
                fi = f[i, vi + 3]
                for w in range(-3, 4):
                    if w == vi:
                        continue
                    d = w - vi
                    nn = net + d
                    if units - abs(vi) + abs(w) > U or nn < lo or nn > hi:
                        continue
                    delta = 2.0 * d * AV[i] + Ad[i] * d * d + f[i, w + 3] - fi
                    if delta < best:
                        best, bi, bdi, bj, bdj = delta, i, d, -1, 0
                for k in range(2):
                    s = 2 * k - 1
                    w = vi + s
                    if w < -3 or w > 3:
                        dc[i, k] = np.inf
                        du[i, k] = 0
                    else:
                        dc[i, k] = 2.0 * s * AV[i] + Ad[i] + f[i, w + 3] - fi
                        du[i, k] = abs(w) - abs(vi)
            room = U - units
            for i in range(g):
                for ki in range(2):
                    di = dc[i, ki]
                    if di == np.inf:
                        continue
                    si = 2 * ki - 1
                    for j in range(i + 1, g):
                        aij = 2.0 * A[i, j] * si
                        for kj in range(2):
                            sj = 2 * kj - 1
                            delta = di + dc[j, kj] + aij * sj
                            if delta < best:
                                nn = net + si + sj
                                if du[i, ki] + du[j, kj] <= room and nn >= lo and nn <= hi:
                                    best, bi, bdi, bj, bdj = delta, i, si, j, sj
            if bi < 0:
                break
            v[bi] += bdi
            units += 0
            for r in range(g):
                AV[r] += bdi * A[r, bi]
            net += bdi
            if bj >= 0:
                v[bj] += bdj
                for r in range(g):
                    AV[r] += bdj * A[r, bj]
                net += bdj
            units = 0
            for r in range(g):
                units += abs(v[r])
            moves += 1
        return v, moves


def day_form(ev: BlockEvaluator, L, S, t: int) -> dict:
    """``big_blocks.block_form(ev, L, S, [t], all assets)`` in closed form (O(7n), not O(7n^3)).

    With every asset in the block the risk is all in ``A``; ``f[i, v]`` holds the
    separable terms of asset ``i`` at net position ``v`` relative to ``v = 0``.
    """
    m = ev.model
    T = m.n_periods
    v = VALS[None, :].astype(float)
    Lg, Sg = np.maximum(v, 0), np.maximum(-v, 0)
    f = Sg * m.sc[t][:, None] + ev.i0 * v
    if t < T - 1:
        f = f - v * m.ret[t][:, None]
    if t == 0 or t == T - 1:
        f = f + (Lg + Sg) * m.tc[t][:, None]
    tc = m.tc
    if 1 <= t <= T - 2:
        Lp, Sp = L[t - 1][:, None], S[t - 1][:, None]
        f = f + (np.abs(Lg - Lp) + np.abs(Sg - Sp) - (Lp + Sp)) * tc[t][:, None]
    if t + 1 <= T - 2:
        Ln, Sn = L[t + 1][:, None], S[t + 1][:, None]
        f = f + (np.abs(Ln - Lg) + np.abs(Sn - Sg) - (Ln + Sn)) * tc[t + 1][:, None]
    return dict(A=ev.Rs[t], f=f, U=int(m.budget), lo=int(m.capital - ev.max_cash), hi=int(m.capital),
                c0=0.0)


class DaySearch:
    """Pair-move steepest descent on one day's form ``F`` (``A``, ``f``, ``U``, ``lo``, ``hi``)."""

    # The objective is convex along any line (PSD risk up to rounding, |Δv| trading costs,
    # linear return / interest), so ±2 pair steps never improve when no ±1 pair step does;
    # ±1 pairs reach the same local optima 4.5x faster on a400.
    def __init__(self, F: dict, pair_deltas=(-1, 1), top: int = 0, big_kicks: float = 0.5):
        self.F, self.top, self.big_kicks = F, top, big_kicks
        A = F["A"]
        g = A.shape[0]
        self.g = g
        self.Ad = np.diag(A).copy()
        self.fast = njit is not None and tuple(pair_deltas) == (-1, 1) and not top
        if self.fast:
            self.Af = np.ascontiguousarray(A, dtype=np.float64)
            self.ff = np.ascontiguousarray(F["f"], dtype=np.float64)
            return
        pd = np.asarray(pair_deltas)
        self.I = np.repeat(np.arange(g), len(pd))
        self.DL = np.tile(pd, g)
        self.P = 2.0 * np.outer(self.DL, self.DL) * A[np.ix_(self.I, self.I)]
        self.P[self.I[:, None] == self.I[None, :]] = np.inf
        self.NS = self.DL[:, None] + self.DL[None, :]

    def cost(self, v) -> float:
        c, ok = form_cost(self.F, v[None])
        assert ok[0]
        return float(c[0])

    def descend(self, v, max_moves: int = 100_000):
        if self.fast:
            F = self.F
            return _descend_nb(self.Af, self.Ad.astype(np.float64), self.ff, int(F["U"]), int(F["lo"]),
                               int(F["hi"]), v.astype(np.int64).copy(), max_moves)
        F, A, f = self.F, self.F["A"], self.F["f"]
        U, lo, hi, g = F["U"], F["lo"], F["hi"], self.g
        v = v.astype(np.int64).copy()
        AV = A @ v
        units, net = int(np.abs(v).sum()), int(v.sum())
        ar = np.arange(g)
        net_ok = (net + self.NS >= lo) & (net + self.NS <= hi)
        moves = 0
        for _ in range(max_moves):
            fcur = f[ar, v + 3]
            D = VALS[None, :] - v[:, None]
            d1 = 2 * D * AV[:, None] + self.Ad[:, None] * D ** 2 + f - fcur[:, None]
            du1 = np.abs(VALS)[None, :] - np.abs(v)[:, None]
            ok1 = (D != 0) & (units + du1 <= U) & (net + D >= lo) & (net + D <= hi)
            s1 = np.where(ok1, d1, np.inf)
            k1 = int(np.argmin(s1))
            b1 = s1.flat[k1]
            # pairs: candidate c = (asset I[c], step DL[c])
            vi = v[self.I]
            nv = vi + self.DL
            inr = np.abs(nv) <= 3
            nvc = np.clip(nv, -3, 3)
            dlc = nvc - vi
            dc = 2 * dlc * AV[self.I] + self.Ad[self.I] * dlc ** 2 + f[self.I, nvc + 3] - fcur[self.I]
            dc = np.where(inr, dc, np.inf)
            du = np.abs(nvc) - np.abs(vi)
            # rows: the ``top`` cheapest candidates (one side of the best pair is almost
            # always among them); all rows only to confirm that nothing improves
            rows = None
            if self.top and self.top < len(dc):
                rows = np.argpartition(dc, self.top)[:self.top]
            while True:
                R = slice(None) if rows is None else rows
                M = dc[R, None] + dc[None, :] + self.P[R]
                bad = ~net_ok[R] | (du[R, None] + du[None, :] > U - units)
                M[bad] = np.inf
                k2 = int(np.argmin(M))
                b2 = M.flat[k2]
                if min(b1, b2) < -1e-9 or rows is None:
                    break
                rows = None
            if min(b1, b2) >= -1e-9:
                break
            if b1 <= b2:
                i, w = divmod(k1, 7)
                steps = [(i, VALS[w] - v[i])]
            else:
                a, b = divmod(k2, M.shape[1])
                a = a if rows is None else rows[a]
                steps = [(self.I[a], self.DL[a]), (self.I[b], self.DL[b])]
            for i, d in steps:
                v[i] += d
                AV += d * A[:, i]
                net += int(d)
            units = int(np.abs(v).sum())
            net_ok = (net + self.NS >= lo) & (net + self.NS <= hi)
            moves += 1
        return v, moves

    def kick(self, v, rng, n_moves: int):
        """``n_moves`` random feasible single / pair unit moves (cost ignored)."""
        F = self.F
        v = v.copy()
        g = self.g
        feasible = lambda w: (np.abs(w).sum() <= F["U"]) and (F["lo"] <= w.sum() <= F["hi"])  # noqa: E731
        held = np.flatnonzero(v)
        r = rng.random()
        if len(held) <= 2 and r < 0.5:  # (near) empty day: plant a random small book
            w = v.copy()
            k = int(rng.integers(min(4, g), min(20, g) + 1))  # g < 20 on a010
            w[rng.choice(g, k, replace=False)] = rng.choice([-2, -1, 1, 2], k)
            if feasible(w):
                return w
        if r < self.big_kicks and len(held) > 1:
            # hedged books are deep local minima: removing one leg costs more than it saves,
            # so take out many legs at once and let the descent rebuild
            w = v.copy()
            if r < self.big_kicks / 2:  # ruin: zero 10-40% of the held positions
                k = max(1, int(len(held) * rng.uniform(0.1, 0.4)))
                w[rng.choice(held, k, replace=False)] = 0
            else:  # shrink: one unit off 20-60% of the held positions
                k = max(1, int(len(held) * rng.uniform(0.2, 0.6)))
                idx = rng.choice(held, k, replace=False)
                w[idx] -= np.sign(w[idx])
            if feasible(w):
                return w
        for _ in range(n_moves):
            for _try in range(50):
                w = v.copy()
                i = rng.integers(g)
                w[i] = np.clip(w[i] + rng.choice([-1, 1]), -3, 3)
                if rng.random() < 0.7:
                    j = rng.integers(g)
                    w[j] = np.clip(w[j] + rng.choice([-1, 1]), -3, 3)
                if (np.abs(w).sum() <= F["U"]) and (F["lo"] <= w.sum() <= F["hi"]) and (w != v).any():
                    v = w
                    break
        return v


def run(model: qt.PortfolioModel, L, S, time_limit: float, seed: int, kick_moves=(2, 8),
        ckpt: Path | None = None, verbose: bool = True, top: int = 0, days=None,
        eps_rel: float = 2.5e-4, two_day: float = 0.2, stall: float | None = None,
        window: int = 2):
    """``stall``: also stop once no improvement was found for this many seconds.
    ``window``: longest run of consecutive days a multi-day move spans (2 = two-day moves).
    """
    rng = np.random.default_rng(seed)
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    days = list(range(T)) if not days else [int(d) for d in days]
    G = np.arange(n)
    V = (L - S).astype(np.int64)
    split = lambda V: (np.maximum(V, 0), np.maximum(-V, 0))  # noqa: E731
    best = model.score_arrays(*split(V)).objective
    t0 = time.time()
    trace = [(0.0, best)]
    last_ck = time.time()
    # phase 1: descend every day until no day improves
    sweep = 0
    while time.time() - t0 < time_limit and not _STOP[0]:
        sweep += 1
        changed = False
        for t in days:
            Ls, Ss = split(V)
            ds = DaySearch(day_form(ev, Ls, Ss, t), top=top)
            v, moves = ds.descend(V[t])
            z = np.zeros_like(v)
            if sweep == 1 and ds.F["lo"] <= 0 <= ds.F["hi"]:  # also rebuild the day from empty
                vz, _ = ds.descend(z)
                if ds.cost(vz) < ds.cost(v) - 1e-6:
                    v, moves = vz, moves + 1
            if moves:
                V[t] = v
                changed = True
        sc = model.score_arrays(*split(V))
        assert sc.feasible and sc.objective <= best, (sc.feasible, sc.objective, best)
        best = sc.objective
        trace.append((round(time.time() - t0, 1), best))
        if verbose:
            print(f"  {time.time() - t0:7.0f}s  descent sweep {sweep}: {best}", flush=True)
        if not changed:
            break
    # phase 2: per-day iterated local search
    kicks = acc = kicks2 = acc2 = 0
    forms, cur = {}, {}
    day_set = set(days)
    U, lo, hi = model.budget, model.capital - ev.max_cash, model.capital

    def fix(w, keep):
        """Bring one day back into the unit / net limits by trimming assets not in ``keep``."""
        free = np.setdiff1d(np.flatnonzero(w), keep)
        for _ in range(2000):
            units, net = int(np.abs(w).sum()), int(w.sum())
            if units <= U and lo <= net <= hi:
                return True
            if len(free) == 0:
                return False
            if net > hi:
                cand = free[w[free] > 0]
            elif net < lo:
                cand = free[w[free] < 0]
            else:
                cand = free[w[free] != 0]
            if len(cand) == 0:
                return False
            i = rng.choice(cand)
            w[i] -= np.sign(w[i])
        return False

    while time.time() - t0 < time_limit and not _STOP[0]:
        if stall and time.time() - t0 - trace[-1][0] > stall:
            break
        t = int(rng.choice(days))
        nb = [d for d in (t - 1, t + 1) if d in day_set]
        if two_day and nb and rng.random() < two_day:
            # two-day move: trading costs couple neighbouring days, so some gains need both
            # days to change at once (hold a position over both, or drop it on both)
            a, b = sorted((t, int(rng.choice(nb))))
            W = V.copy()
            win = [a, b]
            if window > 2 and rng.random() < 0.5:
                # longer window: hold (or drop) the same positions over w consecutive days,
                # e.g. keep a position open three more days and hedge it meanwhile
                w = int(rng.integers(3, window + 1))
                a = int(rng.integers(t - w + 1, t + 1))
                win = [d for d in range(a, a + w) if d in day_set]
                if len(win) < 3 or win != list(range(win[0], win[0] + len(win))):
                    continue
                a, b = win[0], win[-1]
                ends = [d for d in (a - 1, b + 1) if d in day_set]
                src = int(rng.choice(win + ends))
                dst = None
            else:
                src, dst = (a, b) if rng.random() < 0.5 else (b, a)
            held = np.flatnonzero(W[src])
            kicks2 += 1
            if len(held) == 0:
                continue
            idx = rng.choice(held, max(1, int(len(held) * rng.uniform(0.05, 0.4))), replace=False)
            if rng.random() < 0.7:
                for d in ([dst] if dst is not None else win):
                    W[d, idx] = W[src, idx]
            else:
                for d in win:
                    W[d, idx] = 0
            if not all(fix(W[d], idx) for d in win):
                continue
            for _ in range(8):  # alternate exact day descents until the window is settled
                moved = 0
                for d in win:
                    Ls, Ss = split(W)
                    W[d], n_mv = DaySearch(day_form(ev, Ls, Ss, d), top=top).descend(W[d])
                    moved += n_mv
                if not moved:
                    break
            sc = model.score_arrays(*split(W))
            if sc.feasible and sc.objective < best:
                V, best = W, sc.objective
                for d in range(a - 1, b + 2):
                    forms.pop(d, None)
                    cur.pop(d, None)
                acc2 += 1
                trace.append((round(time.time() - t0, 1), best))
                if verbose:
                    print(f"  {time.time() - t0:7.0f}s  two-day move {kicks2} (days {a},{b}): {best}", flush=True)
                if ckpt is not None and time.time() - last_ck > 30:
                    model.write_solution(ckpt, model.to_positions(*split(V)), comment=f"checkpoint {best}")
                    last_ck = time.time()
            continue
        if t not in forms:
            Ls, Ss = split(V)
            forms[t] = DaySearch(day_form(ev, Ls, Ss, t), top=top)
        ds = forms[t]
        c0 = ds.cost(V[t])
        # threshold acceptance: each day keeps a working state that may sit up to
        # eps above the day's best (V[t]); it restarts from V[t] when the form changes
        base, cbase = cur.get(t, (V[t], c0))
        v = ds.kick(base, rng, int(rng.integers(kick_moves[0], kick_moves[1] + 1)))
        v, _ = ds.descend(v)
        cv = ds.cost(v)
        kicks += 1
        eps = eps_rel * max(abs(c0), 1000.0)
        if cv >= c0 - 1e-6:
            if cv < cbase + eps and (v != base).any():
                cur[t] = (v, cv)
            continue
        if cv < c0 - 1e-6:
            V[t] = v
            for d in (t - 1, t, t + 1):  # neighbours' rebalancing terms changed
                forms.pop(d, None)
                cur.pop(d, None)
            sc = model.score_arrays(*split(V))
            assert sc.feasible and sc.objective < best, (sc.feasible, sc.objective, best)
            best = sc.objective
            acc += 1
            trace.append((round(time.time() - t0, 1), best))
            if verbose:
                print(f"  {time.time() - t0:7.0f}s  kick {kicks} (day {t}): {best}", flush=True)
            if ckpt is not None and time.time() - last_ck > 30:
                model.write_solution(ckpt, model.to_positions(*split(V)), comment=f"checkpoint {best}")
                last_ck = time.time()
    Ls, Ss = split(V)
    return dict(score=best, kicks=kicks, accepted=acc, kicks2=kicks2, accepted2=acc2, trace=trace, L=Ls, S=Ss,
                seconds=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a400_t15_orig")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--init", required=True, help='start .sol file, or "empty"')
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/day_swap")
    ap.add_argument("--top", type=int, default=0, help="pair rows scored per move (0 = all)")
    ap.add_argument("--days", type=int, nargs="*", default=None, help="only search these days")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--two-day", type=float, default=0.2, help="share of two-day moves")
    ap.add_argument("--eps-rel", type=float, default=2.5e-4,
                    help="threshold acceptance: working state may be this share of |day cost| worse")
    ap.add_argument("--kick-min", type=int, default=2)
    ap.add_argument("--kick-max", type=int, default=8)
    ap.add_argument("--window", type=int, default=2, help="longest multi-day move, in days")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    model = qt.PortfolioModel(qt.load_portfolio_instance(args.instance), budget=args.budget, lam=args.lam)
    if args.init == "empty":
        L = np.zeros((model.n_periods, len(model.symbols)), np.int64)
        S = L.copy()
    else:
        _, pos = qt.read_portfolio_solution(args.init)
        L, S = (a.astype(np.int64) for a in model.to_arrays(pos))
    sc = model.score_arrays(L, S)
    assert sc.feasible
    print(f"{args.instance} B={args.budget} λ={args.lam}: start {sc.objective} from {args.init}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instance}_b{args.budget:03d}_l{args.lam}"
    r = run(model, L, S, args.time, args.seed, ckpt=out / f"{stem}.ckpt.sol", top=args.top,
            kick_moves=(args.kick_min, args.kick_max), days=args.days, verbose=not args.quiet,
            eps_rel=args.eps_rel, two_day=args.two_day, window=args.window)
    final = model.score_arrays(r["L"], r["S"])
    assert final.feasible and final.objective == r["score"]
    sol = out / f"{stem}_seed{args.seed}.sol"
    model.write_solution(sol, model.to_positions(r["L"], r["S"]),
                         comment=f"day_swap.py: whole-day pair-move descent + per-day ILS from {args.init}")
    rec = dict(instance=args.instance, budget=args.budget, lam=args.lam, init=args.init, start=sc.objective,
               score=r["score"], kicks=r["kicks"], accepted=r["accepted"], kicks2=r["kicks2"],
               accepted2=r["accepted2"], trace=r["trace"],
               seconds=r["seconds"], solution_file=str(sol))
    (out / f"{stem}_seed{args.seed}.json").write_text(json.dumps(rec, indent=2, default=float))
    print(f"  final {r['score']} ({r['kicks']} kicks, {r['accepted']} accepted; {r['kicks2']} two-day moves, "
          f"{r['accepted2']} accepted; {r['seconds']}s); wrote {sol}")


if __name__ == "__main__":
    main()
