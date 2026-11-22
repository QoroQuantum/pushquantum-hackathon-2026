"""Valid-topology moves for QOBLIB 08: every step stays a 2-in/2-out simple digraph.

  exchange_moves  all degree-preserving 2-exchanges (a->b),(c->d) => (a->d),(c->b)
  polish          final local search over 2-exchanges (replaces Divi's single-bit local_search):
                  rank neighbours by a cheap surrogate, LP-score the best few, accept, repeat
  move_sa         simulated annealing in move space (never leaves the valid set)
  node surrogates node-load models for "route heavy flows via quiet nodes" (part C)

    .venv/bin/python network_moves.py polish   --n 12            # attack 3: samples + 2-exchange polish
    .venv/bin/python network_moves.py busy     --n 12 24         # part C: fidelity of node-load surrogates
    .venv/bin/python network_moves.py busyopt  --n 12            # part C: LP of each surrogate's optimum
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import qoblib_tools as qt  # noqa: E402
from network_hopqubo import adj, hop_dist, sample_topologies, shortest_path_counts  # noqa: E402

Arc = tuple[int, int]


# ── moves ────────────────────────────────────────────────────────────────
def exchange_moves(arcs: list[Arc]):
    """Every distinct 2-exchange neighbour (sorted arc list)."""
    s = set(arcs)
    L = sorted(arcs)
    for first in range(len(L)):
        a, b = L[first]
        for second in range(first + 1, len(L)):
            c, d = L[second]
            if a == c or b == d or a == d or c == b or (a, d) in s or (c, b) in s:
                continue
            yield sorted((s - {(a, b), (c, d)}) | {(a, d), (c, b)})


def lp(T, arcs):
    return qt.route_min_congestion(T, arcs).objective


def bkv_gap(n: int, score: float) -> str:
    """Gap to the QOBLIB best-known value (proven optimal for n <= 10)."""
    from network_pipeline import BKV

    return f"{(score - BKV[n]) / BKV[n]:+.2%} vs BKV {BKV[n]}"


# ── routing model shared by the surrogates ───────────────────────────────
def ecmp_loads(T, arcs):
    """Arc loads when every pair splits evenly over all its shortest paths."""
    n = len(T)
    A = adj(n, arcs)
    D = hop_dist(A)
    sig = shortest_path_counts(A, D)
    L = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if k == j or T[k, j] == 0 or D[k, j] >= 10**6:
                continue
            on = (D[k][:, None] + 1 + D[:, j][None, :] == D[k, j]) & (A > 0)
            L += T[k, j] * on * (sig[k][:, None] * sig[:, j][None, :]) / sig[k, j]
    return A, D, L


def make_surrogates(T):
    """name -> f(arcs) (lower is better). Node weights from the demand matrix only."""
    n = len(T)
    off = ~np.eye(n, dtype=bool)
    S, R = T.sum(1).astype(float), T.sum(0).astype(float)  # own send / receive per node
    busy = (S + R) / (S + R).max()  # 0..1, "how loaded are j's own links already"

    def parts(arcs):
        A, D, L = ecmp_loads(T, arcs)
        tr = L.sum(0) - R  # traffic entering j that is not destined to j = transit through j
        return A, D, L, tr

    def hinf(D):
        return float((T * np.minimum(D, len(T)))[off].sum())

    fns = {
        "H2": lambda a: _h2(T, a, off),
        "Hinf": lambda a: hinf(hop_dist(adj(n, a))),
        "ecmp": lambda a: float(ecmp_loads(T, a)[2].max()),
    }

    def busy_lin(beta):  # user's idea: transit through a busy node costs extra
        def f(a):
            _, D, _, tr = parts(a)
            return hinf(D) + beta * float((busy * tr).sum())
        return f

    def nodemax(a):  # out-links of j carry S_j + transit_j, in-links R_j + transit_j (2 links each)
        _, _, _, tr = parts(a)
        return float(np.maximum(S + tr, R + tr).max() / 2)

    def nodeL2(a):
        _, _, _, tr = parts(a)
        return float(((S + tr) ** 2 + (R + tr) ** 2).sum())

    def mix(a, lam=1.0):  # total flow (average load) + node-level max, both in load units
        _, D, _, tr = parts(a)
        return hinf(D) / (2 * n) + lam * float(np.maximum(S + tr, R + tr).max() / 2)

    for b in (0.25, 0.5, 1.0, 2.0):
        fns[f"busy{b}"] = busy_lin(b)
    fns["nodemax"] = nodemax
    fns["nodeL2"] = nodeL2
    fns["avg+nodemax"] = mix
    return fns


def _h2(T, arcs, off):
    n = len(T)
    A = adj(n, arcs)
    A2 = A @ A
    h2 = np.where(A > 0, 1, np.where(A2 > 0, 2, 3))
    return float((T * h2)[off].sum())


# ── polish & move-space SA ───────────────────────────────────────────────
def polish(T, arcs, rank=None, lp_budget=200, k=5):
    """First-improvement local search over 2-exchanges. Neighbours are ranked by `rank` (cheap),
    the best k untried ones are LP-scored per round; ties accepted once per topology (plateaus)."""
    cur = sorted(arcs)
    z = lp(T, cur)
    seen = {tuple(cur)}
    used = 0
    while used < lp_budget:
        nb = [m for m in exchange_moves(cur) if tuple(m) not in seen]
        if not nb:
            break
        order = np.argsort([rank(m) for m in nb], kind="stable") if rank else np.random.default_rng(used).permutation(len(nb))
        moved = False
        for start in range(0, len(nb), k):
            for idx in order[start:start + k]:
                zz = lp(T, nb[idx])
                used += 1
                seen.add(tuple(nb[idx]))
                if zz < z or (zz == z and math.isfinite(z)):  # ties only between feasible topologies
                    cur, z, moved = nb[idx], zz, True
                    break
                if used >= lp_budget:
                    break
            if moved or used >= lp_budget:
                break
        if not moved:
            break
    return cur, z, used


def move_sa(T, energy, rng, steps, start=None, t0_frac=0.02):
    from network_hopqubo import two_exchange

    a = sorted(start) if start is not None else sorted(qt.random_topology(len(T), rng))
    e = energy(a)
    best, be = a, e
    t0 = abs(e) * t0_frac + 1e-9
    for s in range(steps):
        temp = t0 * (1 - s / steps) + 1e-12
        b = two_exchange(a, rng)
        if b is None:
            continue
        eb = energy(b)
        if eb <= e or rng.random() < math.exp(-(eb - e) / temp):
            a, e = b, eb
            if e < be:
                best, be = a, e
    return best, be


# ── experiments ──────────────────────────────────────────────────────────
def cmd_polish(args):
    """Attack 3: does a QUBO sampler start + 2-exchange polish beat random start + polish?"""
    from dwave.samplers import SimulatedAnnealingSampler

    from network_hopqubo import build_path_qubo

    for n in args.n:
        T = qt.load_network_demand(n)
        H2 = make_surrogates(T)["H2"]
        sur, pen, arcs_v = qt.build_network_qubo(T)
        direct = sur.copy()
        direct.update(pen)
        starts = {}
        rng = np.random.default_rng(0)
        starts["random"] = [sorted(qt.random_topology(n, rng)) for _ in range(args.starts)]
        ss = SimulatedAnnealingSampler().sample(direct, num_reads=args.starts, seed=1)
        starts["SA direct QUBO"] = [sorted(qt.repair_topology(n, [arcs_v[v] for v, b in s.items() if b], T)) for s in ss.samples()]
        bqm = build_path_qubo(T, hops=2, penalty_weight=4.0)
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=args.starts, num_sweeps=2000, seed=1)
        starts["SA 2-hop QUBO"] = [sorted(qt.repair_topology(n, [(k[1], k[2]) for k, b in s.items() if b and k[0] == "x"], T)) for s in ss.samples()]
        print(f"\nn={n}: {args.starts} starts each, polish budget {args.budget} LP evals/start, k={args.k}")
        for name, tops in starts.items():
            for rk in ("H2", None):
                t0 = time.time()
                z0 = [lp(T, a) for a in tops]
                res = [polish(T, a, rank=H2 if rk else None, lp_budget=args.budget, k=args.k) for a in tops]
                z1 = [r[1] for r in res]
                print(f"  {name:<15} rank={'H2' if rk else 'random':<6} start best {min(z0):.0f} -> polished best {min(z1):.0f} ({bkv_gap(n, min(z1))}) "
                      f"mean {np.mean(z1):.0f} (LP evals {sum(r[2] for r in res)}, {time.time() - t0:.0f}s)", flush=True)


def cmd_busy(args):
    """Part C: Spearman rho of node-load surrogates vs LP."""
    rng = np.random.default_rng(0)
    for n in args.n:
        T = qt.load_network_demand(n)
        fns = make_surrogates(T)
        rows = []
        for tag, a in sample_topologies(n, rng, args.rand, args.near):
            z = lp(T, a)
            if math.isinf(z):
                continue
            rows.append((tag, z, {k: f(a) for k, f in fns.items()}))
        print(f"\nn={n}: {len(rows)} topologies")
        for tag in sorted({r[0] for r in rows}):
            sub = [r for r in rows if r[0] == tag]
            z = np.array([r[1] for r in sub])
            print(f"  {tag:<5} N={len(sub)}  " + "  ".join(
                f"{k}={spearmanr([r[2][k] for r in sub], z).statistic:+.2f}" for k in fns), flush=True)


def cmd_busyopt(args):
    """Part C: LP score of each surrogate's move-space-SA optimum (+ LP polish)."""
    for n in args.n:
        T = qt.load_network_demand(n)
        fns = make_surrogates(T)
        print(f"\nn={n}: move-space SA {args.steps} steps x {args.restarts} restarts, then polish {args.budget} LP evals")
        for name in args.surr:
            rng = np.random.default_rng(1)
            t0 = time.time()
            raw, pol = [], []
            for _ in range(args.restarts):
                a, _ = move_sa(T, fns[name], rng, args.steps)
                raw.append(lp(T, a))
                pol.append(polish(T, a, rank=fns["H2"], lp_budget=args.budget)[1])
            print(f"  {name:<12} surrogate-opt LP best {min(raw):.0f} mean {np.mean(raw):.0f} | after polish best {min(pol):.0f} "
                  f"({bkv_gap(n, min(pol))}) "
                  f"mean {np.mean(pol):.0f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    polish_p = sub.add_parser("polish")
    polish_p.add_argument("--n", type=int, nargs="+", default=[12])
    polish_p.add_argument("--starts", type=int, default=10)
    polish_p.add_argument("--budget", type=int, default=150)
    polish_p.add_argument("--k", type=int, default=5)
    b = sub.add_parser("busy")
    b.add_argument("--n", type=int, nargs="+", default=[12, 24])
    b.add_argument("--rand", type=int, default=200)
    b.add_argument("--near", type=int, default=200)
    o = sub.add_parser("busyopt")
    o.add_argument("--n", type=int, nargs="+", default=[12])
    o.add_argument("--surr", nargs="+", default=["H2", "Hinf", "busy0.5", "busy1.0", "nodemax", "nodeL2", "avg+nodemax"])
    o.add_argument("--steps", type=int, default=1500)
    o.add_argument("--restarts", type=int, default=3)
    o.add_argument("--budget", type=int, default=100)
    args = ap.parse_args()
    {"polish": cmd_polish, "busy": cmd_busy, "busyopt": cmd_busyopt}[args.cmd](args)
