"""Path-variable surrogates for QOBLIB 08: do auxiliary 2-hop (3-hop) path variables make the
QUBO objective track the real LP score better?

Surrogates (all expressed as "min over auxiliary path variables" given the topology x):
  direct  : -sum t_ij x_ij                              (notebook 02 baseline, QUBO, no aux)
  H2      : total flow, pair cost 1 / 2 / 3 hops         (QUBO: + y_ijk two-hop vars + slack s_ik)
  H3      : same with 3-hop vars, cost 1/2/3/4           (QUBO: + y_ijlk, still quadratic penalties)
  Hinf    : exact total flow = sum t * shortest-hop-dist (not a QUBO; reference)
  H2L2    : H2 + mu * sum_a load_a^2, load from 1/2-hop routed traffic (QUBO)
  ecmp    : max arc load, even split over all shortest paths (not a QUBO; reference)

    .venv/bin/python network_hopqubo.py fidelity --n 5 8 12 24
    .venv/bin/python network_hopqubo.py surropt --n 12 --restarts 4
    .venv/bin/python network_hopqubo.py qubo --n 12 --reads 100
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

warnings.filterwarnings("ignore")
import dimod  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import qoblib_tools as qt  # noqa: E402


# ── topology helpers ─────────────────────────────────────────────────────
def adj(n, arcs):
    A = np.zeros((n, n), dtype=np.int64)
    for i, j in arcs:
        A[i, j] = 1
    return A


def hop_dist(A):
    n = len(A)
    D = np.full((n, n), 10**6, dtype=np.int64)
    np.fill_diagonal(D, 0)
    reach = np.eye(n, dtype=np.int64)
    frontier = np.eye(n, dtype=np.int64)
    for h in range(1, n):
        frontier = ((frontier @ A) > 0).astype(np.int64) * (1 - reach)
        if not frontier.any():
            break
        D[frontier > 0] = h
        reach |= frontier
    return D


def two_exchange(arcs, rng, tries=100):
    """Degree-preserving (a->b),(c->d) => (a->d),(c->b); returns new arc list or None."""
    s = set(arcs)
    L = list(arcs)
    for _ in range(tries):
        (a, b), (c, d) = (L[k] for k in rng.choice(len(L), 2, replace=False))
        if a == c or b == d or a == d or c == b or (a, d) in s or (c, b) in s:
            continue
        s2 = (s - {(a, b), (c, d)}) | {(a, d), (c, b)}
        return sorted(s2)
    return None


# ── closed-form surrogates: E(x) = min over aux given x ──────────────────
def surrogates(T, arcs, mu=None):
    n = len(T)
    A = adj(n, arcs)
    A2 = A @ A
    A3 = A2 @ A
    off = ~np.eye(n, dtype=bool)
    out = {}
    out["direct"] = -float((T * A).sum())
    h2 = np.where(A > 0, 1, np.where(A2 > 0, 2, 3))
    out["H2"] = float((T * h2)[off].sum())
    h3 = np.where(A > 0, 1, np.where(A2 > 0, 2, np.where(A3 > 0, 3, 4)))
    out["H3"] = float((T * h3)[off].sum())
    for al in (0.25, 0.5, 1.0):
        out[f"Q2a{al}"] = float((T * (-2 * A - al * A2 * (A == 0)))[off].sum())
        out[f"Q2full{al}"] = float((T * (-2 * A - al * A2))[off].sum())
    D = hop_dist(A)
    out["Hinf"] = float((T * np.minimum(D, n))[off].sum())
    # L2 of load from 1-/2-hop routed traffic, 2-hop pairs split evenly over their 2-hop paths
    load = (T * A).astype(float)
    two = (A == 0) & (A2 > 0) & off & (T > 0)
    for i, k in zip(*np.nonzero(two)):
        mids = np.nonzero(A[i] * A[:, k])[0]
        w = T[i, k] / len(mids)
        for j in mids:
            load[i, j] += w
            load[j, k] += w
    out["L2"] = float((load**2).sum())
    # ECMP max load over shortest paths (path-count split)
    sig = shortest_path_counts(A, D)
    eload = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if k == j or T[k, j] == 0 or D[k, j] >= 10**6:
                continue
            on = (D[k][:, None] + 1 + D[:, j][None, :] == D[k, j]) & (A > 0)
            eload += T[k, j] * on * (sig[k][:, None] * sig[:, j][None, :]) / sig[k, j]
    out["ecmp"] = float(eload.max())
    return out


def shortest_path_counts(A, D):
    n = len(A)
    sig = np.zeros((n, n))
    np.fill_diagonal(sig, 1.0)
    maxd = int(D[D < 10**6].max())
    for h in range(1, maxd + 1):
        prev = (D == h - 1).astype(float) * sig
        sig = np.where(D == h, prev @ A, sig)
    return sig


# ── fidelity: Spearman rho vs LP ─────────────────────────────────────────
def ref_topology(n):
    for suf in ("opt", "bst"):
        try:
            _, arcs, _ = qt.read_network_solution(qt.fetch_qoblib_file(f"08-network/solutions/network{n:02d}.{suf}.sol"))
            return sorted(map(tuple, arcs))
        except Exception:
            pass
    return None


def sample_topologies(n, rng, n_rand, n_near):
    if n == 5:
        import itertools
        rows = [list(itertools.combinations([j for j in range(n) if j != i], 2)) for i in range(n)]
        out = []
        for ch in itertools.product(*rows):
            a = sorted((i, j) for i, pair in enumerate(ch) for j in pair)
            if qt.is_valid_topology(n, a):
                out.append(("all", a))
        return out
    out = [("rand", sorted(qt.random_topology(n, rng))) for _ in range(n_rand)]
    ref = ref_topology(n)
    if ref is not None:
        out.append(("near", ref))
        for _ in range(n_near):
            a = ref
            for _ in range(rng.integers(1, 7)):
                a = two_exchange(a, rng) or a
            out.append(("near", a))
    return out


def cmd_fidelity(args):
    rng = np.random.default_rng(0)
    names = ["direct", "Q2full0.25", "Q2full0.5", "Q2full1.0", "H2", "H3", "Hinf", "ecmp"]
    for n in args.n:
        T = qt.load_network_demand(n)
        rows = []
        t0 = time.time()
        for tag, a in sample_topologies(n, rng, args.rand, args.near):
            z = qt.route_min_congestion(T, a).objective
            if math.isinf(z):
                continue
            rows.append((tag, z, surrogates(T, a)))
        print(f"\nn={n}: {len(rows)} topologies ({time.time() - t0:.0f}s)")
        for tag in sorted({r[0] for r in rows}):
            sub = [r for r in rows if r[0] == tag]
            z = np.array([r[1] for r in sub])
            line = f"  {tag:<5} N={len(sub):4d} rho: "
            for nm in names:
                v = np.array([r[2][nm] for r in sub])
                line += f"{nm}={spearmanr(v, z).statistic:+.2f} "
            # best mix H2 / Hinf + mu * normalized L2
            for base in ("H2", "Hinf"):
                b = np.array([r[2][base] for r in sub])
                l2 = np.array([r[2]["L2"] for r in sub])
                best = max(((spearmanr(b / b.std() + m * l2 / l2.std(), z).statistic, m) for m in (0.25, 0.5, 1, 2, 4)))
                line += f"| {base}+mu*L2 best={best[0]:+.2f}@mu={best[1]} "
            print(line)


# ── surrogate optimum: SA over 2-exchanges on closed-form energy ─────────
def energy_fn(T, name, mu):
    def E(a):
        s = surrogates(T, a)
        if name.endswith("L2"):
            base = s[name[:-2]]
            return base + mu * s["L2"] / max(T.max(), 1)
        return s[name]
    return E


def swap_sa(n, E, rng, steps, t0=None):
    a = sorted(qt.random_topology(n, rng))
    e = E(a)
    best, be = a, e
    t0 = t0 or abs(e) * 0.02
    for s in range(steps):
        T = t0 * (1 - s / steps) + 1e-9
        b = two_exchange(a, rng)
        if b is None:
            continue
        eb = E(b)
        if eb <= e or rng.random() < math.exp(-(eb - e) / T):
            a, e = b, eb
            if e < be:
                best, be = a, e
    return best, be


def cmd_surropt(args):
    for n in args.n:
        T = qt.load_network_demand(n)
        print(f"\nn={n}")
        for name in args.surr:
            rng = np.random.default_rng(1)
            E = energy_fn(T, name, args.mu)
            zs = []
            t0 = time.time()
            for _ in range(args.restarts):
                a, _ = swap_sa(n, E, rng, args.steps)
                zs.append(qt.route_min_congestion(T, a).objective)
            print(f"  {name:<7} LP of surrogate optimum: best {min(zs):.0f} mean {np.mean(zs):.0f} ({time.time() - t0:.0f}s)")


# ── full QUBO with auxiliary path variables ──────────────────────────────
def build_path_qubo(T, hops=2, mu=0.0, penalty_weight=3.0):
    """Vars: ('x',i,j) arcs; ('y',i,j,k) 2-hop; ('y3',i,j,l,k) 3-hop; ('s',i,k) slack (longer path).
    Objective: sum_{ik} t_ik * hops(path chosen) (+ mu * sum_a load_a^2), t scaled to max 1.
    Constraints (penalty weight penalty_weight each, squared): out/in degree 2; one-hot x_ik + sum y + s = 1 per pair with t>0;
    consistency y <= x on each arc of the path: penalty_weight * y (1 - x_a)."""
    n = len(T)
    t = T / T.max()
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    X = lambda i, j: ("x", i, j)  # noqa: E731
    for i in range(n):
        bqm.add_linear_equality_constraint([(X(i, j), 1) for j in range(n) if j != i], penalty_weight, -2)
        bqm.add_linear_equality_constraint([(X(j, i), 1) for j in range(n) if j != i], penalty_weight, -2)
    load = {}  # arc -> list of (var, coeff)
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] == 0:
                continue
            terms = [(X(i, k), 1)]
            bqm.add_linear(X(i, k), t[i, k] * 1.0)
            load.setdefault((i, k), []).append((X(i, k), t[i, k]))
            for j in range(n):
                if j in (i, k):
                    continue
                y = ("y", i, j, k)
                terms.append((y, 1))
                bqm.add_linear(y, t[i, k] * 2.0)
                for a in ((i, j), (j, k)):
                    bqm.add_linear(y, penalty_weight)
                    bqm.add_quadratic(y, X(*a), -penalty_weight)
                    load.setdefault(a, []).append((y, t[i, k]))
                if hops >= 3:
                    for l in range(n):  # noqa: E741
                        if l in (i, j, k):
                            continue
                        y3 = ("y3", i, j, l, k)
                        terms.append((y3, 1))
                        bqm.add_linear(y3, t[i, k] * 3.0)
                        for a in ((i, j), (j, l), (l, k)):
                            bqm.add_linear(y3, penalty_weight)
                            bqm.add_quadratic(y3, X(*a), -penalty_weight)
                            load.setdefault(a, []).append((y3, t[i, k]))
            s = ("s", i, k)
            terms.append((s, 1))
            bqm.add_linear(s, t[i, k] * (hops + 1.0))
            bqm.add_linear_equality_constraint(terms, penalty_weight, -1)
    if mu:
        for a, terms in load.items():  # mu * (sum c v)^2, binary v^2 = v
            for idx, (u, cu) in enumerate(terms):
                bqm.add_linear(u, mu * cu * cu)
                for v, cv in terms[idx + 1:]:
                    bqm.add_quadratic(u, v, 2 * mu * cu * cv)
    return bqm


def build_q2_qubo(T, alpha=0.5, penalty_weight=3.0):
    """Aux-free: sum_ik t_ik (-2 x_ik - alpha * sum_j x_ij x_jk) + degree penalties. n(n-1) vars."""
    n = len(T)
    t = T / T.max()
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    X = lambda i, j: ("x", i, j)  # noqa: E731
    for i in range(n):
        bqm.add_linear_equality_constraint([(X(i, j), 1) for j in range(n) if j != i], penalty_weight, -2)
        bqm.add_linear_equality_constraint([(X(j, i), 1) for j in range(n) if j != i], penalty_weight, -2)
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] == 0:
                continue
            bqm.add_linear(X(i, k), -2 * t[i, k])
            for j in range(n):
                if j not in (i, k):
                    bqm.add_quadratic(X(i, j), X(j, k), -alpha * t[i, k])
    return bqm


def build_soft_qubo(T, penalty_weight=3.0, c=1.2):
    """Soft 2-hop aux: pair cost t(3 - 2 x_ik - sum_j y_ijk), no slack.
    y_ijk <= x_ij, x_jk via c*t*y(1-x); at-most-one via c*t * (sum of pairwise products incl. x_ik).
    For c >= 1: min over y given x == H2 exactly. Penalties scale with t_ik -> small barriers."""
    n = len(T)
    t = T / T.max()
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    X = lambda i, j: ("x", i, j)  # noqa: E731
    for i in range(n):
        bqm.add_linear_equality_constraint([(X(i, j), 1) for j in range(n) if j != i], penalty_weight, -2)
        bqm.add_linear_equality_constraint([(X(j, i), 1) for j in range(n) if j != i], penalty_weight, -2)
    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] == 0:
                continue
            w = t[i, k]
            bqm.add_linear(X(i, k), -2 * w)
            ys = [("y", i, j, k) for j in range(n) if j not in (i, k)]
            for y in ys:
                _, _, j, _ = y
                bqm.add_linear(y, -w)
                for a in ((i, j), (j, k)):
                    bqm.add_linear(y, c * w)
                    bqm.add_quadratic(y, X(*a), -c * w)
                bqm.add_quadratic(y, X(i, k), c * w)
            for p_, u in enumerate(ys):
                for v in ys[p_ + 1:]:
                    bqm.add_quadratic(u, v, c * w)
    return bqm


def aux_assignment(T, arcs, bqm):
    """Topology -> full sample with optimal aux (path vars) for build_path_qubo."""
    n = len(T)
    A = adj(n, arcs)
    smp = {v: 0 for v in bqm.variables}
    for i, j in arcs:
        smp[("x", i, j)] = 1
    if not any(v[0] == "s" for v in bqm.variables):  # soft encoding: one y per 2-hop-only pair
        for i in range(n):
            for k in range(n):
                if i != k and not A[i, k] and T[i, k] > 0:
                    mids = [j for j in range(n) if j not in (i, k) and A[i, j] and A[j, k]]
                    if mids:
                        smp[("y", i, mids[0], k)] = 1
        return smp
    for v in bqm.variables:
        if v[0] == "s":
            i, k = v[1], v[2]
            if A[i, k]:
                continue
            mids = [j for j in range(n) if j not in (i, k) and A[i, j] and A[j, k]]
            if mids:
                smp[("y", i, mids[0], k)] = 1
            else:
                smp[v] = 1
    return smp


def cmd_qubo(args):
    from dwave.samplers import SimulatedAnnealingSampler

    for n in args.n:
        T = qt.load_network_demand(n)
        print(f"\nn={n}")
        configs = [("direct", None)] + [(f"Q2(a={a})", ("q2", a)) for a in args.alpha] + [(f"H2soft(c={c})", ("soft", c)) for c in args.soft] + [(f"H{h}" + (f"+L2(mu={m})" if m else ""), (h, m)) for h in args.hops for m in args.mu]
        for label, cfg in configs:
            t0 = time.time()
            if cfg is None:
                sur, pen, arcs_v = qt.build_network_qubo(T)
                bqm = sur.copy()
                bqm.update(pen)
                decode = lambda s: [arcs_v[v] for v, b in s.items() if b]  # noqa: E731
            elif cfg[0] == "soft":
                bqm = build_soft_qubo(T, penalty_weight=args.penalty_weight, c=cfg[1])
                decode = lambda s: [(k[1], k[2]) for k, b in s.items() if b and k[0] == "x"]  # noqa: E731
            elif cfg[0] == "q2":
                bqm = build_q2_qubo(T, alpha=cfg[1], penalty_weight=args.penalty_weight)
                decode = lambda s: [(k[1], k[2]) for k, b in s.items() if b and k[0] == "x"]  # noqa: E731
            else:
                bqm = build_path_qubo(T, hops=cfg[0], mu=cfg[1], penalty_weight=args.penalty_weight)
                decode = lambda s: [(k[1], k[2]) for k, b in s.items() if b and k[0] == "x"]  # noqa: E731
            if args.solver == "tabu":
                from dwave.samplers import TabuSampler
                ss = TabuSampler().sample(bqm, num_reads=args.reads, timeout=args.tabu_ms, seed=1)
            else:
                ss = SimulatedAnnealingSampler().sample(bqm, num_reads=args.reads, num_sweeps=args.sweeps, seed=1)
            tops = {tuple(sorted(qt.repair_topology(n, decode(s), T))) for s in ss.samples()}
            valid_raw = sum(qt.is_valid_topology(n, decode(s)) for s in ss.samples())
            zs = [qt.route_min_congestion(T, a).objective for a in tops]
            if cfg is not None and cfg[0] in ("soft", 2, 3) and args.ref:
                ref = [tuple(map(int, a)) for a in __import__("json").load(open(args.ref))["arcs"]]
                e_ref = bqm.energy(aux_assignment(T, ref, bqm))
                print(f"    energy: SA best {ss.first.energy:.3f} vs reference topology w/ optimal aux {e_ref:.3f} "
                      f"(ref LP {qt.route_min_congestion(T, ref).objective:.0f})")
            print(f"  {label:<16} vars={bqm.num_variables:6d} quad={bqm.num_interactions:7d} | raw-valid {valid_raw}/{args.reads} "
                  f"| {len(tops)} topologies | LP best {min(zs):.0f} median {np.median(zs):.0f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__" and __import__("sys").argv[1:2] != ["lns"]:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fidelity_p = sub.add_parser("fidelity")
    fidelity_p.add_argument("--n", type=int, nargs="+", default=[5, 8, 12])
    fidelity_p.add_argument("--rand", type=int, default=300)
    fidelity_p.add_argument("--near", type=int, default=300)
    surropt_p = sub.add_parser("surropt")
    surropt_p.add_argument("--n", type=int, nargs="+", default=[12])
    surropt_p.add_argument("--surr", nargs="+", default=["direct", "H2", "H3", "Hinf", "H2L2", "ecmp"])
    surropt_p.add_argument("--mu", type=float, default=1.0)
    surropt_p.add_argument("--restarts", type=int, default=4)
    surropt_p.add_argument("--steps", type=int, default=3000)
    qubo_p = sub.add_parser("qubo")
    qubo_p.add_argument("--n", type=int, nargs="+", default=[12])
    qubo_p.add_argument("--hops", type=int, nargs="*", default=[2])
    qubo_p.add_argument("--mu", type=float, nargs="+", default=[0.0])
    qubo_p.add_argument("--penalty-weight", type=float, default=3.0)
    qubo_p.add_argument("--reads", type=int, default=100)
    qubo_p.add_argument("--alpha", type=float, nargs="*", default=[])
    qubo_p.add_argument("--soft", type=float, nargs="*", default=[])
    qubo_p.add_argument("--solver", choices=["sa", "tabu"], default="sa")
    qubo_p.add_argument("--tabu-ms", type=int, default=2000)
    qubo_p.add_argument("--ref", default=None, help="runs/*.json topology for energy-gap check")
    qubo_p.add_argument("--sweeps", type=int, default=2000)
    args = ap.parse_args()
    {"fidelity": cmd_fidelity, "surropt": cmd_surropt, "qubo": cmd_qubo}[args.cmd](args)


# ── surrogate as LNS proposal ranker (equal LP budget) ───────────────────
def cmd_lns(args):
    import itertools

    for n in args.n:
        T = qt.load_network_demand(n)
        print(f"\nn={n}  r={args.r} K={args.K} steps={args.steps} (LP evals = steps*K)")
        for rank in args.rank:
            finals = []
            t0 = time.time()
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                a = sorted(qt.random_topology(n, rng))
                z = qt.route_min_congestion(T, a).objective
                for _ in range(args.steps):
                    free = rng.choice(len(a), args.r, replace=False)
                    keep = [a[i] for i in range(len(a)) if i not in set(free)]
                    tails = [a[i][0] for i in free]
                    heads = [a[i][1] for i in free]
                    ks = set(keep)
                    cands = []
                    for perm in itertools.permutations(heads):
                        new = list(zip(tails, perm))
                        if any(i == j for i, j in new) or len(set(new)) < len(new) or any(e in ks for e in new):
                            continue
                        b = sorted(keep + new)
                        if b != a:
                            cands.append(b)
                    if not cands:
                        continue
                    if rank == "random":
                        order = rng.permutation(len(cands))
                    else:
                        order = np.argsort([surrogates(T, b)[rank] for b in cands], kind="stable")
                    for idx in order[: args.K]:
                        zb = qt.route_min_congestion(T, cands[idx]).objective
                        if zb <= z:
                            a, z = cands[idx], zb
                finals.append(z)
            print(f"  rank={rank:<7} final LP mean {np.mean(finals):.0f} best {min(finals):.0f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__" and __import__("sys").argv[1:2] == ["lns"]:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--n", type=int, nargs="+", default=[12])
    ap.add_argument("--rank", nargs="+", default=["random", "direct", "H2", "Hinf"])
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cmd_lns(ap.parse_args())
