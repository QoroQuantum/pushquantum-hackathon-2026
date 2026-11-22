"""Quantum-proposer LNS for network design, on a block QUBO learned from a cheap exact bound.

One step on the current topology:

1. free r arcs (half on LP bottlenecks): their tails stay and their heads are permuted, so every
   node keeps 2 in / 2 out whatever the permutation;
2. learn the block: score S random valid rewirings with the LP weak-duality bound
   sum_ij d_ij dist_w(i,j) / sum_a w_a (incumbent duals, shortest paths, no LP solve) and fit a
   QUBO on the r*r assignment bits, linear + pairwise, by ridge regression. That QUBO is the
   compressed model the sampler sees;
3. three proposers on the same block, each producing candidate rewirings:
     quantum  Divi PCE (poly encoding, about sqrt(2)*r qubits) or QAOA (W-state + XY mixer,
              r*r qubits), local simulator or Qoro cloud;
     sa       simulated annealing on the SAME learned QUBO, in permutation space;
     random   uniform valid rewirings;
   every bitstring is decoded to the nearest permutation (Hungarian), so every proposal is a
   valid topology;
4. each proposer's candidates are ranked by the exact bound and its top K go to the exact LP:
   the same K for all three, so they spend the same LP budget;
5. the walk follows ``--follow`` (default: the quantum proposer) and accepts a rewiring that is
   not worse; the other proposers are logged on the same block as controls. Every improvement
   is banked through the official checker (``network_pipeline.bank``, into ``solutions/``).

State is saved after every step under ``runs/goat/lns/<tag>/``, so a killed run resumes.

    venv/bin/python TheGOAT_Testing/goat_lns.py --tag pce_r10 --solver pce --r 10 --steps 40
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from block_fidelity import flow_bound  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.sa_search import lp_eval  # noqa: E402
from network_lns import load_start, pick_free  # noqa: E402

OUT = ROOT / "runs" / "goat" / "lns"


# ── the block ──────────────────────────────────────────────────────────────────────────


class Block:
    """r freed arcs of a topology: tails fixed, heads permuted among the r slots."""

    def __init__(self, arcs, free, n):
        self.arcs, self.free, self.n, self.r = list(arcs), list(free), n, len(free)
        self.tails = [arcs[f][0] for f in free]
        self.slots = [arcs[f][1] for f in free]
        fixed = {a for i, a in enumerate(arcs) if i not in set(free)}
        # (f, s) is forbidden when arc f -> slot s is a self-loop or duplicates a fixed arc
        self.forbidden = np.array([[self.tails[f] == self.slots[s] or (self.tails[f], self.slots[s]) in fixed
                                    for s in range(self.r)] for f in range(self.r)])

    def topology(self, perm):
        new = list(self.arcs)
        for f, s in enumerate(perm):
            new[self.free[f]] = (self.tails[f], self.slots[s])
        if len(set(new)) != len(new) or not qt.is_valid_topology(self.n, new):
            return None
        return new

    def decode(self, scores):
        """Nearest permutation to an r*r score matrix (Hungarian); forbidden cells excluded."""
        M = np.asarray(scores, float).reshape(self.r, self.r) - 1e6 * self.forbidden
        _, cols = linear_sum_assignment(-M)
        return tuple(int(c) for c in cols)

    def unique_valid(self, perms):
        """Valid perms, one per distinct topology (repeated heads make some perms coincide)."""
        out, seen = [], set()
        for p in perms:
            t = self.topology(p)
            if t is None:
                continue
            key = tuple(sorted(t))
            if key not in seen:
                seen.add(key)
                out.append((p, t))
        return out

    def random_perms(self, k, rng):
        return self.unique_valid(tuple(int(x) for x in rng.permutation(self.r)) for _ in range(20 * k))[:k]


# ── learned QUBO ───────────────────────────────────────────────────────────────────────


def compatible_pairs(r):
    """Assignment-bit pairs in different rows and columns: the only pairs a permutation sets."""
    return np.array([(i, j) for i in range(r * r) for j in range(i + 1, r * r)
                     if i // r != j // r and i % r != j % r])


def learn_qubo(perms, y, r, rng, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Ridge fit of y (standardised) on assignment bits and their compatible pairs.

    Returns (h, J, held-out Spearman of the chosen fit): h has r*r entries, J is upper
    triangular r*r x r*r. The penalty is chosen on a 25% hold-out, then refitted on all data.
    """
    X = np.zeros((len(perms), r * r))
    for k, p in enumerate(perms):
        X[k, [f * r + s for f, s in enumerate(p)]] = 1.0
    pairs = compatible_pairs(r)
    F = np.hstack([X, X[:, pairs[:, 0]] * X[:, pairs[:, 1]]])
    ys = (y - y.mean()) / (y.std() + 1e-12)

    def fits(A, b, alphas):
        """Ridge coefficients for every alpha, in dual (kernel) form: there are more features
        (r^2 + pairs, 4150 at r = 10) than samples, so one eigendecomposition of the n x n
        Gram matrix serves all penalties."""
        Ac = A - A.mean(axis=0)
        lam, U = np.linalg.eigh(Ac @ Ac.T)
        Uy = U.T @ (b - b.mean())
        return [Ac.T @ (U @ (Uy / (np.maximum(lam, 0.0) + alpha))) for alpha in alphas]

    idx = rng.permutation(len(ys))
    cut = max(len(ys) // 4, 10)
    te, tr = idx[:cut], idx[cut:]
    best = (-np.inf, alphas[0])
    for alpha, beta in zip(alphas, fits(F[tr], ys[tr], alphas)):
        rho = spearmanr(F[te] @ beta, ys[te]).statistic
        best = max(best, (float(np.nan_to_num(rho, nan=-1.0)), alpha))
    beta = fits(F, ys, [best[1]])[0]
    h = beta[: r * r].copy()
    J = np.zeros((r * r, r * r))
    J[pairs[:, 0], pairs[:, 1]] = beta[r * r:]
    return h, J, best[0]


def qubo_matrix(h, J, block, penalty):
    """Upper-triangular QUBO: learned terms + one-hot rows and columns + forbidden cells."""
    r = block.r
    Q = J.copy()
    Q[np.diag_indices(r * r)] += h
    P = penalty * max(np.abs(h).max(), 1e-9)
    for f in range(r):
        for s in range(r):
            i = f * r + s
            Q[i, i] += -2 * P + (3 * P if block.forbidden[f, s] else 0.0)   # row + column (x^2 = x)
            for s2 in range(s + 1, r):
                Q[i, f * r + s2] += 2 * P
            for f2 in range(f + 1, r):
                Q[i, f2 * r + s] += 2 * P
    return Q


def perm_energy(p, h, Jsym, r):
    idx = [f * r + s for f, s in enumerate(p)]
    return float(h[idx].sum() + 0.5 * Jsym[np.ix_(idx, idx)].sum())


# ── proposers ──────────────────────────────────────────────────────────────────────────


def local_search(block, perms, h, J):
    """Steepest descent by swaps on the learned QUBO from each start (the post-processing PCE
    papers always apply). Same routine for every start set, so 'quantum+ls' vs 'random+ls'
    isolates what the quantum starts contribute."""
    r = block.r
    Jsym = J + J.T
    hh = h + 1e3 * block.forbidden.ravel()
    out = []
    for p in perms:
        p = list(p)
        e = perm_energy(p, hh, Jsym, r)
        while True:
            best = (e, None)
            for a in range(r):
                for b in range(a + 1, r):
                    p[a], p[b] = p[b], p[a]
                    e2 = perm_energy(p, hh, Jsym, r)
                    if e2 < best[0] - 1e-12:
                        best = (e2, (a, b))
                    p[a], p[b] = p[b], p[a]
            if best[1] is None:
                break
            a, b = best[1]
            p[a], p[b] = p[b], p[a]
            e = best[0]
        out.append(tuple(int(x) for x in p))
    return block.unique_valid(out)


def propose_sa(block, h, J, k, rng, sweeps=400, restarts=8):
    """Simulated annealing on the learned QUBO, moving by swaps inside the permutation."""
    r = block.r
    Jsym = J + J.T
    hh = h + 1e3 * block.forbidden.ravel()
    found = {}
    for _ in range(restarts):
        p = list(rng.permutation(r))
        e = perm_energy(p, hh, Jsym, r)
        T0 = max(np.abs(h).max(), 1e-6)
        for t in range(sweeps * r):
            T = T0 * (1 - t / (sweeps * r)) + 1e-6
            a, b = rng.choice(r, 2, replace=False)
            p[a], p[b] = p[b], p[a]
            e2 = perm_energy(p, hh, Jsym, r)
            if e2 <= e or rng.random() < math.exp(-(e2 - e) / T):
                e = e2
                found[tuple(int(x) for x in p)] = e
            else:
                p[a], p[b] = p[b], p[a]
    ranked = sorted(found, key=found.get)
    return block.unique_valid(ranked)[:k]


OPTIMIZERS = ("mc", "cobyla", "lbfgs", "qng", "qnspsa", "spsa")


def make_optimizer(name, pop=20):
    """Classical optimizer of the circuit parameters (all from divi 0.16.2)."""
    from divi.qprog import optimizers as o

    if name == "mc":
        return o.MonteCarloOptimizer(population_size=pop)
    if name in ("cobyla", "lbfgs"):
        return o.ScipyOptimizer(method={"cobyla": o.ScipyMethod.COBYLA, "lbfgs": o.ScipyMethod.L_BFGS_B}[name])
    if name == "qng":
        return o.QNGOptimizer()               # quantum natural gradient (Fubini-Study metric)
    if name == "qnspsa":
        return o.QNSPSAOptimizer()            # natural-gradient SPSA (Gacon et al.)
    if name == "spsa":
        return o.SPSAOptimizer()
    raise ValueError(f"unknown optimizer {name!r}")


def perturbed_assignments(block, scores, k, rng, temp=0.5):
    """The max-score permutation (Hungarian), then near-max ones from Gumbel-perturbed scores."""
    scores = np.asarray(scores, float).ravel()
    scale = temp * (scores.std() + 1e-12)
    perms = [block.decode(scores)]
    perms += [block.decode(scores + scale * rng.gumbel(size=scores.size)) for _ in range(4 * k)]
    return block.unique_valid(perms)[:k]


_WARM: dict = {}
"""Best circuit parameters of the previous block, per (solver, r, layers, optimizer)."""


def warm_params(prog, args, rng, r):
    """Angle warm start: the previous block's best parameters (blocks of the same size give
    circuits of the same shape), tiled over the optimizer's parameter sets, with small noise on
    every set but the first. None (a fresh random start) when disabled or on the first block."""
    prev = _WARM.get((args.solver, r, args.layers, args.optimizer))
    if not getattr(args, "warm_angles", 0) or prev is None:
        return None
    shape = prog.get_expected_param_shape()
    if prev.size != shape[-1]:
        return None
    init = np.tile(prev, (shape[0], 1))
    init[1:] += 0.1 * rng.standard_normal(init[1:].shape)
    return init


def propose_quantum(block, Q, h, k, args, seed, rng):
    """Divi sampler on the block QUBO; bitstrings -> nearest permutations (Hungarian).

    For PCE the measured bitstrings are a poor read-out: with the poly encoding each of the r*r
    cells is the parity of one or two qubits, so the cells are rigid functions of ~sqrt(2)*r
    bits that know nothing of rows and columns. The compressed information PCE optimises is in
    the per-cell expectation values <P_i>, so those are also decoded, as soft assignment scores
    (Hungarian, plus Gumbel-perturbed near-optimal assignments).
    """
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.algorithms import PCE, QAOA, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    r = block.r
    backend = (QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster=args.cloud, tag="goat-lns"))
               if args.cloud else MaestroSimulator(shots=args.shots))
    common = dict(n_layers=args.layers, optimizer=make_optimizer(args.optimizer, args.pop),
                  max_iterations=args.iters, backend=backend, seed=seed)
    prob = BinaryOptimizationProblem(Q)
    if args.solver == "qaoa":
        from divi.hamiltonians import xy_mixer
        from divi.qprog.initial_states import WState
        prog = QAOA(prob, initial_state=WState(r, r), **common)
        prog.mixer_hamiltonian = xy_mixer(
            [(f * r + s, f * r + s2) for f in range(r) for s in range(r) for s2 in range(s + 1, r)],
            n_qubits=r * r)
        prog.run()
        sols = prog.get_top_solutions(n=args.top)
    elif args.solver == "qaoa_swap":
        # Warm start: the incumbent (identity permutation) as a basis state, and a mixer of
        # 2-exchanges between freed arcs, so every amplitude stays on a permutation.
        from divi.qprog.initial_states import CustomPerQubitState
        start = "".join("1" if q // r == q % r else "0" for q in range(r * r))
        prog = QAOA(prob, initial_state=CustomPerQubitState(start), **common)
        prog.mixer_hamiltonian = swap_mixer(r)
        prog.run()
        sols = prog.get_top_solutions(n=args.top)
    else:
        prog = PCE(prob, encoding=PolyEncoding(), objective=SoftObjective(alpha=args.alpha), **common)
        prog.run(initial_params=warm_params(prog, args, rng, r))
        _WARM[(args.solver, r, args.layers, args.optimizer)] = np.array(prog.best_params, float)
        sols = prog.get_top_solutions(n=args.top, sort_by="energy")
    prior = -h.reshape(r, r)
    prior = 1e-3 * prior / (np.abs(prior).max() + 1e-12)       # breaks ties toward the model
    bits = [np.array([int(c) for c in s.bitstring], float) for s in sols]
    perms = [block.decode(b + prior.ravel()) for b in bits]
    soft = []
    if args.solver == "pce" and args.soft:
        # x_i = 1 where the parity is even, i.e. where <P_i> = +1: <P_i> is the soft score.
        expect = prog._variable_expectations(np.atleast_2d(prog.best_params))[0]
        soft = [p for p, _ in perturbed_assignments(block, expect, k // 2, rng)]
    perms = soft + perms                       # half soft read-out at most, sampled bitstrings after
    # share of raw samples that already are permutations (before any repair)
    is_perm = [b.size == r * r and (b.reshape(r, r).sum(0) == 1).all() and (b.reshape(r, r).sum(1) == 1).all()
               for b in bits]
    n_qubits = int(getattr(prog, "n_qubits", r * r))
    info = {"qubits": n_qubits, "circuits": int(prog.total_circuit_count),
            "raw_perm_share": float(np.mean(is_perm)) if is_perm else 0.0, "soft_cands": len(soft)}
    return block.unique_valid(perms)[:k], info


def swap_mixer(r):
    """Sum over freed-arc pairs f<g and slot pairs s<t of the exchange
    (f->s, g->t) <-> (f->t, g->s): s-_fs s-_gt s+_ft s+_gs + h.c., 8 Pauli strings each.
    Never simplify() the sum: each term's strings must stay together (as in network_mixer)."""
    from network_mixer import _sigma
    from qiskit.quantum_info import SparsePauliOp

    n_q = r * r
    labels, coeffs = [], []
    for f in range(r):
        for g in range(f + 1, r):
            for s in range(r):
                for t in range(s + 1, r):
                    fs, gt, ft, gs = f * r + s, g * r + t, f * r + t, g * r + s
                    op = (_sigma(fs, n_q, False).compose(_sigma(gt, n_q, False))
                          .compose(_sigma(ft, n_q, True)).compose(_sigma(gs, n_q, True)))
                    term = (op + op.adjoint()).simplify(atol=1e-12)
                    labels += list(term.paulis)
                    coeffs += list(term.coeffs.real)
    return SparsePauliOp(labels, np.array(coeffs))


# ── one step ───────────────────────────────────────────────────────────────────────────


def lp_top(demand, cands, bound_w, K):
    """Rank candidates by the exact bound, LP the best K. Returns (best z, its arcs, n LPs)."""
    if not cands:
        return math.inf, None, 0
    tops = [t for _, t in cands]
    lb = flow_bound(demand, tops, [bound_w] * len(tops))
    best = (math.inf, None)
    order = np.argsort(lb)[:K]
    for i in order:
        z, _ = lp_eval(demand, tops[i])
        if z < best[0]:
            best = (z, tops[i])
    return best[0], best[1], len(order)


def step(demand, cur, z_cur, hot, args, rng, seed):
    t0 = time.time()
    free = pick_free(cur, hot, args.r, rng)
    block = Block(cur, free, args.n)
    if args.weights == "duals":
        _, duals = npl.score_topology(cur, demand)
        dual_of = dict(zip(sorted(cur), duals))
        w = [dual_of[a] for a in cur]                  # weights follow the arc index (its tail)
    else:
        # Unit weights: demand-weighted hop count. Measured on network24 blocks (block_fidelity):
        # Spearman 0.85-0.89 against the LP, versus 0.19-0.57 with carried-over duals.
        w = np.ones(len(cur))

    train = block.random_perms(args.samples, rng)
    y = flow_bound(demand, [t for _, t in train], [w] * len(train))
    h, J, fit_rho = learn_qubo([p for p, _ in train], y, args.r, rng)
    Q = qubo_matrix(h, J, block, args.penalty)
    t_learn = time.time() - t0

    rec = {"free": free, "n_train": len(train), "fit_rho_vs_bound": fit_rho, "t_learn": round(t_learn, 1)}
    props = {}
    t = time.time()
    props["quantum"], info = propose_quantum(block, Q, h, args.cands, args, seed, rng)
    rec |= info
    rec["t_quantum"] = round(time.time() - t, 1)
    props["sa"] = propose_sa(block, h, J, args.cands, rng)
    # classical twin of the soft read-out: the same perturbed Hungarian on the linear terms
    props["hungarian"] = perturbed_assignments(block, -h, args.cands, rng)
    props["random"] = block.random_perms(args.cands, rng)
    # The hardest classical control: the very samples the QUBO was learned from, best by bound.
    props["sampled"] = [train[i] for i in np.argsort(y)[: args.cands]]
    if args.ls:
        props["quantum+ls"] = local_search(block, [p for p, _ in props["quantum"]], h, J)
        props["random+ls"] = local_search(block, [p for p, _ in props["random"]], h, J)

    best = {}
    for name, cands in props.items():
        z, arcs, n_lp = lp_top(demand, cands, w, args.K)
        best[name] = (z, arcs)
        rec[name] = {"z": z, "distinct": len(cands), "lp": n_lp}
    rec["t_step"] = round(time.time() - t0, 1)
    return rec, best


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True, help="run name: state goes in runs/goat/lns/<tag>/")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--start", default="solutions", choices=["solutions", "runs", "published", "greedy"])
    p.add_argument("--solver", default="pce", choices=["pce", "qaoa", "qaoa_swap"],
                   help="pce: poly-encoded PCE; qaoa: W-state + XY mixer per arc; "
                        "qaoa_swap: incumbent start + 2-exchange mixer (warm, always a permutation)")
    p.add_argument("--r", type=int, default=10)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--samples", type=int, default=1500, help="bound-scored rewirings to learn from")
    p.add_argument("--cands", type=int, default=60, help="distinct candidates kept per proposer")
    p.add_argument("--K", type=int, default=10, help="LP solves per proposer per step")
    p.add_argument("--penalty", type=float, default=2.0)
    p.add_argument("--weights", default="hops", choices=["hops", "duals"],
                   help="bound weights: unit (hop count, measured strong) or incumbent duals (measured weak)")
    p.add_argument("--follow", default="quantum",
                   choices=["quantum", "quantum+ls", "sa", "hungarian", "random", "random+ls", "sampled", "best"])
    p.add_argument("--ls", type=int, default=1, help="also log quantum+ls and random+ls (1/0)")
    p.add_argument("--shots", type=int, default=5000)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--optimizer", default="mc", choices=OPTIMIZERS,
                   help="mc: Monte Carlo (pop x iters circuits); cobyla / lbfgs: scipy; qng: quantum natural "
                        "gradient; qnspsa / spsa: stochastic. iters = max iterations")
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--top", type=int, default=200, help="samples read back from the quantum program")
    p.add_argument("--alpha", type=float, default=4.0, help="PCE SoftObjective sharpness")
    p.add_argument("--soft", type=int, default=1, help="PCE: also decode the per-cell expectation values (1/0)")
    p.add_argument("--warm-angles", type=int, default=0,
                   help="PCE: start each block from the previous block's best parameters (1/0)")
    p.add_argument("--start-file", default=None, help="start from this .sol instead of --start")
    p.add_argument("--cloud", default=None, help="Qoro cluster (qoro_maestro | qoro_cpu | qoro_gpu); default local")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    run = OUT / a.tag
    run.mkdir(parents=True, exist_ok=True)
    state_path, log_path = run / "state.json", run / "steps.jsonl"
    demand = qt.load_network_demand(a.n)
    if state_path.exists():
        st = json.loads(state_path.read_text())
        cur, done, best_z = [tuple(x) for x in st["arcs"]], st["step"], st["best_z"]
        print(f"resuming {a.tag} at step {done}", flush=True)
    elif a.start_file:
        _, arcs, _ = qt.read_network_solution(Path(a.start_file))
        cur, done, best_z = [tuple(x) for x in arcs], 0, None
    else:
        cur, done = load_start(a.n, a.start, 0), 0
        best_z = None
    z_cur, hot = lp_eval(demand, cur)
    best_z = z_cur if best_z is None else best_z
    (run / "args.json").write_text(json.dumps(vars(a)))
    service = None
    if a.cloud:
        from divi.backends import QoroService
        service = QoroService()
        print(f"credits at start: {service.get_credit_balance()['balance']}", flush=True)
    print(f"start {math.ceil(z_cur - 1e-6)}  r={a.r}  solver={a.solver}  cloud={a.cloud}", flush=True)

    for k in range(done + 1, a.steps + 1):
        rng = np.random.default_rng([a.seed, k])
        rec, best = step(demand, cur, z_cur, hot, a, rng, a.seed * 1000 + k)
        lead = min(best, key=lambda m: best[m][0]) if a.follow == "best" else a.follow
        z_new, arcs_new = best[lead]
        moved = ""
        if arcs_new is not None and z_new <= z_cur + 1e-6:
            moved = "down" if z_new < z_cur - 1e-6 else "sideways"
            cur = arcs_new
            z_cur, hot = lp_eval(demand, cur)
            if z_cur < best_z - 1e-6:
                best_z = z_cur
                moved = "NEW BEST " + npl.bank(a.n, demand, cur, f"goat_lns:{a.tag}")[2]
        rec |= {"step": k, "z_cur": z_cur, "moved": moved, "lead": lead}
        if service is not None:
            rec["credits"] = float(service.get_credit_balance()["balance"])
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        state_path.write_text(json.dumps({"arcs": cur, "step": k, "best_z": best_z}))
        print(f"step {k:3d} {rec['qubits']}q perm {rec['raw_perm_share']:.0%} {rec['t_step']:6.1f}s "
              f"fit {rec['fit_rho_vs_bound']:+.2f} | "
              + "  ".join(f"{m} {rec[m]['z']:9.0f}" for m in
                          ("quantum", "quantum+ls", "random+ls", "sa", "hungarian", "sampled", "random") if m in rec)
              + f" | cur {math.ceil(z_cur - 1e-6)} {moved}"
              + (f" | credits {rec['credits']:.1f}" if "credits" in rec else ""), flush=True)


if __name__ == "__main__":
    main()
