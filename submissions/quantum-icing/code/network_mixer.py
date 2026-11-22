"""Degree-preserving exchange mixer for QOBLIB 08 QAOA (no penalties, every state a valid topology).

Encoding: one qubit per candidate arc (i, j). A 2-exchange (a->b),(c->d) => (a->d),(c->b) is the
operator  M_q = s-_ab s-_cd s+_ad s+_cb + h.c.   (s+ = |1><0| adds an arc, s- removes one).
M_q expands into 8 mutually commuting 4-qubit Pauli strings (X/Y with an even number of Y), so
Divi's term-by-term exponentiation applies each exp(-i beta M_q) exactly as long as the 8 strings
stay contiguous. The layer is the product over q of these factors (first-order Trotter across
non-commuting exchanges, not exp(-i beta sum_q M_q)); each factor maps valid topologies to valid
topologies, so the product does too: in/out degrees never change and a simple digraph stays simple
(s+ on an existing arc gives 0).

Cost: H2 (total flow, hops capped at 3) as an exact degree-4 HUBO on the valid subspace:
  pair (i,k): tau_ik [3 - 2 x_ik - sum_j p_j + sum_{j<j'} p_j p_j' + x_ik sum_j p_j],  p_j = x_ij x_jk
(exact because i has 2 out-arcs: at most two 2-hop paths, at most one if x_ik = 1).

    .venv/bin/python network_mixer.py verify                 # exactness checks (local statevector)
    .venv/bin/python network_mixer.py full5 --cloud qoro_gpu # full network05 QAOA: X mixer vs exchange
    .venv/bin/python network_mixer.py qlns --n 12 --r 4 --steps 15 --cloud qoro_gpu
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

import qoblib_tools as qt  # noqa: E402
from network_moves import bkv_gap, exchange_moves, lp, make_surrogates  # noqa: E402

Arc = tuple[int, int]


# ── mixer ────────────────────────────────────────────────────────────────
def _sigma(q, n_q, plus):
    # s+ = |1><0| = (X - iY)/2 ; s- = |0><1| = (X + iY)/2   (qubit q, qiskit indexing)
    return SparsePauliOp.from_sparse_list([("X", [q], 0.5), ("Y", [q], -0.5j if plus else 0.5j)], num_qubits=n_q)


def exchange_quads(qubit_of: dict[Arc, int]):
    """All 2-exchanges among candidate arcs: (ab, cd, ad, cb) qubit quadruples, each 4-set once."""
    arcs = sorted(qubit_of)
    out, seen = [], set()
    for (a, b), (c, d) in itertools.combinations(arcs, 2):
        if a == c or b == d or a == d or c == b:
            continue
        if (a, d) not in qubit_of or (c, b) not in qubit_of:
            continue
        key = frozenset([(a, b), (c, d), (a, d), (c, b)])
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(qubit_of[x] for x in ((a, b), (c, d), (a, d), (c, b))))
    return out


def exchange_mixer(qubit_of: dict[Arc, int], n_q: int | None = None) -> SparsePauliOp:
    n_q = n_q or len(qubit_of)
    quads = exchange_quads(qubit_of)
    if not quads:
        return None
    labels, coeffs = [], []
    for ab, cd, ad, cb in quads:
        op = _sigma(ab, n_q, False).compose(_sigma(cd, n_q, False)).compose(_sigma(ad, n_q, True)).compose(_sigma(cb, n_q, True))
        term = (op + op.adjoint()).simplify(atol=1e-12)
        assert np.allclose(term.coeffs.imag, 0) and len(term) == 8
        labels += list(term.paulis)
        coeffs += list(term.coeffs.real)
    return SparsePauliOp(labels, np.array(coeffs))  # no simplify(): keeps each term's 8 strings contiguous


# ── cost: exact H2 HUBO on the valid subspace ────────────────────────────
def _mul(poly_a, poly_b):
    out = {}
    for a, ca in poly_a.items():
        for b, cb in poly_b.items():
            k = tuple(sorted(set(a) | set(b)))
            out[k] = out.get(k, 0.0) + ca * cb
    return out


def h2_hubo(T, var_of: dict[Arc, object], fixed: dict[Arc, int] | None = None):
    """HUBO {tuple(vars): coeff} of H2 in normalised units; arcs in `fixed` are constants 0/1,
    arcs absent from both dicts are 0."""
    n = len(T)
    t = T / T.max()
    fixed = fixed or {}

    def x(i, j):  # polynomial of a single arc
        if (i, j) in var_of:
            return {(var_of[(i, j)],): 1.0}
        return {(): 1.0} if fixed.get((i, j), 0) else {}

    hubo: dict[tuple, float] = {}

    def add(poly, w):
        for k, c in poly.items():
            if c:
                hubo[k] = hubo.get(k, 0.0) + w * c

    for i in range(n):
        for k in range(n):
            if i == k or t[i, k] == 0:
                continue
            w = t[i, k]
            add({(): 3.0}, w)
            xik = x(i, k)
            add(xik, -2 * w)
            path_polys = [pp for j in range(n) if j not in (i, k) if (pp := _mul(x(i, j), x(j, k)))]
            for path in path_polys:
                add(path, -w)
                add(_mul(xik, path), w)
            for path_a, path_b in itertools.combinations(path_polys, 2):
                add(_mul(path_a, path_b), w)
    return {k: v for k, v in hubo.items() if abs(v) > 1e-12}


def hubo_energy(hubo, bits: dict):
    return sum(c * all(bits[v] for v in k) for k, c in hubo.items())


# ── Divi QAOA run with a given mixer ─────────────────────────────────────
def run_qaoa(hubo, n_q, mixer, init_bits, n_layers, cloud, shots, pop, iters, seed, penalty_problem=None,
             optimizer_name="mc", mps_bond=0):
    from divi.backends import JobConfig, MaestroConfig, MaestroSimulator, QoroService
    from divi.qprog.algorithms import QAOA
    from divi.qprog.initial_states import CustomPerQubitState, SuperpositionState
    from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    h = dict(hubo)
    for v in range(n_q):  # every qubit must appear in the cost so the wire count matches
        h[(v,)] = h.get((v,), 0.0) + 1e-9
    h.pop((), None)
    problem = penalty_problem or BinaryOptimizationProblem(h, hamiltonian_builder="native")
    maestro = (MaestroConfig(simulation_type="MatrixProductState", max_bond_dimension=mps_bond) if mps_bond else None)
    backend = (QoroService(job_config=JobConfig(shots=shots, simulator_cluster=cloud), maestro_config=maestro) if cloud
               else MaestroSimulator(shots=shots, maestro_config=maestro))
    qaoa = QAOA(problem, n_layers=n_layers,
             initial_state=CustomPerQubitState("".join(map(str, init_bits))) if init_bits is not None else SuperpositionState(),
             optimizer=(ScipyOptimizer(method=ScipyMethod.COBYLA) if optimizer_name == "cobyla"
                        else MonteCarloOptimizer(population_size=pop)),
             max_iterations=iters, backend=backend, seed=seed)
    if mixer is not None:
        qaoa.mixer_hamiltonian = mixer
    qaoa.run()
    return qaoa


# ── commands ─────────────────────────────────────────────────────────────
def cmd_verify(args):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import LieTrotter

    from network_pipeline import all_valid_topologies

    # 1) HUBO == H2 on every valid topology (n=5)
    T = qt.load_network_demand(5)
    arcs = qt.all_arcs(5)
    qubit_of = {a: i for i, a in enumerate(arcs)}
    hubo = h2_hubo(T, qubit_of)
    H2 = make_surrogates(T)["H2"]
    tops = list(all_valid_topologies(5))
    err = max(abs(hubo_energy(hubo, {qubit_of[a]: int(a in set(tp)) for a in arcs}) - H2(tp) / T.max()) for tp in tops)
    print(f"HUBO vs H2 on {len(tops)} valid topologies: max |diff| = {err:.2e}; HUBO terms {len(hubo)}, "
          f"max degree {max(len(k) for k in hubo)}")
    # 2) exchange mixer keeps the state inside the valid set (n=4: 12 qubits; n=5: 20 qubits)
    for n in (4, 5):
        arcs = qt.all_arcs(n)
        qubit_of = {a: i for i, a in enumerate(arcs)}
        mixer = exchange_mixer(qubit_of)
        valid = {sum(1 << qubit_of[a] for a in tp) for tp in all_valid_topologies(n)}
        start = sorted(qt.random_topology(n, np.random.default_rng(0)))
        qc = QuantumCircuit(len(arcs))
        for a in start:
            qc.x(qubit_of[a])
        for beta in (0.37, 1.1):
            qc.compose(LieTrotter().synthesize(PauliEvolutionGate(mixer, time=beta)), inplace=True)
        probs = Statevector(qc).probabilities()
        p_in = sum(probs[s] for s in valid)
        reached = sum(probs[s] > 1e-9 for s in valid)
        print(f"n={n}: {len(arcs)} qubits, {len(mixer) // 8} exchange terms ({len(mixer)} Pauli strings) | "
              f"P(valid) after 2 mixer layers = {p_in:.12f} | valid topologies reached {reached}/{len(valid)}")


def cmd_full5(args):
    """network05 end-to-end: X mixer + penalties (notebook) vs exchange mixer + penalty-free HUBO."""
    from network_pipeline import all_valid_topologies

    n = 5
    T = qt.load_network_demand(n)
    arcs = qt.all_arcs(n)
    qubit_of = {a: i for i, a in enumerate(arcs)}
    hubo = h2_hubo(T, qubit_of)
    tops = list(all_valid_topologies(n))
    lp_of = {tuple(sorted(tp)): lp(T, tp) for tp in tops}
    opt = min(lp_of.values())
    H2 = make_surrogates(T)["H2"]
    h2_opt = min(H2(tp) for tp in tops)
    start = sorted(qt.random_topology(n, np.random.default_rng(args.seed)))
    init = [int(a in set(start)) for a in arcs]
    print(f"start topology LP {lp(T, start):.0f} (rank {sum(v < lp(T, start) for v in lp_of.values()) + 1}/{len(tops)}); "
          f"LP optimum {opt:.0f}; backend {args.cloud or 'local'}")
    mixer = exchange_mixer(qubit_of)
    for n_layers in args.layers:
        t0 = time.time()
        qaoa = run_qaoa(hubo, len(arcs), mixer, init, n_layers, args.cloud, args.shots, args.pop, args.iters, args.seed)
        sols = qaoa.get_top_solutions(n=0)
        p_valid = p_opt = p_h2 = 0.0
        best = math.inf
        for s in sols:
            tp = tuple(sorted(a for a, b in zip(arcs, s.bitstring) if b == "1"))
            if tp in lp_of:
                p_valid += s.prob
                p_opt += s.prob * (lp_of[tp] == opt)
                p_h2 += s.prob * (H2(list(tp)) == h2_opt)
                best = min(best, lp_of[tp])
        print(f"  exchange mixer, {n_layers} layers: P(valid) {p_valid:.4%} | P(LP-opt) {p_opt:.3%} | P(H2-opt) {p_h2:.3%} | "
              f"distinct samples {len(sols)} | best LP sampled {best:.0f} ({time.time() - t0:.0f}s)", flush=True)
    print("  reference, notebook X mixer + penalties, 2 layers, blind: P(valid) 1.8%, P(LP-opt) 0.005%")


def cmd_reliable(args):
    """Success rate of the full pipeline on a small instance, over random starts.
    Pipeline: random valid start -> exchange-mixer QAOA (H2 HUBO) -> LP the top_k most probable
    distinct topologies -> 2-exchange LP polish. Baseline: polish only, same start, same LP budget."""
    from network_moves import polish
    from network_pipeline import all_valid_topologies

    n = args.n
    T = qt.load_network_demand(n)
    arcs = qt.all_arcs(n)
    qubit_of = {a: i for i, a in enumerate(arcs)}
    hubo = h2_hubo(T, qubit_of)
    mixer = exchange_mixer(qubit_of)
    H2 = make_surrogates(T)["H2"]
    all_tops = [sorted(tp) for tp in all_valid_topologies(n)] if n <= 6 else []
    from network_pipeline import BKV

    opt = min(lp(T, tp) for tp in all_tops) if all_tops else (args.target or BKV[n])
    h2_min = min(H2(tp) for tp in all_tops) if all_tops else None
    budget = args.top_k + args.polish_budget
    print(f"network{n:02d}: optimum {opt:.0f} | top_k {args.top_k} + polish {args.polish_budget} = {budget} LP evals "
          f"| backend {args.cloud or 'local'} | select by {args.select} | optimizer {args.optimizer} "
          f"(iters {args.iters}, pop {args.pop})", flush=True)
    for n_layers in args.layers:
        rows = []
        for seed in args.seeds:
            start = sorted(qt.random_topology(n, np.random.default_rng(1000 + seed)))
            init = [int(a in set(start)) for a in arcs]
            t0 = time.time()
            qaoa = run_qaoa(hubo, len(arcs), mixer, init, n_layers, args.cloud, args.shots, args.pop, args.iters, seed,
                            optimizer_name=args.optimizer, mps_bond=args.mps_bond)
            sols = sorted(qaoa.get_top_solutions(n=0), key=lambda s: -s.prob)
            tops, seen_t, p_ground = [], set(), 0.0
            for s in sols:
                tp = tuple(sorted(a for a, b in zip(arcs, s.bitstring) if b == "1"))
                if h2_min is not None and H2(list(tp)) == h2_min:
                    p_ground += s.prob
                if tp not in seen_t:
                    seen_t.add(tp)
                    tops.append(list(tp))
            if args.select == "energy":  # cost is classical and cheap: rank distinct samples by it
                tops.sort(key=H2)
            scored = [(lp(T, tp), tp) for tp in tops[: args.top_k]]
            z_q, best_q = min(scored)
            z_final = polish(T, best_q, rank=H2, lp_budget=args.polish_budget)[1]
            z_base = polish(T, start, rank=H2, lp_budget=budget)[1]
            in_top = any(z == opt for z, _ in scored)
            rows.append((z_q, z_final, z_base, in_top, p_ground, len(tops)))
            print(f"  {n_layers} layers seed {seed}: start {lp(T, start):.0f} | P(H2 ground) {p_ground:.2%}, "
                  f"{len(tops)} distinct | top-{args.top_k} best {z_q:.0f}"
                  f"{' (OPT)' if in_top else ''} -> polished {z_final:.0f} ({bkv_gap(n, z_final)}) | polish-only baseline {z_base:.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        hit = lambda col: sum(r[col] == opt for r in rows)  # noqa: E731
        print(f"  == {n_layers} layers: optimum found by QAOA top-{args.top_k} {sum(r[3] for r in rows)}/{len(rows)} | "
              f"pipeline {hit(1)}/{len(rows)} | polish-only baseline {hit(2)}/{len(rows)} | "
              f"mean P(H2 ground) {np.mean([r[4] for r in rows]):.2%}", flush=True)


def cmd_qlns(args):
    """Attacks 2+4: LNS where QAOA (exchange mixer on the freed arcs, H2 HUBO with fixed arcs
    substituted) proposes reassignments; same LP budget as random and classical-H2 proposers."""
    from network_lns import pick_free
    from network_design.sa_search import lp_eval

    n = args.n
    T = qt.load_network_demand(n)
    H2 = make_surrogates(T)["H2"]
    rng0 = np.random.default_rng(args.seed)
    start = sorted(qt.random_topology(n, rng0))
    if args.start_json:
        start = sorted(tuple(map(int, a)) for a in json.load(open(args.start_json))["arcs"])
    log = {"args": vars(args), "steps": []}
    print(f"n={n} r={args.r} K={args.K} steps={args.steps} start LP {lp(T, start):.0f} backend {args.cloud or 'local'}", flush=True)
    from pathlib import Path

    Path("runs").mkdir(exist_ok=True)
    for proposer in args.proposers:
        rng = np.random.default_rng(args.seed)  # freed-arc choice only (same stream for every proposer)
        rng_rank = np.random.default_rng(args.seed + 10_000)  # random ranking / padding
        cur = list(start)
        z = lp(T, cur)
        evals, p_valid_per_step, n_quantum_props = 0, [], 0
        t0 = time.time()
        for step in range(args.steps):
            z_lp, hot = lp_eval(T, cur)
            free = pick_free(cur, hot, args.r, rng)
            kept = [a for i, a in enumerate(cur) if i not in set(free)]
            tails = sorted({cur[i][0] for i in free})
            heads = sorted({cur[i][1] for i in free})
            cand = [(a, b) for a in tails for b in heads if a != b and (a, b) not in set(kept)]
            qubit_of = {a: i for i, a in enumerate(cand)}
            # enumerate the valid reassignments (used by random/classical proposers and as ground truth)
            freed = [cur[i] for i in free]
            opts = set()
            for perm in itertools.permutations([b for _, b in freed]):
                new = list(zip([a for a, _ in freed], perm))
                if all(a != b for a, b in new) and len(set(new)) == len(new) and not set(new) & set(kept):
                    opts.add(tuple(sorted(kept + new)))
            opts.discard(tuple(cur))
            opts = [list(o) for o in opts]
            if not opts:
                continue
            if proposer == "random":
                ranked = [opts[i] for i in rng_rank.permutation(len(opts))]
            elif proposer == "classical":
                ranked = sorted(opts, key=H2)
            else:  # quantum
                fixed = {a: 1 for a in kept}
                hubo = h2_hubo(T, qubit_of, fixed)
                init = [int(a in set(freed)) for a in cand]
                mixer = exchange_mixer(qubit_of)
                if mixer is None:
                    continue
                qaoa = run_qaoa(hubo, len(cand), mixer, init, args.layers, args.cloud, args.shots, args.pop, args.iters, args.seed + step)
                sols = sorted(qaoa.get_top_solutions(n=0), key=lambda s: -s.prob)
                ranked, seen_t, p_valid = [], set(), 0.0
                for s in sols:
                    new = [a for a, b in zip(cand, s.bitstring) if b == "1"]
                    tp = tuple(sorted(kept + new))
                    if qt.is_valid_topology(n, tp):
                        p_valid += s.prob
                        if tp != tuple(cur) and tp not in seen_t:
                            seen_t.add(tp)
                            ranked.append(list(tp))
                p_valid_per_step.append(p_valid)
                n_quantum_props += min(len(ranked), args.K)
                # equal LP budget: pad with random options (never classical) if QAOA gave < K new ones
                rest = [opts[i] for i in rng_rank.permutation(len(opts)) if tuple(opts[i]) not in seen_t]
                ranked += rest[: max(0, args.K - len(ranked))]
            for c in ranked[: args.K]:
                zz = lp(T, c)
                evals += 1
                if zz <= z:
                    cur, z = c, zz
            log["steps"].append({"proposer": proposer, "step": step, "z": z, "n_opts": len(opts), "qubits": len(cand)})
        extra = (f" | P(valid) min {min(p_valid_per_step):.4f} | {n_quantum_props} of {evals} LP evals were QAOA proposals"
                 if p_valid_per_step else "")
        print(f"  {proposer:<9} final LP {z:.0f} ({bkv_gap(n, z)}) after {evals} LP evals ({time.time() - t0:.0f}s){extra}", flush=True)
        if args.final_polish:
            from network_moves import polish

            z_before = z
            cur, z, used = polish(T, cur, rank=H2, lp_budget=args.final_polish)
            print(f"  {proposer:<9} + final 2-exchange polish ({used} LP evals): {z_before:.0f} -> {z:.0f} "
                  f"({bkv_gap(n, z)})", flush=True)
        log[proposer] = {"z": z, "arcs": cur}
    json.dump(log, open(f"runs/mixer_qlns_n{n}_r{args.r}_s{args.seed}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    f = sub.add_parser("full5")
    f.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    rel_p = sub.add_parser("reliable")
    rel_p.add_argument("--n", type=int, default=5)
    rel_p.add_argument("--layers", type=int, nargs="+", default=[3, 4])
    rel_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    rel_p.add_argument("--top-k", type=int, default=10)
    rel_p.add_argument("--polish-budget", type=int, default=20)
    rel_p.add_argument("--target", type=float, default=0)
    rel_p.add_argument("--select", choices=["prob", "energy"], default="prob")
    rel_p.add_argument("--optimizer", choices=["mc", "cobyla"], default="mc")
    qlns_p = sub.add_parser("qlns")
    qlns_p.add_argument("--n", type=int, default=12)
    qlns_p.add_argument("--r", type=int, default=4)
    qlns_p.add_argument("--K", type=int, default=3)
    qlns_p.add_argument("--steps", type=int, default=15)
    qlns_p.add_argument("--layers", type=int, default=2)
    qlns_p.add_argument("--proposers", nargs="+", default=["random", "classical", "quantum"])
    qlns_p.add_argument("--start-json", default=None)
    qlns_p.add_argument("--final-polish", type=int, default=0, help="LP evals for a closing 2-exchange polish")
    for sub_parser in (f, rel_p, qlns_p):
        sub_parser.add_argument("--cloud", default=None)
        sub_parser.add_argument("--shots", type=int, default=5000)
        sub_parser.add_argument("--pop", type=int, default=20)
        sub_parser.add_argument("--iters", type=int, default=5)
        sub_parser.add_argument("--seed", type=int, default=0)
        sub_parser.add_argument("--mps-bond", type=int, default=0, help="0 = statevector; >0 = MPS bond dimension")
    a = ap.parse_args()
    {"verify": cmd_verify, "full5": cmd_full5, "reliable": cmd_reliable, "qlns": cmd_qlns}[a.cmd](a)
