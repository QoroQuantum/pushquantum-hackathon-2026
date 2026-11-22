"""Network design: quantum-proposer LNS. Divi proposes batches of rewires, the exact LP decides.

One step:
  1. free r arcs of the current topology (biased to LP bottleneck arcs); tails stay,
     their r heads become r "slots" to re-assign (a permutation).
  2. build a sub-QUBO on x[f,s] (arc f takes slot s), r*r binaries:
       - linear: marginal value of arc (t_f, head_s), measured by exact LP probes
         (absent arc: z drop when added; present arc: z rise when removed),
       - quadratic: two-hop reward, new arcs f->s then g->s' with tail_g == head_s
         carry demand[tail_f, head_s'] in two hops,
       - penalties: one slot per arc, one arc per slot, no self-loop / duplicate arc.
  3. Divi samples it: QAOA with W-state + XY mixer (one-hot per arc kept by the
     circuit) for small r, PCE (poly encoding, ~sqrt(2 r^2) qubits) for larger r.
  4. top-K samples -> nearest permutation -> exact LP; best accepted if not worse.
     Same number of random permutations scored as the baseline.

    python network_qlns.py --n 24 --start solutions --solver pce --r 8 --steps 30
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
import qoblib_tools as qt  # noqa: E402
from network_lns import load_start, pick_free  # noqa: E402
from network_design.sa_search import RUN_DIR, lp_eval  # noqa: E402


# ── sub-QUBO ─────────────────────────────────────────────────────────────
def build_subqubo(demand, arcs, free, z0, two_hop=0.5):
    r = len(free)
    tails = [arcs[f][0] for f in free]
    slots = [arcs[f][1] for f in free]
    fixed = {a for i, a in enumerate(arcs) if i not in set(free)}
    current = set(arcs)
    idx = lambda f, s: f * r + s  # noqa: E731

    # marginal value of each distinct candidate arc, by exact LP probes
    value = {}
    for t in set(tails):
        for h in set(slots):
            if t == h or (t, h) in fixed or (t, h) in value:
                continue
            if (t, h) in current:
                z, _ = lp_eval(demand, [a for a in arcs if a != (t, h)])
                value[t, h] = (min(z, 2 * z0) - z0)  # removal hurts -> keep it
            else:
                z, _ = lp_eval(demand, list(arcs) + [(t, h)])
                value[t, h] = z0 - z
    vmax = max([abs(v) for v in value.values()] + [1e-9])
    Q = np.zeros((r * r, r * r))
    P = 3.0  # penalty, in units of vmax-normalised values (max |linear| = 1)
    bad = []
    for f in range(r):
        for s in range(r):
            a = (tails[f], slots[s])
            if a in value:
                Q[idx(f, s), idx(f, s)] -= value[a] / vmax
            else:
                Q[idx(f, s), idx(f, s)] += P  # self-loop or duplicates a fixed arc
                bad.append(idx(f, s))
    # two-hop reward between freed arcs
    dmax = demand.max()
    for f in range(r):
        for s in range(r):
            for g in range(r):
                if g == f or tails[g] != slots[s]:
                    continue
                for s2 in range(r):
                    if s2 != s:
                        i, j = sorted((idx(f, s), idx(g, s2)))
                        Q[i, j] -= two_hop * demand[tails[f], slots[s2]] / dmax
    # one-hot: each arc one slot (rows), each slot one arc (cols): P*(sum-1)^2
    for f in range(r):
        for s in range(r):
            Q[idx(f, s), idx(f, s)] -= 2 * P  # -P from the row, -P from the column
            for s2 in range(s + 1, r):
                Q[idx(f, s), idx(f, s2)] += 2 * P
            for f2 in range(f + 1, r):
                Q[idx(f, s), idx(f2, s)] += 2 * P
    # two freed arcs with the same tail must not get the same head
    for f in range(r):
        for g in range(f + 1, r):
            if tails[f] != tails[g]:
                continue
            for s in range(r):
                for s2 in range(r):
                    if s != s2 and slots[s] == slots[s2]:
                        i, j = sorted((idx(f, s), idx(g, s2)))
                        Q[i, j] += P
    return Q, tails, slots


def to_arcs(bits, arcs, free, tails, slots, prior):
    """Bits -> nearest permutation (Hungarian, ties by QUBO prior) -> topology or None."""
    r = len(free)
    M = np.asarray(bits, float).reshape(r, r) + 1e-3 * prior
    rows, cols = linear_sum_assignment(-M)
    new = list(arcs)
    for f, s in zip(rows, cols):
        new[free[f]] = (tails[f], slots[s])
    n = max(max(a) for a in arcs) + 1
    return new if len(set(new)) == len(new) and qt.is_valid_topology(n, new) else None


# ── Divi samplers ────────────────────────────────────────────────────────
def make_backend(shots, cloud):
    """Local Maestro simulator, or Qoro cloud (USE_CLOUD=True in the notebooks; uses credits)."""
    from divi.backends import JobConfig, MaestroSimulator, QoroService
    if cloud:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro",
                                                tag="network24-qlns"))
    return MaestroSimulator(shots=shots)


def divi_sample(Q, r, solver, k, seed, shots, layers, pop, iters, cloud=False):
    from divi.qprog.algorithms import PCE, QAOA, PolyEncoding, SoftObjective
    from divi.qprog.initial_states import WState
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    prob = BinaryOptimizationProblem(Q)
    common = dict(n_layers=layers, optimizer=MonteCarloOptimizer(population_size=pop),
                  max_iterations=iters, backend=make_backend(shots, cloud), seed=seed)
    if solver == "qaoa":
        from divi.hamiltonians import xy_mixer
        prog = QAOA(prob, initial_state=WState(r, r), **common)
        # XY mixer inside each arc's block keeps exactly one slot per freed arc
        prog.mixer_hamiltonian = xy_mixer(
            [(f * r + s, f * r + s2) for f in range(r) for s in range(r) for s2 in range(s + 1, r)],
            n_qubits=r * r)
    else:
        prog = PCE(prob, encoding=PolyEncoding(), objective=SoftObjective(alpha=4.0), **common)
    prog.run()
    sols = prog.get_top_solutions(n=k, sort_by="energy") if solver == "pce" else prog.get_top_solutions(n=k)
    return [[int(c) for c in s.bitstring] for s in sols], getattr(prog, "n_qubits", r * r), prog.total_circuit_count


# ── driver ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--start", default="solutions", choices=["greedy", "solutions", "runs", "published"])
    p.add_argument("--solver", default="pce", choices=["pce", "qaoa"])
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--k", type=int, default=40, help="proposals per step sent to the LP")
    p.add_argument("--shots", type=int, default=5000)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.0, help="annealing temperature, fraction of start z")
    p.add_argument("--cloud", action="store_true", help="run circuits on Qoro cloud (credits)")
    p.add_argument("--min-credits", type=float, default=200, help="stop when balance drops below")
    a = p.parse_args()
    service = None
    if a.cloud:
        from divi.backends import QoroService
        service = QoroService()
        print("credits at start:", service.get_credit_balance()["balance"], flush=True)

    rng = np.random.default_rng(a.seed)
    demand = qt.load_network_demand(a.n)
    cur = load_start(a.n, a.start, 0)
    cur_e, hot = lp_eval(demand, cur)
    start_e = cur_e
    best_e, best_arcs = cur_e, list(cur)
    out = RUN_DIR / f"network{a.n:02d}" / f"q{a.solver}{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = []
    print(f"start {math.ceil(cur_e - 1e-6)} ({a.start}); solver {a.solver}, r={a.r}", flush=True)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        free = pick_free(cur, hot, a.r, rng)
        Q, tails, slots = build_subqubo(demand, cur, free, cur_e)
        prior = -np.diag(Q).reshape(a.r, a.r)
        samples, nq, ncirc = divi_sample(Q, a.r, a.solver, a.k, a.seed + step, a.shots, a.layers,
                                         a.pop, a.iters, cloud=a.cloud)
        tq = time.time() - t0

        def best_of(cands):
            seen, best = set(), (math.inf, None, None)
            for c in cands:
                if c is None or tuple(c) in seen:
                    continue
                seen.add(tuple(c))
                e, h = lp_eval(demand, c)
                if e < best[0]:
                    best = (e, c, h)
            return best, len(seen)

        (qe, qc, qh), nq_dist = best_of(to_arcs(b, cur, free, tails, slots, prior) for b in samples)
        # fair baseline: as many distinct *valid* random permutations as quantum gave
        rand, seen = [], set()
        for _ in range(5000):
            if len(rand) >= max(nq_dist, 1):
                break
            c = to_arcs(np.eye(a.r)[rng.permutation(a.r)].ravel(), cur, free, tails, slots, prior)
            if c is not None and tuple(c) not in seen:
                seen.add(tuple(c))
                rand.append(c)
        (re_, _, _), nr = best_of(rand)
        moved = ""
        T = a.temp * start_e * (1 - (step - 1) / a.steps)  # linear cooling; temp=0 -> greedy
        if qc is not None and (qe <= cur_e + 1e-6 or (T > 0 and rng.random() < math.exp(-(qe - cur_e) / T))):
            moved = "down" if qe < cur_e - 1e-6 else ("sideways" if qe <= cur_e + 1e-6 else "uphill")
            cur, cur_e, hot = qc, qe, qh
            if cur_e < best_e - 1e-6:
                best_e, best_arcs, moved = cur_e, list(cur), "NEW BEST"
                out.write_text(json.dumps({"z_lp": best_e, "score": math.ceil(best_e - 1e-6), "arcs": best_arcs}))
        log.append({"step": step, "q_best": qe, "rand_best": re_, "cur": cur_e, "n_q": nq_dist, "n_r": nr})
        print(f"step {step:3d}  {nq} qubits {ncirc} circuits {tq:5.1f}s | quantum best {qe:9.0f} "
              f"({nq_dist} distinct) | random best {re_:9.0f} ({nr}) | current {math.ceil(cur_e - 1e-6)} {moved}",
              flush=True)
        if service is not None:
            bal = float(service.get_credit_balance()["balance"])
            print(f"          credits left: {bal:.2f}", flush=True)
            if bal < a.min_credits:
                print("credit floor reached, stopping", flush=True)
                break
    wins = sum(l["q_best"] < l["rand_best"] - 1e-6 for l in log)
    ties = sum(abs(l["q_best"] - l["rand_best"]) <= 1e-6 for l in log)
    if best_e < start_e - 1e-6:
        import network_pipeline as npipe
        print(" ", npipe.bank(a.n, demand, best_arcs, "network_qlns")[2])
    print(f"done: {math.ceil(start_e - 1e-6)} -> best {math.ceil(best_e - 1e-6)} | quantum beat random in {wins}/{len(log)} steps, tied {ties}")
    (RUN_DIR / f"network{a.n:02d}" / f"q{a.solver}{a.seed}.steplog").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
