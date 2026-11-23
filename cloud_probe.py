"""Latency / cost probe: one MonteCarlo-QAOA ensemble step on N blocks of a400_t15.

Each block is 3 assets x 1 day of the current a400 solution (signed 3-bit encoding,
9 qubits, mu = 0). All N programs go into one ProgramEnsemble, so every optimiser
iteration is one merged backend submission. Records wall time, submissions, circuits,
time spent in submit/poll, credits used, and how often the best of the top-64 samples
is the exact (enumerated) block optimum.

    python cloud_probe.py --cluster local --n 10
    python cloud_probe.py --cluster qoro_gpu --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DIVI_DISABLE_PROGRESS", "1")
import numpy as np

import qoblib_tools as qt
from asset_groups import BlockEvaluator, _load, _net_configs
from qaoa_signed_blocks import block_qubo, decode, tqa_params
from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.qprog.algorithms import QAOA
from divi.qprog.ensembles import BatchConfig, ProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem


class TQAQAOA(QAOA):
    """QAOA started at the TQA ramp (population = TQA + jitter)."""

    def _initialize_param_sets(self):
        p0 = tqa_params(self.n_layers).reshape(1, -1)
        k = self.optimizer.n_param_sets
        jit = self._rng.normal(0, 0.15, (k, p0.shape[1]))
        jit[0] = 0
        return p0 + jit


class Step(ProgramEnsemble):
    def __init__(self, backend, bqms, pop, iters):
        super().__init__(backend)
        self.bqms, self.pop, self.iters = bqms, pop, iters

    def create_programs(self, state=None):
        super().create_programs()
        self.programs = {
            f"b{i}": TQAQAOA(BinaryOptimizationProblem(b), n_layers=2,
                             optimizer=MonteCarloOptimizer(population_size=self.pop, n_best_sets=5,
                                                           keep_best_params=True),
                             max_iterations=self.iters, backend=self.backend, seed=i)
            for i, b in enumerate(self.bqms)}

    def aggregate_results(self):
        return None


def instrument(backend):
    st = dict(submits=0, circuits=0, submit_s=0.0, poll_s=0.0, results_s=0.0)

    def wrap(name, key, count=False):
        orig = getattr(backend, name, None)
        if orig is None:
            return

        def f(*a, **k):
            t0 = time.time()
            if count:
                st["submits"] += 1
                st["circuits"] += len(a[0]) if a else len(k.get("circuits", []))
            try:
                return orig(*a, **k)
            finally:
                st[key] += time.time() - t0
        setattr(backend, name, f)

    wrap("submit_circuits", "submit_s", count=True)
    wrap("poll_job_status", "poll_s")
    wrap("get_job_results", "results_s")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="local", help="local | qoro_maestro | qoro_cpu | qoro_gpu")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance", default="po_a400_t15_orig")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--sol", default="solutions/po_a400_t15_orig_b100_l1e-3.sol")
    ap.add_argument("--out", default="results/cloud_probe")
    args = ap.parse_args()

    model, _ = _load(args.instance, args.budget, args.lam)
    _, pos = qt.read_portfolio_solution(args.sol)
    L, S = model.to_arrays(pos)
    start_obj = model.score_arrays(L, S).objective
    ev = BlockEvaluator(model)
    T, n, g = model.n_periods, len(model.symbols), args.group
    rng = np.random.default_rng(args.seed)

    blocks, seen = [], set()
    while len(blocks) < args.n:
        t = int(rng.integers(0, T))
        held = np.flatnonzero((L[t] + S[t]) > 0)
        k = min(len(held), g // 2 + 1)
        G = np.sort(np.concatenate([rng.choice(held, k, replace=False),
                                    rng.choice(np.setdiff1d(np.arange(n), held), g - k, replace=False)]))
        key = (t, tuple(G))
        if key in seen:
            continue
        seen.add(key)
        bqm, W, C, _ = block_qubo(ev, L, S, [t], G, 0.0)
        sc = max(np.abs(list(bqm.linear.values())).max(), np.abs(list(bqm.quadratic.values())).max())
        bn = bqm.copy()
        bn.scale(1 / sc)
        Vall = _net_configs(g, model.ub)
        c, ok = ev.costs(L, S, [t], G, Vall)
        cur = float(ev.costs(L, S, [t], G, (L[t, G] - S[t, G])[None])[0][0])
        blocks.append(dict(bqm=bn, W=W, C=C, t=t, G=G, exact=float(np.min(np.where(ok, c, np.inf))), cur=cur))

    if args.cluster == "local":
        backend, service = MaestroSimulator(shots=args.shots), None
    else:
        service = QoroService()
        backend = QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster=args.cluster,
                                                   tag=f"probe_n{args.n}"))
    st = instrument(backend)
    bal0 = float(service.get_credit_balance()["balance"]) if service else None

    ens = Step(backend, [b["bqm"] for b in blocks], args.pop, args.iters)
    ens.create_programs()
    t0 = time.time()
    ens.run_one_round(blocking=True, batch_config=BatchConfig(max_concurrent_programs=-1))
    wall = time.time() - t0
    bal1 = float(service.get_credit_balance()["balance"]) if service else None

    hits = improv = 0
    gain = 0.0
    for b, prog in zip(blocks, ens.programs.values()):
        tops = prog.get_top_solutions(n=64)
        order = list(b["bqm"].variables)
        bits = np.array([[int(ch) for ch in s.bitstring] for s in tops], dtype=np.int8)
        out = np.zeros_like(bits)
        out[:, order] = bits
        cq, okq = ev.costs(L, S, [b["t"]], b["G"], decode(out, b["W"], b["C"]))
        best = float(np.min(np.where(okq, cq, np.inf)))
        hits += int(np.isclose(best, b["exact"]))
        if best < b["cur"] - 1e-9:
            improv += 1
            gain += b["cur"] - best

    rec = dict(vars(args), qubits=3 * g, start_objective=start_obj, wall_s=round(wall, 1),
               submits=st["submits"], circuits=st["circuits"], submit_s=round(st["submit_s"], 1),
               poll_s=round(st["poll_s"], 1), results_s=round(st["results_s"], 1),
               s_per_submit=round(wall / max(st["submits"], 1), 2),
               credits_used=None if bal0 is None else round(bal0 - bal1, 3),
               exact_hits=hits, improvable_found=improv, sum_block_gain=round(gain, 1),
               blocks_with_slack=int(sum(b["exact"] < b["cur"] - 1e-9 for b in blocks)))
    print(json.dumps(rec), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "probe.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
