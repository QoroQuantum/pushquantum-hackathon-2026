"""Whole-day QAOA blocks on Qoro's cloud: every asset of one day in one quantum program.

Cutting the portfolio by period gives one block per day (all assets, the other days
fixed). With the signed 3-bit encoding (``qaoa_signed_blocks.block_qubo``) a whole a010
day is 10 assets x 3 = 30 qubits, an exact QUBO up to the budget price ``mu``. Each
(day, mu) pair is one TQA-started QAOA program with a MonteCarlo optimiser; all of them
go into one ``ProgramEnsemble``, so each optimiser iteration is one merged cloud job.

The budget is priced, not enforced: ``mu = step * unit`` with ``unit`` the block's mean
|cost change| of one unit on one asset (as in ``QAOABlockSolver``), one program per step.
Per day the proposal is the best feasible decoded sample over all its programs.

Ground truth (diagnostic only): the budget allows at most B units, so enumerating every
net vector with sum|v| <= B gives the exact day optimum (8,341 vectors on a010). The
random baseline draws as many uniform bitstrings as the QAOA program took shots.

Angle tuning (local Maestro, 21-qubit day subsets), then the 30-qubit cloud run at
fixed angles (2 circuits per program, 40 circuits, 2 merged qoro_gpu jobs, 618 s,
65.45 credits). QAOA had ~8x random's feasible shots and a closer best sample on 19/20
programs, but sampled no day optimum (see handin/QUANTUM_STORY.md):

    python day_qaoa_cloud.py --cluster local --group 7 --days 3,8 --mu-steps 0.1,0.25,1 \
        --pop 8 --iters 5 --tqa-dt 2.0
    python day_qaoa_cloud.py --cluster qoro_gpu --days all --mu-steps 0.25,1 --pop 1 --iters 1 \
        --angles 0.4519,4.8095,0.7296,5.9768 --force-sampling

Cost on qoro_gpu is GPU seconds x 0.065 credits: ~48 s per 30-qubit expectation-value
circuit, ~20-30 s per sampled one (``--force-sampling`` is cheaper).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DIVI_DISABLE_PROGRESS", "1")
import numpy as np

import qoblib_tools as qt
from asset_groups import BlockEvaluator, _load
from cloud_probe import instrument
from qaoa_signed_blocks import block_qubo, decode, tqa_params
from divi.qprog.algorithms import QAOA
from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.qprog.ensembles import BatchConfig, ProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem


def ball(n: int, B: int, ub: int = 3) -> np.ndarray:
    """Every net vector in Z^n with |v_i| <= ub and sum|v_i| <= B."""
    pts = [np.zeros(n, int)]
    for k in range(1, B + 1):
        for idx in itertools.combinations(range(n), k):
            for mags in itertools.product(range(1, ub + 1), repeat=k):
                if sum(mags) > B:
                    continue
                for sg in itertools.product((1, -1), repeat=k):
                    v = np.zeros(n, int)
                    v[list(idx)] = np.array(mags) * np.array(sg)
                    pts.append(v)
    return np.array(pts)


class TQAQAOA(QAOA):
    """QAOA whose MonteCarlo population starts at the TQA ramp (+ jitter on all but one)."""

    def __init__(self, *a, tqa_dt: float = 1.0, angles=None, **k):
        self.tqa_dt, self.angles = tqa_dt, angles
        super().__init__(*a, **k)

    def _initialize_param_sets(self):
        p0 = (np.asarray(self.angles, float) if self.angles is not None
              else tqa_params(self.n_layers, self.tqa_dt)).reshape(1, -1)
        jit = self._rng.normal(0, 0.15, (self.optimizer.n_param_sets, p0.shape[1]))
        jit[0] = 0
        return p0 + jit


class Ensemble(ProgramEnsemble):
    def __init__(self, backend, bqms, layers, pop, iters, seed, tqa_dt, angles=None):
        super().__init__(backend)
        self.bqms, self.layers, self.pop, self.iters, self.seed = bqms, layers, pop, iters, seed
        self.tqa_dt, self.angles = tqa_dt, angles

    def create_programs(self, state=None):
        super().create_programs()
        self.programs = {
            f"p{i}": TQAQAOA(BinaryOptimizationProblem(b), n_layers=self.layers,
                             optimizer=MonteCarloOptimizer(population_size=self.pop,
                                                           n_best_sets=max(1, self.pop // 4),
                                                           keep_best_params=self.pop > 1),
                             max_iterations=self.iters, backend=self.backend, seed=self.seed + i,
                             tqa_dt=self.tqa_dt, angles=self.angles)
            for i, b in enumerate(self.bqms)}

    def aggregate_results(self):
        return None


def normalise(bqm, how: str):
    """Rescale to O(1) energies (argmin unchanged). ``max``: largest coefficient = 1.
    ``std``: the energy's standard deviation over uniform bitstrings = 1, i.e.
    sqrt(sum h^2 + sum J^2) of the Ising form, so angles transfer between block sizes."""
    if how == "max":
        sc = max(np.abs(list(bqm.linear.values())).max(), np.abs(list(bqm.quadratic.values())).max())
    else:
        h, J, _ = bqm.to_ising()
        sc = float(np.sqrt(sum(v * v for v in h.values()) + sum(v * v for v in J.values())))
    bn = bqm.copy()
    bn.scale(1 / sc)
    return bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a010_t10_orig")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--sol", default="handin/po_a010_t10_orig_b004_l1e-3.sol",
                    help="other days fixed from this file; 'empty' for an empty portfolio")
    ap.add_argument("--days", default="all", help="comma list or 'all'")
    ap.add_argument("--group", type=int, default=0, help="assets per block (0 = all; smaller for local tests)")
    ap.add_argument("--mu-steps", default="0,0.1,0.25,1", help="budget price steps x unit cost")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--top", type=int, default=0, help="decode the top-N samples (0 = every distinct sample)")
    ap.add_argument("--tqa-dt", type=float, default=1.0)
    ap.add_argument("--angles", default=None, help="fixed start angles g0,b0,g1,b1,... (with --pop 1: run exactly these)")
    ap.add_argument("--norm", choices=["max", "std"], default="std")
    ap.add_argument("--force-sampling", action="store_true", help="estimate the cost from shots (cloud)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", default="local", help="local | qoro_maestro | qoro_cpu | qoro_gpu")
    ap.add_argument("--out", default="results/day_qaoa_cloud")
    args = ap.parse_args()

    model, bkv = _load(args.instance, args.budget, args.lam)
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    if args.sol == "empty":
        L = np.zeros((T, n), int)
        S = np.zeros((T, n), int)
    else:
        _, pos = qt.read_portfolio_solution(args.sol)
        L, S = model.to_arrays(pos)
    start_obj = model.score_arrays(L, S).objective
    days = list(range(T)) if args.days == "all" else [int(d) for d in args.days.split(",")]
    steps = [float(s) for s in args.mu_steps.split(",")]
    rng = np.random.default_rng(args.seed)

    blocks, bqms = [], []
    for t in days:
        if args.group and args.group < n:  # local tests: held assets first, then the lowest others
            held = np.flatnonzero(L[t] + S[t])
            G = np.sort(np.concatenate([held, np.setdiff1d(np.arange(n), held)])[:args.group])
        else:
            G = np.arange(n)
        g = len(G)
        V1 = np.concatenate([np.eye(g, dtype=np.int64), -np.eye(g, dtype=np.int64)])
        c1, _ = ev.costs(L, S, [t], G, V1)
        c0, _ = ev.costs(L, S, [t], G, np.zeros((1, g), np.int64))
        unit = float(np.mean(np.abs(c1 - c0[0]))) or 1.0
        Vb = ball(g, args.budget, model.ub)
        cb, okb = ev.costs(L, S, [t], G, Vb)
        j = int(np.argmin(np.where(okb, cb, np.inf)))
        cur = (L[t, G] - S[t, G]).astype(np.int64)
        cur_c = float(ev.costs(L, S, [t], G, cur[None])[0][0])
        for st in steps:
            bqm, W, C, margins = block_qubo(ev, L, S, [t], G, st * unit)
            bqms.append(normalise(bqm, args.norm))
            blocks.append(dict(day=t, G=G, mu_step=st, mu=st * unit, W=W, C=C, min_margin=float(margins.min()),
                               exact=float(cb[j]), exact_v=Vb[j].tolist(), cur=cur_c, nq=3 * g,
                               couplings=len(bqm.quadratic)))
    nq = max(b["nq"] for b in blocks)
    print(f"{args.instance} λ{args.lam} start {start_obj} (bkv {bkv}); {len(bqms)} programs of {nq} qubits "
          f"({len(days)} days x {len(steps)} mu steps), pop {args.pop} x iters {args.iters}, p={args.layers}, "
          f"cluster {args.cluster}", flush=True)

    if args.cluster == "local":
        backend, service = MaestroSimulator(shots=args.shots, force_sampling=args.force_sampling), None
    else:
        service = QoroService()
        backend = QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster=args.cluster,
                                                   force_sampling=args.force_sampling, tag=f"day_qaoa_{nq}q"))
    st = instrument(backend)
    bal0 = float(service.get_credit_balance()["balance"]) if service else None

    angles = None if args.angles is None else [float(x) for x in args.angles.split(",")]
    ens = Ensemble(backend, bqms, args.layers, args.pop, args.iters, args.seed, args.tqa_dt, angles)
    ens.create_programs()
    t0 = time.time()
    ens.run_one_round(blocking=True, batch_config=BatchConfig(max_concurrent_programs=-1))
    wall = time.time() - t0
    bal1 = float(service.get_credit_balance()["balance"]) if service else None

    rows, per_day = [], {}
    for b, prog in zip(blocks, ens.programs.values()):
        tops = prog.get_top_solutions(n=args.top or args.shots)
        order = list(range(b["nq"]))
        bits = np.array([[int(ch) for ch in s.bitstring] for s in tops], dtype=np.int8)
        out = np.zeros_like(bits)
        out[:, order] = bits
        probs = np.array([s.prob for s in tops])
        Vq = decode(out, b["W"], b["C"])
        cq, okq = ev.costs(L, S, [b["day"]], b["G"], Vq)
        kbest = int(np.argmin(np.where(okq, cq, np.inf))) if okq.any() else None
        best = float(cq[kbest]) if kbest is not None else None
        rb = rng.integers(0, 2, (args.shots, b["nq"]), dtype=np.int8)
        cr, okr = ev.costs(L, S, [b["day"]], b["G"], decode(rb, b["W"], b["C"]))
        rbest = float(np.min(cr[okr])) if okr.any() else None
        row = dict(day=b["day"], qubits=b["nq"], couplings=b["couplings"], mu_step=b["mu_step"],
                   mu=round(b["mu"], 3), min_margin=round(b["min_margin"], 3), exact=b["exact"],
                   exact_v=b["exact_v"], current=b["cur"],
                   qaoa_best_gap=None if best is None else best - b["exact"],
                   qaoa_best_v=None if kbest is None else Vq[kbest].tolist(),
                   p_opt_top=float(probs[okq & np.isclose(cq, b["exact"])].sum()),
                   p_feasible_top=float(probs[okq].sum()), feasible_in_top=int(okq.sum()), distinct=len(tops),
                   random_best_gap=None if rbest is None else rbest - b["exact"],
                   random_feasible=int(okr.sum()), random_samples=args.shots,
                   best_params=[round(float(x), 4) for x in np.ravel(prog.best_params)],
                   best_loss=float(prog.best_loss))
        rows.append(row)
        print(json.dumps(row), flush=True)
        d = per_day.setdefault(b["day"], dict(exact=b["exact"], best=np.inf, rbest=np.inf))
        if best is not None:
            d["best"] = min(d["best"], best)
        if rbest is not None:
            d["rbest"] = min(d["rbest"], rbest)
    hits = sum(np.isclose(d["best"], d["exact"]) for d in per_day.values())
    rhits = sum(np.isclose(d["rbest"], d["exact"]) for d in per_day.values())
    rec = dict(vars(args), qubits=nq, programs=len(bqms), start_objective=start_obj, wall_s=round(wall, 1),
               submits=st["submits"], circuits=st["circuits"], submit_s=round(st["submit_s"], 1),
               poll_s=round(st["poll_s"], 1), results_s=round(st["results_s"], 1),
               credits_used=None if bal0 is None else round(bal0 - bal1, 3),
               days_at_optimum=int(hits), random_days_at_optimum=int(rhits), n_days=len(per_day),
               rows=rows)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
