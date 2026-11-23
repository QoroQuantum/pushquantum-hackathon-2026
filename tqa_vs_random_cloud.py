#!/usr/bin/env python
"""Random vs TQA initialisation of the block QAOA on Qoro's cloud (one problem, a few λ).

Same solver as tqa_vs_random.py (colleagues' signed 3-bit block QUBO, ``QAOABlockSolver``,
empty start, asset groups of 5, sweeps until 3 sweeps improve nothing), but every QAOA
circuit runs on Qoro's cloud and each run keeps ``--in-flight`` blocks running at once:
a block is solved on a snapshot of the portfolio and its result is accepted only if it
still improves the *current* portfolio. All runs (λ x init x seed) go at the same time.
Measured: the cloud solves ~6 blocks per minute whatever is in flight, ~0.06 credits per block.

    python tqa_vs_random_cloud.py --quick        # calibration: 1 λ, 1 seed, 2 sweeps max (a few credits)
    python tqa_vs_random_cloud.py                # po_a050_t10_orig_b020, λ 1e-6 1e-4 1e-2, seeds 1-2
    python plot_tqa_vs_random.py AutoData/po_a050_t10_orig_b020_p3_cloud   # replot only

Safety: no new blocks once the balance is below --min-credits or after --hours; runs
stopped that way are marked "converged": false in their run.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import auto_runner as ar
import qoblib_tools as qt
from asset_groups import BlockEvaluator, block_schedule, grouping_fn
from auto_worker import load_instance_dir
from qaoa_signed_blocks import QAOABlockSolver, dt_summary

REPO = Path(__file__).resolve().parent
load_dotenv(REPO / ".env")

from divi.backends import JobConfig, QoroService  # noqa: E402  (needs the API key loaded)

INITS = ["random", "tqa"]


class Budget:
    """Shared stop conditions: credit floor and wall-clock deadline (checked before each block)."""

    def __init__(self, min_credits: float, hours: float):
        self.min_credits, self.deadline = min_credits, time.time() + hours * 3600
        self.lock, self.last_check, self.balance, self.reason = threading.Lock(), 0.0, None, None

    def ok(self) -> bool:
        with self.lock:
            if self.reason:
                return False
            if time.time() > self.deadline:
                self.reason = "deadline"
            elif time.time() - self.last_check > 30:
                self.last_check, self.balance = time.time(), credits()
                if self.balance is not None and self.balance < self.min_credits:
                    self.reason = f"credit balance {self.balance} < {self.min_credits}"
            return self.reason is None


def credits() -> float | None:
    try:
        return float(QoroService().get_credit_balance()["balance"])
    except Exception:
        return None


class CloudRun:
    """One (λ, init, seed) run: sweeps of day x group blocks, ``in_flight`` solved at once."""

    def __init__(self, model, task, init, seed, args, pool, budget, out_dir: Path):
        self.model, self.task, self.init, self.seed, self.args = model, task, init, seed, args
        self.pool, self.budget, self.out_dir = pool, budget, out_dir
        self.ev = BlockEvaluator(model)
        T, n = model.n_periods, len(model.symbols)
        self.L, self.S = np.zeros((T, n), np.int64), np.zeros((T, n), np.int64)
        self.best = model.score_arrays(self.L, self.S).objective
        self.lock = threading.Lock()
        self.cache: dict = {}
        self.stats: dict = {}
        self.dt_chosen: list[float] = []
        self.dt_curves: list[list[float]] = []
        self.blocks = self.accepted = 0
        self.history = [self.best]
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(out_dir / "run.log", "w", encoding="utf-8")

    def say(self, msg: str) -> None:
        self.log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.flush()

    def solve_block(self, days, G):
        with self.lock:
            L, S = self.L.copy(), self.S.copy()
        backend = QoroService(job_config=JobConfig(shots=self.args.shots, simulator_cluster="qoro_maestro"))
        s = QAOABlockSolver(layers=self.args.layers, iters=self.args.iters, shots=self.args.shots,
                            top=self.args.top, init=self.init, backend=backend, seed=self.seed,
                            optimizer="montecarlo", population=self.args.population,
                            tqa_dt_range=tuple(self.args.dt_range), tqa_dt_sampling=self.args.dt_sampling)
        self.dts = s.dts
        s.cache = self.cache  # shared across the run's blocks: unchanged blocks cost nothing
        V = s(self.ev, L, S, days, G).V
        with self.lock:
            for k, v in s.stats.items():
                self.stats[k] = self.stats.get(k, 0) + v
            self.dt_chosen += s.dt_chosen
            self.dt_curves += s.dt_curves
            self.blocks += 1
            L2, S2 = self.L.copy(), self.S.copy()
            L2[np.ix_(days, G)], S2[np.ix_(days, G)] = np.maximum(V, 0), np.maximum(-V, 0)
            if not (np.array_equal(L2, self.L) and np.array_equal(S2, self.S)):
                new = self.model.score_arrays(L2, S2)
                if new.feasible and new.objective < self.best:  # still better on the current portfolio
                    self.L, self.S, self.best = L2, S2, new.objective
                    self.accepted += 1

    def run(self) -> dict:
        t0, stale, converged, groups = time.time(), 0, False, grouping_fn(self.model, 5, "contiguous")
        schedule = block_schedule(self.model.n_periods, (1,))
        gate = threading.Semaphore(self.args.in_flight)
        sweep = 0
        for sweep in range(self.args.max_sweeps):
            before, futures = self.best, []
            for days in schedule:
                for G in groups(sweep):
                    if not self.budget.ok():
                        break
                    gate.acquire()
                    fut = self.pool.submit(self.solve_block, days, G)
                    fut.add_done_callback(lambda _f: gate.release())
                    futures.append(fut)
            wait(futures)
            for f in futures:
                if f.exception():
                    self.say(f"block error: {type(f.exception()).__name__}: {f.exception()}")
            self.history.append(self.best)
            self.say(f"sweep {sweep + 1}: {self.best}  ({self.blocks} blocks, {self.accepted} accepted, "
                     f"{self.stats.get('qaoa_runs', 0)} QAOA runs, {self.stats.get('cache_hits', 0)} cached)")
            if self.budget.reason:
                break
            stale = stale + 1 if self.best >= before else 0
            if stale >= self.args.patience:
                converged = True
                break
        return self.finish(t0, sweep + 1, converged)

    def finish(self, t0: float, sweeps: int, converged: bool) -> dict:
        task, model = self.task, self.model
        sol = self.out_dir / f"{task.name}.sol"
        model.write_solution(sol, model.to_positions(self.L, self.S),
                             comment=f"tqa_vs_random_cloud {self.init} seed {self.seed}, p={self.args.layers}")
        checker = ar.run_checker(sol)
        objective, bkv = checker.get("objective"), self.args.bkv[task.lam]
        st = dict(self.stats, init=self.init, layers=self.args.layers, population=self.args.population,
                  iters=self.args.iters, shots=self.args.shots, top=self.args.top)
        st.update(dt_summary(self.dt_chosen, self.dt_curves, tuple(self.args.dt_range), getattr(self, "dts", [])))
        rec = {"instance": task.name, "lam": task.lam, "init": self.init, "seed": self.seed, "exit_code": 0,
               "wall_s": round(time.time() - t0, 1), "objective": objective, "feasible": checker.get("feasible"),
               "best_known": bkv, "gap": None if objective is None or not bkv else (objective - bkv) / abs(bkv),
               "checker": checker, "solver": st,
               "pipeline": {"start": self.history[0], "score": self.best, "sweeps": sweeps, "converged": converged,
                            "stopped": self.budget.reason, "blocks": self.blocks, "accepted": self.accepted,
                            "in_flight": self.args.in_flight, "history": self.history, "backend": "qoro_cloud"}}
        (self.out_dir / "run.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        self.say(f"done: {objective} gap {rec['gap']} converged={converged} stopped={self.budget.reason}")
        self.log.close()
        return rec


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="po_a050_t10_orig_b020", help="problem without _l<λ>")
    ap.add_argument("--lams", nargs="+", default=["1e-6", "1e-4", "1e-2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--dt-range", type=float, nargs=2, default=[0.05, 8.0])
    ap.add_argument("--dt-sampling", choices=["geom", "random", "stratified"], default="geom",
                    help="Δt values: log grid, uniform random draws, or one random draw per equal "
                         "slice of the range (seeded per run)")
    ap.add_argument("--in-flight", type=int, default=16, help="blocks of one run solved at once")
    ap.add_argument("--max-blocks-total", type=int, default=128, help="blocks in flight over all runs")
    ap.add_argument("--max-sweeps", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many sweeps without improvement")
    ap.add_argument("--min-credits", type=float, default=100.0)
    ap.add_argument("--hours", type=float, default=5.5, help="no new blocks after this many hours")
    ap.add_argument("--inits", nargs="+", choices=INITS, default=INITS, help="initialisations to run")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--root", default=str(ar.AUTODATA), help="parent folder of the output (default AutoData)")
    ap.add_argument("--quick", action="store_true", help="calibration: λ 1e-2, seed 1, 2 sweeps")
    args = ap.parse_args(argv)
    if args.quick:
        args.lams, args.seeds, args.max_sweeps = ["1e-2"], [1], 2

    lams = [ar.norm_lam(lam) for lam in args.lams]
    tasks = {lam: ar.make_task(f"{args.instance}_l{lam}", "orig", lam) for lam in lams}
    rows = ar.load_csv()
    args.bkv = {lam: ar.best_known(t, rows)["value"] for lam, t in tasks.items()}
    base = next(iter(tasks.values())).base
    inst = load_instance_dir(ar.INSTANCES_DIR / base, base)
    problem_dir = Path(args.root) / f"{args.instance}_p{args.layers}_cloud{args.tag}{'_quick' if args.quick else ''}"
    budget = Budget(args.min_credits, args.hours)
    start_credits = credits()
    jobs = [(lam, seed, init) for lam in lams for seed in args.seeds for init in args.inits]
    print(f"{args.instance} on Qoro's cloud: {len(jobs)} runs (λ {lams} x seeds {args.seeds} x {args.inits}), "
          f"p={args.layers}, Monte Carlo {args.population}x{args.iters}, {args.in_flight} blocks in flight per run, "
          f"{args.max_blocks_total} in total; best known {args.bkv}; credits {start_credits}; "
          f"output {problem_dir}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_blocks_total) as block_pool, \
            ThreadPoolExecutor(max_workers=len(jobs)) as run_pool:
        runs = {run_pool.submit(CloudRun(qt.PortfolioModel(inst, budget=tasks[lam].budget, lam=lam), tasks[lam],
                                         init, seed, args, block_pool, budget,
                                         problem_dir / f"l{lam}" / f"{init}{seed}").run): (lam, seed, init)
                for lam, seed, init in jobs}
        pending = set(runs)
        while pending:
            done, pending = wait(pending, timeout=300)
            for f in done:
                lam, seed, init = runs[f]
                try:
                    r = f.result()
                    gap = "n/a" if r["gap"] is None else f"{100 * r['gap']:+.3f}%"
                    status = (f"{r['objective']} gap {gap}, {r['pipeline']['sweeps']} sweeps, "
                              f"{r['pipeline']['blocks']} blocks, converged={r['pipeline']['converged']}")
                except Exception as err:
                    status = f"{type(err).__name__}: {err}"
                print(f"[{(time.time() - t0) / 60:6.1f} min] λ={lam:5s} {init:6s} seed {seed}: {status}", flush=True)
            if pending:
                print(f"[{(time.time() - t0) / 60:6.1f} min] {len(pending)} runs still going; credits "
                      f"{budget.balance}; per-run progress in {problem_dir}/l*/*/run.log", flush=True)

    end_credits = credits()
    summary = {"instance": args.instance, "lams": lams, "seeds": args.seeds, "config": {
        k: v for k, v in vars(args).items() if k != "bkv"}, "best_known": args.bkv, "elapsed_s": round(time.time() - t0),
        "credits": {"start": start_credits, "end": end_credits,
                    "used": None if None in (start_credits, end_credits) else start_credits - end_credits},
        "stopped": budget.reason}
    (problem_dir / "cloud_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min; credits used {summary['credits']['used']}; "
          f"stopped: {budget.reason}", flush=True)
    import plot_tqa_vs_random

    plot_tqa_vs_random.plot(problem_dir)
    print(f"replot without re-running: python plot_tqa_vs_random.py {problem_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
