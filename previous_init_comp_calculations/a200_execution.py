#!/usr/bin/env python
"""a200 on Qoro's cloud: block QAOA from TQA starts with a Monte Carlo search over the time step.

Uses the team's block pipeline: SA start (``asset_groups.sa_start``), day x asset-group
blocks (``asset_groups.make_groups``, contiguous groups of 5, every day) and the exact
signed 3-bit block QUBO (``qaoa_signed_blocks.block_qubo``). Every circuit runs on Qoro's
cloud, and each job is timed from submission until its results come back.

    python a200_execution.py                  # every block of the instance
    python a200_execution.py --max-blocks 8   # short timing run

Writes ``results/a200_execution_<time>.json`` (timings, blocks, credits) and the ``.sol``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import qoblib_tools as qt
from asset_groups import BlockEvaluator, _net_configs, make_groups, sa_start
from auto_worker import load_instance_dir
from qaoa_signed_blocks import block_qubo, decode, lagrange_mu, tqa_params

REPO = Path(__file__).resolve().parent
load_dotenv(REPO / ".env")

from divi.backends import JobConfig, QoroService  # noqa: E402  (needs the API key loaded)
from divi.qprog.algorithms import QAOA  # noqa: E402
from divi.qprog.optimizers import MonteCarloOptimizer  # noqa: E402
from divi.qprog.problems import BinaryOptimizationProblem  # noqa: E402


class TimedQoroService(QoroService):
    """QoroService that records when each job is sent and when its results come back."""

    jobs: list[dict] = []
    _lock = threading.Lock()

    def submit_circuits(self, payloads, **kwargs):
        sent = time.perf_counter()
        result = super().submit_circuits(payloads, **kwargs)
        try:
            n_payloads = len(payloads)
        except TypeError:
            n_payloads = None
        with self._lock:
            self.jobs.append({"job_id": result.job_id, "block": getattr(self, "block", None),
                              "payloads": n_payloads, "sent": sent, "uploaded": time.perf_counter(),
                              "returned": None})
        return result

    def get_job_results(self, execution_result):
        result = super().get_job_results(execution_result)
        now = time.perf_counter()
        with self._lock:
            for job in self.jobs:
                if job["job_id"] == execution_result.job_id and job["returned"] is None:
                    job["returned"] = now
        return result


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p90": round(ordered[max(0, -(-9 * len(ordered) // 10) - 1)], 2),
            "min": round(ordered[0], 2), "max": round(ordered[-1], 2)}


def job_timings(block: int | None = None) -> dict:
    done = [j for j in TimedQoroService.jobs
            if j["returned"] is not None and (block is None or j["block"] == block)]
    return {"round_trip_s": stats([j["returned"] - j["sent"] for j in done]),
            "upload_s": stats([j["uploaded"] - j["sent"] for j in done]),
            "queue_and_run_s": stats([j["returned"] - j["uploaded"] for j in done])}


def credits() -> float | None:
    try:
        return float(QoroService().get_credit_balance()["balance"])
    except Exception:
        return None


def solve_block(index: int, day: int, group: np.ndarray, L, S, ev: BlockEvaluator, args) -> dict:
    """Build the block QUBO from a snapshot, run QAOA on the cloud, return decoded candidates."""
    t0 = time.perf_counter()
    configs = _net_configs(len(group), ev.model.ub)
    cost, ok = ev.costs(L, S, [day], group, configs)
    exact = float(np.min(np.where(ok, cost, np.inf)))
    mu, _ = lagrange_mu(configs, cost, ok)
    bqm, W, C, _ = block_qubo(ev, L, S, day, group, mu)
    scale = max(np.abs(list(bqm.linear.values())).max(), np.abs(list(bqm.quadratic.values())).max())
    bqm.scale(1.0 / scale)

    dts = np.geomspace(args.dt_min, args.dt_max, args.population)
    rows = np.array([tqa_params(args.layers, dt) for dt in dts])
    backend = TimedQoroService(job_config=JobConfig(shots=args.shots, simulator_cluster="qoro_maestro"))
    backend.block = index
    qaoa = QAOA(BinaryOptimizationProblem(bqm), n_layers=args.layers,
                optimizer=MonteCarloOptimizer(population_size=args.population),
                max_iterations=args.iterations, backend=backend, seed=args.seed)
    qaoa.run(initial_params=rows)

    first = qaoa.losses_history[0]
    best_start = int(np.argmin([first[str(i)] for i in range(len(first))]))
    tops = qaoa.get_top_solutions(n=args.top)
    bits = np.array([[int(ch) for ch in s.bitstring] for s in tops])
    return {"block": index, "day": day, "group": group.tolist(), "qubits": 3 * len(group),
            "candidates": decode(bits, W, C), "exact": exact, "dt_best_start": float(dts[best_start]),
            "circuits": qaoa.total_circuit_count, "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="po_a200_t10_orig")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--max-blocks", type=int, help="stop after this many blocks (default: all)")
    ap.add_argument("--concurrency", type=int, default=8, help="blocks in flight at once")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--population", type=int, default=20, help="TQA time steps per Monte Carlo population")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--dt-min", type=float, default=0.05)
    ap.add_argument("--dt-max", type=float, default=8.0)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--min-credits", type=float, default=20.0, help="stop sending blocks below this balance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    start_credits = credits()
    print(f"credits: {start_credits}", flush=True)
    data = REPO / "instances" / args.instance
    model = qt.PortfolioModel(load_instance_dir(data, args.instance), budget=args.budget, lam=args.lam)
    ev = BlockEvaluator(model)
    L, S, sa_info = sa_start(model, seed=args.seed)
    start_score = model.score_arrays(L, S).objective

    groups = make_groups(model, args.group_size, "contiguous")
    blocks = [(day, g) for g in groups for day in range(model.n_periods)]
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]
    print(f"{args.instance} B={args.budget} λ={args.lam}: start {start_score}; {len(blocks)} blocks "
          f"({len(groups)} groups of {args.group_size} x {model.n_periods} days), "
          f"{3 * args.group_size} qubits each, {args.concurrency} in flight", flush=True)

    state_lock = threading.Lock()
    rows, stopped, t_start = [], None, time.perf_counter()

    def submit(pool, index):
        day, group = blocks[index]
        with state_lock:
            snapshot = (L.copy(), S.copy())
        return pool.submit(solve_block, index, day, group, *snapshot, ev, args)

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    pending = {}
    next_block = 0
    try:
        while next_block < len(blocks) and len(pending) < args.concurrency:
            pending[submit(pool, next_block)] = next_block
            next_block += 1
        while pending:
            future = next(as_completed(pending))
            index = pending.pop(future)
            try:
                res = future.result()
            except Exception as err:
                rows.append({"block": index, "error": f"{type(err).__name__}: {err}"})
                print(f"block {index}: {type(err).__name__}: {err}", flush=True)
            else:
                day, group = res["day"], np.array(res["group"])
                with state_lock:
                    cost, ok = ev.costs(L, S, [day], group, res["candidates"])
                    current = float(ev.costs(L, S, [day], group, (L[day, group] - S[day, group])[None])[0][0])
                    best = int(np.argmin(np.where(ok, cost, np.inf)))
                    gain = current - float(cost[best]) if ok[best] else 0.0
                    if gain > 1e-9:
                        V = res["candidates"][best]
                        L[day, group], S[day, group] = np.maximum(V, 0), np.maximum(-V, 0)
                    score = model.score_arrays(L, S).objective
                timing = job_timings(index)["round_trip_s"] or {}
                row = {k: v for k, v in res.items() if k != "candidates"}
                row.update(qaoa_gap_to_exact=float(np.min(np.where(ok, cost, np.inf))) - res["exact"]
                           if ok.any() else None, accepted_gain=gain, score=score,
                           jobs=timing.get("n"), round_trip_mean_s=timing.get("mean"))
                rows.append(row)
                print(f"block {index + 1:3d}/{len(blocks)}  day {day}  {res['qubits']} qubits  "
                      f"{timing.get('n')} jobs, round trip {timing.get('mean')} s  "
                      f"block {res['wall_s']} s  dt*={res['dt_best_start']:.2f}  gain {gain:.0f}  "
                      f"score {score}", flush=True)
            if next_block < len(blocks) and stopped is None:
                balance = credits()
                if balance is not None and balance < args.min_credits:
                    stopped = f"credit balance {balance} < {args.min_credits}"
                    print(f"stopping: {stopped}", flush=True)
                else:
                    pending[submit(pool, next_block)] = next_block
                    next_block += 1
    except KeyboardInterrupt:
        stopped = "Ctrl+C"
        print("Ctrl+C: no new blocks; waiting for the jobs in flight ...", flush=True)
        pool.shutdown(wait=True, cancel_futures=True)
    pool.shutdown(wait=True)

    elapsed = time.perf_counter() - t_start
    end_credits = credits()
    final = model.score_arrays(L, S)
    stamp = time.strftime("%Y%m%d_%H%M")
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    sol = out_dir / f"a200_execution_{stamp}.sol"
    model.write_solution(sol, model.to_positions(L, S), comment="a200_execution: cloud block QAOA, TQA + Monte Carlo")
    summary = {
        "instance": args.instance, "budget": args.budget, "lam": args.lam, "config": vars(args),
        "sa_start": sa_info, "start_score": start_score, "final_score": final.objective,
        "feasible": final.feasible, "blocks_done": len(rows), "blocks_planned": len(blocks),
        "stopped": stopped, "elapsed_s": round(elapsed, 1),
        "credits": {"start": start_credits, "end": end_credits,
                    "used": None if None in (start_credits, end_credits) else start_credits - end_credits},
        "timings": job_timings(),
        "jobs": [{**j, "sent": j["sent"] - t_start, "uploaded": j["uploaded"] - t_start,
                  "returned": None if j["returned"] is None else j["returned"] - t_start}
                 for j in TimedQoroService.jobs],
        "blocks": rows,
    }
    out = out_dir / f"a200_execution_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    t = summary["timings"]
    print(f"\n{len(rows)}/{len(blocks)} blocks in {elapsed / 60:.1f} min; score {start_score} -> "
          f"{final.objective} ({'feasible' if final.feasible else 'INFEASIBLE'})")
    print(f"jobs: {len(TimedQoroService.jobs)}; round trip {t['round_trip_s']}")
    print(f"      upload {t['upload_s']}; queue + run {t['queue_and_run_s']}")
    print(f"credits used: {summary['credits']['used']}")
    print(f"wrote {out}\nwrote {sol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
