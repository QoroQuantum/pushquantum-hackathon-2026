#!/usr/bin/env python
"""Random vs TQA initialisation of the block QAOA (colleagues' pipeline, auto_worker.py).

For one problem, every λ of the QOBLIB grid, both initialisations and 4 seeds: run
``auto_worker.py`` (empty start, QAOA blocks with p = 3, Monte Carlo optimiser; TQA
searches its time step through the Monte Carlo starting population), check each .sol,
and store one run.json per run. Plotting is separate (plot_tqa_vs_random.py); it is
called once at the end and can be rerun on its own without executing anything again.

    python tqa_vs_random.py                                    # po_a010_t10_orig_b004, 64 runs
    python tqa_vs_random.py --quick                            # smoke test: 2 λ, 1 seed, 10 s limit
    python plot_tqa_vs_random.py AutoData/po_a010_t10_orig_b004_p3  # plot only

Output: AutoData/<problem>_p<p>/l<λ>/{random,tqa}<seed>/ (sol, result json, run.json, run.log);
the plot and the CSVs go in AutoData/<problem>_p<p>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import auto_runner as ar

REPO = Path(__file__).resolve().parent
WORKER = REPO / "auto_worker.py"
INITS = ["random", "tqa"]
FIXED = ["--no-sa", "--solver", "qaoa", "--qaoa-optimizer", "montecarlo"]


def run_one(task: ar.Task, init: str, seed: int, run_dir: Path, extra: list[str], rows, lock) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = os.path.relpath(ar.INSTANCES_DIR / task.base, run_dir)
    cmd = [sys.executable, str(WORKER), "--instance", task.name, "--data-dir", data,
           "--assets", str(task.assets), "--days", str(task.days), "--lam", task.lam,
           "--budget", str(task.budget), "--cloud", "false", "--seed", str(seed),
           "--qaoa-init", init, *FIXED, *extra]
    t0 = time.time()
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    sol = run_dir / f"{task.name}.sol"
    result_path = run_dir / f"{task.name}.result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    checker = ar.run_checker(sol) if sol.exists() else None
    with lock:
        bkv = ar.best_known(task, rows)["value"]
    objective = (checker or {}).get("objective")
    rec = {"instance": task.name, "lam": task.lam, "init": init, "seed": seed, "exit_code": code,
           "wall_s": round(time.time() - t0, 1), "objective": objective,
           "feasible": (checker or {}).get("feasible"), "best_known": bkv,
           "gap": None if objective is None or not bkv else (objective - bkv) / abs(bkv),
           "checker": checker, "solver": (result.get("extra") or {}).get("solver"),
           "pipeline": {k: v for k, v in (result.get("extra") or {}).items() if k != "solver"}}
    (run_dir / "run.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # λ on Windows consoles
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="po_a010_t10_orig_b004", help="problem without _l<λ>")
    ap.add_argument("--lams", nargs="+", default=ar.RISKS, help=f"λ values (default all: {ar.RISKS})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--layers", type=int, default=3, help="QAOA layers p")
    ap.add_argument("--population", type=int, default=50, help="Monte Carlo population per block")
    ap.add_argument("--iters", type=int, default=10, help="Monte Carlo iterations per block")
    ap.add_argument("--dt-range", type=float, nargs=2, default=[0.05, 8.0], help="TQA Δt range of the population")
    ap.add_argument("--dt-sampling", choices=["geom", "random", "stratified"], default="geom",
                    help="Δt values: log grid, uniform random draws, or one random draw per equal "
                         "slice of the range (seeded per run)")
    ap.add_argument("--time", type=float, default=0.0,
                    help="time limit per run (s); 0 (default) = none: sweep blocks until converged")
    ap.add_argument("--kicks", type=int, default=0, help="with --time 0: kicks after the sweeps (default 0)")
    ap.add_argument("--workers", type=int, help="runs at once (default: (cores - 1) // 2, memory permitting)")
    ap.add_argument("--gpu", action="store_true", help="QAOA blocks on the local GPU")
    ap.add_argument("--inits", nargs="+", choices=INITS, default=INITS, help="initialisations to run")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--root", default=str(ar.AUTODATA), help="parent folder of the output (default AutoData)")
    ap.add_argument("--quick", action="store_true", help="smoke test: λ 1e-6 and 1e-2, seed 1, 10 s search")
    args = ap.parse_args(argv)

    if args.quick:
        args.lams, args.seeds, args.time = ["1e-6", "1e-2"], [1], 10.0
    lams = [ar.norm_lam(lam) for lam in args.lams]
    tasks = {lam: ar.make_task(f"{args.instance}_l{lam}", "orig", lam) for lam in lams}
    problem_dir = Path(args.root) / f"{args.instance}_p{args.layers}{args.tag}"
    rows = ar.load_csv()
    res = ar.Resources(cloud=False, max_parallel=args.workers, low_credits=0)
    res.assess()
    workers = args.workers or res.slots
    extra = ["--qaoa-layers", str(args.layers), "--qaoa-population", str(args.population),
             "--qaoa-iters", str(args.iters),
             "--tqa-dt-range", *map(str, args.dt_range), "--tqa-dt-sampling", args.dt_sampling, "--time", str(args.time), "--kicks", str(args.kicks)] + (["--qaoa-backend", "gpu"] if args.gpu else [])
    jobs = [(lam, seed, init) for lam in lams for seed in args.seeds for init in args.inits]  # interleaved
    print(f"{args.instance}, p = {args.layers}, Monte Carlo {args.population} x {args.iters}: {len(lams)} λ x {len(args.seeds)} seeds x {len(args.inits)} inits = {len(jobs)} runs, "
          f"{workers} at once ({res.info['decision']}); output {problem_dir}", flush=True)

    lock, t0 = threading.Lock(), time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, tasks[lam], init, seed, problem_dir / f"l{lam}" / f"{init}{seed}",
                               extra, rows, lock): (lam, seed, init) for lam, seed, init in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            lam, seed, init = futures[fut]
            try:
                r = fut.result()
                gap = "n/a" if r["gap"] is None else f"{100 * r['gap']:+.3f}%"
                status = f"{r['objective']} ({'feasible' if r['feasible'] else 'INFEASIBLE'}), gap {gap}"
            except Exception as err:
                status = f"{type(err).__name__}: {err}"
            print(f"[{done:3d}/{len(jobs)}  {(time.time() - t0) / 60:5.1f} min]  λ={lam:5s} {init:6s} seed {seed}: "
                  f"{status}", flush=True)
    import plot_tqa_vs_random

    plot_tqa_vs_random.plot(problem_dir)
    print(f"replot without re-running: python plot_tqa_vs_random.py {problem_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
