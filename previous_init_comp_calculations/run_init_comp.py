#!/usr/bin/env python
"""Run tonight's initialisation comparison and plot it.

Starts the runs below at the same time (each through auto_runner.py with init_comp.py as the
worker), and as soon as one finishes, draws its plot with ``init_comp.py --plot``.

    python experiments/run_init_comp.py              # terminal windows per instance (shows progress)
    python experiments/run_init_comp.py --silent     # no extra windows; progress in AutoData/<run>/*.log

Ctrl+C stops all runs; whatever finished is still plotted.
Smoke test (minutes instead of hours): set INIT_COMP_QUICK=1 before running.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # repo root
PY = sys.executable

# (label, runner arguments, runs in parallel inside each instance, expected duration)
JOBS = [
    ("small", ["--tests", "local_small_risk_tests"], 2, "~1 h  (3 x a010_t10, lambda 1e-6 / 1e-4 / 1e-2)"),
    ("a050", ["--manual", "20"], 3, "~2.5-3 h  (a050_t10_orig, lambda 1e-4)"),
]


def run_job(label: str, runner_args: list[str], workers: int, silent: bool, out: dict) -> None:
    """One auto_runner.py call; streams its output with a prefix, then plots its run directory."""
    env = {**os.environ, "INIT_COMP_WORKERS": str(workers), "PYTHONIOENCODING": "utf-8"}
    cmd = [PY, str(REPO / "auto_runner.py"), "experiments/init_comp.py", *runner_args, *(["--silent"] if silent else [])]
    run_dirs = []
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    out[label] = {"proc": proc, "run_dirs": run_dirs}
    for line in proc.stdout:
        print(f"[{label}] {line}", end="", flush=True)
        if m := re.search(r"results: (.+?)\s*$", line):
            run_dirs.append(m.group(1))
    proc.wait()
    for run_dir in run_dirs:
        if not any(Path(run_dir).glob("*.json")):
            print(f"[{label}] no results JSON in {run_dir}: nothing to plot", flush=True)
            continue
        plot = subprocess.run([PY, str(REPO / "experiments" / "init_comp.py"), "--plot", run_dir], cwd=REPO, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in (plot.stdout + plot.stderr).strip().splitlines():
            print(f"[{label}] {line}", flush=True)


def main() -> int:
    silent = "--silent" in sys.argv[1:]
    quick = os.environ.get("INIT_COMP_QUICK") == "1"
    start = time.time()
    print(f"Started {time.strftime('%H:%M')}{'  [QUICK smoke test]' if quick else ''}")
    for label, args, workers, eta in JOBS:
        print(f"  {label:6s} auto_runner.py experiments/init_comp.py {' '.join(args)}   "
              f"INIT_COMP_WORKERS={workers}   {eta}")
    print("Ctrl+C stops everything and still plots what finished.\n", flush=True)

    out: dict = {}
    threads = [threading.Thread(target=run_job, args=(label, args, workers, silent, out), daemon=True)
               for label, args, workers, _ in JOBS]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:  # the runners get Ctrl+C too: they stop their tasks and write partial JSON
        print("\nCtrl+C: waiting for the runners to stop and plotting what finished ...", flush=True)
        for t in threads:
            t.join(timeout=120)

    print(f"\nDone after {(time.time() - start) / 60:.0f} min. Plots and CSVs:")
    for label in out:
        for run_dir in out[label]["run_dirs"]:
            print(f"  {label:6s} {Path(run_dir) / 'init_comp_gap.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
