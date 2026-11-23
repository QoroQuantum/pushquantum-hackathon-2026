#!/usr/bin/env python
"""Auto runner: execute a solver script on many QOBLIB portfolio instances (spec: AutoRunner.md).

    python auto_runner.py WORKER.py --tests local_size_tests            # test by name
    python auto_runner.py WORKER.py --tests 0 4 --seed s01 --risk 6      # tests by index, seed, λ index
    python auto_runner.py WORKER.py --tests all                          # every test, every seed
    python auto_runner.py WORKER.py --manual 12 20                       # rows of the QOBLIB table (--list)
    python auto_runner.py WORKER.py --tests cloud_size_tests --cloud     # Qoro cloud instead of local
    python auto_runner.py WORKER.py --list                               # tests, λ values, table rows
    python auto_runner.py WORKER.py --tests CompareToBest --dry-run      # show the plan, run nothing

WORKER contract
---------------
The runner starts ``python WORKER.py`` once per instance, from the run directory
``AutoData/<test>_<HHMM>/``, in its own terminal (or hidden with ``--silent``), with::

    --instance po_a010_t10_orig_b004_l1e-4    full instance name
    --data-dir ../../instances/po_a010_t10_orig   instance data, relative to the run directory
    --assets 10 --days 10 --lam 1e-4 --budget 4   parameters that define the instance
    --cloud false                                 true = run on Qoro's cloud

The worker writes ``<instance>.sol`` into its working directory and reports, as one stdout line::

    AUTORUNNER_RESULT {"score": -69372, "best_sol": "po_a010_t10_orig_b004_l1e-4.sol", "extra": {...}}

(or the same JSON in ``<instance>.result.json``). ``best_sol`` defaults to ``<instance>.sol``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent
AUTODATA = REPO / "AutoData"
INSTANCES_DIR = REPO / "instances"
CSV_PATH = REPO / "qoblib_instance_benchmarks.csv"
CHECKER = REPO / "solution_checker.py"

RESULT_MARKER = "AUTORUNNER_RESULT "

# Risk λ, indexed 0..7 (spelled as in QOBLIB's file names, e.g. po_a010_t10_orig_b004_l1e-4.sol).
RISKS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]
DEFAULT_RISK = 4  # 1e-4
SEEDS = ["orig", "s00", "s01", "s02"]  # data seeds that exist in instances/

# The tests of AutoRunner.md, in its order (index = position). "XX" = --seed; entries without
# "_l…" take λ from --risk. Spec typos normalised: "p0_a10_…" → "a010_…", "l1e-05" → 1e-5.
TESTS: dict[str, list[str]] = {
    "CompareToBest": ["a010_t10_s02_b004_l1e-05", "a010_t15_XX_b004", "a050_t10_s02_b020_l0"],
    "CompareToOptimal": ["a010_t10_s02_b004_l1e-05", "a010_t15_orig_b004_l1e-06", "a050_t15_s02_b020_l0"],
    "local_size_tests": ["a010_t10_XX_b004", "a050_t10_XX_b020", "a200_t10_XX_b050"],
    "cloud_size_tests": ["a050_t15_XX_b020", "a200_t15_XX_b050", "a400_t15_XX_b100"],
    "local_small_risk_tests": ["a010_t10_XX_b004_l1e-06", "a010_t10_XX_b004_l1e-04", "a010_t10_XX_b004_l1e-02"],
    "local_big_risk_tests": ["a050_t15_XX_b020_l1e-06", "a050_t15_XX_b020_l1e-04", "a050_t15_XX_b020_l1e-02"],
    "cloud_small_risk_tests": ["a200_t15_XX_b050_l1e-06", "a200_t15_XX_b050_l1e-04", "a200_t15_XX_b050_l1e-02"],
    "cloud_big_risk_tests": ["a400_t15_XX_b100_l1e-06", "a400_t15_XX_b100_l1e-04", "a400_t15_XX_b100_l1e-02"],
}
TEST_NAMES = list(TESTS)
SEED_FIXED_TESTS = {"CompareToOptimal"}  # "This overwrites the Seed parameter!"

NAME_RE = re.compile(r"^(?:po_|p0_)?a(\d+)_t(\d+)_(orig|s\d\d|XX)_b(\d+)(?:_l([0-9.eE+-]+))?$")

# Resource model, scaled from notebook 01: po_a010_t10_orig (710 variables) → 70 sub-programs ×
# 501 circuits ≈ 35 000 circuits per run, "under a minute" locally.
REF_VARS, REF_SUBPROGRAMS, CIRCUITS_PER_SUBPROGRAM, REF_MINUTES = 710, 70, 501, 1.0
CORES_PER_TASK = 2  # a Divi/Maestro worker keeps about two cores busy


# ── instances ──────────────────────────────────────────────────────────────────────────


def norm_lam(text: str) -> str:
    """'1e-05' / '0.0001' / '1e-4' → the canonical spelling in RISKS."""
    value = float(text)
    for r in RISKS:
        if float(r) == value:
            return r
    raise ValueError(f"λ = {text} is not one of {RISKS}")


@dataclass
class Task:
    name: str      # po_a010_t10_orig_b004_l1e-4
    base: str      # po_a010_t10_orig   (instances/<base>/ holds the data)
    assets: int
    days: int
    seed: str
    budget: int
    lam: str

    @property
    def n_vars(self) -> int:  # official QUBO: 6nT position bits + (4 cash + 7 budget) slack bits per day
        return 6 * self.assets * self.days + 11 * self.days

    @property
    def subprograms(self) -> int:
        return math.ceil(self.n_vars * REF_SUBPROGRAMS / REF_VARS)

    @property
    def circuits(self) -> int:
        return self.subprograms * CIRCUITS_PER_SUBPROGRAM

    @property
    def est_minutes(self) -> float:
        return REF_MINUTES * self.subprograms / REF_SUBPROGRAMS

    @property
    def est_mb(self) -> float:  # the dense intra-day risk block ((6n)² couplings per day) dominates
        return 400 + 3.6e-5 * (6 * self.assets) ** 2 * self.days

    @property
    def csv_instance(self) -> str:  # a010_t10_orig_b004, as in the CSV's "Instance" column
        return f"{self.base[3:]}_b{self.budget:03d}"


def make_task(text: str, seed: str, lam_default: str) -> Task:
    m = NAME_RE.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse instance name {text!r}")
    a, t, xx, b, lam = m.groups()
    xx = seed if xx == "XX" else xx
    base = f"po_a{int(a):03d}_t{int(t):02d}_{xx}"
    lam = norm_lam(lam) if lam else lam_default
    return Task(f"{base}_b{int(b):03d}_l{lam}", base, int(a), int(t), xx, int(b), lam)


def tasks_for_test(test: str, seeds: list[str], lam: str) -> list[Task]:
    tasks, seen = [], set()
    for entry in TESTS[test]:
        for seed in (["orig"] if test in SEED_FIXED_TESTS else seeds):
            task = make_task(entry, seed, lam)
            if task.name not in seen:
                seen.add(task.name)
                tasks.append(task)
    return tasks


def load_csv() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tasks_for_manual(indices: list[str], rows: list[dict]) -> list[Task]:
    """Rows of the QOBLIB table; ``IDX`` uses the default λ, ``IDX:R`` uses RISKS[R]."""
    tasks = []
    for item in indices:
        idx, _, risk = item.partition(":")
        row = rows[int(idx)]
        tasks.append(make_task(row["Instance"], "orig", RISKS[int(risk)] if risk else RISKS[DEFAULT_RISK]))
    return tasks


# ── best known (CSV, then QOBLIB's curated reference) ──────────────────────────────────

_BKV: dict[str, dict] = {}


def _qt():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import qoblib_tools as qt

    return qt


def _local_model(task: Task):
    """Official model for ``task``, reading instances/<base>/ like qt.load_portfolio_instance."""
    qt = _qt()
    folder = INSTANCES_DIR / task.base
    if not folder.is_dir():
        return qt.PortfolioModel(qt.load_portfolio_instance(task.base), budget=task.budget, lam=task.lam)
    symbols, index, entries = [], {}, []
    for day, sym, price in qt._data_rows(folder / "stock_prices.txt.gz"):
        if sym not in index:
            index[sym] = len(symbols)
            symbols.append(sym)
        entries.append((index[sym], int(day), Fraction(price)))
    n_periods = max(t for _, t, _ in entries) + 1
    prices = [[None] * n_periods for _ in symbols]
    for i, t, p in entries:
        prices[i][t] = p
    cov = {}
    for day, s1, s2, value in qt._data_rows(folder / "covariance_matrices.txt.gz"):
        if s1 in index and s2 in index and int(day) < n_periods:
            cov[index[s1], index[s2], int(day)] = Fraction(value)
    inst = qt.PortfolioInstance(task.base, symbols, n_periods, prices, cov)
    return qt.PortfolioModel(inst, budget=task.budget, lam=task.lam)


def best_known(task: Task, rows: list[dict]) -> dict:
    if task.name in _BKV:
        return _BKV[task.name]
    row = next((r for r in rows if r["Instance"] == task.csv_instance), None)
    out = {"value": None, "source": None, "optimal": None,
           "csv_instance": row["Instance"] if row else None,
           "csv_parameters": row["Parameters"] if row else None,
           "csv_status": row["Status"] if row else None, "note": None}
    if row:
        found = dict(re.findall(r"(Assets|Periods|Budget) (\d+)", row["Parameters"]))
        expected = {"Assets": str(task.assets), "Periods": str(task.days), "Budget": str(task.budget)}
        if found != expected:
            out["note"] = f"CSV Parameters {found} do not match the instance {expected}"
        if row["Best objective"].strip():
            value = float(row["Best objective"])
            out.update(value=int(value) if value.is_integer() else value, source="csv",
                       optimal=(row["Optimal"].strip() or None))
    if out["value"] is None:  # the CSV column is empty: score QOBLIB's curated reference instead
        try:
            qt = _qt()
            path = qt.fetch_portfolio_reference(task.base, budget=task.budget, lam_tag=task.lam)
            _, positions = qt.read_portfolio_solution(path)
            out.update(value=int(_local_model(task).score(positions).objective),
                       source="qoblib_reference", optimal=".opt." in Path(path).name)
        except Exception as err:  # a200/a400 have no published values
            out["note"] = (out["note"] + "; " if out["note"] else "") + f"no best-known value ({type(err).__name__})"
    _BKV[task.name] = out
    return out


# ── solution checker ───────────────────────────────────────────────────────────────────

SCORE_RE = re.compile(r"Your score:\s*objective=(-?\d+)\s*\((feasible|INFEASIBLE)(?:;\s*(.*))?\)\s*$", re.M)
VIOLATIONS_RE = re.compile(r"Constraint violations:\s*(\[.*\])", re.S)


def run_checker(sol: Path) -> dict:
    """``python solution_checker.py <sol>`` and its printout as a dict."""
    try:
        proc = subprocess.run([sys.executable, str(CHECKER), str(sol)], cwd=REPO, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=1800,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "solution_checker.py timed out"}
    out = {"ok": proc.returncode == 0, "returncode": proc.returncode, "objective": None, "feasible": None,
           "breakdown": {}, "violations": None, "stdout": proc.stdout.strip(),
           "stderr": proc.stderr.strip()[-2000:] or None}
    m = SCORE_RE.search(proc.stdout)
    if m:
        out["objective"] = int(m.group(1))
        out["feasible"] = m.group(2) == "feasible"
        out["breakdown"] = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(3) or "")}
    v = VIOLATIONS_RE.search(proc.stdout)
    if v:
        try:
            out["violations"] = ast.literal_eval(v.group(1).strip())
        except (ValueError, SyntaxError):
            out["violations"] = v.group(1).strip()
    return out


# ── helpers ────────────────────────────────────────────────────────────────────────────


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def unique_path(path: Path) -> Path:
    """``path``, or ``path_1``, ``path_2``, … if taken: never overwrite."""
    if not path.exists():
        return path
    k = 1
    while (cand := path.with_name(f"{path.stem}_{k}{path.suffix}")).exists():
        k += 1
    return cand


def credit_status() -> dict | None:
    try:
        from divi.backends import QoroService

        raw = QoroService().get_credit_balance()
        return {"balance": float(raw["balance"]), "total_used": float(raw.get("total_used") or 0)}
    except Exception as err:
        print(f"[auto_runner] could not read the Qoro credit balance: {err}")
        return None


# ── one task's terminal (child mode) ───────────────────────────────────────────────────


def child_main(spec_path: str) -> int:
    """Runs inside the task's terminal: start the worker, tee its output, record the result."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    start, code, result = now(), None, None
    print(f"[auto_runner] {spec['name']}  started {start}\n[auto_runner] {shlex.join(spec['cmd'])}\n", flush=True)
    try:
        with open(spec["log"], "w", encoding="utf-8") as log:
            proc = subprocess.Popen(spec["cmd"], cwd=spec["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if line.startswith(RESULT_MARKER):
                    try:
                        result = json.loads(line[len(RESULT_MARKER):])
                    except json.JSONDecodeError:
                        pass
            code = proc.wait()
    except Exception as err:
        print(f"[auto_runner] could not run the worker: {err!r}", flush=True)
        code = -1
    finally:
        tmp = Path(spec["done"] + ".tmp")
        tmp.write_text(json.dumps({"exit_code": code, "start": start, "end": now(), "result": result}),
                       encoding="utf-8")
        os.replace(tmp, spec["done"])
    print(f"\n[auto_runner] {spec['name']} finished (exit code {code})", flush=True)
    if not spec.get("silent"):
        time.sleep(3)
    return 0


def open_terminal(cmd: list[str], cwd: Path):
    """Start ``cmd`` in a new terminal window. Returns (process | None, process_tracks_task)."""
    system = platform.system()
    if system == "Windows":
        return subprocess.Popen(cmd, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE), True
    if system == "Darwin":
        script = f"cd {shlex.quote(str(cwd))} && {shlex.join(cmd)}".replace("\\", "\\\\").replace('"', '\\"')
        return subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{script}"']), False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        for term, prefix, tracks in (("gnome-terminal", ["--wait", "--"], True), ("konsole", ["--nofork", "-e"], True),
                                     ("xterm", ["-e"], True), ("x-terminal-emulator", ["-e"], False)):
            if exe := shutil.which(term):
                return subprocess.Popen([exe, *prefix, *cmd], cwd=cwd), tracks
    return None, False  # no window system: caller falls back to silent


def kill_tree(proc) -> None:
    if proc is None:
        return
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        family = parent.children(recursive=True) + [parent]
        for p in family:
            p.terminate()
        psutil.wait_procs(family, timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ── scheduling ─────────────────────────────────────────────────────────────────────────


@dataclass
class Running:
    task: Task
    proc: subprocess.Popen | None
    tracks: bool
    start: str
    credits_used_before: float | None = None
    overlapped: bool = False
    gone_since: float | None = None


@dataclass
class Resources:
    """How many tasks may run at once: local cores and memory, or cloud credits."""
    cloud: bool
    max_parallel: int | None
    low_credits: float
    slots: int = 1
    mem_budget_mb: float = float("inf")
    info: dict = field(default_factory=dict)

    def assess(self) -> None:
        if self.cloud:
            status = credit_status()
            self.info["credits_at_start"] = status
            if status is None or status["balance"] < self.low_credits:
                self.slots = 1
                self.info["decision"] = (f"credits {status['balance'] if status else 'unknown'} < "
                                         f"{self.low_credits}: one instance at a time")
            else:
                self.slots = self.max_parallel or 4
                self.info["decision"] = f"credits {status['balance']}: up to {self.slots} instances in parallel"
            return
        try:
            import psutil

            logical = psutil.cpu_count(logical=True) or os.cpu_count() or 2
            avail_mb = psutil.virtual_memory().available / 2**20
        except ImportError:
            logical, avail_mb = os.cpu_count() or 2, float("inf")
        cores = max(1, logical - 1)  # all cores but one
        self.slots = max(1, cores // CORES_PER_TASK)
        if self.max_parallel:
            self.slots = min(self.slots, self.max_parallel)
        self.mem_budget_mb = 0.8 * avail_mb
        self.info.update(logical_cores=logical, cores_used=cores, cores_per_task=CORES_PER_TASK,
                         available_mb=round(avail_mb), memory_budget_mb=round(self.mem_budget_mb),
                         slots=self.slots,
                         decision=f"{cores} cores / {CORES_PER_TASK} per task = {self.slots} parallel; "
                                  "summed memory estimates ≤ 80% of free RAM")

    def fits(self, task: Task, running: list[Running]) -> bool:
        if not running:
            return True
        if len(running) >= self.slots:
            return False
        return self.cloud or sum(r.task.est_mb for r in running) + task.est_mb <= self.mem_budget_mb


def run_test(test: str, tasks: list[Task], args, rows: list[dict]) -> Path | None:
    worker = Path(args.worker).resolve()
    tasks = sorted(tasks, key=lambda t: t.n_vars, reverse=True)  # most complex first
    res = Resources(cloud=args.cloud, max_parallel=args.max_parallel, low_credits=args.low_credits)
    res.assess()

    print(f"\n=== {test}: {len(tasks)} instance(s), {'CLOUD' if args.cloud else 'local'}, "
          f"{'silent' if args.silent else 'terminal windows'} ===\n    {res.info['decision']}")
    for t in tasks:
        data = "ok" if (INSTANCES_DIR / t.base).is_dir() else "MISSING"
        print(f"  {t.name:32s} {t.n_vars:6d} vars  ~{t.subprograms:5d} sub-programs  ~{t.circuits:9,d} circuits  "
              f"~{t.est_minutes:6.1f} min/run locally  ~{t.est_mb:5.0f} MB  data {data}")
    if args.dry_run:
        for t in tasks:
            print(f"  best known {t.name}: {best_known(t, rows)}")
        return None

    AUTODATA.mkdir(exist_ok=True)
    run_dir = unique_path(AUTODATA / f"{test}_{dt.datetime.now():%H%M}")
    (run_dir / "_tasks").mkdir(parents=True)
    print(f"    results: {run_dir}")

    pending, running, finished = list(tasks), [], {}
    silent, interrupted = args.silent, False
    try:
        while pending or running:
            for r in list(running):
                done = run_dir / f"{r.task.name}.done.json"
                if done.exists():
                    finished[r.task.name] = collect(r, json.loads(done.read_text(encoding="utf-8")), run_dir, args, rows)
                    running.remove(r)
                elif r.tracks and r.proc is not None and r.proc.poll() is not None:
                    r.gone_since = r.gone_since or time.time()
                    if time.time() - r.gone_since > 10:  # window closed before the task finished
                        finished[r.task.name] = collect(r, {"exit_code": None, "start": r.start, "end": now(),
                                                            "error": "terminal closed before the task finished"},
                                                        run_dir, args, rows)
                        running.remove(r)
            while pending and res.fits(pending[0], running):
                task = pending.pop(0)
                spec = {
                    "name": task.name, "cwd": str(run_dir), "silent": silent,
                    "log": str(run_dir / f"{task.name}.log"), "done": str(run_dir / f"{task.name}.done.json"),
                    "cmd": [sys.executable, str(worker),
                            "--instance", task.name,
                            "--data-dir", os.path.relpath(INSTANCES_DIR / task.base, run_dir),
                            "--assets", str(task.assets), "--days", str(task.days),
                            "--lam", task.lam, "--budget", str(task.budget),
                            "--cloud", "true" if args.cloud else "false"],
                }
                spec_path = run_dir / "_tasks" / f"{task.name}.json"
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                child = [sys.executable, str(Path(__file__).resolve()), "--_child", str(spec_path)]
                proc, tracks = (None, True) if silent else open_terminal(child, run_dir)
                if proc is None:
                    if not silent:
                        print("[auto_runner] no terminal window available: continuing silently")
                        silent = True
                    proc = subprocess.Popen(child, cwd=run_dir, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                before = (credit_status() or {}).get("total_used") if args.cloud else None
                r = Running(task, proc, tracks, now(), before, overlapped=bool(running))
                for other in running:
                    other.overlapped = True
                running.append(r)
                print(f"[{now()}] started  {task.name}  ({len(running)} running, {len(pending)} queued)")
            time.sleep(2)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[auto_runner] Ctrl+C: stopping {len(running)} running instance(s)")
        for r in running:
            kill_tree(r.proc)

    # Final JSON: TYPE_TIME.json, TIME = hour and minute when all outputs are done; never overwritten.
    report = {
        "test": test,
        "instances": [t.name for t in tasks],
        "settings": {"worker": str(worker), "local": not args.cloud, "silent": args.silent, "seed": args.seed,
                     "risk": RISKS[args.risk], "manual": args.manual, "python": sys.executable,
                     "platform": platform.platform(), "interrupted": interrupted},
        "results": {t.name: finished.get(t.name) for t in tasks},
        "extra": {"resources": res.info},
    }
    if args.cloud:
        end = credit_status()
        start = res.info.get("credits_at_start")
        report["extra"]["credits_used_total"] = end["total_used"] - start["total_used"] if end and start else None
    json_path = unique_path(run_dir / f"{test}_{dt.datetime.now():%H%M}.json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_summary(report)
    print(f"\nwrote {json_path}")
    return json_path


def collect(r: Running, done: dict, run_dir: Path, args, rows: list[dict]) -> dict:
    """One instance's block for the final JSON."""
    task, result = r.task, done.get("result")
    if result is None and (path := run_dir / f"{task.name}.result.json").exists():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    block = {"start": done.get("start", r.start), "end": done.get("end"), "exit_code": done.get("exit_code")}
    if args.cloud:
        after = (credit_status() or {}).get("total_used")
        block["credits_used"] = ({"value": after - r.credits_used_before, "approximate": r.overlapped}
                                 if after is not None and r.credits_used_before is not None else None)
    best_sol = (result or {}).get("best_sol") or f"{task.name}.sol"
    sol_path = run_dir / best_sol
    block["best_sol"] = best_sol if sol_path.exists() else None
    block["score"] = (result or {}).get("score")
    block["checker"] = run_checker(sol_path) if sol_path.exists() else None
    block["best_known"] = best_known(task, rows)
    objective = (block["checker"] or {}).get("objective")
    if objective is None:
        objective = block["score"]
    bkv = block["best_known"]["value"]
    block["gap"] = (objective - bkv) / abs(bkv) if objective is not None and bkv not in (None, 0) else None
    if done.get("error") or (result is None and done.get("exit_code") != 0):
        block["error"] = done.get("error") or f"worker exit code {done.get('exit_code')}, no result reported"
    block["extra"] = (result or {}).get("extra")
    gap = "" if block["gap"] is None else f"  gap {block['gap']:+.2%}"
    print(f"[{now()}] finished {task.name}: score {block['score']}, best known {bkv}{gap}")
    return block


def print_summary(report: dict) -> None:
    print(f"\n{'instance':34s} {'score':>11s} {'checker':>11s} {'best known':>11s} {'gap':>9s}  feasible")
    for name, b in report["results"].items():
        if not b:
            print(f"{name:34s}  (not run)")
            continue
        chk = b.get("checker") or {}
        gap = "" if b.get("gap") is None else f"{b['gap']:+.2%}"
        print(f"{name:34s} {str(b.get('score')):>11s} {str(chk.get('objective')):>11s} "
              f"{str(b['best_known']['value']):>11s} {gap:>9s}  {chk.get('feasible')}")


# ── CLI ────────────────────────────────────────────────────────────────────────────────


def resolve_tests(items: list[str]) -> list[str]:
    names = {n.lower(): n for n in TEST_NAMES}
    out = []
    for item in items:
        if item.isdigit() and int(item) < len(TEST_NAMES):
            out.append(TEST_NAMES[int(item)])
        elif item.lower() in names:
            out.append(names[item.lower()])
        else:
            raise SystemExit(f"unknown test {item!r}: use a name or index from --list")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--_child"]:
        return child_main(argv[1])

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("worker", help="python file executed once per instance")
    p.add_argument("--tests", nargs="+", metavar="TEST", help="test names or indices (see --list), or 'all'")
    p.add_argument("--manual", nargs="+", metavar="ROW", help="rows of qoblib_instance_benchmarks.csv (see --list), "
                   "optionally ROW:RISK; ignores --tests/--seed/--risk")
    p.add_argument("--silent", action="store_true", help="no terminal windows")
    p.add_argument("--seed", default="orig", choices=SEEDS, help="data seed for XX (default orig)")
    p.add_argument("--cloud", action="store_true", help="run on Qoro's cloud with credits (default: local)")
    p.add_argument("--risk", type=int, default=DEFAULT_RISK, choices=range(len(RISKS)), metavar="0-7",
                   help=f"λ index into {RISKS} (default {DEFAULT_RISK} = {RISKS[DEFAULT_RISK]})")
    p.add_argument("--max-parallel", type=int, help="upper limit on instances running at once")
    p.add_argument("--low-credits", type=float, default=100.0, help="cloud: below this balance run one at a time")
    p.add_argument("--dry-run", action="store_true", help="show tasks, resource plan and best-known values only")
    p.add_argument("--list", action="store_true", help="list tests, λ values and the CSV rows")
    args = p.parse_args(argv)

    rows = load_csv()
    if args.list:
        print("Tests (index: name):")
        for i, name in enumerate(TEST_NAMES):
            print(f"  {i}: {name:24s} {', '.join(TESTS[name])}")
        print("\nRisk λ (--risk): " + ", ".join(f"{i}={r}" for i, r in enumerate(RISKS)))
        print("\nQOBLIB table rows (--manual):")
        for i, r in enumerate(rows):
            print(f"  {i:2d}: {r['Instance']:22s} {r['Parameters']:38s} {r['Status']}")
        return 0
    if not Path(args.worker).is_file():
        raise SystemExit(f"worker script {args.worker!r} not found")

    if args.manual:
        ignored = [flag for flag, used in (("--tests", args.tests), ("--seed", args.seed != "orig"),
                                           ("--risk", args.risk != DEFAULT_RISK)) if used]
        if ignored:
            print(f"[auto_runner] --manual overrides {', '.join(ignored)}")
        plan = [("manual", tasks_for_manual(args.manual, rows))]
    elif args.tests:
        if any(t.lower() == "all" for t in args.tests):  # every test, every seed
            plan = [(t, tasks_for_test(t, SEEDS, RISKS[args.risk])) for t in TEST_NAMES]
        else:
            plan = [(t, tasks_for_test(t, [args.seed], RISKS[args.risk])) for t in resolve_tests(args.tests)]
    else:
        raise SystemExit("choose instances with --tests or --manual (see --list)")

    for test, tasks in plan:
        run_test(test, tasks, args, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
