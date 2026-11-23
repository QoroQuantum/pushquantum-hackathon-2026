#!/usr/bin/env python3
"""Automated multi-instance runner for QOBLIB Portfolio Optimization solvers.

Executes a target solver script across predefined benchmark suites or manually
selected rows from ``qoblib_instance_benchmarks.csv``, manages local resource
budgeting or Qoro cloud credit checks, launches tasks in parallel (silently or
in OS-agnostic visible terminal windows), validates produced ``.sol`` files via
``solution_checker.py``, and aggregates all metrics into a non-overwriting JSON
report inside ``AutoData/<test_name>_<HHMM>/``.

Target Solver CLI Contract
--------------------------
Each instance task is launched with its working directory set to the run's
output directory (``AutoData/<run_dir>/``) and ``PYTHONPATH`` including the
repository root. ``auto_runner.py`` invokes the target Python script as::

    python <target_script.py> \\
        --instance <full_instance_name> \\
        --instance-dir <relative_path_to_instance_dir> \\
        --assets <int> \\
        --days <int> \\
        --lambda <str_value> \\
        --budget <int> \\
        --cloud <true|false> \\
        [--solver-seed <int>]   # included when --solver-seeds is used

Expected outputs from the target script:
  1. Indirect: writes a ``.sol`` file in the current working directory named
     ``<full_instance_name>.sol`` (e.g. ``po_a010_t10_orig_b004_l1e-4.sol``).
  2. Direct (stdout): optionally prints a JSON object (either the entire stdout,
     a line prefixed with ``AUTORUNNER_RESULT:``, or the final JSON object in
     stdout) containing ``score`` (numeric) and/or ``extra`` (arbitrary JSON
     object). Plain ``score`` + JSON output is also accepted.

Pass solver-specific options after ``--``. ``auto_worker.py``, ``day_swap.py``
and ``day_swap_parallel.py`` accept a warm-start ``--init`` with a .sol file
or a directory containing ``<full_instance_name>.sol`` for each task.
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
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants & Benchmark Metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_CSV = REPO_ROOT / "qoblib_instance_benchmarks.csv"
INSTANCES_DIR = REPO_ROOT / "instances"
QOBLIB_DATA_INSTANCES_DIR = REPO_ROOT / "qoblib_data" / "06-portfolio" / "instances"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
SOLUTION_CHECKER = REPO_ROOT / "solution_checker.py"
DAY_SWAP_SCRIPTS = {"day_swap.py", "day_swap_parallel.py"}
WARM_START_SCRIPTS = DAY_SWAP_SCRIPTS | {"auto_worker.py"}
AUTODATA_DIR = REPO_ROOT / "AutoData"

# Hardcoded risk-aversion lambda grid (indexed 0..7)
RISK_LAMBDAS: tuple[float, ...] = (
    0.0,
    1e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    1e-2,
)
RISK_LAMBDA_TAGS: tuple[str, ...] = (
    "0",
    "1e-6",
    "1e-5",
    "5e-5",
    "1e-4",
    "5e-4",
    "1e-3",
    "1e-2",
)
DEFAULT_RISK_INDEX = 4  # 1e-4

VALID_SEEDS: tuple[str, ...] = ("orig", "s00", "s01", "s02")
DEFAULT_SEED = "orig"
DEFAULT_SOLVER_SEEDS: tuple[int, ...] = (1, 2, 3)

BUDGET_BY_ASSETS: dict[int, int] = {
    3: 3,
    4: 4,
    5: 4,
    10: 4,
    50: 20,
    200: 50,
    400: 100,
}

# Predefined test suites in the exact order specified in AutoRunner.md
TEST_SUITES_ORDERED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "CompareToBest",
        (
            "a010_t10_s02_b004_l1e-05",
            "p0_a10_t15_XX_b004",
            "a050_t10_s02_b020_l0",
        ),
    ),
    (
        "CompareToOptimal",
        (
            "a010_t10_s02_b004_l1e-05",
            "a010_t15_orig_b004_l1e-06",
            "a050_t15_s02_b020_l0",
        ),
    ),
    (
        "local_size_tests",
        (
            "po_a010_t10_XX_b004",
            "po_a050_t10_XX_b020",
            "p0_a200_t10_XX_b050",
        ),
    ),
    (
        "cloud_size_tests",
        (
            "po_a050_t15_XX_b020",
            "po_a200_t15_XX_b050",
            "po_a400_t15_XX_b100",
        ),
    ),
    (
        "local_small_risk_tests",
        (
            "a010_t10_XX_b004_l1e-06",
            "a010_t10_XX_b004_l1e-04",
            "a010_t10_XX_b004_l1e-02",
        ),
    ),
    (
        "local_big_risk_tests",
        (
            "a050_t15_XX_b020_l1e-06",
            "a050_t15_XX_b020_l1e-04",
            "a050_t15_XX_b020_l1e-02",
        ),
    ),
    (
        "cloud_small_risk_tests",
        (
            "a200_t15_XX_b050_l1e-06",
            "a200_t15_XX_b050_l1e-04",
            "a200_t15_XX_b050_l1e-02",
        ),
    ),
    (
        "cloud_big_risk_tests",
        (
            "a400_t15_XX_b100_l1e-06",
            "a400_t15_XX_b100_l1e-04",
            "a400_t15_XX_b100_l1e-02",
        ),
    ),
    ("all", ()),  # Dynamically populated from all rows in qoblib_instance_benchmarks.csv
)

TEST_SUITE_NAMES: tuple[str, ...] = tuple(name for name, _ in TEST_SUITES_ORDERED)
TEST_SUITES_MAP: dict[str, tuple[str, ...]] = dict(TEST_SUITES_ORDERED)

# Regex matching all instance formats in AutoRunner.md and qoblib_instance_benchmarks.csv
INSTANCE_SPEC_RE = re.compile(
    r"^(?:p[o0]_)?"
    r"a(?P<assets>\d+)_"
    r"t(?P<days>\d+)_"
    r"(?P<seed>orig|s\d+|XX)"
    r"(?:_b(?P<budget>\d+))?"
    r"(?:_l(?P<lam>[0-9eE.+-]+))?$"
)

CSV_PARAMS_RE = re.compile(
    r"Assets\s+(?P<assets>\d+)\s*·\s*Periods\s+(?P<days>\d+)\s*·\s*Budget\s+(?P<budget>\d+)"
)

LOW_CREDIT_THRESHOLD_DEFAULT = 100.0
CLOUD_MAX_PARALLEL_DEFAULT = 3

# GPU monitoring & anti-throttling defaults
GPU_MAX_UTIL_DEFAULT = 85.0
GPU_VRAM_HEADROOM_PCT_DEFAULT = 15.0
GPU_MAX_TEMP_C_DEFAULT = 82.0
GPU_TEMP_HYSTERESIS_C = 3.0
GPU_POLL_INTERVAL_S = 0.5
GPU_WARMUP_RESERVATION_S = 5.0
GPU_THROTTLE_COOLDOWN_S = 5.0

# NVIDIA clocks_throttle_reasons.active bits that indicate real thermal/power throttling
# (ignores 0x1 GPU Idle and 0x2 Applications Clocks Setting)
_NV_THROTTLE_MASK = 0x0000000000000004 | 0x0000000000000008 | 0x0000000000000020 | 0x0000000000000040 | 0x0000000000000080


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvBenchmarkRow:
    """One row from ``qoblib_instance_benchmarks.csv``."""

    index: int
    problem_id: str
    problem: str
    instance: str  # e.g. a010_t10_orig_b004
    parameters: str  # e.g. Assets 10 · Periods 10 · Budget 4
    assets: int
    days: int
    budget: int
    best_objective: int | float | None
    optimal: str
    source: str
    source_url: str
    status: str
    raw_url: str


@dataclass(frozen=True)
class InstanceTask:
    """Fully resolved portfolio optimization instance to execute."""

    assets: int
    days: int
    seed: str
    budget: int
    lam: float
    lam_tag: str
    raw_spec: str
    solver_seed: int | None = None

    @property
    def base_name(self) -> str:
        """Physical directory name under ``instances/`` (e.g. ``po_a010_t10_orig``)."""
        return f"po_a{self.assets:03d}_t{self.days:02d}_{self.seed}"

    @property
    def csv_instance_key(self) -> str:
        """Key matching ``Instance`` column in ``qoblib_instance_benchmarks.csv``."""
        return f"a{self.assets:03d}_t{self.days:02d}_{self.seed}_b{self.budget:03d}"

    @property
    def csv_parameters_key(self) -> str:
        """Key matching ``Parameters`` column in ``qoblib_instance_benchmarks.csv``."""
        return f"Assets {self.assets} · Periods {self.days} · Budget {self.budget}"

    @property
    def full_name(self) -> str:
        """Canonical full instance name including budget and lambda."""
        return f"{self.base_name}_b{self.budget:03d}_l{self.lam_tag}"

    @property
    def run_key(self) -> str:
        """Unique identifier for one scheduled task (distinguishes solver RNG seeds)."""
        if self.solver_seed is None:
            return self.full_name
        return f"{self.full_name}__seed{self.solver_seed}"

    @property
    def display_name(self) -> str:
        """Human-readable task label for logs."""
        if self.solver_seed is None:
            return self.full_name
        return f"{self.full_name} [solver_seed={self.solver_seed}]"

    @property
    def instance_dir(self) -> Path:
        return INSTANCES_DIR / self.base_name

    @property
    def estimated_qubo_vars(self) -> int:
        """Official QOBLIB portfolio QUBO variable count: T * (6*N + 11)."""
        return self.days * (6 * self.assets + 11)


@dataclass
class GPUDeviceStatus:
    """Live telemetry snapshot for one local GPU device."""

    index: int
    name: str
    backend: str  # "cuda", "rocm", "mps"
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    utilization_pct: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    throttling: bool = False
    throttle_reason: str | None = None
    unified_memory: bool = False

    @property
    def device_str(self) -> str:
        if self.backend == "mps":
            return "mps"
        return f"{self.backend}:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "backend": self.backend,
            "device": self.device_str,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "used_vram_gb": round(self.used_vram_gb, 2),
            "utilization_pct": (
                round(self.utilization_pct, 1) if self.utilization_pct is not None else None
            ),
            "temperature_c": (
                round(self.temperature_c, 1) if self.temperature_c is not None else None
            ),
            "power_draw_w": (
                round(self.power_draw_w, 1) if self.power_draw_w is not None else None
            ),
            "power_limit_w": (
                round(self.power_limit_w, 1) if self.power_limit_w is not None else None
            ),
            "throttling": self.throttling,
            "throttle_reason": self.throttle_reason,
            "unified_memory": self.unified_memory,
        }


@dataclass(frozen=True)
class GPUThrottlePolicy:
    """Configuration controlling passive GPU monitoring and anti-throttling guardrails."""

    enabled: bool = True
    max_util_pct: float = GPU_MAX_UTIL_DEFAULT
    vram_headroom_pct: float = GPU_VRAM_HEADROOM_PCT_DEFAULT
    max_temp_c: float = GPU_MAX_TEMP_C_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_util_pct": self.max_util_pct,
            "vram_headroom_pct": self.vram_headroom_pct,
            "max_temp_c": self.max_temp_c,
        }


@dataclass
class TaskExecutionResult:
    """Collected runtime, checker, and benchmark output for one instance task."""

    task: InstanceTask
    start_time: str
    end_time: str
    duration_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    credits_used: float | None = None
    best_solution_file: str | None = None
    solution_rel_path: str | None = None
    reported_score: int | float | None = None
    checker_result: dict[str, Any] | None = None
    best_known_score: int | float | None = None
    benchmark_metadata: dict[str, Any] = field(default_factory=dict)
    extra: Any = field(default_factory=dict)
    gpu_telemetry: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# CSV Benchmark & Submission Best-Known Loading
# ---------------------------------------------------------------------------


def _parse_numeric(value: str | None) -> int | float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    if num.is_integer():
        return int(num)
    return num


def load_benchmark_csv(csv_path: Path = BENCHMARKS_CSV) -> list[CsvBenchmarkRow]:
    """Read ``qoblib_instance_benchmarks.csv`` preserving exact row order."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    rows: list[CsvBenchmarkRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, raw in enumerate(reader):
            inst = (raw.get("Instance") or "").strip()
            params = (raw.get("Parameters") or "").strip()
            match = CSV_PARAMS_RE.search(params)
            if match:
                assets = int(match.group("assets"))
                days = int(match.group("days"))
                budget = int(match.group("budget"))
            else:
                spec_match = INSTANCE_SPEC_RE.match(inst)
                if not spec_match:
                    raise ValueError(
                        f"Row {idx} in {csv_path.name} has unparseable Instance/Parameters: "
                        f"{inst!r} / {params!r}"
                    )
                assets = int(spec_match.group("assets"))
                days = int(spec_match.group("days"))
                budget = (
                    int(spec_match.group("budget"))
                    if spec_match.group("budget")
                    else BUDGET_BY_ASSETS[assets]
                )

            rows.append(
                CsvBenchmarkRow(
                    index=idx,
                    problem_id=(raw.get("Problem ID") or "").strip(),
                    problem=(raw.get("Problem") or "").strip(),
                    instance=inst,
                    parameters=params,
                    assets=assets,
                    days=days,
                    budget=budget,
                    best_objective=_parse_numeric(raw.get("Best objective")),
                    optimal=(raw.get("Optimal") or "").strip(),
                    source=(raw.get("Source") or "").strip(),
                    source_url=(raw.get("Source URL") or "").strip(),
                    status=(raw.get("Status") or "").strip(),
                    raw_url=(raw.get("Raw URL") or "").strip(),
                )
            )
    return rows


def lookup_best_known_score(
    task: InstanceTask,
    csv_by_instance: dict[str, CsvBenchmarkRow],
) -> tuple[int | float | None, dict[str, Any]]:
    """Extract the best known score from ``qoblib_instance_benchmarks.csv``.

    Matches both the ``Instance`` and ``Parameters`` columns of the CSV.
    When the CSV's ``Best objective`` column is blank (as in QOBLIB's base-level
    table where each base spans 8 lambda values), falls back to scanning local
    QOBLIB submission summary files under ``submissions/`` for the exact
    ``(instance, budget, lambda)`` combination.
    """
    row = csv_by_instance.get(task.csv_instance_key)
    metadata: dict[str, Any] = {
        "csv_instance": task.csv_instance_key,
        "csv_parameters": task.csv_parameters_key,
        "csv_matched": row is not None,
    }

    if row is not None:
        metadata["csv_parameters_match"] = row.parameters == task.csv_parameters_key
        metadata["csv_status"] = row.status or None
        metadata["csv_optimal"] = row.optimal or None
        metadata["csv_source"] = row.source or None
        metadata["csv_raw_url"] = row.raw_url or None
        if row.best_objective is not None:
            metadata["score_source"] = BENCHMARKS_CSV.name
            return row.best_objective, metadata

    # Fallback: check local QOBLIB submission summaries for this exact lambda
    if SUBMISSIONS_DIR.exists():
        best_val: int | float | None = None
        best_source: str | None = None
        is_proven_optimal = False
        pattern = f"{task.full_name}_summary.csv"
        for summary_path in sorted(SUBMISSIONS_DIR.rglob(pattern)):
            try:
                with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
                    for srow in csv.DictReader(fh):
                        obj = _parse_numeric(srow.get("Best Objective Value"))
                        bound = _parse_numeric(srow.get("Optimality Bound"))
                        if obj is None:
                            continue
                        # QOBLIB portfolio objective is minimization (lower is better).
                        if best_val is None or obj < best_val:
                            best_val = obj
                            # Submission folder name is 2 levels up from instance folder
                            rel_sub = summary_path.relative_to(SUBMISSIONS_DIR).parts[0]
                            best_source = f"submissions/{rel_sub}"
                            if bound is not None and abs(float(obj) - float(bound)) < 1e-6:
                                is_proven_optimal = True
            except OSError:
                continue

        if best_val is not None:
            metadata["score_source"] = best_source
            metadata["proven_optimal"] = is_proven_optimal
            return best_val, metadata

    metadata["score_source"] = None
    return None, metadata


# ---------------------------------------------------------------------------
# Seed, Risk Lambda, and Instance Spec Resolution
# ---------------------------------------------------------------------------


def normalize_seed(seed: str) -> str:
    """Normalize user seed strings such as ``orig``, ``s01``, or ``01``."""
    s = seed.strip().lower()
    if s == "orig":
        return "orig"
    if re.fullmatch(r"\d{2}", s):
        s = f"s{s}"
    if s not in VALID_SEEDS:
        raise ValueError(
            f"Invalid seed {seed!r}. Expected one of: {', '.join(VALID_SEEDS)} "
            "(or two-digit suffix like '00', '01', '02')."
        )
    return s


def risk_from_index(risk_index: int) -> tuple[float, str]:
    """Return ``(lambda_float, lambda_tag)`` for a risk index in ``0..7``."""
    if not 0 <= risk_index < len(RISK_LAMBDAS):
        choices = ", ".join(f"{i}={t}" for i, t in enumerate(RISK_LAMBDA_TAGS))
        raise ValueError(f"Risk index {risk_index} out of range 0..7 ({choices}).")
    return RISK_LAMBDAS[risk_index], RISK_LAMBDA_TAGS[risk_index]


def normalize_lambda(raw_lam: str) -> tuple[float, str]:
    """Parse a lambda string (e.g. ``1e-05``, ``0``, ``1e-4``) to canonical form."""
    val = float(raw_lam)
    for grid_val, grid_tag in zip(RISK_LAMBDAS, RISK_LAMBDA_TAGS):
        if math.isclose(val, grid_val, rel_tol=1e-9, abs_tol=1e-12):
            return grid_val, grid_tag
    if val == 0.0:
        return 0.0, "0"
    tag = format(val, "g").replace("e-0", "e-").replace("e+0", "e+")
    return val, tag


def parse_instance_spec(
    spec: str,
    *,
    default_seed: str,
    default_lam: float,
    default_lam_tag: str,
) -> InstanceTask:
    """Parse an instance string from a test suite or CSV into an ``InstanceTask``."""
    match = INSTANCE_SPEC_RE.match(spec.strip())
    if not match:
        raise ValueError(f"Unrecognized instance specification: {spec!r}")

    assets = int(match.group("assets"))
    days = int(match.group("days"))
    raw_seed = match.group("seed")
    seed = default_seed if raw_seed == "XX" else normalize_seed(raw_seed)

    budget_str = match.group("budget")
    if budget_str is not None:
        budget = int(budget_str)
    elif assets in BUDGET_BY_ASSETS:
        budget = BUDGET_BY_ASSETS[assets]
    else:
        raise ValueError(f"Cannot infer budget for asset count {assets} in {spec!r}")

    lam_str = match.group("lam")
    if lam_str is not None:
        lam, lam_tag = normalize_lambda(lam_str)
    else:
        lam, lam_tag = default_lam, default_lam_tag

    task = InstanceTask(
        assets=assets,
        days=days,
        seed=seed,
        budget=budget,
        lam=lam,
        lam_tag=lam_tag,
        raw_spec=spec,
    )
    if not task.instance_dir.is_dir():
        raise FileNotFoundError(
            f"Physical instance directory not found for {task.full_name}: {task.instance_dir}"
        )
    return task


def resolve_test_suite_name(test_arg: str) -> str:
    """Resolve either an integer index or a suite name to its canonical name."""
    raw = test_arg.strip()
    if raw.isdigit():
        idx = int(raw)
        if not 0 <= idx < len(TEST_SUITE_NAMES):
            listing = ", ".join(f"{i}={n}" for i, n in enumerate(TEST_SUITE_NAMES))
            raise ValueError(f"Test index {idx} out of range ({listing}).")
        return TEST_SUITE_NAMES[idx]

    lower_map = {name.lower(): name for name in TEST_SUITE_NAMES}
    lower_map["test_all"] = "all"
    lower_map["test all"] = "all"
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]

    raise ValueError(
        f"Unknown test suite {test_arg!r}. Valid choices (by name or index 0..{len(TEST_SUITE_NAMES) - 1}): "
        + ", ".join(f"{i}:{n}" for i, n in enumerate(TEST_SUITE_NAMES))
    )


def parse_manual_indices(raw_tokens: list[str], total_rows: int) -> list[int]:
    """Parse space- and/or comma-separated manual CSV row indices."""
    indices: list[int] = []
    for token in raw_tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if not re.fullmatch(r"\d+", part):
                raise ValueError(f"Manual index must be a non-negative integer, got {part!r}.")
            idx = int(part)
            if not 0 <= idx < total_rows:
                raise ValueError(
                    f"Manual index {idx} out of range for {BENCHMARKS_CSV.name} "
                    f"(valid indices: 0..{total_rows - 1})."
                )
            indices.append(idx)
    if not indices:
        raise ValueError("Manual selection requires at least one instance index.")
    return indices


def parse_solver_seeds(raw_tokens: list[str] | None) -> list[int] | None:
    """Parse solver RNG seed(s) for multi-seed optimization (e.g. ``1 2 3``).

    Returns ``None`` if ``--solver-seeds`` was omitted, ``[1, 2, 3]`` if passed
    with no values, or a deduplicated list of non-negative integers otherwise.
    """
    if raw_tokens is None:
        return None
    if len(raw_tokens) == 0:
        return list(DEFAULT_SOLVER_SEEDS)

    seeds: list[int] = []
    seen: set[int] = set()
    for token in raw_tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if not re.fullmatch(r"\d+", part):
                raise ValueError(
                    f"Solver RNG seed must be a non-negative integer, got {part!r}."
                )
            seed_val = int(part)
            if seed_val not in seen:
                seen.add(seed_val)
                seeds.append(seed_val)
    if not seeds:
        raise ValueError("Expected at least one integer solver RNG seed.")
    return seeds


def _expand_tasks_with_solver_seeds(
    base_tasks: list[InstanceTask],
    solver_seeds: list[int] | None,
) -> list[InstanceTask]:
    if solver_seeds is None:
        return base_tasks
    return [replace(task, solver_seed=s) for task in base_tasks for s in solver_seeds]


def build_task_stack(
    *,
    csv_rows: list[CsvBenchmarkRow],
    manual_tokens: list[str] | None,
    test_arg: str | None,
    seed_arg: str,
    risk_index: int,
    solver_seeds: list[int] | None = None,
    instance_specs: list[str] | None = None,
) -> tuple[str, list[InstanceTask], dict[str, Any]]:
    """Build the ordered task stack according to CLI selection rules."""
    if instance_specs:
        if manual_tokens is not None or test_arg is not None:
            raise ValueError("--instance cannot be combined with --manual or --tests")
        lam, lam_tag = risk_from_index(risk_index)
        seed = normalize_seed(seed_arg)
        tasks_by_name = {}
        for spec in instance_specs:
            task = parse_instance_spec(
                spec, default_seed=seed, default_lam=lam, default_lam_tag=lam_tag,
            )
            tasks_by_name.setdefault(task.full_name, task)
        return "instances", _expand_tasks_with_solver_seeds(list(tasks_by_name.values()), solver_seeds), {
            "mode": "instances",
            "instance_specs": instance_specs,
            "seed_used": seed,
            "risk_index_used": risk_index,
            "risk_lambda_used": lam,
            "solver_seeds": solver_seeds,
        }

    if manual_tokens is not None:
        # Manual selection overwrites tests, seed, and risk parameters.
        indices = parse_manual_indices(manual_tokens, len(csv_rows))
        default_lam, default_lam_tag = risk_from_index(DEFAULT_RISK_INDEX)
        base_tasks = [
            parse_instance_spec(
                csv_rows[i].instance,
                default_seed=DEFAULT_SEED,
                default_lam=default_lam,
                default_lam_tag=default_lam_tag,
            )
            for i in indices
        ]
        tasks = _expand_tasks_with_solver_seeds(base_tasks, solver_seeds)
        meta = {
            "mode": "manual",
            "manual_indices": indices,
            "seed_used": None,
            "risk_index_used": DEFAULT_RISK_INDEX,
            "risk_lambda_used": default_lam,
            "solver_seeds": solver_seeds,
        }
        return "manual", tasks, meta

    if test_arg is None:
        raise ValueError("Specify --tests <name|index>, --manual <indices...>, or --instance <spec>.")

    suite_name = resolve_test_suite_name(test_arg)
    lam, lam_tag = risk_from_index(risk_index)
    seed = normalize_seed(seed_arg)

    if suite_name == "all":
        # "test all" uses each CSV row's own seed (ignores --seed)
        base_tasks = [
            parse_instance_spec(
                row.instance,
                default_seed=DEFAULT_SEED,
                default_lam=lam,
                default_lam_tag=lam_tag,
            )
            for row in csv_rows
        ]
        tasks = _expand_tasks_with_solver_seeds(base_tasks, solver_seeds)
        meta = {
            "mode": "suite",
            "suite_name": suite_name,
            "seed_used": None,
            "risk_index_used": risk_index,
            "risk_lambda_used": lam,
            "solver_seeds": solver_seeds,
        }
        return suite_name, tasks, meta

    specs = TEST_SUITES_MAP[suite_name]
    # CompareToOptimal explicitly overwrites the Seed parameter (all specs carry fixed seeds).
    base_tasks = [
        parse_instance_spec(
            spec,
            default_seed=seed,
            default_lam=lam,
            default_lam_tag=lam_tag,
        )
        for spec in specs
    ]
    tasks = _expand_tasks_with_solver_seeds(base_tasks, solver_seeds)
    meta = {
        "mode": "suite",
        "suite_name": suite_name,
        "seed_used": None if suite_name == "CompareToOptimal" else seed,
        "risk_index_used": risk_index,
        "risk_lambda_used": lam,
        "solver_seeds": solver_seeds,
    }
    return suite_name, tasks, meta


# ---------------------------------------------------------------------------
# Resource Estimation (Local) & Credit Checking (Cloud)
# ---------------------------------------------------------------------------


def estimate_local_task_resources(task: InstanceTask) -> tuple[float, int]:
    """Heuristically estimate ``(ram_gb, cpu_cores)`` occupied by one local task.

    Calibrated from notebook 01:
      - ``po_a010_t10`` has 710 QUBO variables (~70 sub-programs, runs in <1 min
        locally with modest memory ~0.5 GB).
      - Larger instances scale quadratically in asset count for dense risk
        tensors (``T * N^2``) and linearly in ``n_vars = T * (6*N + 11)``.
    """
    n_vars = task.estimated_qubo_vars
    risk_matrix_gb = (task.days * task.assets * task.assets * 8) / (1024**3)
    # Base Python + Divi + Maestro simulator + QUBO & sub-program memory
    est_ram_gb = 0.35 + (n_vars / 1000.0) * 0.16 + risk_matrix_gb * 8.0
    est_ram_gb = round(max(0.45, est_ram_gb), 2)

    if task.assets <= 10:
        est_cores = 2
    elif task.assets <= 50:
        est_cores = 2
    elif task.assets <= 200:
        est_cores = 4
    else:
        est_cores = 6

    return est_ram_gb, est_cores


def estimate_task_gpu_resources(
    task: InstanceTask,
    *,
    vram_scale: float = 1.0,
    util_scale: float = 1.0,
) -> tuple[float, float]:
    """Heuristically estimate ``(vram_gb, gpu_util_pct)`` for one task on a GPU.

    Scales with QUBO variable count ``T * (6*N + 11)`` and dense risk tensor
    footprint ``T * N^2``. Optional adaptive EMA scales refine estimates online
    from observed GPU load during a run.
    """
    n_vars = task.estimated_qubo_vars
    risk_tensor_gb = (task.days * task.assets * task.assets * 8) / (1024**3)
    base_vram = 0.35 + (n_vars / 1000.0) * 0.14 + risk_tensor_gb * 6.0
    est_vram_gb = round(max(0.40, base_vram * vram_scale), 2)

    if task.assets <= 5:
        base_util = 18.0
    elif task.assets <= 10:
        base_util = 24.0
    elif task.assets <= 50:
        base_util = 38.0
    elif task.assets <= 200:
        base_util = 58.0
    else:
        base_util = 74.0

    day_factor = 1.0 + max(0, task.days - 10) * 0.02
    est_util_pct = round(min(95.0, max(10.0, base_util * day_factor * util_scale)), 1)
    return est_vram_gb, est_util_pct


def _parse_float_safe(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.upper() in ("N/A", "[N/A]", "[NOT SUPPORTED]", "NOT SUPPORTED"):
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _query_nvidia_gpus(max_temp_c: float = GPU_MAX_TEMP_C_DEFAULT) -> list[GPUDeviceStatus]:
    """Query NVIDIA GPUs via ``nvidia-smi`` (works on Linux & Windows)."""
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw,power.limit,"
        "clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    devices: list[GPUDeviceStatus] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        total_mb = _parse_float_safe(parts[2]) or 0.0
        free_mb = _parse_float_safe(parts[3]) or 0.0
        used_mb = _parse_float_safe(parts[4]) or 0.0
        util = _parse_float_safe(parts[5])
        temp = _parse_float_safe(parts[6])
        power_draw = _parse_float_safe(parts[7]) if len(parts) > 7 else None
        power_limit = _parse_float_safe(parts[8]) if len(parts) > 8 else None
        throttle_raw = parts[9] if len(parts) > 9 else ""

        throttling = False
        reasons: list[str] = []
        if throttle_raw:
            try:
                mask = int(throttle_raw, 16)
                if mask & _NV_THROTTLE_MASK:
                    throttling = True
                    reasons.append(f"nv_throttle_mask={throttle_raw}")
            except ValueError:
                pass
        if temp is not None and temp >= max_temp_c:
            throttling = True
            reasons.append(f"temp={temp:.0f}C>={max_temp_c:.0f}C")
        if power_draw is not None and power_limit and power_limit > 0:
            if (power_draw / power_limit) >= 0.97:
                throttling = True
                reasons.append(f"power={power_draw:.0f}W/{power_limit:.0f}W")

        devices.append(
            GPUDeviceStatus(
                index=idx,
                name=name,
                backend="cuda",
                total_vram_gb=round(total_mb / 1024.0, 2),
                free_vram_gb=round(free_mb / 1024.0, 2),
                used_vram_gb=round(used_mb / 1024.0, 2),
                utilization_pct=util,
                temperature_c=temp,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                throttling=throttling,
                throttle_reason="; ".join(reasons) if reasons else None,
                unified_memory=False,
            )
        )
    return devices


def _query_rocm_gpus(max_temp_c: float = GPU_MAX_TEMP_C_DEFAULT) -> list[GPUDeviceStatus]:
    """Query AMD ROCm GPUs via ``rocm-smi``."""
    if shutil.which("rocm-smi") is None:
        return []
    cmd = ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    devices: list[GPUDeviceStatus] = []
    for card_key, info in sorted(payload.items()):
        if not isinstance(info, dict) or not card_key.lower().startswith("card"):
            continue
        m = re.search(r"(\d+)", card_key)
        idx = int(m.group(1)) if m else len(devices)
        total_b = _parse_float_safe(str(info.get("VRAM Total Memory (B)", ""))) or 0.0
        used_b = _parse_float_safe(str(info.get("VRAM Total Used Memory (B)", ""))) or 0.0
        util = _parse_float_safe(str(info.get("GPU use (%)", "")))
        temp: float | None = None
        for k, v in info.items():
            if "temperature" in k.lower():
                parsed_t = _parse_float_safe(str(v))
                if parsed_t is not None:
                    temp = parsed_t if temp is None else max(temp, parsed_t)
        total_gb = total_b / (1024**3)
        used_gb = used_b / (1024**3)
        free_gb = max(0.0, total_gb - used_gb)
        throttling = temp is not None and temp >= max_temp_c
        devices.append(
            GPUDeviceStatus(
                index=idx,
                name=str(info.get("Card series") or f"AMD ROCm {card_key}"),
                backend="rocm",
                total_vram_gb=round(total_gb, 2),
                free_vram_gb=round(free_gb, 2),
                used_vram_gb=round(used_gb, 2),
                utilization_pct=util,
                temperature_c=temp,
                throttling=throttling,
                throttle_reason=f"temp={temp:.0f}C>={max_temp_c:.0f}C" if throttling and temp else None,
                unified_memory=False,
            )
        )
    return devices


def _query_apple_mps_gpu() -> list[GPUDeviceStatus]:
    """Query Apple Silicon Metal/MPS GPU telemetry on macOS without sudo."""
    if platform.system() != "Darwin":
        return []

    model_name: str | None = None
    util_pct: float | None = None
    used_bytes: float = 0.0
    gpu_detected = False

    # 1. Non-privileged IOKit accelerator telemetry (Device Utilization %, In use system memory)
    if shutil.which("ioreg") is not None:
        try:
            proc = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            out = proc.stdout
            if proc.returncode == 0 and out:
                m_model = re.search(r'"model"\s*=\s*"([^"]+)"', out)
                if m_model:
                    model_name = m_model.group(1)
                    gpu_detected = True
                m_util = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
                if not m_util:
                    m_util = re.search(r'"Renderer Utilization %"\s*=\s*(\d+)', out)
                if m_util:
                    util_pct = float(m_util.group(1))
                    gpu_detected = True
                m_mem = re.search(r'"In use system memory"\s*=\s*(\d+)', out)
                if not m_mem:
                    m_mem = re.search(r'"Alloc system memory"\s*=\s*(\d+)', out)
                if m_mem:
                    used_bytes = float(m_mem.group(1))
                    gpu_detected = True
        except (OSError, subprocess.SubprocessError):
            pass

    # 2. Fallback check via sysctl Apple Silicon brand or torch.backends.mps
    recommended_bytes: float | None = None
    try:
        import torch  # type: ignore import-not-found

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_detected = True
            if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
                recommended_bytes = float(torch.mps.recommended_max_memory())
            if used_bytes == 0.0 and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                used_bytes = float(torch.mps.current_allocated_memory())
    except Exception:
        pass

    if not gpu_detected and shutil.which("sysctl") is not None:
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            brand = proc.stdout.strip()
            if proc.returncode == 0 and brand.startswith("Apple "):
                model_name = f"{brand} GPU"
                gpu_detected = True
        except (OSError, subprocess.SubprocessError):
            pass

    if not gpu_detected:
        return []

    sys_res = get_system_resources()
    total_sys_gb = float(sys_res["total_ram_gb"])
    avail_sys_gb = float(sys_res["available_ram_gb"])

    # Unified memory GPU budget: recommended Metal working set (~67% of unified RAM)
    total_vram_gb = (
        recommended_bytes / (1024**3)
        if recommended_bytes and recommended_bytes > 0
        else round(total_sys_gb * 0.67, 2)
    )
    used_vram_gb = used_bytes / (1024**3)
    free_vram_gb = max(0.0, min(total_vram_gb - used_vram_gb, avail_sys_gb))

    # Non-privileged macOS thermal pressure check via pmset
    throttling = False
    reasons: list[str] = []
    if shutil.which("pmset") is not None:
        try:
            proc = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            therm_out = proc.stdout
            m_speed = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm_out)
            if m_speed and int(m_speed.group(1)) < 100:
                throttling = True
                reasons.append(f"CPU_Speed_Limit={m_speed.group(1)}%")
            m_tlevel = re.search(r"Thermal_Level\s*=\s*(\d+)", therm_out)
            if m_tlevel and int(m_tlevel.group(1)) > 0:
                throttling = True
                reasons.append(f"Thermal_Level={m_tlevel.group(1)}")
        except (OSError, subprocess.SubprocessError):
            pass

    return [
        GPUDeviceStatus(
            index=0,
            name=model_name or "Apple Silicon Metal GPU",
            backend="mps",
            total_vram_gb=round(total_vram_gb, 2),
            free_vram_gb=round(free_vram_gb, 2),
            used_vram_gb=round(used_vram_gb, 2),
            utilization_pct=util_pct,
            temperature_c=None,
            throttling=throttling,
            throttle_reason="; ".join(reasons) if reasons else None,
            unified_memory=True,
        )
    ]


def query_gpu_devices(max_temp_c: float = GPU_MAX_TEMP_C_DEFAULT) -> list[GPUDeviceStatus]:
    """Discover and query live load/VRAM/thermal status of all local GPUs."""
    nv = _query_nvidia_gpus(max_temp_c=max_temp_c)
    if nv:
        return nv
    rocm = _query_rocm_gpus(max_temp_c=max_temp_c)
    if rocm:
        return rocm
    return _query_apple_mps_gpu()


def get_system_resources() -> dict[str, float | int]:
    """Query available CPU cores and RAM (GB) in an OS-agnostic way."""
    cpu_count = os.cpu_count() or 2
    if psutil is not None:
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        avail_ram_gb = vm.available / (1024**3)
    else:  # pragma: no cover - fallback when psutil is missing
        total_ram_gb = 8.0
        avail_ram_gb = 4.0
    return {
        "cpu_count": cpu_count,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(avail_ram_gb, 2),
    }


class PassiveGPUMonitor:
    """Passive background GPU monitor for black-box anti-throttling & telemetry.

    Polls local GPU telemetry on a background daemon thread without modifying
    solver CLI flags or device environment variables. Uses live GPU load, free
    VRAM, temperature, and throttle flags—combined with CPU/RAM headroom and
    per-task resource estimates—to pace parallel launches so the system runs as
    many tasks as possible without throttling.
    """

    def __init__(
        self,
        policy: GPUThrottlePolicy,
        *,
        gpu_query_fn=None,
        sys_res_fn=None,
        poll_interval_s: float = GPU_POLL_INTERVAL_S,
        warmup_s: float = GPU_WARMUP_RESERVATION_S,
        cooldown_s: float = GPU_THROTTLE_COOLDOWN_S,
        clock_fn=time.monotonic,
    ) -> None:
        self.policy = policy
        self._gpu_query_fn = gpu_query_fn or (
            lambda: query_gpu_devices(max_temp_c=self.policy.max_temp_c)
        )
        self._sys_res_fn = sys_res_fn or get_system_resources
        self._poll_interval_s = poll_interval_s
        self._warmup_s = warmup_s
        self._cooldown_s = cooldown_s
        self._clock = clock_fn

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_jobs: list[ActiveJob] = []

        self._last_poll_mono: float = -1e9
        self._cached_gpus: list[GPUDeviceStatus] = []
        self._gpu_cooldown_until: dict[int, float] = {}
        self._throttle_logged: set[int] = set()
        self.throttle_events: list[dict[str, Any]] = []

    def start(self, active_jobs: list[ActiveJob]) -> None:
        """Start passive background polling in a daemon thread."""
        if not self.policy.enabled:
            return
        with self._lock:
            self._active_jobs = active_jobs
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            name="autorunner-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_gpus(force=True)
                with self._lock:
                    jobs = list(self._active_jobs)
                self._observe_jobs(jobs)
            except Exception:
                pass
            self._stop_event.wait(self._poll_interval_s)

    def poll_gpus(self, *, force: bool = False) -> list[GPUDeviceStatus]:
        if not self.policy.enabled:
            return []
        now = self._clock()
        with self._lock:
            if not force and (now - self._last_poll_mono) < self._poll_interval_s:
                return list(self._cached_gpus)

        gpus = self._gpu_query_fn()
        cool_target = self.policy.max_temp_c - GPU_TEMP_HYSTERESIS_C

        with self._lock:
            self._cached_gpus = list(gpus)
            self._last_poll_mono = now
            for g in gpus:
                is_hot = g.temperature_c is not None and g.temperature_c >= self.policy.max_temp_c
                if g.throttling or is_hot:
                    self._gpu_cooldown_until[g.index] = now + self._cooldown_s
                    if g.index not in self._throttle_logged:
                        self._throttle_logged.add(g.index)
                        reason = g.throttle_reason or f"temp={g.temperature_c}C"
                        print(
                            f"[THROTTLE] GPU {g.index} ({g.name}) throttling ({reason}); "
                            "pausing new parallel launches until cooled.",
                            file=sys.stderr,
                        )
                        self.throttle_events.append(
                            {
                                "time": dt.datetime.now().isoformat(),
                                "gpu_index": g.index,
                                "gpu_name": g.name,
                                "reason": reason,
                                "temperature_c": g.temperature_c,
                                "utilization_pct": g.utilization_pct,
                            }
                        )
                else:
                    if g.index in self._gpu_cooldown_until:
                        temp_ok = g.temperature_c is None or g.temperature_c <= cool_target
                        if now >= self._gpu_cooldown_until[g.index] and temp_ok:
                            del self._gpu_cooldown_until[g.index]
                            self._throttle_logged.discard(g.index)
            return list(self._cached_gpus)

    def is_gpu_cooling(self, gpu: GPUDeviceStatus) -> bool:
        if gpu.throttling:
            return True
        if gpu.temperature_c is not None and gpu.temperature_c >= self.policy.max_temp_c:
            return True
        with self._lock:
            until = self._gpu_cooldown_until.get(gpu.index)
        if until is None:
            return False
        now = self._clock()
        cool_target = self.policy.max_temp_c - GPU_TEMP_HYSTERESIS_C
        temp_ok = gpu.temperature_c is None or gpu.temperature_c <= cool_target
        return not (now >= until and temp_ok)

    def _observe_jobs(self, jobs: list[ActiveJob]) -> None:
        if not jobs:
            return
        with self._lock:
            gpus = list(self._cached_gpus)
        if not gpus:
            return
        max_util = max(
            (g.utilization_pct for g in gpus if g.utilization_pct is not None),
            default=None,
        )
        any_throttling = any(g.throttling or self.is_gpu_cooling(g) for g in gpus)
        for job in jobs:
            if max_util is not None:
                if job.peak_gpu_util_pct is None or max_util > job.peak_gpu_util_pct:
                    job.peak_gpu_util_pct = max_util
            if any_throttling:
                job.throttling_observed = True

    def latest_snapshot(self) -> list[GPUDeviceStatus]:
        with self._lock:
            return list(self._cached_gpus)

    def pick_next_task(
        self,
        pending: list[InstanceTask],
        active: list[ActiveJob],
        max_parallel: int,
    ) -> int | None:
        """Return index in ``pending`` safe to launch now, or ``None`` to wait."""
        if not pending or len(active) >= max_parallel:
            return None

        sys_res = self._sys_res_fn()
        sys_total_ram = float(sys_res["total_ram_gb"])
        sys_avail_ram = float(sys_res["available_ram_gb"])
        cpu_count = int(sys_res["cpu_count"])
        usable_cores = max(1, cpu_count - 1)
        now = self._clock()

        gpus = self.poll_gpus()
        any_cooling = any(self.is_gpu_cooling(g) for g in gpus)
        if any_cooling:
            # Pause new launches while any GPU is thermally/power throttling
            return None

        if not active:
            # Never deadlock when idle and no GPU is throttling
            return 0

        active_cpu_cores = sum(estimate_local_task_resources(j.task)[1] for j in active)
        warmup_ram = sum(
            estimate_local_task_resources(j.task)[0]
            for j in active
            if (now - j.start_mono) < self._warmup_s
        )
        eff_avail_ram = max(0.0, sys_avail_ram - warmup_ram)
        ram_reserve = sys_total_ram * 0.15

        warmup_gpu_jobs = [j for j in active if (now - j.start_mono) < self._warmup_s]
        warmup_vram = sum(estimate_task_gpu_resources(j.task)[0] for j in warmup_gpu_jobs)
        warmup_util = sum(estimate_task_gpu_resources(j.task)[1] for j in warmup_gpu_jobs)

        for idx, task in enumerate(pending):
            est_ram, est_cores = estimate_local_task_resources(task)
            cores_ok = (active_cpu_cores + est_cores) <= usable_cores
            ram_ok = (eff_avail_ram - est_ram) >= ram_reserve

            if active and not (cores_ok and ram_ok):
                continue

            if gpus:
                est_vram, est_util = estimate_task_gpu_resources(task)
                gpu_ok = False
                for g in gpus:
                    vram_reserve = g.total_vram_gb * (self.policy.vram_headroom_pct / 100.0)
                    eff_free_vram = max(0.0, g.free_vram_gb - warmup_vram)
                    if g.unified_memory:
                        eff_free_vram = min(eff_free_vram, max(0.0, eff_avail_ram - ram_reserve))
                        vram_reserve = 0.0
                    live_util = g.utilization_pct if g.utilization_pct is not None else 0.0
                    eff_util = min(100.0, live_util + warmup_util)

                    vram_fits = (eff_free_vram - est_vram) >= vram_reserve
                    util_fits = (eff_util + est_util) <= self.policy.max_util_pct
                    if not active and live_util < self.policy.max_util_pct and eff_free_vram >= est_vram:
                        vram_fits = True
                        util_fits = True
                    if vram_fits and util_fits:
                        gpu_ok = True
                        break
                if not gpu_ok:
                    continue

            return idx

        return None


def calculate_local_concurrency(
    tasks: list[InstanceTask],
    *,
    gpu_policy: GPUThrottlePolicy | None = None,
    gpus: list[GPUDeviceStatus] | None = None,
    max_workers_override: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Compute how many local tasks can safely run in parallel."""
    policy = gpu_policy or GPUThrottlePolicy()
    sys_res = get_system_resources()
    cpu_count = int(sys_res["cpu_count"])
    avail_ram_gb = float(sys_res["available_ram_gb"])

    # Keep 20% RAM headroom and at least 1 CPU core for OS & runner
    usable_ram_gb = max(0.5, avail_ram_gb * 0.80)
    usable_cores = max(1, cpu_count - 1)

    cpu_estimates = [estimate_local_task_resources(t) for t in tasks]
    gpu_estimates = [estimate_task_gpu_resources(t) for t in tasks]

    min_task_ram = min(ram for ram, _ in cpu_estimates)
    min_task_cores = min(cores for _, cores in cpu_estimates)

    ram_slots = max(1, int(usable_ram_gb // min_task_ram))
    cpu_slots = max(1, int(usable_cores // min(min_task_cores, usable_cores)))
    computed_max = max(1, min(len(tasks), ram_slots, cpu_slots))

    detected_gpus = (
        []
        if not policy.enabled
        else (gpus if gpus is not None else query_gpu_devices(max_temp_c=policy.max_temp_c))
    )

    if max_workers_override is not None:
        final_max = max(1, min(len(tasks), max_workers_override))
    else:
        final_max = computed_max

    report = {
        "system": sys_res,
        "usable_ram_gb": round(usable_ram_gb, 2),
        "usable_cores": usable_cores,
        "gpu_policy": policy.to_dict(),
        "gpus": [g.to_dict() for g in detected_gpus],
        "per_task_estimates": {
            t.run_key: {
                "instance": t.full_name,
                "solver_seed": t.solver_seed,
                "estimated_qubo_vars": t.estimated_qubo_vars,
                "estimated_ram_gb": ram,
                "estimated_cores": cores,
                "estimated_vram_gb": vram,
                "estimated_gpu_util_pct": gpu_util,
            }
            for t, (ram, cores), (vram, gpu_util) in zip(tasks, cpu_estimates, gpu_estimates)
        },
        "computed_max_parallel": computed_max,
        "selected_max_parallel": final_max,
    }
    return final_max, report


def _load_repo_dotenv() -> None:
    """Load ``QORO_API_KEY`` from ``<REPO_ROOT>/.env`` if not already in env."""
    if os.environ.get("QORO_API_KEY"):
        return
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def query_qoro_credits() -> tuple[float | None, dict[str, Any]]:
    """Query Qoro cloud credit balance via ``divi.backends.QoroService``.

    Returns ``(balance_float_or_None, info_dict)``.
    """
    _load_repo_dotenv()
    try:
        from divi.backends import QoroService  # type: ignore import-not-found

        service = QoroService()
        raw = service.get_credit_balance()
        if isinstance(raw, dict):
            balance = _parse_numeric(str(raw.get("balance")))
            total_used = _parse_numeric(str(raw.get("total_used")))
            return (
                float(balance) if balance is not None else None,
                {
                    "available": True,
                    "balance": float(balance) if balance is not None else None,
                    "total_used": float(total_used) if total_used is not None else None,
                    "raw": raw,
                },
            )
        bal = _parse_numeric(str(raw))
        return (
            float(bal) if bal is not None else None,
            {"available": True, "balance": float(bal) if bal is not None else None, "raw": raw},
        )
    except Exception as exc:
        return None, {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def calculate_cloud_concurrency(
    tasks: list[InstanceTask],
    *,
    low_credit_threshold: float = LOW_CREDIT_THRESHOLD_DEFAULT,
    max_workers_override: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Check Qoro cloud credits and decide concurrency (1 at a time if low)."""
    balance, credit_info = query_qoro_credits()

    # Heuristic estimate of relative circuit load: ~70 sub-programs for 710 vars
    est_subprograms = {
        t.run_key: max(1, math.ceil(t.estimated_qubo_vars / 10)) for t in tasks
    }

    low_on_credits = balance is None or balance <= low_credit_threshold
    if low_on_credits:
        computed_max = 1
        reason = (
            "credit balance unavailable; defaulting to sequential execution"
            if balance is None
            else f"credit balance ({balance:.2f}) <= threshold ({low_credit_threshold:.2f}); "
            "running one instance at a time"
        )
    else:
        computed_max = max(1, min(len(tasks), CLOUD_MAX_PARALLEL_DEFAULT))
        reason = (
            f"credit balance ({balance:.2f}) > threshold ({low_credit_threshold:.2f}); "
            f"running up to {computed_max} instances in parallel"
        )

    if max_workers_override is not None:
        final_max = max(1, min(len(tasks), max_workers_override))
    else:
        final_max = computed_max

    report = {
        "credit_check": credit_info,
        "low_credit_threshold": low_credit_threshold,
        "low_on_credits": low_on_credits,
        "estimated_subprograms": est_subprograms,
        "reason": reason,
        "computed_max_parallel": computed_max,
        "selected_max_parallel": final_max,
    }
    return final_max, report


# ---------------------------------------------------------------------------
# Directory & Unique Filename Management
# ---------------------------------------------------------------------------


def create_run_directory(test_name: str, now: dt.datetime | None = None) -> Path:
    """Create ``<REPO_ROOT>/AutoData/<test_name>_<HHMM>`` (unique if exists)."""
    AUTODATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = (now or dt.datetime.now()).strftime("%H%M")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_name).strip("_") or "run"
    base = AUTODATA_DIR / f"{safe_name}_{stamp}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = AUTODATA_DIR / f"{safe_name}_{stamp}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def get_unique_json_path(run_dir: Path, test_name: str, finished_at: dt.datetime) -> Path:
    """Return a non-overwriting path ``<TYPE>_<HHMM>[_<n>].json`` inside ``run_dir``."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_name).strip("_") or "run"
    time_tag = finished_at.strftime("%H%M")
    candidate = run_dir / f"{safe_name}_{time_tag}.json"
    counter = 1
    while candidate.exists():
        candidate = run_dir / f"{safe_name}_{time_tag}_{counter}.json"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Solution Checker & Direct Output Parsing
# ---------------------------------------------------------------------------


def ensure_qoblib_data_cache(task: InstanceTask) -> None:
    """Mirror local ``instances/<base>`` into ``qoblib_data/`` if missing.

    ``solution_checker.py`` calls ``qoblib_tools.load_portfolio_instance``,
    which looks in ``qoblib_data/06-portfolio/instances/<base>`` before falling
    back to downloading from GitHub. Copying already-present local files ensures
    ``solution_checker.py`` works instantaneously and offline.
    """
    src_dir = task.instance_dir
    dst_dir = QOBLIB_DATA_INSTANCES_DIR / task.base_name
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("stock_prices.txt.gz", "covariance_matrices.txt.gz"):
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        if src_file.is_file() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)


def find_solution_file(run_dir: Path, task: InstanceTask, existing_before: set[Path]) -> Path | None:
    """Locate the ``.sol`` file generated for ``task`` inside ``run_dir``."""
    lam_sci = f"{task.lam:.0e}" if task.lam != 0 else "0"
    lam_dec = format(task.lam, "f").rstrip("0").rstrip(".") if task.lam != 0 else "0"

    preferred_names = [
        f"{task.full_name}.sol",
        f"{task.full_name}_day_swap/{task.full_name}.sol",
        f"{task.csv_instance_key}_l{task.lam_tag}.sol",
        f"{task.base_name}_b{task.budget:03d}_l{lam_sci}.sol",
        f"{task.csv_instance_key}_l{lam_sci}.sol",
        f"{task.base_name}_b{task.budget:03d}_l{lam_dec}.sol",
        f"{task.base_name}.sol",
    ]
    for name in preferred_names:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate

    # Fallback: any newly created .sol file referencing this instance base
    base_no_prefix = task.base_name.removeprefix("po_")
    candidates = [
        p
        for p in run_dir.glob("*.sol")
        if p not in existing_before and not p.name.endswith(".ckpt.sol")
        and (base_no_prefix in p.name or task.base_name in p.name)
    ]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def parse_solution_checker_output(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    """Parse terminal output produced by ``solution_checker.py``.

    Expected stdout lines from ``solution_checker.py``::

        Your score:  objective=-69215 (feasible; risk=23400, return=-97784, ...)
        Constraint violations:  []
    """
    result: dict[str, Any] = {
        "returncode": returncode,
        "objective": None,
        "feasible": None,
        "breakdown": {},
        "violations": [],
        "raw_stdout": stdout.strip(),
    }
    if stderr.strip():
        result["raw_stderr"] = stderr.strip()

    score_match = re.search(
        r"Your score:\s*objective=(-?\d+)\s*\((feasible|INFEASIBLE);\s*([^)]*)\)",
        stdout,
    )
    if score_match:
        result["objective"] = int(score_match.group(1))
        result["feasible"] = score_match.group(2) == "feasible"
        breakdown: dict[str, int] = {}
        for part in score_match.group(3).split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    breakdown[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        result["breakdown"] = breakdown

    viol_match = re.search(r"Constraint violations:\s*(\[.*\])", stdout, re.DOTALL)
    if viol_match:
        raw_list = viol_match.group(1).splitlines()[0].strip()
        try:
            parsed_viol = ast.literal_eval(raw_list)
            if isinstance(parsed_viol, list):
                result["violations"] = [str(v) for v in parsed_viol]
        except (SyntaxError, ValueError):
            result["violations"] = [raw_list]

    return result


def run_solution_checker(sol_path: Path, task: InstanceTask) -> dict[str, Any]:
    """Execute ``solution_checker.py <sol_path>`` and return parsed dictionary."""
    ensure_qoblib_data_cache(task)
    cmd = [sys.executable, str(SOLUTION_CHECKER), str(sol_path.resolve())]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_solution_checker_output(proc.stdout, proc.stderr, proc.returncode)


def _extract_last_json_object(text: str) -> tuple[Any | None, str]:
    """Extract the last JSON object/array from stdout text, returning ``(obj, remaining)``."""
    stripped = text.strip()
    if not stripped:
        return None, ""

    # 1. Explicit marker line: AUTORUNNER_RESULT: {...}
    for line in reversed(stripped.splitlines()):
        if "AUTORUNNER_RESULT:" in line:
            payload = line.split("AUTORUNNER_RESULT:", 1)[1].strip()
            try:
                return json.loads(payload), stripped
            except json.JSONDecodeError:
                pass

    # 2. Whole stdout is valid JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, (dict, list)):
            return obj, ""
    except json.JSONDecodeError:
        pass

    # 3. Scan backwards for a JSON object starting with '{'
    decoder = json.JSONDecoder()
    for idx in range(len(stripped) - 1, -1, -1):
        if stripped[idx] == "{":
            try:
                obj, end = decoder.raw_decode(stripped, idx)
                if stripped[end:].strip() == "":
                    return obj, stripped[:idx].strip()
            except json.JSONDecodeError:
                continue

    return None, stripped


def parse_direct_task_output(stdout: str) -> tuple[int | float | None, Any]:
    """Parse direct outputs (score + arbitrary JSON object) from task stdout."""
    parsed_json, remaining = _extract_last_json_object(stdout)
    reported_score: int | float | None = None
    extra: Any = {}

    if isinstance(parsed_json, dict):
        for key in ("score", "best_score", "objective"):
            if key in parsed_json and isinstance(parsed_json[key], (int, float)):
                reported_score = parsed_json[key]
                break
        if "extra" in parsed_json:
            extra = parsed_json["extra"]
        else:
            extra = {k: v for k, v in parsed_json.items() if k not in ("score", "best_score")}
    elif parsed_json is not None:
        extra = parsed_json

    if reported_score is None:
        search_text = remaining or stdout
        score_patterns = [
            r"(?:best[_\s]*)?(?:score|objective)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            r"^\s*(-?\d+(?:\.\d+)?)\s*$",
        ]
        for pat in score_patterns:
            matches = re.findall(pat, search_text, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                reported_score = _parse_numeric(matches[-1])
                break

    return reported_score, extra


# ---------------------------------------------------------------------------
# OS-Agnostic Terminal Launching & Worker Shim
# ---------------------------------------------------------------------------


def get_task_work_dir(run_dir: Path, task: InstanceTask, multi_seed: bool) -> Path:
    """Return the execution working directory for ``task``.

    When multiple solver RNG seeds run for the same instance, each seed runs in
    an isolated ``seed_<N>/`` subdirectory inside ``run_dir`` so parallel tasks
    cannot overwrite each other's ``<instance>.sol`` file.
    """
    if multi_seed and task.solver_seed is not None:
        task_dir = run_dir / f"seed_{task.solver_seed}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir
    return run_dir


def resolve_warm_start_args(
    script_path: Path, task: InstanceTask, extra_args: list[str] | None,
) -> list[str]:
    """Resolve warm starts before workers change their working directory."""
    out = list(extra_args or [])
    if script_path.name not in WARM_START_SCRIPTS:
        return out
    for i, arg in enumerate(out):
        if arg == "--init":
            if i + 1 == len(out) or out[i + 1].startswith("--"):
                raise ValueError("--init requires a .sol file or directory")
            index, value = i + 1, out[i + 1]
        elif arg.startswith("--init="):
            index, value = i, arg.removeprefix("--init=")
        else:
            continue
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            path /= f"{task.full_name}.sol"
        if not path.is_file():
            raise FileNotFoundError(f"Warm-start solution for {task.full_name} not found: {path}")
        out[index] = f"--init={path}" if index == i else str(path)
        return out
    if script_path.name in DAY_SWAP_SCRIPTS:
        raise ValueError(f"{script_path.name} requires --init after the runner's -- separator")
    return out


def build_target_command(
    script_path: Path,
    task: InstanceTask,
    task_cwd: Path,
    local: bool,
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Construct the command line and environment for executing one task."""
    rel_instance_dir = os.path.relpath(task.instance_dir, task_cwd)
    cloud_flag = "false" if local else "true"

    solver_args = resolve_warm_start_args(script_path, task, extra_args)
    cmd = [
        sys.executable,
        str(script_path.resolve()),
        "--instance",
        task.full_name,
        "--instance-dir",
        rel_instance_dir,
        "--assets",
        str(task.assets),
        "--days",
        str(task.days),
        "--lambda",
        task.lam_tag,
        "--budget",
        str(task.budget),
        "--cloud",
        cloud_flag,
    ]
    if task.solver_seed is not None:
        cmd.extend(["--solver-seed", str(task.solver_seed)])
    if script_path.name in DAY_SWAP_SCRIPTS:
        # The standalone day-swap CLIs take a base name and use the referee cache.
        cmd = [
            sys.executable, str(script_path.resolve()), "--instance", task.base_name,
            "--budget", str(task.budget), "--lam", task.lam_tag,
        ]
        if task.solver_seed is not None:
            cmd.extend(["--seed", str(task.solver_seed)])
        # Keep the detected final solution in the job directory and isolate
        # the parallel workers' intermediate files by instance.
        cmd.extend(solver_args)
        if script_path.name == "day_swap_parallel.py":
            cmd.extend(["--out", f"{task.full_name}_day_swap"])
        else:
            cmd.extend(["--out", ".", "--canonical-sol"])
    else:
        cmd.extend(solver_args)

    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH", "")
    repo_str = str(REPO_ROOT)
    if existing_py_path:
        env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_py_path}"
    else:
        env["PYTHONPATH"] = repo_str

    # Mirror parameters in environment variables for convenience
    env["AUTORUNNER_INSTANCE"] = task.full_name
    env["AUTORUNNER_INSTANCE_DIR"] = rel_instance_dir
    env["AUTORUNNER_ASSETS"] = str(task.assets)
    env["AUTORUNNER_DAYS"] = str(task.days)
    env["AUTORUNNER_LAMBDA"] = task.lam_tag
    env["AUTORUNNER_BUDGET"] = str(task.budget)
    env["AUTORUNNER_CLOUD"] = cloud_flag
    if task.solver_seed is not None:
        env["AUTORUNNER_SOLVER_SEED"] = str(task.solver_seed)
    return cmd, env


def run_worker_shim(spec_path: Path) -> int:
    """Internal entry point executed inside a visible OS terminal window.

    Streams the target script's stdout/stderr live to the terminal while also
    recording them to log files and writing a ``.done.json`` completion file so
    the parent ``auto_runner.py`` process knows the exact timestamps, exit code,
    and outputs across macOS, Linux, and Windows.
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cmd: list[str] = spec["cmd"]
    cwd: str = spec["cwd"]
    env_updates: dict[str, str] = spec.get("env_updates", {})
    stdout_log = Path(spec["stdout_log"])
    stderr_log = Path(spec["stderr_log"])
    done_file = Path(spec["done_file"])
    pid_file = Path(spec["pid_file"])
    instance_name: str = spec["instance_name"]

    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    print("=" * 72, flush=True)
    print(f"[AutoRunner] Instance : {instance_name}", flush=True)
    print(f"[AutoRunner] Workdir  : {cwd}", flush=True)
    print(f"[AutoRunner] Command  : {' '.join(cmd)}", flush=True)
    print("=" * 72, flush=True)

    env = os.environ.copy()
    env.update(env_updates)

    start_dt = dt.datetime.now()
    t0 = time.monotonic()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Read stdout & stderr concurrently via two lightweight threads
        import threading

        def _pump(stream, sink_list: list[str], target_io) -> None:
            try:
                for line in iter(stream.readline, ""):
                    sink_list.append(line)
                    target_io.write(line)
                    target_io.flush()
            finally:
                stream.close()

        t_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_chunks, sys.stdout))
        t_err = threading.Thread(target=_pump, args=(proc.stderr, stderr_chunks, sys.stderr))
        t_out.start()
        t_err.start()
        returncode = proc.wait()
        t_out.join()
        t_err.join()
    except Exception as exc:  # pragma: no cover
        returncode = 1
        err_msg = f"[AutoRunner Worker Error] {type(exc).__name__}: {exc}\n"
        stderr_chunks.append(err_msg)
        sys.stderr.write(err_msg)
        sys.stderr.flush()

    duration = time.monotonic() - t0
    end_dt = dt.datetime.now()

    stdout_log.write_text("".join(stdout_chunks), encoding="utf-8")
    stderr_log.write_text("".join(stderr_chunks), encoding="utf-8")

    done_payload = {
        "returncode": returncode,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_seconds": round(duration, 4),
    }
    tmp_done = done_file.with_suffix(".tmp")
    tmp_done.write_text(json.dumps(done_payload), encoding="utf-8")
    tmp_done.replace(done_file)

    print("=" * 72, flush=True)
    print(f"[AutoRunner] Finished {instance_name} (exit={returncode}, {duration:.2f}s)", flush=True)
    print("=" * 72, flush=True)
    return returncode


@dataclass
class ActiveJob:
    """Tracks one in-flight task (either silent subprocess or visible terminal)."""

    task: InstanceTask
    run_dir: Path
    task_cwd: Path
    existing_sols_before: set[Path]
    start_dt: dt.datetime
    start_mono: float
    credits_before: float | None
    mode: str  # "silent" or "visible"
    gpu_util_before_pct: float | None = None
    gpu_vram_free_before_gb: float | None = None
    peak_gpu_util_pct: float | None = None
    throttling_observed: bool = False
    popen: subprocess.Popen[str] | None = None
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    done_file: Path | None = None
    pid_file: Path | None = None
    launch_mono: float = field(default_factory=time.monotonic)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _launch_visible_terminal(
    spec_path: Path,
    tmp_dir: Path,
    task_name: str,
) -> tuple[bool, subprocess.Popen[str] | None, str]:
    """Attempt to launch ``auto_runner.py --_worker <spec_path>`` in a native OS terminal.

    Returns ``(launched_ok, optional_popen, diagnostic_message)``.
    """
    system = platform.system()
    shim_cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(spec_path)]

    if system == "Windows":
        create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        try:
            proc = subprocess.Popen(shim_cmd, creationflags=create_new_console)
            return True, proc, "Windows CREATE_NEW_CONSOLE"
        except OSError as exc:
            return False, None, f"Windows terminal launch failed: {exc}"

    # POSIX helper script avoids nested shell/AppleScript quoting hazards
    launcher_sh = tmp_dir / f"launch_{task_name}.sh"
    quoted_cmd = " ".join(shlex.quote(arg) for arg in shim_cmd)
    launcher_sh.write_text(f"#!/bin/sh\n{quoted_cmd}\n", encoding="utf-8")
    launcher_sh.chmod(0o755)

    if system == "Darwin":
        if shutil.which("osascript") is None:
            return False, None, "osascript not available"
        applescript_cmd = f"sh {shlex.quote(str(launcher_sh))}"
        escaped = applescript_cmd.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "Terminal" to do script "{escaped}"'
        res = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return True, None, "macOS Terminal.app"
        return False, None, f"macOS Terminal launch failed: {res.stderr.strip()}"

    # Linux / BSD desktop terminals
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, None, "No DISPLAY/WAYLAND_DISPLAY set (headless Linux)"

    linux_terminals: list[tuple[str, list[str]]] = [
        ("x-terminal-emulator", ["-e", "sh", str(launcher_sh)]),
        ("gnome-terminal", ["--", "sh", str(launcher_sh)]),
        ("konsole", ["-e", "sh", str(launcher_sh)]),
        ("xfce4-terminal", ["-e", f"sh {shlex.quote(str(launcher_sh))}"]),
        ("alacritty", ["-e", "sh", str(launcher_sh)]),
        ("kitty", ["sh", str(launcher_sh)]),
        ("xterm", ["-e", "sh", str(launcher_sh)]),
        ("mate-terminal", ["--", "sh", str(launcher_sh)]),
        ("lxterminal", ["-e", f"sh {shlex.quote(str(launcher_sh))}"]),
    ]
    for binary, args in linux_terminals:
        found = shutil.which(binary)
        if found:
            try:
                proc = subprocess.Popen([found, *args])
                return True, proc, f"Linux terminal ({binary})"
            except OSError:
                continue

    return False, None, "No supported Linux terminal emulator found"


def start_task_job(
    *,
    script_path: Path,
    task: InstanceTask,
    run_dir: Path,
    task_cwd: Path,
    tmp_dir: Path,
    local: bool,
    silent: bool,
    track_credits: bool = False,
    gpu_snapshots: list[GPUDeviceStatus] | None = None,
    extra_args: list[str] | None = None,
) -> ActiveJob:
    """Start one instance task either silently or in a visible OS terminal."""
    if script_path.name in DAY_SWAP_SCRIPTS:
        ensure_qoblib_data_cache(task)
    cmd, env = build_target_command(
        script_path, task, task_cwd, local, extra_args=extra_args
    )
    existing_sols = set(task_cwd.glob("*.sol"))

    credits_before: float | None = None
    if not local and track_credits:
        credits_before, _ = query_qoro_credits()

    start_dt = dt.datetime.now()
    start_mono = time.monotonic()

    snaps = gpu_snapshots or []
    gpu_util_before = max(
        (g.utilization_pct for g in snaps if g.utilization_pct is not None),
        default=None,
    )
    gpu_vram_before = (
        round(sum(g.free_vram_gb for g in snaps), 2) if snaps else None
    )

    if not silent:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.run_key)
        spec_path = tmp_dir / f"{safe}.spec.json"
        stdout_log = tmp_dir / f"{safe}.stdout.log"
        stderr_log = tmp_dir / f"{safe}.stderr.log"
        done_file = tmp_dir / f"{safe}.done.json"
        pid_file = tmp_dir / f"{safe}.pid"

        env_updates = {
            k: env[k]
            for k in (
                "PYTHONPATH",
                "AUTORUNNER_INSTANCE",
                "AUTORUNNER_INSTANCE_DIR",
                "AUTORUNNER_ASSETS",
                "AUTORUNNER_DAYS",
                "AUTORUNNER_LAMBDA",
                "AUTORUNNER_BUDGET",
                "AUTORUNNER_CLOUD",
                "AUTORUNNER_SOLVER_SEED",
            )
            if k in env
        }
        spec_path.write_text(
            json.dumps(
                {
                    "instance_name": task.display_name,
                    "cmd": cmd,
                    "cwd": str(task_cwd),
                    "env_updates": env_updates,
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "done_file": str(done_file),
                    "pid_file": str(pid_file),
                }
            ),
            encoding="utf-8",
        )

        launched, proc, diag = _launch_visible_terminal(spec_path, tmp_dir, safe)
        if launched:
            print(f"[LAUNCH] {task.display_name} in visible terminal ({diag})")
            return ActiveJob(
                task=task,
                run_dir=run_dir,
                task_cwd=task_cwd,
                existing_sols_before=existing_sols,
                start_dt=start_dt,
                start_mono=start_mono,
                credits_before=credits_before,
                mode="visible",
                gpu_util_before_pct=gpu_util_before,
                gpu_vram_free_before_gb=gpu_vram_before,
                peak_gpu_util_pct=gpu_util_before,
                popen=proc,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                done_file=done_file,
                pid_file=pid_file,
            )
        print(
            f"[WARN] Could not open visible terminal for {task.display_name} ({diag}); "
            "falling back to silent execution.",
            file=sys.stderr,
        )

    # Silent execution
    print(f"[LAUNCH] {task.display_name} (silent)")
    proc = subprocess.Popen(
        cmd,
        cwd=str(task_cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return ActiveJob(
        task=task,
        run_dir=run_dir,
        task_cwd=task_cwd,
        existing_sols_before=existing_sols,
        start_dt=start_dt,
        start_mono=start_mono,
        credits_before=credits_before,
        mode="silent",
        gpu_util_before_pct=gpu_util_before,
        gpu_vram_free_before_gb=gpu_vram_before,
        peak_gpu_util_pct=gpu_util_before,
        popen=proc,
    )


def poll_active_job(job: ActiveJob) -> tuple[bool, int, str, str, str, str, float]:
    """Poll an ``ActiveJob``.

    Returns ``(finished, returncode, stdout, stderr, start_iso, end_iso, duration_s)``.
    """
    if job.mode == "silent":
        assert job.popen is not None
        rc = job.popen.poll()
        if rc is None:
            return False, 0, "", "", "", "", 0.0
        stdout, stderr = job.popen.communicate()
        end_dt = dt.datetime.now()
        duration = round(time.monotonic() - job.start_mono, 4)
        return True, rc, stdout, stderr, job.start_dt.isoformat(), end_dt.isoformat(), duration

    # Visible terminal mode
    assert job.done_file is not None
    assert job.stdout_log is not None
    assert job.stderr_log is not None
    assert job.pid_file is not None

    if job.done_file.is_file():
        info = json.loads(job.done_file.read_text(encoding="utf-8"))
        stdout = job.stdout_log.read_text(encoding="utf-8") if job.stdout_log.is_file() else ""
        stderr = job.stderr_log.read_text(encoding="utf-8") if job.stderr_log.is_file() else ""
        return (
            True,
            int(info.get("returncode", 0)),
            stdout,
            stderr,
            str(info.get("start_time", job.start_dt.isoformat())),
            str(info.get("end_time", dt.datetime.now().isoformat())),
            float(info.get("duration_seconds", round(time.monotonic() - job.start_mono, 4))),
        )

    # Detect prematurely closed terminal window once worker PID has been written
    if job.pid_file.is_file():
        try:
            worker_pid = int(job.pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            worker_pid = -1
        if worker_pid > 0 and not _pid_is_alive(worker_pid) and not job.done_file.is_file():
            end_dt = dt.datetime.now()
            duration = round(time.monotonic() - job.start_mono, 4)
            stdout = job.stdout_log.read_text(encoding="utf-8") if job.stdout_log.is_file() else ""
            stderr = job.stderr_log.read_text(encoding="utf-8") if job.stderr_log.is_file() else ""
            stderr += "\n[AutoRunner] Terminal window closed before task completion."
            return True, -1, stdout, stderr, job.start_dt.isoformat(), end_dt.isoformat(), duration

    return False, 0, "", "", "", "", 0.0


# ---------------------------------------------------------------------------
# Parallel Scheduler & Report Aggregation
# ---------------------------------------------------------------------------


def _result_rank_key(result: TaskExecutionResult) -> tuple[int, int, float, int]:
    """Ranking key for picking the best seed run (smaller tuple is better).

    Prefers feasible checked solutions, then lowest QOBLIB objective (minimization),
    falling back to reported score and clean exit code.
    """
    checker = result.checker_result or {}
    checked_obj = checker.get("objective")
    feasible = checker.get("feasible") is True and checked_obj is not None
    score = checked_obj if checked_obj is not None else result.reported_score
    has_score = score is not None
    numeric_score = float(score) if score is not None else float("inf")
    exit_rank = 0 if result.exit_code == 0 else 1
    return (0 if feasible else 1, 0 if has_score else 1, numeric_score, exit_rank)


def _group_results_by_instance(
    results: list[TaskExecutionResult],
) -> list[tuple[str, list[TaskExecutionResult]]]:
    """Group execution results by canonical instance name preserving first-seen order."""
    grouped: dict[str, list[TaskExecutionResult]] = {}
    for res in results:
        grouped.setdefault(res.task.full_name, []).append(res)
    return list(grouped.items())


def _compute_gap(
    score: int | float | None,
    best_known: int | float | None,
) -> float | None:
    if score is None or best_known is None or best_known == 0:
        return None
    return round((float(score) - float(best_known)) / abs(float(best_known)), 6)


def _format_gap(
    score: int | float | None,
    best_known: int | float | None,
) -> str:
    gap = _compute_gap(score, best_known)
    return "N/A" if gap is None else f"{gap:+.2%}"


def execute_task_stack(
    *,
    script_path: Path,
    tasks: list[InstanceTask],
    run_dir: Path,
    local: bool,
    silent: bool,
    max_parallel: int,
    csv_by_instance: dict[str, CsvBenchmarkRow],
    track_credits: bool = False,
    gpu_monitor: PassiveGPUMonitor | None = None,
    extra_args: list[str] | None = None,
) -> list[TaskExecutionResult]:
    """Run all tasks with bounded parallelism, checking each solution as it completes."""
    tmp_dir = run_dir / ".autorunner_tmp"
    pending = list(tasks)
    active: list[ActiveJob] = []
    results_by_key: dict[str, TaskExecutionResult] = {}
    multi_seed = len({t.solver_seed for t in tasks if t.solver_seed is not None}) > 1
    monitor = gpu_monitor if local else None

    if monitor is not None:
        monitor.start(active)

    try:
        while pending or active:
            launched_any = False

            # Launch new tasks while slots and live CPU/RAM/GPU headroom allow
            while pending and len(active) < max_parallel:
                if monitor is not None:
                    pending_idx = monitor.pick_next_task(pending, active, max_parallel)
                    if pending_idx is None:
                        break
                    next_task = pending.pop(pending_idx)
                    gpu_snaps = monitor.latest_snapshot()
                else:
                    next_task = pending.pop(0)
                    gpu_snaps = None

                task_cwd = get_task_work_dir(run_dir, next_task, multi_seed)
                job = start_task_job(
                    script_path=script_path,
                    task=next_task,
                    run_dir=run_dir,
                    task_cwd=task_cwd,
                    tmp_dir=tmp_dir,
                    local=local,
                    silent=silent,
                    track_credits=track_credits,
                    gpu_snapshots=gpu_snaps,
                    extra_args=extra_args,
                )
                active.append(job)
                if monitor is not None:
                    monitor.start(active)
                launched_any = True

            still_active: list[ActiveJob] = []
            for job in active:
                finished, rc, stdout, stderr, start_iso, end_iso, duration = poll_active_job(job)
                if not finished:
                    still_active.append(job)
                    continue

                credits_used: float | None = None
                if not local and track_credits and job.credits_before is not None:
                    credits_after, _ = query_qoro_credits()
                    if credits_after is not None:
                        credits_used = round(max(0.0, job.credits_before - credits_after), 4)

                gpu_telemetry: dict[str, Any] | None = None
                if monitor is not None:
                    gpus_after = monitor.poll_gpus(force=True)
                    if gpus_after:
                        util_after = max(
                            (g.utilization_pct for g in gpus_after if g.utilization_pct is not None),
                            default=None,
                        )
                        vram_after = round(sum(g.free_vram_gb for g in gpus_after), 2)
                        temps = [g.temperature_c for g in gpus_after if g.temperature_c is not None]
                        gpu_telemetry = {
                            "util_before_pct": job.gpu_util_before_pct,
                            "peak_util_pct": job.peak_gpu_util_pct,
                            "util_after_pct": util_after,
                            "vram_free_before_gb": job.gpu_vram_free_before_gb,
                            "vram_free_after_gb": vram_after,
                            "max_temp_after_c": max(temps) if temps else None,
                            "throttling_observed": job.throttling_observed,
                        }

                reported_score, extra = parse_direct_task_output(stdout)
                sol_path = find_solution_file(job.task_cwd, job.task, job.existing_sols_before)

                checker_dict: dict[str, Any] | None = None
                if sol_path is not None:
                    checker_dict = run_solution_checker(sol_path, job.task)
                else:
                    checker_dict = {
                        "returncode": None,
                        "objective": None,
                        "feasible": False,
                        "breakdown": {},
                        "violations": ["No .sol file produced by target script"],
                        "raw_stdout": "",
                    }

                best_known, bench_meta = lookup_best_known_score(job.task, csv_by_instance)
                sol_rel_path = (
                    str(sol_path.relative_to(run_dir)) if sol_path is not None else None
                )

                exec_result = TaskExecutionResult(
                    task=job.task,
                    start_time=start_iso,
                    end_time=end_iso,
                    duration_seconds=duration,
                    exit_code=rc,
                    stdout=stdout,
                    stderr=stderr,
                    credits_used=credits_used,
                    best_solution_file=sol_path.name if sol_path is not None else None,
                    solution_rel_path=sol_rel_path,
                    reported_score=reported_score,
                    checker_result=checker_dict,
                    best_known_score=best_known,
                    benchmark_metadata=bench_meta,
                    extra=extra,
                    gpu_telemetry=gpu_telemetry,
                )
                results_by_key[job.task.run_key] = exec_result

                checked_obj = checker_dict.get("objective") if checker_dict else None
                feasible = checker_dict.get("feasible") if checker_dict else None
                print(
                    f"[DONE]   {job.task.display_name} | exit={rc} | {duration:.2f}s | "
                    f"sol={exec_result.solution_rel_path} | score={checked_obj} "
                    f"(feasible={feasible}) | best_known={best_known} | "
                    f"gap={_format_gap(checked_obj, best_known)}"
                )

            active = still_active
            if monitor is not None:
                monitor.start(active)
            if active or (pending and not launched_any):
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user; terminating running tasks...", file=sys.stderr)
        for job in active:
            if job.popen is not None and job.popen.poll() is None:
                job.popen.terminate()
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    ordered = [results_by_key[t.run_key] for t in tasks if t.run_key in results_by_key]

    # When running multiple solver RNG seeds per instance, copy the winning .sol
    # into the run root directory so the best solution is immediately available.
    if multi_seed:
        for inst_name, group in _group_results_by_instance(ordered):
            best_r = min(group, key=_result_rank_key)
            if best_r.solution_rel_path:
                src_sol = run_dir / best_r.solution_rel_path
                dst_sol = run_dir / src_sol.name
                if src_sol.is_file() and src_sol.resolve() != dst_sol.resolve():
                    shutil.copy2(src_sol, dst_sol)
            best_obj = (best_r.checker_result or {}).get("objective")
            score_display = best_obj if best_obj is not None else best_r.reported_score
            print(
                f"[BEST]   {inst_name} | best_solver_seed={best_r.task.solver_seed} | "
                f"score={score_display} | sol={best_r.best_solution_file} | "
                f"best_known={best_r.best_known_score} | "
                f"gap={_format_gap(score_display, best_r.best_known_score)}"
            )

    return ordered


def build_final_report(
    *,
    test_name: str,
    script_path: Path,
    run_dir: Path,
    local: bool,
    silent: bool,
    selection_meta: dict[str, Any],
    resource_report: dict[str, Any],
    results: list[TaskExecutionResult],
    throttle_events: list[dict[str, Any]] | None = None,
    solver_args: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the final JSON structure specified in ``AutoRunner.md``."""
    grouped = _group_results_by_instance(results)
    instance_names = [name for name, _ in grouped]
    per_instance_blocks: dict[str, Any] = {}
    aggregated_extra: dict[str, Any] = {}

    for inst_name, group in grouped:
        best_r = min(group, key=_result_rank_key)
        has_solver_seeds = any(r.task.solver_seed is not None for r in group)

        checked_obj = (best_r.checker_result or {}).get("objective")
        benchmark_score = checked_obj if checked_obj is not None else best_r.reported_score
        gap_to_best_known = _compute_gap(benchmark_score, best_r.best_known_score)

        inst_credits: float | None = None
        if not local:
            known = [r.credits_used for r in group if r.credits_used is not None]
            if known:
                inst_credits = round(sum(known), 4)

        params_block: dict[str, Any] = {
            "assets": best_r.task.assets,
            "days": best_r.task.days,
            "seed": best_r.task.seed,
            "budget": best_r.task.budget,
            "lambda": best_r.task.lam,
            "lambda_tag": best_r.task.lam_tag,
        }
        output_block: dict[str, Any] = {
            "best_solution_file": best_r.best_solution_file,
            "benchmark_score": benchmark_score,
            "reported_score": best_r.reported_score,
            "gap_to_best_known": gap_to_best_known,
            "solution_checker": best_r.checker_result,
        }

        if has_solver_seeds:
            params_block["solver_seeds"] = [r.task.solver_seed for r in group]
            params_block["best_solver_seed"] = best_r.task.solver_seed
            output_block["best_solver_seed"] = best_r.task.solver_seed

        block: dict[str, Any] = {
            "start_time": min(r.start_time for r in group),
            "end_time": max(r.end_time for r in group),
            "duration_seconds": round(sum(r.duration_seconds for r in group), 4),
            "exit_code": best_r.exit_code,
            "parameters": params_block,
            "credits_used": inst_credits,
            "output": output_block,
            "best_known_score": best_r.best_known_score,
            "benchmark_metadata": best_r.benchmark_metadata,
            "extra": best_r.extra,
        }
        if best_r.gpu_telemetry is not None:
            block["gpu_telemetry"] = best_r.gpu_telemetry

        if has_solver_seeds:
            seed_entries: list[dict[str, Any]] = []
            for r in group:
                s_checked = (r.checker_result or {}).get("objective")
                s_bench = s_checked if s_checked is not None else r.reported_score
                seed_entry: dict[str, Any] = {
                    "solver_seed": r.task.solver_seed,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration_seconds": r.duration_seconds,
                    "exit_code": r.exit_code,
                    "credits_used": None if local else r.credits_used,
                    "solution_file": r.solution_rel_path,
                    "benchmark_score": s_bench,
                    "reported_score": r.reported_score,
                    "gap_to_best_known": _compute_gap(s_bench, r.best_known_score),
                    "solution_checker": r.checker_result,
                    "extra": r.extra,
                }
                if r.gpu_telemetry is not None:
                    seed_entry["gpu_telemetry"] = r.gpu_telemetry
                if r.exit_code != 0 and r.stderr.strip():
                    seed_entry["stderr"] = r.stderr.strip()
                seed_entries.append(seed_entry)
            block["seed_runs"] = seed_entries
        elif best_r.exit_code != 0 and best_r.stderr.strip():
            block["stderr"] = best_r.stderr.strip()

        per_instance_blocks[inst_name] = block
        if best_r.extra:
            aggregated_extra[inst_name] = best_r.extra

    total_credits_used: float | None = None
    if not local:
        known_credits = [r.credits_used for r in results if r.credits_used is not None]
        if known_credits:
            total_credits_used = round(sum(known_credits), 4)

    resources_out = dict(resource_report)
    if throttle_events:
        resources_out["gpu_throttle_events"] = throttle_events

    return {
        "test_name": test_name,
        "instances": instance_names,
        "run_config": {
            "script": str(script_path.resolve()),
            "solver_args": list(solver_args or []),
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "local": local,
            "silent": silent,
            "selection": selection_meta,
            "resources": resources_out,
            "total_credits_used": total_credits_used,
        },
        "results": per_instance_blocks,
        "extra": aggregated_extra,
    }


# ---------------------------------------------------------------------------
# CLI Definition & Entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    risk_help = ", ".join(f"{i}={t}" for i, t in enumerate(RISK_LAMBDA_TAGS))
    tests_help = ", ".join(f"{i}={n}" for i, n in enumerate(TEST_SUITE_NAMES))

    parser = argparse.ArgumentParser(
        prog="auto_runner.py",
        description=(
            "Run a QOBLIB portfolio optimization solver across multiple benchmark "
            "instances in parallel, verify solutions with solution_checker.py, and "
            "write structured JSON reports under AutoData/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pass solver options after -- (for example: -- --init handin --day-blocks --day-tscale 0.5).",
    )

    parser.add_argument(
        "script",
        nargs="?",
        help="Path to the Python solver script to execute on each selected instance.",
    )
    parser.add_argument(
        "--manual",
        nargs="+",
        metavar="IDX",
        default=None,
        help=(
            "Manual selection: 0-based row indices from qoblib_instance_benchmarks.csv "
            "(space- or comma-separated). Overwrites --tests, --seed, and --risk "
            "(keeping --local)."
        ),
    )
    parser.add_argument(
        "--tests",
        "-t",
        default=None,
        help=f"Predefined test suite by name or index ({tests_help}).",
    )
    parser.add_argument(
        "--instance",
        action="append",
        metavar="SPEC",
        help="Run an explicit instance, e.g. po_a400_t15_orig_b100_l1e-3. Repeat for multiple instances.",
    )
    parser.add_argument(
        "--silent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute tasks silently without opening terminal windows "
            "(default: False, opens OS-native terminal windows; pass --silent for headless)."
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=DEFAULT_SEED,
        help=(
            f"Data seed used wherever an instance has 'XX' ({', '.join(VALID_SEEDS)} "
            f"or '00'/'01'/'02'; default: {DEFAULT_SEED}). Ignored by --manual, "
            "CompareToOptimal, and 'all'."
        ),
    )
    parser.add_argument(
        "--solver-seeds",
        "--solver-seed",
        "--rng-seeds",
        nargs="*",
        metavar="SEED",
        default=None,
        help=(
            "Solver RNG seed(s) passed to the target script via --solver-seed "
            "(matching section 1.7 of 01_portfolio_optimisation.ipynb). Pass one "
            "or more integers (e.g. --solver-seeds 1 2 3 or 1,2,3), or pass "
            "--solver-seeds with no values to default to 1 2 3. Each seed runs "
            "independently and the best feasible .sol is promoted."
        ),
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run locally (--local, default) or on Qoro cloud (--no-local).",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        default=False,
        help="Shorthand alias for --no-local (run on Qoro cloud using credits).",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=int,
        default=DEFAULT_RISK_INDEX,
        help=(
            f"Risk aversion lambda index 0..7 ({risk_help}; default: "
            f"{DEFAULT_RISK_INDEX}={RISK_LAMBDA_TAGS[DEFAULT_RISK_INDEX]}). "
            "Overwritten by risk test suites."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for maximum parallel workers (default: auto-calculated).",
    )
    parser.add_argument(
        "--gpu-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Passively monitor local GPU load/VRAM/temperature in the background "
            "to pace launches and prevent throttling (default: True; --no-gpu-monitor to disable)."
        ),
    )
    parser.add_argument(
        "--gpu-max-util",
        type=float,
        default=GPU_MAX_UTIL_DEFAULT,
        help=(
            "Maximum projected GPU utilization %% before holding back new parallel launches "
            f"(default: {GPU_MAX_UTIL_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--gpu-vram-headroom",
        type=float,
        default=GPU_VRAM_HEADROOM_PCT_DEFAULT,
        help=(
            "Minimum %% of GPU VRAM kept free to prevent OOM/throttling "
            f"(default: {GPU_VRAM_HEADROOM_PCT_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--gpu-max-temp",
        type=float,
        default=GPU_MAX_TEMP_C_DEFAULT,
        help=(
            "GPU temperature threshold (°C) above which new parallel launches pause to cool "
            f"(default: {GPU_MAX_TEMP_C_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--low-credit-threshold",
        type=float,
        default=LOW_CREDIT_THRESHOLD_DEFAULT,
        help=(
            "Cloud credit balance below which tasks run sequentially one at a time "
            f"(default: {LOW_CREDIT_THRESHOLD_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="Print all predefined test suites and exit.",
    )
    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="Print all indexed rows from qoblib_instance_benchmarks.csv and exit.",
    )
    parser.add_argument(
        "--_worker",
        dest="worker_spec",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_script_path(raw_script: str) -> Path:
    """Resolve the target Python script relative to CWD or repository root."""
    candidate = Path(raw_script)
    if candidate.is_file():
        return candidate.resolve()
    repo_candidate = REPO_ROOT / raw_script
    if repo_candidate.is_file():
        return repo_candidate.resolve()
    raise FileNotFoundError(f"Target solver script not found: {raw_script!r}")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        runner_argv = raw_argv[:sep]
        script_extra_args = raw_argv[sep + 1 :]
    else:
        runner_argv = raw_argv
        script_extra_args = []

    parser = build_arg_parser()
    args = parser.parse_args(runner_argv)

    # Internal worker mode invoked inside visible OS terminals
    if args.worker_spec is not None:
        return run_worker_shim(Path(args.worker_spec))

    csv_rows = load_benchmark_csv(BENCHMARKS_CSV)
    csv_by_instance = {row.instance: row for row in csv_rows}

    if args.list_tests:
        print("Available test suites (pass name or index to --tests):")
        for idx, (name, specs) in enumerate(TEST_SUITES_ORDERED):
            detail = f"{len(csv_rows)} CSV instances" if name == "all" else ", ".join(specs)
            print(f"  [{idx}] {name}: {detail}")
        return 0

    if args.list_instances:
        print(f"Instances in {BENCHMARKS_CSV.name} (pass 0-based indices to --manual):")
        for row in csv_rows:
            print(f"  [{row.index:2d}] {row.instance:<20s} | {row.parameters} | status={row.status}")
        return 0

    if not args.script:
        parser.error("the following arguments are required: script")

    try:
        script_path = resolve_script_path(args.script)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    local = False if args.cloud else bool(args.local)
    silent = bool(args.silent)

    if args.manual is not None and (args.tests is not None):
        print(
            "[INFO] --manual specified: overwriting --tests, --seed, and --risk "
            "(keeping --local).",
            file=sys.stderr,
        )

    try:
        solver_seeds = parse_solver_seeds(args.solver_seeds)
        test_name, tasks, selection_meta = build_task_stack(
            csv_rows=csv_rows,
            manual_tokens=args.manual,
            test_arg=args.tests,
            seed_arg=args.seed,
            risk_index=args.risk,
            solver_seeds=solver_seeds,
            instance_specs=args.instance,
        )
        for task in tasks:
            resolve_warm_start_args(script_path, task, script_extra_args)
        if script_path.name in DAY_SWAP_SCRIPTS and not local:
            raise ValueError(f"{script_path.name} is a local-only solver")
        if script_path.name == "day_swap_parallel.py" and args.max_workers is None:
            # Each benchmark job launches its own pool of day shards.
            args.max_workers = 1
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    gpu_monitor: PassiveGPUMonitor | None = None
    if local:
        gpu_policy = GPUThrottlePolicy(
            enabled=bool(args.gpu_monitor),
            max_util_pct=args.gpu_max_util,
            vram_headroom_pct=args.gpu_vram_headroom,
            max_temp_c=args.gpu_max_temp,
        )
        gpu_monitor = PassiveGPUMonitor(gpu_policy)
        initial_gpus = gpu_monitor.poll_gpus(force=True)
        max_parallel, resource_report = calculate_local_concurrency(
            tasks,
            gpu_policy=gpu_policy,
            gpus=initial_gpus,
            max_workers_override=args.max_workers,
        )
        gpu_summary = (
            ", ".join(
                f"{g.device_str}({g.name}, {g.free_vram_gb:.1f}/{g.total_vram_gb:.1f}GB free"
                + (f", util={g.utilization_pct:.0f}%" if g.utilization_pct is not None else "")
                + (f", temp={g.temperature_c:.0f}C" if g.temperature_c is not None else "")
                + (", THROTTLING" if g.throttling else "")
                + ")"
                for g in initial_gpus
            )
            if initial_gpus
            else ("disabled" if not gpu_policy.enabled else "none")
        )
        print(
            f"[RESOURCES] Local mode: "
            f"{resource_report['system']['cpu_count']} CPUs, "
            f"{resource_report['system']['available_ram_gb']} GB RAM available, "
            f"GPU monitor=[{gpu_summary}] -> running up to {max_parallel} task(s) in parallel."
        )
    else:
        max_parallel, resource_report = calculate_cloud_concurrency(
            tasks,
            low_credit_threshold=args.low_credit_threshold,
            max_workers_override=args.max_workers,
        )
        print(f"[CREDITS] Cloud mode: {resource_report['reason']}")

    run_dir = create_run_directory(test_name)
    print(f"[RUN] Test={test_name} | Tasks={len(tasks)} | Output dir={run_dir}")

    track_credits = bool(
        not local and resource_report.get("credit_check", {}).get("available", False)
    )
    results = execute_task_stack(
        script_path=script_path,
        tasks=tasks,
        run_dir=run_dir,
        local=local,
        silent=silent,
        max_parallel=max_parallel,
        csv_by_instance=csv_by_instance,
        track_credits=track_credits,
        gpu_monitor=gpu_monitor,
        extra_args=script_extra_args,
    )

    finished_at = dt.datetime.now()
    report = build_final_report(
        test_name=test_name,
        script_path=script_path,
        run_dir=run_dir,
        local=local,
        silent=silent,
        selection_meta=selection_meta,
        resource_report=resource_report,
        results=results,
        throttle_events=gpu_monitor.throttle_events if gpu_monitor is not None else None,
        solver_args=script_extra_args,
    )
    json_path = get_unique_json_path(run_dir, test_name, finished_at)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[SAVED] Report written to {json_path}")

    return 0 if all(r.exit_code == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
