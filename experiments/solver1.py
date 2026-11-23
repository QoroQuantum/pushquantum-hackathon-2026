#!/usr/bin/env python3
"""Portfolio optimization solver matching ``01_portfolio_optimisation.ipynb`` (Section 1.7).

Implements the CLI and output contract expected by ``auto_runner.py``:
  - Builds the official ``PortfolioModel`` and ``PortfolioQUBO`` from the
    instance directory passed via ``--instance-dir``.
  - Applies ``penalised(q, weight=100.0, normalize=True)``.
  - Solves across seeds ``(1, 2, 3)`` using partitioned QAOA
    (``CommunityDecomposer`` with ``max_n_variables_per_cluster=12``,
    ``SplatComposer``, ``local_search=True``, ``n_layers=2``,
    ``MonteCarloOptimizer(population_size=50)``, ``max_iterations=10``,
    ``shots=10_000``) on either ``MaestroSimulator`` or ``QoroService``.
  - Aggregates top solutions via ``BeamSearchStrategy(beam_width=10,
    n_partition_candidates=5)``, decodes and polishes with ``model.local_search``,
    and keeps the best feasible portfolio across seeds.
  - Writes ``<full_instance_name>.sol`` in the current working directory and
    prints ``AUTORUNNER_RESULT: {...}`` to stdout.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # repo root
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dimod
import hybrid
import maestro

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from divi.backends import (
    CircuitRunner,
    ExecutionResult,
    JobConfig,
    MaestroConfig,
    MaestroSimulator,
    QoroService,
)
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    CommunityDecomposer,
    QUBOPartitioningConfig,
)
import qoblib_tools as qt


# Notebook 01 (Section 1.7) hyperparameters
SHOTS = 10_000
MC_POPULATION = 50
MC_ITERATIONS = 10
PENALTY_WEIGHT = 100.0
NORMALIZE_QUBO = True
ROUTINE = "qaoa"
CLUSTER_SIZE = 12
N_LAYERS = 2
TOP_SOLUTIONS = 10
BEAM_WIDTH = 10
N_PARTITION_CANDIDATES = 5
SEEDS: tuple[int, ...] = (1, 2, 3)


def _parse_bool(value: str | bool) -> bool:
    """Parse boolean CLI/environment values safely."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _load_repo_dotenv(repo_dir: Path = SCRIPT_DIR) -> None:
    """Load environment variables from ``<repo_dir>/.env`` if not already set."""
    if os.environ.get("QORO_API_KEY"):
        return
    env_file = repo_dir / ".env"
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


def resolve_base_instance_name(
    instance_arg: str, instance_dir: Path | None
) -> str:
    """Return the canonical QOBLIB base instance name (e.g. ``po_a010_t10_orig``)."""
    if instance_dir is not None:
        dir_name = instance_dir.resolve().name
        if dir_name.startswith("po_"):
            return dir_name
    stripped = re.sub(r"_b\d+_l.*$", "", instance_arg.strip())
    if stripped.startswith("po_"):
        return stripped
    return stripped or instance_arg.strip()


def load_portfolio_instance_from_dir(
    instance_dir: Path | None, base_name: str
) -> qt.PortfolioInstance:
    """Load market data from ``instance_dir`` or fall back to ``qoblib_tools``."""
    candidate_dirs: list[Path] = []
    if instance_dir is not None:
        candidate_dirs.append(instance_dir)
    candidate_dirs.append(SCRIPT_DIR / "instances" / base_name)

    for directory in candidate_dirs:
        prices_path = directory / "stock_prices.txt.gz"
        cov_path = directory / "covariance_matrices.txt.gz"
        if not (prices_path.is_file() and cov_path.is_file()):
            continue

        symbols: list[str] = []
        index: dict[str, int] = {}
        entries: list[tuple[int, int, Fraction]] = []
        for day, sym, price in qt._data_rows(prices_path):
            if sym not in index:
                index[sym] = len(symbols)
                symbols.append(sym)
            entries.append((index[sym], int(day), Fraction(price)))

        n_periods = max(t for _, t, _ in entries) + 1
        prices: list[list[Fraction | None]] = [[None] * n_periods for _ in symbols]
        for i, t, p in entries:
            prices[i][t] = p
        if any(p is None for row in prices for p in row):
            raise ValueError(f"{base_name}: missing prices in {prices_path}")

        cov: dict[tuple[int, int, int], Fraction] = {}
        for day, s1, s2, value in qt._data_rows(cov_path):
            t = int(day)
            if s1 in index and s2 in index and t < n_periods:
                cov[index[s1], index[s2], t] = Fraction(value)

        return qt.PortfolioInstance(
            base_name,
            symbols,
            n_periods,
            prices,  # type: ignore[arg-type]
            cov,
        )

    return qt.load_portfolio_instance(base_name)


@dataclass(frozen=True)
class LocalComputeConfig:
    """Resolved local simulator backend configuration (MPS -> CUDA -> CPU)."""

    use_gpu: bool
    backend_name: str  # "torch_mps", "maestro_cuda", "torch_cuda", or "maestro_cpu"
    gpu_backend: str | None  # "mps", "cuda", or None
    gpu_device: int | None
    device_str: str  # "mps", "cuda:0", or "cpu"
    reason: str


# ---------------------------------------------------------------------------
# Batched PyTorch Statevector Simulator (Apple Silicon MPS & NVIDIA CUDA)
# ---------------------------------------------------------------------------

_QASM_GATE_RE = re.compile(
    r"^(?P<name>[a-zA-Z][a-zA-Z0-9_]*)"
    r"(?:\((?P<params>[^)]*)\))?"
    r"\s+(?P<qubits>.+)$"
)
_QASM_QREG_RE = re.compile(r"^qreg\s+\w+\[(\d+)\]$")
_QASM_QUBIT_RE = re.compile(r"\w+\[(\d+)\]")

_DIAGONAL_GATES = frozenset({"rz", "rzz", "p", "u1", "z", "s", "sdg", "t", "tdg", "cz", "cp", "cu1"})
_SUPPORTED_GATES = _DIAGONAL_GATES | frozenset(
    {"h", "x", "y", "rx", "ry", "u", "u2", "u3", "cx", "cnot", "swap", "id", "barrier"}
)


def _safe_eval_angle(expr: str) -> float:
    """Safely evaluate OpenQASM numeric angle expressions like ``(-0.02)*1.75`` or ``pi/2``."""
    node = ast.parse(expr.strip(), mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "pi":
            return math.pi
        if isinstance(n, ast.UnaryOp):
            val = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +val
            if isinstance(n.op, ast.USub):
                return -val
        if isinstance(n, ast.BinOp):
            left, right = _eval(n.left), _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return left**right
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = getattr(math, n.func.id, None)
            if fn in (math.sin, math.cos, math.tan, math.sqrt, math.exp, math.log):
                return float(fn(*(_eval(a) for a in n.args)))
        raise ValueError(f"Unsupported angle expression: {expr!r}")

    return _eval(node)


def _parse_openqasm_circuit(
    circuit: Any,
) -> tuple[int, list[tuple[str, tuple[int, ...], tuple[float, ...]]]] | None:
    """Parse an OpenQASM 2.0 string (or Qiskit circuit) into ``(n_qubits, ops)``.

    Returns ``None`` if any gate is unsupported so the caller can fall back cleanly.
    """
    if not isinstance(circuit, str):
        if hasattr(circuit, "qasm"):
            circuit = circuit.qasm()
        else:
            try:
                from qiskit import qasm2

                circuit = qasm2.dumps(circuit)
            except Exception:
                return None

    n_qubits = 0
    ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    for raw_stmt in circuit.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        lines = [ln.strip() for ln in stmt.splitlines() if ln.strip() and not ln.strip().startswith("//")]
        stmt = " ".join(lines).strip()
        if not stmt:
            continue
        if stmt.startswith(("OPENQASM", "include", "creg", "measure")):
            continue
        m_qreg = _QASM_QREG_RE.match(stmt)
        if m_qreg:
            n_qubits = max(n_qubits, int(m_qreg.group(1)))
            continue
        m_gate = _QASM_GATE_RE.match(stmt)
        if not m_gate:
            return None
        name = m_gate.group("name").lower()
        if name not in _SUPPORTED_GATES:
            return None
        if name in ("barrier", "id"):
            continue
        qubits = tuple(int(q) for q in _QASM_QUBIT_RE.findall(m_gate.group("qubits")))
        if not qubits:
            return None
        params_str = m_gate.group("params")
        params: tuple[float, ...] = ()
        if params_str is not None and params_str.strip():
            params = tuple(_safe_eval_angle(p) for p in params_str.split(","))
        ops.append((name, qubits, params))

    if n_qubits <= 0:
        return None
    return n_qubits, ops


class TorchGPUCircuitRunner(CircuitRunner):
    """Batched PyTorch statevector ``CircuitRunner`` for Apple Silicon MPS & NVIDIA CUDA.

    Uses split ``(real, imag)`` ``float32`` tensors for full Metal/MPS and CUDA
    compatibility, fuses consecutive diagonal QAOA gates (``rz``/``rzz``) into
    a single batched phase rotation, and evaluates all Pauli-Z ``ham_ops`` via
    one GPU matrix multiply. Falls back to ``MaestroSimulator`` for circuits
    exceeding ``max_qubits`` or containing unsupported gates.
    """

    def __init__(
        self,
        *,
        device: str = "mps",
        shots: int = SHOTS,
        force_sampling: bool = False,
        max_qubits: int = 20,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for TorchGPUCircuitRunner.")
        super().__init__(shots=shots)
        self._device = torch.device(device)
        self._force_sampling = force_sampling
        self._max_qubits = max_qubits
        self._fallback = MaestroSimulator(shots=shots, force_sampling=force_sampling)
        self._z_cache: dict[int, torch.Tensor] = {}
        self._bit_cache: dict[int, torch.Tensor] = {}
        self._perm_cache: dict[tuple[str, int, int, int], torch.Tensor] = {}

    @property
    def supports_expval(self) -> bool:
        return not self._force_sampling

    @property
    def is_async(self) -> bool:
        return False

    def _z_signs(self, n_qubits: int) -> torch.Tensor:
        """Shape ``(n_qubits, 2**n_qubits)`` with ``+1`` for bit 0 and ``-1`` for bit 1 (q[0] MSB)."""
        cached = self._z_cache.get(n_qubits)
        if cached is not None:
            return cached
        idx = torch.arange(1 << n_qubits, device=self._device, dtype=torch.int64)
        shifts = torch.arange(n_qubits - 1, -1, -1, device=self._device, dtype=torch.int64)
        bits = ((idx.unsqueeze(0) >> shifts.unsqueeze(1)) & 1).to(torch.float32)
        z = 1.0 - 2.0 * bits
        self._bit_cache[n_qubits] = bits
        self._z_cache[n_qubits] = z
        return z

    def _bits(self, n_qubits: int) -> torch.Tensor:
        if n_qubits not in self._bit_cache:
            self._z_signs(n_qubits)
        return self._bit_cache[n_qubits]

    def _cx_perm(self, n_qubits: int, ctrl: int, tgt: int) -> torch.Tensor:
        key = ("cx", n_qubits, ctrl, tgt)
        perm = self._perm_cache.get(key)
        if perm is None:
            idx = torch.arange(1 << n_qubits, device=self._device, dtype=torch.int64)
            c_bit = (idx >> (n_qubits - 1 - ctrl)) & 1
            perm = idx ^ (c_bit << (n_qubits - 1 - tgt))
            self._perm_cache[key] = perm
        return perm

    def _swap_perm(self, n_qubits: int, q1: int, q2: int) -> torch.Tensor:
        key = ("swap", n_qubits, min(q1, q2), max(q1, q2))
        perm = self._perm_cache.get(key)
        if perm is None:
            idx = torch.arange(1 << n_qubits, device=self._device, dtype=torch.int64)
            s1, s2 = n_qubits - 1 - q1, n_qubits - 1 - q2
            b1 = (idx >> s1) & 1
            b2 = (idx >> s2) & 1
            diff = b1 ^ b2
            perm = idx ^ (diff << s1) ^ (diff << s2)
            self._perm_cache[key] = perm
        return perm

    def _apply_1q_h(
        self, re: torch.Tensor, im: torch.Tensor, n_qubits: int, q: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = re.shape[0]
        inv_s2 = 0.7071067811865476
        r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        r0, r1 = r[:, :, 0, :], r[:, :, 1, :]
        i0, i1 = i[:, :, 0, :], i[:, :, 1, :]
        out_r = torch.stack(((r0 + r1) * inv_s2, (r0 - r1) * inv_s2), dim=2).reshape(B, -1)
        out_i = torch.stack(((i0 + i1) * inv_s2, (i0 - i1) * inv_s2), dim=2).reshape(B, -1)
        return out_r, out_i

    def _apply_1q_rx(
        self, re: torch.Tensor, im: torch.Tensor, n_qubits: int, q: int, thetas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = re.shape[0]
        half = (0.5 * thetas).view(B, 1, 1)
        c, s = torch.cos(half), torch.sin(half)
        r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        r0, r1 = r[:, :, 0, :], r[:, :, 1, :]
        i0, i1 = i[:, :, 0, :], i[:, :, 1, :]
        nr0 = c * r0 + s * i1
        ni0 = c * i0 - s * r1
        nr1 = c * r1 + s * i0
        ni1 = c * i1 - s * r0
        return (
            torch.stack((nr0, nr1), dim=2).reshape(B, -1),
            torch.stack((ni0, ni1), dim=2).reshape(B, -1),
        )

    def _apply_1q_ry(
        self, re: torch.Tensor, im: torch.Tensor, n_qubits: int, q: int, thetas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = re.shape[0]
        half = (0.5 * thetas).view(B, 1, 1)
        c, s = torch.cos(half), torch.sin(half)
        r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        r0, r1 = r[:, :, 0, :], r[:, :, 1, :]
        i0, i1 = i[:, :, 0, :], i[:, :, 1, :]
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        return (
            torch.stack((nr0, nr1), dim=2).reshape(B, -1),
            torch.stack((ni0, ni1), dim=2).reshape(B, -1),
        )

    def _apply_1q_u(
        self,
        re: torch.Tensor,
        im: torch.Tensor,
        n_qubits: int,
        q: int,
        theta: torch.Tensor,
        phi: torch.Tensor,
        lam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = re.shape[0]
        ct = torch.cos(0.5 * theta).view(B, 1, 1)
        st = torch.sin(0.5 * theta).view(B, 1, 1)
        cp, sp = torch.cos(phi).view(B, 1, 1), torch.sin(phi).view(B, 1, 1)
        cl, sl = torch.cos(lam).view(B, 1, 1), torch.sin(lam).view(B, 1, 1)
        cpl = torch.cos(phi + lam).view(B, 1, 1)
        spl = torch.sin(phi + lam).view(B, 1, 1)

        r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
        r0, r1 = r[:, :, 0, :], r[:, :, 1, :]
        i0, i1 = i[:, :, 0, :], i[:, :, 1, :]

        # u01 = -e^{i*lam} * st, u10 = e^{i*phi} * st, u11 = e^{i*(phi+lam)} * ct
        u01_r, u01_i = -cl * st, -sl * st
        u10_r, u10_i = cp * st, sp * st
        u11_r, u11_i = cpl * ct, spl * ct

        nr0 = ct * r0 + (u01_r * r1 - u01_i * i1)
        ni0 = ct * i0 + (u01_r * i1 + u01_i * r1)
        nr1 = (u10_r * r0 - u10_i * i0) + (u11_r * r1 - u11_i * i1)
        ni1 = (u10_r * i0 + u10_i * r0) + (u11_r * i1 + u11_i * r1)
        return (
            torch.stack((nr0, nr1), dim=2).reshape(B, -1),
            torch.stack((ni0, ni1), dim=2).reshape(B, -1),
        )

    def _simulate_batch(
        self,
        n_qubits: int,
        batch_ops: list[list[tuple[str, tuple[int, ...], tuple[float, ...]]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate a batch of structurally identical circuits simultaneously on GPU."""
        B = len(batch_ops)
        dim = 1 << n_qubits
        re = torch.zeros((B, dim), dtype=torch.float32, device=self._device)
        im = torch.zeros((B, dim), dtype=torch.float32, device=self._device)
        re[:, 0] = 1.0

        z = self._z_signs(n_qubits)
        bits = self._bits(n_qubits)
        n_ops = len(batch_ops[0])
        phase: torch.Tensor | None = None

        def _flush_phase() -> None:
            nonlocal re, im, phase
            if phase is None:
                return
            c = torch.cos(phase)
            s = torch.sin(phase)
            re, im = re * c - im * s, re * s + im * c
            phase = None

        for op_idx in range(n_ops):
            name, qubits, _ = batch_ops[0][op_idx]
            if name in _DIAGONAL_GATES:
                if phase is None:
                    phase = torch.zeros((B, dim), dtype=torch.float32, device=self._device)
                if name == "rz":
                    ang = torch.tensor(
                        [ops[op_idx][2][0] for ops in batch_ops],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    phase = phase + (-0.5 * ang).unsqueeze(1) * z[qubits[0]].unsqueeze(0)
                elif name == "rzz":
                    ang = torch.tensor(
                        [ops[op_idx][2][0] for ops in batch_ops],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    zz = z[qubits[0]] * z[qubits[1]]
                    phase = phase + (-0.5 * ang).unsqueeze(1) * zz.unsqueeze(0)
                elif name in ("p", "u1"):
                    ang = torch.tensor(
                        [ops[op_idx][2][0] for ops in batch_ops],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    phase = phase + ang.unsqueeze(1) * bits[qubits[0]].unsqueeze(0)
                elif name == "z":
                    phase = phase + math.pi * bits[qubits[0]].unsqueeze(0)
                elif name == "s":
                    phase = phase + (0.5 * math.pi) * bits[qubits[0]].unsqueeze(0)
                elif name == "sdg":
                    phase = phase + (-0.5 * math.pi) * bits[qubits[0]].unsqueeze(0)
                elif name == "t":
                    phase = phase + (0.25 * math.pi) * bits[qubits[0]].unsqueeze(0)
                elif name == "tdg":
                    phase = phase + (-0.25 * math.pi) * bits[qubits[0]].unsqueeze(0)
                elif name == "cz":
                    bb = bits[qubits[0]] * bits[qubits[1]]
                    phase = phase + math.pi * bb.unsqueeze(0)
                elif name in ("cp", "cu1"):
                    ang = torch.tensor(
                        [ops[op_idx][2][0] for ops in batch_ops],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    bb = bits[qubits[0]] * bits[qubits[1]]
                    phase = phase + ang.unsqueeze(1) * bb.unsqueeze(0)
                continue

            _flush_phase()

            if name == "h":
                re, im = self._apply_1q_h(re, im, n_qubits, qubits[0])
            elif name == "x":
                q = qubits[0]
                r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
                i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
                re = torch.flip(r, dims=[2]).reshape(B, dim)
                im = torch.flip(i, dims=[2]).reshape(B, dim)
            elif name == "y":
                q = qubits[0]
                r = re.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
                i = im.view(B, 1 << q, 2, 1 << (n_qubits - 1 - q))
                nr = torch.stack((i[:, :, 1, :], -i[:, :, 0, :]), dim=2).reshape(B, dim)
                ni = torch.stack((-r[:, :, 1, :], r[:, :, 0, :]), dim=2).reshape(B, dim)
                re, im = nr, ni
            elif name == "rx":
                ang = torch.tensor(
                    [ops[op_idx][2][0] for ops in batch_ops],
                    dtype=torch.float32,
                    device=self._device,
                )
                re, im = self._apply_1q_rx(re, im, n_qubits, qubits[0], ang)
            elif name == "ry":
                ang = torch.tensor(
                    [ops[op_idx][2][0] for ops in batch_ops],
                    dtype=torch.float32,
                    device=self._device,
                )
                re, im = self._apply_1q_ry(re, im, n_qubits, qubits[0], ang)
            elif name in ("u", "u3"):
                th = torch.tensor(
                    [ops[op_idx][2][0] for ops in batch_ops], dtype=torch.float32, device=self._device
                )
                ph = torch.tensor(
                    [ops[op_idx][2][1] for ops in batch_ops], dtype=torch.float32, device=self._device
                )
                lm = torch.tensor(
                    [ops[op_idx][2][2] for ops in batch_ops], dtype=torch.float32, device=self._device
                )
                re, im = self._apply_1q_u(re, im, n_qubits, qubits[0], th, ph, lm)
            elif name == "u2":
                th = torch.full((B,), 0.5 * math.pi, dtype=torch.float32, device=self._device)
                ph = torch.tensor(
                    [ops[op_idx][2][0] for ops in batch_ops], dtype=torch.float32, device=self._device
                )
                lm = torch.tensor(
                    [ops[op_idx][2][1] for ops in batch_ops], dtype=torch.float32, device=self._device
                )
                re, im = self._apply_1q_u(re, im, n_qubits, qubits[0], th, ph, lm)
            elif name in ("cx", "cnot"):
                perm = self._cx_perm(n_qubits, qubits[0], qubits[1])
                re = torch.index_select(re, 1, perm)
                im = torch.index_select(im, 1, perm)
            elif name == "swap":
                perm = self._swap_perm(n_qubits, qubits[0], qubits[1])
                re = torch.index_select(re, 1, perm)
                im = torch.index_select(im, 1, perm)

        _flush_phase()
        return re, im

    def _compute_expvals(
        self, re: torch.Tensor, im: torch.Tensor, n_qubits: int, terms: list[str]
    ) -> list[dict[str, float]]:
        B = re.shape[0]
        z = self._z_signs(n_qubits)
        z_only_terms = [t for t in terms if set(t) <= {"I", "Z"}]
        general_terms = [t for t in terms if not (set(t) <= {"I", "Z"})]

        out: list[dict[str, float]] = [{} for _ in range(B)]

        if z_only_terms:
            probs = re * re + im * im
            cols = []
            for term in z_only_terms:
                vec = torch.ones(1 << n_qubits, dtype=torch.float32, device=self._device)
                for q, ch in enumerate(term):
                    if ch == "Z":
                        vec = vec * z[q]
                cols.append(vec)
            sign_mat = torch.stack(cols, dim=1)  # (2**n, M)
            vals = (probs @ sign_mat).detach().cpu().tolist()
            for b in range(B):
                for m, term in enumerate(z_only_terms):
                    out[b][term] = float(vals[b][m])

        if general_terms:
            bits = self._bits(n_qubits)
            for term in general_terms:
                tr, ti = re, im
                vec = torch.ones(1 << n_qubits, dtype=torch.float32, device=self._device)
                for q, ch in enumerate(term):
                    if ch == "X":
                        tr, ti = self._apply_1q_h(tr, ti, n_qubits, q)
                        vec = vec * z[q]
                    elif ch == "Y":
                        # Sdg then H rotates Y eigenbasis to Z eigenbasis
                        # Sdg multiplies |1> by -i: (r, i) -> (i, -r) on bit 1
                        b1 = bits[q].unsqueeze(0)
                        b0 = 1.0 - b1
                        tr, ti = b0 * tr + b1 * ti, b0 * ti - b1 * tr
                        tr, ti = self._apply_1q_h(tr, ti, n_qubits, q)
                        vec = vec * z[q]
                    elif ch == "Z":
                        vec = vec * z[q]
                probs = tr * tr + ti * ti
                vals = (probs @ vec).detach().cpu().tolist()
                for b in range(B):
                    out[b][term] = float(vals[b])

        return out

    def _sample_counts(
        self, re: torch.Tensor, im: torch.Tensor, n_qubits: int
    ) -> list[dict[str, int]]:
        B = re.shape[0]
        probs = (re * re + im * im).clamp(min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        samples = torch.multinomial(probs, num_samples=self.shots, replacement=True)
        out: list[dict[str, int]] = []
        fmt = f"0{n_qubits}b"
        for b in range(B):
            uniq, cnts = torch.unique(samples[b], return_counts=True)
            out.append(
                {format(int(u), fmt): int(c) for u, c in zip(uniq.cpu().tolist(), cnts.cpu().tolist())}
            )
        return out

    @staticmethod
    def _resolve_per_circuit_ham_ops(
        n_circuits: int,
        ham_ops: str | None,
        circuit_ham_map: Any | None,
    ) -> list[str] | None:
        """Resolve each circuit's semicolon-separated observable group string."""
        if ham_ops is None:
            return None
        groups = [g.strip() for g in ham_ops.split("|")]
        if circuit_ham_map is None:
            combined = ";".join(g for g in groups if g)
            return [combined] * n_circuits
        if len(circuit_ham_map) != len(groups):
            raise ValueError(
                f"circuit_ham_map length ({len(circuit_ham_map)}) must match "
                f"number of ham_ops groups ({len(groups)})."
            )
        per_circuit: list[str | None] = [None] * n_circuits
        for group_str, (start, end) in zip(groups, circuit_ham_map):
            for idx in range(start, end):
                per_circuit[idx] = group_str
        if any(g is None for g in per_circuit):
            raise ValueError("circuit_ham_map does not cover every circuit.")
        return per_circuit  # type: ignore[return-value]

    @staticmethod
    def _encode_ham_ops(
        per_circuit_ham: list[str],
    ) -> tuple[str, list[list[int]] | None]:
        """Re-encode per-circuit observable strings into ``(ham_ops, circuit_ham_map)``."""
        groups: list[str] = []
        ham_map: list[list[int]] = []
        for idx, group_str in enumerate(per_circuit_ham):
            if groups and groups[-1] == group_str:
                ham_map[-1][1] = idx + 1
            else:
                groups.append(group_str)
                ham_map.append([idx, idx + 1])
        if len(groups) == 1:
            return groups[0], None
        return "|".join(groups), ham_map

    @staticmethod
    def _subset_shot_groups(
        shot_groups: Any,
        n_circuits: int,
        indices: list[int],
    ) -> list[list[int]]:
        """Slice ``shot_groups`` ``[start, end, shots]`` triples for a fallback subset."""
        per_circuit_shots: list[int | None] = [None] * n_circuits
        for start, end, shots in shot_groups:
            for idx in range(start, end):
                per_circuit_shots[idx] = int(shots)
        encoded: list[list[int]] = []
        for new_idx, orig_idx in enumerate(indices):
            shots = per_circuit_shots[orig_idx]
            if shots is None:
                continue
            if encoded and encoded[-1][2] == shots and encoded[-1][1] == new_idx:
                encoded[-1][1] = new_idx + 1
            else:
                encoded.append([new_idx, new_idx + 1, shots])
        return encoded

    def submit_circuits(
        self,
        circuits,
        ham_ops: str | None = None,
        circuit_ham_map=None,
        **kwargs,
    ) -> ExecutionResult:
        if isinstance(circuits, dict):
            items = list(circuits.items())
        else:
            items = [(str(i), c) for i, c in enumerate(circuits)]

        n_circuits = len(items)
        per_circuit_ham = self._resolve_per_circuit_ham_ops(
            n_circuits, ham_ops, circuit_ham_map
        )
        eval_expvals = per_circuit_ham is not None and not self._force_sampling

        # Group structurally identical circuits (and matching observable groups)
        # so all parameter sets execute in one GPU batch.
        groups: dict[
            tuple,
            list[tuple[str, list[tuple[str, tuple[int, ...], tuple[float, ...]]]]],
        ] = defaultdict(list)
        fallback_entries: list[tuple[int, str, Any, str | None]] = []

        for idx, (label, circ) in enumerate(items):
            ham_str = per_circuit_ham[idx] if per_circuit_ham is not None else None
            parsed = _parse_openqasm_circuit(circ)
            if parsed is None or parsed[0] > self._max_qubits:
                fallback_entries.append((idx, label, circ, ham_str))
                continue
            n_qubits, ops = parsed
            if eval_expvals and ham_str is not None:
                circuit_terms = [t.strip() for t in ham_str.split(";") if t.strip()]
                if any(len(t) != n_qubits for t in circuit_terms):
                    # Mismatched qubit width: delegate cleanly to MaestroSimulator
                    fallback_entries.append((idx, label, circ, ham_str))
                    continue
            sig = (
                n_qubits,
                tuple((name, qubits) for name, qubits, _ in ops),
                ham_str if eval_expvals else None,
            )
            groups[sig].append((label, ops))

        results_by_label: dict[str, dict[str, Any]] = {}

        with torch.no_grad():
            for (n_qubits, _, ham_str), grp in groups.items():
                labels = [lbl for lbl, _ in grp]
                batch_ops = [ops for _, ops in grp]
                re, im = self._simulate_batch(n_qubits, batch_ops)
                if eval_expvals and ham_str is not None:
                    terms = [t.strip() for t in ham_str.split(";") if t.strip()]
                    exp_list = self._compute_expvals(re, im, n_qubits, terms)
                    for lbl, res_dict in zip(labels, exp_list):
                        results_by_label[lbl] = {"label": lbl, "results": res_dict}
                else:
                    cnt_list = self._sample_counts(re, im, n_qubits)
                    for lbl, res_dict in zip(labels, cnt_list):
                        results_by_label[lbl] = {"label": lbl, "results": res_dict}

        if fallback_entries:
            fallback_entries.sort(key=lambda e: e[0])
            fallback_items = {lbl: circ for _, lbl, circ, _ in fallback_entries}
            fb_kwargs = dict(kwargs)
            fb_ham_ops: str | None = None
            if per_circuit_ham is not None:
                fb_ham_ops, fb_ham_map = self._encode_ham_ops(
                    [h for _, _, _, h in fallback_entries if h is not None]
                )
                if fb_ham_map is not None:
                    fb_kwargs["circuit_ham_map"] = fb_ham_map
            if "shot_groups" in fb_kwargs and fb_kwargs["shot_groups"] is not None:
                fb_kwargs["shot_groups"] = self._subset_shot_groups(
                    fb_kwargs["shot_groups"],
                    n_circuits,
                    [idx for idx, _, _, _ in fallback_entries],
                )
            fb_res = self._fallback.submit_circuits(
                fallback_items, ham_ops=fb_ham_ops, **fb_kwargs
            )
            for entry in fb_res.results:
                results_by_label[entry["label"]] = entry

        ordered_results = [results_by_label[lbl] for lbl, _ in items]
        return ExecutionResult(results=ordered_results)


def _pick_best_gpu_index(visible_count: int) -> int:
    """Pick the least-loaded visible CUDA GPU index (fallback: 0)."""
    if visible_count <= 1 or shutil.which("nvidia-smi") is None:
        return 0
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0
        rows: list[tuple[int, float, float]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))

        # Map CUDA_VISIBLE_DEVICES logical indices (0..visible_count-1) if filtered
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_env:
            chosen = [int(tok.strip()) for tok in visible_env.split(",") if tok.strip().isdigit()]
            by_phys = {idx: (util, free) for idx, util, free in rows}
            candidates = [
                (logical_idx, by_phys[phys][0], by_phys[phys][1])
                for logical_idx, phys in enumerate(chosen[:visible_count])
                if phys in by_phys
            ]
        else:
            candidates = [(idx, util, free) for idx, util, free in rows if 0 <= idx < visible_count]

        if candidates:
            # Lowest utilization first, then most free VRAM
            candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
            return candidates[0][0]
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return 0


def resolve_local_compute_config(
    *,
    allow_gpu: bool = True,
    preferred_gpu: int | None = None,
) -> LocalComputeConfig:
    """Select local execution backend with strict priority: MPS -> CUDA -> CPU."""
    if not allow_gpu:
        return LocalComputeConfig(
            use_gpu=False,
            backend_name="maestro_cpu",
            gpu_backend=None,
            gpu_device=None,
            device_str="cpu",
            reason="GPU disabled via --no-gpu",
        )

    # 1. Priority 1: Apple Silicon Metal (MPS) via PyTorch
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, dtype=torch.float32, device="mps")
            return LocalComputeConfig(
                use_gpu=True,
                backend_name="torch_mps",
                gpu_backend="mps",
                gpu_device=0,
                device_str="mps",
                reason="Apple Silicon MPS GPU active (batched PyTorch statevector)",
            )
        except Exception:
            pass

    # 2. Priority 2a: NVIDIA CUDA via Maestro native GPU simulator
    try:
        gpu_count = int(maestro.get_gpu_device_count())
    except Exception:
        gpu_count = 0

    if gpu_count > 0:
        gpu_id = preferred_gpu if preferred_gpu is not None else _pick_best_gpu_index(gpu_count)
        if not 0 <= gpu_id < gpu_count:
            raise ValueError(
                f"Requested --gpu-device {gpu_id} out of range (visible CUDA GPUs: 0..{gpu_count - 1})."
            )
        try:
            maestro.select_gpu_device(gpu_id)
            if bool(maestro.init_gpu()) and bool(maestro.is_gpu_available()):
                return LocalComputeConfig(
                    use_gpu=True,
                    backend_name="maestro_cuda",
                    gpu_backend="cuda",
                    gpu_device=gpu_id,
                    device_str=f"cuda:{gpu_id}",
                    reason=f"Maestro CUDA GPU device {gpu_id} active ({gpu_count} visible)",
                )
        except Exception:
            pass

    # 2b. Priority 2b: NVIDIA CUDA via PyTorch GPU simulator (if Maestro CUDA library missing)
    if torch is not None and torch.cuda.is_available():
        cuda_count = int(torch.cuda.device_count())
        if cuda_count > 0:
            gpu_id = preferred_gpu if preferred_gpu is not None else _pick_best_gpu_index(cuda_count)
            if not 0 <= gpu_id < cuda_count:
                raise ValueError(
                    f"Requested --gpu-device {gpu_id} out of range (visible CUDA GPUs: 0..{cuda_count - 1})."
                )
            try:
                _ = torch.zeros(1, dtype=torch.float32, device=f"cuda:{gpu_id}")
                gpu_name = torch.cuda.get_device_name(gpu_id)
                return LocalComputeConfig(
                    use_gpu=True,
                    backend_name="torch_cuda",
                    gpu_backend="cuda",
                    gpu_device=gpu_id,
                    device_str=f"cuda:{gpu_id}",
                    reason=f"PyTorch CUDA GPU {gpu_id} ({gpu_name}) active",
                )
            except Exception:
                pass

    # 3. Priority 3: CPU fallback via Maestro QCSim
    return LocalComputeConfig(
        use_gpu=False,
        backend_name="maestro_cpu",
        gpu_backend=None,
        gpu_device=None,
        device_str="cpu",
        reason="no MPS or CUDA GPU available; falling back to CPU QCSim",
    )


def make_backend(
    *,
    use_cloud: bool,
    shots: int = SHOTS,
    force_sampling: bool = False,
    compute_cfg: LocalComputeConfig | None = None,
):
    """Create the Divi backend: Qoro cloud, MPS GPU, CUDA GPU, or CPU."""
    if use_cloud:
        _load_repo_dotenv()
        return QoroService(
            job_config=JobConfig(
                shots=shots,
                simulator_cluster="qoro_maestro",
                force_sampling=force_sampling,
            )
        )
    if compute_cfg is None:
        compute_cfg = resolve_local_compute_config(allow_gpu=True)
    if compute_cfg.backend_name == "torch_mps":
        return TorchGPUCircuitRunner(
            device="mps", shots=shots, force_sampling=force_sampling
        )
    if compute_cfg.backend_name == "maestro_cuda":
        return MaestroSimulator(
            shots=shots,
            force_sampling=force_sampling,
            maestro_config=MaestroConfig(
                simulator_type="Gpu",
                gpu_device=compute_cfg.gpu_device,
            ),
        )
    if compute_cfg.backend_name == "torch_cuda":
        return TorchGPUCircuitRunner(
            device=compute_cfg.device_str, shots=shots, force_sampling=force_sampling
        )
    return MaestroSimulator(shots=shots, force_sampling=force_sampling)


def penalised(
    q: qt.PortfolioQUBO, weight: float, normalize: bool = False
) -> dimod.BinaryQuadraticModel:
    """Combine objective + weight * penalty into a single BQM."""
    bqm = q.objective.copy()
    penalty = q.penalty.copy()
    penalty.scale(weight)
    bqm.update(penalty)
    if normalize:
        bqm.normalize()
    return bqm


def best_polished(
    model: qt.PortfolioModel,
    q: qt.PortfolioQUBO,
    samples,
    top: int = TOP_SOLUTIONS,
) -> tuple[dict, qt.PortfolioScore]:
    """Decode the first ``top`` samples, apply ``model.local_search``, and return the best."""
    sample_list = list(samples)[:top]
    candidates = [model.local_search(q.decode(s)) for s in sample_list]
    if not candidates:
        candidates = [model.local_search({})]
    best = min(candidates, key=lambda p: model.score(p).objective)
    return best, model.score(best)


def solve_partitioned(
    bqm: dimod.BinaryQuadraticModel,
    routine: str = ROUTINE,
    cluster_size: int = CLUSTER_SIZE,
    n_layers: int = N_LAYERS,
    backend=None,
    decomposer=None,
    mc_population: int = MC_POPULATION,
    mc_iterations: int = MC_ITERATIONS,
    **routine_kwargs,
) -> PartitioningProgramEnsemble:
    """Partition ``bqm``, solve every part with ``routine``, and return the ensemble."""
    if backend is None:
        backend = make_backend(use_cloud=False)
    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=decomposer
        or CommunityDecomposer(
            QUBOPartitioningConfig(max_n_variables_per_cluster=cluster_size)
        ),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine=routine,
        n_layers=n_layers,
        optimizer=MonteCarloOptimizer(population_size=mc_population),
        max_iterations=mc_iterations,
        backend=backend,
        **routine_kwargs,
    )
    ensemble.run()
    return ensemble


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "QOBLIB Portfolio Optimization solver implementing the partitioned QAOA + "
            "local search pipeline from 01_portfolio_optimisation.ipynb."
        )
    )
    parser.add_argument(
        "--instance",
        default=os.environ.get("AUTORUNNER_INSTANCE", "po_a010_t10_orig"),
        help="Full or base instance name (e.g. po_a010_t10_orig_b004_l1e-4).",
    )
    parser.add_argument(
        "--instance-dir",
        default=os.environ.get("AUTORUNNER_INSTANCE_DIR"),
        help="Path to the instance directory containing stock_prices and covariance_matrices.",
    )
    parser.add_argument(
        "--assets",
        type=int,
        default=(
            int(os.environ["AUTORUNNER_ASSETS"])
            if os.environ.get("AUTORUNNER_ASSETS")
            else None
        ),
        help="Expected asset count (optional sanity check).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=(
            int(os.environ["AUTORUNNER_DAYS"])
            if os.environ.get("AUTORUNNER_DAYS")
            else None
        ),
        help="Expected period/day count (optional sanity check).",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        default=os.environ.get("AUTORUNNER_LAMBDA", "1e-4"),
        help="Risk aversion weight lambda (e.g. 0, 1e-6, 1e-4, 1e-2).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=int(os.environ.get("AUTORUNNER_BUDGET", "4")),
        help="Portfolio position budget B.",
    )
    parser.add_argument(
        "--cloud",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=_parse_bool(os.environ.get("AUTORUNNER_CLOUD", "false")),
        help="Whether to run quantum circuits on Qoro cloud (true/false).",
    )
    parser.add_argument(
        "--solver-seed",
        type=int,
        default=(
            int(os.environ["AUTORUNNER_SOLVER_SEED"])
            if os.environ.get("AUTORUNNER_SOLVER_SEED")
            else None
        ),
        help="Optional single RNG seed override (default: run all seeds 1, 2, 3).",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local Maestro GPU simulator when available (default: True; --no-gpu forces CPU).",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help="Optional CUDA GPU index for Maestro GPU simulation (default: auto-select least loaded).",
    )
    return parser


def resolve_solution_path(
    instance_arg: str, base_name: str, budget: int, lam: str
) -> Path:
    """Return the output ``.sol`` path in the current working directory."""
    if re.search(r"_b\d+_l", instance_arg):
        return Path(f"{instance_arg}.sol")
    lam_tag = lam.replace("-0", "-").replace("+0", "+")
    return Path(f"{base_name}_b{budget:03d}_l{lam_tag}.sol")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t_total_start = time.time()

    instance_dir = Path(args.instance_dir) if args.instance_dir else None
    base_name = resolve_base_instance_name(args.instance, instance_dir)
    inst = load_portfolio_instance_from_dir(instance_dir, base_name)

    if args.assets is not None and inst.n_assets != args.assets:
        raise ValueError(
            f"Asset count mismatch for {base_name}: expected {args.assets}, got {inst.n_assets}"
        )
    if args.days is not None and inst.n_periods != args.days:
        raise ValueError(
            f"Day/period count mismatch for {base_name}: expected {args.days}, got {inst.n_periods}"
        )

    use_cloud = bool(args.cloud)
    compute_cfg: LocalComputeConfig | None = None
    if use_cloud:
        _load_repo_dotenv()
        print("Backend: QoroService (cloud)")
    else:
        compute_cfg = resolve_local_compute_config(
            allow_gpu=bool(args.gpu),
            preferred_gpu=args.gpu_device,
        )
        print(f"Backend: {compute_cfg.backend_name} ({compute_cfg.reason})")

    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    q = qt.PortfolioQUBO(model)
    print(f"{base_name}, B={args.budget}, λ={args.lam}: {q.n_vars} QUBO variables")

    empty_ls_obj = int(model.score(model.local_search({})).objective)
    print(f"Local search from an empty portfolio: {empty_ls_obj}")

    bqm = penalised(q, PENALTY_WEIGHT, normalize=NORMALIZE_QUBO)

    seeds_to_run: tuple[int, ...] = (
        (args.solver_seed,) if args.solver_seed is not None else SEEDS
    )
    runs: list[tuple[int, int, dict, qt.PortfolioScore]] = []
    seed_runs: list[dict] = []
    for seed in seeds_to_run:
        t0 = time.time()
        backend = make_backend(
            use_cloud=use_cloud,
            shots=SHOTS,
            compute_cfg=compute_cfg,
        )
        ensemble = solve_partitioned(
            bqm,
            routine=ROUTINE,
            cluster_size=CLUSTER_SIZE,
            n_layers=N_LAYERS,
            backend=backend,
            seed=seed,
        )
        tops = ensemble.get_top_solutions(
            n=TOP_SOLUTIONS,
            strategy=BeamSearchStrategy(
                beam_width=BEAM_WIDTH,
                n_partition_candidates=N_PARTITION_CANDIDATES,
            ),
        )
        samples = [dict(zip(bqm.variables, solution)) for solution, _ in tops]
        positions, score = best_polished(model, q, samples, top=TOP_SOLUTIONS)
        elapsed = round(time.time() - t0, 3)
        n_subprograms = int(len(ensemble.programs))
        total_circuits = int(ensemble.total_circuit_count)

        runs.append((int(score.objective), seed, positions, score))
        seed_runs.append(
            {
                "seed": seed,
                "objective": int(score.objective),
                "feasible": bool(score.feasible),
                "sub_programs": n_subprograms,
                "circuits": total_circuits,
                "duration_seconds": elapsed,
            }
        )
        print(
            f"seed {seed}: {score.objective} ({n_subprograms} sub-programs, "
            f"{total_circuits} circuits, {elapsed:.1f}s)"
        )

    best_obj, best_seed, best_positions, best_score = min(
        runs, key=lambda r: (0 if r[3].feasible else 1, r[0])
    )

    sol_path = resolve_solution_path(args.instance, base_name, args.budget, args.lam)
    written_score = model.write_solution(
        sol_path,
        best_positions,
        comment="Qoro x QOBLIB hackathon: partitioned QAOA + local search",
    )
    print(f"\nBest of {len(runs)} runs (seed {best_seed}): {best_obj}")
    print(f"Wrote {sol_path}: {written_score}")

    payload = {
        "score": int(written_score.objective),
        "extra": {
            "instance": args.instance,
            "base_instance": base_name,
            "assets": inst.n_assets,
            "days": inst.n_periods,
            "budget": args.budget,
            "lambda": args.lam,
            "cloud": use_cloud,
            "gpu_used": bool(compute_cfg and compute_cfg.use_gpu),
            "gpu_backend": compute_cfg.gpu_backend if compute_cfg else None,
            "gpu_device": compute_cfg.gpu_device if compute_cfg else None,
            "device": "cloud" if use_cloud else (compute_cfg.device_str if compute_cfg else "cpu"),
            "simulator_backend": (
                "qoro_cloud"
                if use_cloud
                else (compute_cfg.backend_name if compute_cfg else "maestro_cpu")
            ),
            "backend_detail": (
                "QoroService"
                if use_cloud
                else (compute_cfg.reason if compute_cfg else None)
            ),
            "feasible": bool(written_score.feasible),
            "breakdown": {k: int(v) for k, v in written_score.breakdown.items()},
            "violations": list(written_score.violations),
            "solution_file": sol_path.name,
            "n_qubo_vars": int(q.n_vars),
            "penalty_weight": PENALTY_WEIGHT,
            "normalize": NORMALIZE_QUBO,
            "routine": ROUTINE,
            "cluster_size": CLUSTER_SIZE,
            "n_layers": N_LAYERS,
            "shots": SHOTS,
            "mc_population": MC_POPULATION,
            "mc_iterations": MC_ITERATIONS,
            "beam_width": BEAM_WIDTH,
            "n_partition_candidates": N_PARTITION_CANDIDATES,
            "top_solutions": TOP_SOLUTIONS,
            "empty_local_search_objective": empty_ls_obj,
            "best_seed": best_seed,
            "total_circuits": sum(item["circuits"] for item in seed_runs),
            "total_duration_seconds": round(time.time() - t_total_start, 3),
            "seed_runs": seed_runs,
        },
    }
    print(f"\nAUTORUNNER_RESULT: {json.dumps(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
