"""Divi QAOA sub-solver for the move-selection QUBOs of the portfolio QLNS.

Two circuit families:
  * ``xy`` (fixed Hamming weight k): Dicke initial state |D^M_k> and a ring XY mixer, which
    conserve the number of selected moves.  Used when a window must add exactly k units to a
    saturated day budget: the budget then holds by construction, with no penalty and no slack.
  * ``x``  (free weight): uniform superposition and X mixer; budget and capital rules become
    Divi ``LinearConstraint``s (penalty + slack qubits).

Angles: QAOA angles concentrate across sub-QUBOs of the same family once the Hamiltonian is
normalised (here by the standard deviation of the window energy).  Each family keeps one set
of p-layer angles, trained with COBYLA on a real window at start-up and re-trained
(warm-started) every ``retrain_every`` windows.  A window then costs two circuits: one loss
evaluation at the library angles and the final sampling.  Every distinct sampled bitstring
goes back to the caller, which scores it exactly.
"""
from __future__ import annotations

import json
import logging
import math
import time
import warnings
from pathlib import Path

import dimod
import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from divi.backends import JobConfig, MaestroSimulator, QoroService  # noqa: E402
from divi.hamiltonians import xy_mixer  # noqa: E402
from divi.qprog.algorithms import QAOA  # noqa: E402
from divi.qprog.initial_states import CustomPerQubitState, DickeState, InitialState  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer  # noqa: E402
from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint  # noqa: E402

import qlns_core as qc  # noqa: E402


class GroupedDicke(InitialState):
    """Product of Dicke states, one per qubit group: |D^{n_1}_{k_1}> x |D^{n_2}_{k_2}> x ...

    With one XY ring per group the mixer conserves every group's count, so a rounding window
    keeps both the day budget (total count) and the capital rule (count of '+' moves) exactly.
    """

    def __init__(self, groups, counts):
        self.groups, self.counts = [list(g) for g in groups], [int(k) for k in counts]

    def build(self, wires):
        qc = QuantumCircuit(len(wires))
        for g, k in zip(self.groups, self.counts):
            if not g or k == 0:
                continue
            sub = DickeState(k).build(list(range(len(g))))
            qc.compose(sub, qubits=g, inplace=True)
        return qc


def linear_ramp(p, gmax=0.8, bmax=0.8):
    g = [gmax * (k + 0.5) / p for k in range(p)]
    b = [bmax * (1 - (k + 0.5) / p) for k in range(p)]
    return np.array([x for gb in zip(g, b) for x in gb])


def energy_std(a, b, M, hw, rng, n=512):
    """Std of the window energy over random bitstrings of the relevant space."""
    X = np.zeros((n, M), dtype=np.int64)
    for r in range(n):
        if hw is None:
            X[r] = rng.integers(0, 2, M)
        else:
            X[r, rng.choice(M, size=hw, replace=False)] = 1
    E = X @ a + np.einsum("ri,ij,rj->r", X, b, X)
    return float(max(E.std(), 1.0))


def flip_qubo(a, b, flip):
    """QUBO in y with x = c + F y (c = flip, F = diag(1 - 2c)): exact affine change of variables."""
    c = np.asarray(flip, dtype=np.int64)
    F = 1 - 2 * c
    bs = b + b.T
    a2 = F * (a + bs @ c)
    b2 = (F[:, None] * b) * F[None, :]
    return a2, b2


def fixed_weight(s, moves):
    """(hw, flip) turning a saturated single-day budget into a Hamming-weight constraint.

    Moves that free a unit (|v| decreases) are flipped (y = 1 - x): with the budget kept
    saturated, sum(x_use) + sum(y_free) = room + n_free.  Returns (None, None) when the
    budget cannot bind inside the window or the window spans several days.
    """
    days = {t for t, _, _ in moves}
    if len(days) != 1:
        return None, None
    t = next(iter(days))
    bud = np.array([abs(s.v[tt, i] + d) - abs(s.v[tt, i]) for tt, i, d in moves], dtype=np.int64)
    room = int(s.p.budget - s.absv[t])
    n_use = int((bud > 0).sum())
    if n_use <= room:
        return None, None
    flip = (bud < 0).astype(np.int64)
    k = room + int(flip.sum())
    if k <= 0 or k >= len(moves):
        return None, None
    return k, flip


class QAOASolver:
    name = "qaoa"

    def __init__(self, layers=5, shots=4000, cloud=False, seed=0, penalty=2.0, keep=4000,
                 retrain_every=300, train_iters=120, angles_file=None, log=print, grouped_dicke=True):
        self.layers, self.shots, self.cloud = layers, shots, cloud
        self.penalty, self.keep = penalty, keep
        self.grouped_dicke = grouped_dicke
        self.plain_when_free = True
        self.retrain_every, self.train_iters = retrain_every, train_iters
        self.rng = np.random.default_rng(seed)
        self.angles_file = Path(angles_file) if angles_file else None
        self.library = {"xy": None, "x": None, "xyw": None}
        self.since_train = {"xy": 0, "x": 0, "xyw": 0}
        if self.angles_file and self.angles_file.exists():
            data = json.loads(self.angles_file.read_text())
            for fam in self.library:
                if data.get(f"{fam}_p{layers}") is not None:
                    self.library[fam] = np.array(data[f"{fam}_p{layers}"])
        self.circuits = 0
        self.train_circuits = 0
        self.calls = 0
        self.max_qubits = 0
        self.log = log
        if cloud:
            self.backend = QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro"))
        else:
            self.backend = MaestroSimulator(shots=shots)

    # ── program construction ─────────────────────────────────────────
    def _program(self, s, moves, hw, optimizer, iters, flip=None, seed_y=None, groups=None):
        M = len(moves)
        a, b, bud = qc.build_qubo(s, moves)
        if flip is not None:
            a, b = flip_qubo(a, b, flip)
        scale = energy_std(a, b, M, hw, self.rng)
        quad = {(m, m): a[m] / scale for m in range(M)}
        for m in range(M):
            for k in range(m + 1, M):
                if b[m, k]:
                    quad[(m, k)] = b[m, k] / scale
        bqm = dimod.BinaryQuadraticModel.from_qubo(quad)
        for m in range(M):
            bqm.add_variable(m, 0.0)
        seed = int(self.rng.integers(1 << 30))
        if hw is not None:
            problem = BinaryOptimizationProblem(bqm)
            if seed_y is None:
                init = DickeState(hw)
            elif groups is not None and self.grouped_dicke:  # uniform over the feasible counts
                init = GroupedDicke(groups, [int(seed_y[g].sum()) for g in groups])
            else:
                init = CustomPerQubitState("".join(map(str, seed_y)))
            prog = QAOA(problem, initial_state=init, n_layers=self.layers, optimizer=optimizer,
                        max_iterations=iters, backend=self.backend, seed=seed)
            edges = []
            for g in (groups or [list(range(M))]):  # one XY ring per conserved group
                if len(g) == 2:
                    edges.append((g[0], g[1]))
                elif len(g) > 2:
                    edges += [(g[k], g[(k + 1) % len(g)]) for k in range(len(g))]
            prog.mixer_hamiltonian = xy_mixer(edges or [(0, 1)], n_qubits=M)
        else:
            days, md, dd, room, lo, hi = qc.window_days(s, moves)
            cons = []
            for w in range(len(days)):
                idx = [m for m in range(M) if md[m] == w]
                coeffs = {m: int(bud[m]) for m in idx if bud[m]}
                if coeffs and sum(max(c, 0) for c in coeffs.values()) > room[w]:
                    cons.append(LinearConstraint(coeffs, "<=", int(room[w])))
                nets = {m: int(dd[m]) for m in idx}
                if sum(max(c, 0) for c in nets.values()) > hi[w]:
                    cons.append(LinearConstraint(nets, "<=", int(hi[w])))
                if sum(min(c, 0) for c in nets.values()) < lo[w]:
                    cons.append(LinearConstraint(nets, ">=", int(lo[w])))
            problem = BinaryOptimizationProblem(bqm, constraints=cons or None, penalty_weight=self.penalty)
            prog = QAOA(problem, n_layers=self.layers, optimizer=optimizer, max_iterations=iters,
                        backend=self.backend, seed=seed)
        return prog

    def _train(self, fam, s, moves, hw, flip=None, seed_y=None, groups=None):
        start = self.library[fam] if self.library[fam] is not None else (
            linear_ramp(self.layers, 0.3, 0.3) if fam == "xyw" else linear_ramp(self.layers))
        t0 = time.time()
        prog = self._program(s, moves, hw, ScipyOptimizer(method=ScipyMethod.COBYLA), self.train_iters, flip,
                             seed_y, groups)
        prog.run(initial_params=start[None, :], perform_final_computation=False)
        self.train_circuits += prog.total_circuit_count
        self.circuits += prog.total_circuit_count
        self.library[fam] = np.asarray(prog.best_params, dtype=float).ravel()
        self.since_train[fam] = 0
        self.log(f"  [qaoa] trained {fam} p={self.layers} on {len(moves)} qubits: loss {prog.best_loss:.3f}, "
                 f"{prog.total_circuit_count} circuits, {time.time() - t0:.0f}s")
        if self.angles_file:
            data = json.loads(self.angles_file.read_text()) if self.angles_file.exists() else {}
            data[f"{fam}_p{self.layers}"] = self.library[fam].tolist()
            self.angles_file.write_text(json.dumps(data, indent=1))

    # ── public API ─────────────────────────────────────────────────────
    def solve(self, s, moves, any_sign, hw=None, flip=None, seed=None, groups=None):
        """seed: optional warm-start bitstring (x-space) of Hamming weight hw (after the flip).
        groups: qubit groups whose counts the mixer conserves separately (needs a seed)."""
        M = len(moves)
        c = np.zeros(M, dtype=np.int64) if flip is None else np.asarray(flip, dtype=np.int64)
        if hw is not None and (hw <= 0 or hw >= M):
            y = np.full(M, 1 if hw >= M else 0, dtype=np.int64)
            return [np.where(c == 1, 1 - y, y)]
        seed_y = None
        if hw is not None and seed is not None:
            seed_y = np.where(c == 1, 1 - np.asarray(seed), np.asarray(seed))
            if int(seed_y.sum()) != hw:
                seed_y = None
        if seed_y is not None and groups is None and self.grouped_dicke and self.plain_when_free:
            seed_y = None  # capital cannot bind: plain Dicke over the whole window
        if seed_y is None:
            groups = None
        if seed_y is not None and groups is not None and self.grouped_dicke:
            fam = "xy"  # grouped Dicke: same family as Dicke (transferred angles)
        else:
            fam = "x" if hw is None else ("xy" if seed_y is None else "xyw")
        if self.library[fam] is None or self.since_train[fam] >= self.retrain_every:
            self._train(fam, s, moves, hw, flip, seed_y, groups)
        prog = self._program(s, moves, hw, MonteCarloOptimizer(population_size=1, n_best_sets=1), 1, flip,
                             seed_y, groups)
        prog.run(initial_params=self.library[fam][None, :])
        self.calls += 1
        self.since_train[fam] += 1
        self.circuits += prog.total_circuit_count
        self.max_qubits = max(self.max_qubits, int(getattr(prog, "n_qubits", M)))
        out = []
        for sol in prog.get_top_solutions(n=self.keep):
            bits = sol.bitstring[:M]
            y = np.frombuffer(bits.encode(), dtype=np.uint8).astype(np.int64) - 48
            out.append(np.where(c == 1, 1 - y, y))
        return out
