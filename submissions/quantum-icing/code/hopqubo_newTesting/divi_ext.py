"""Extensions to Divi 0.16.2 used by this pipeline (no edits to the installed package).

1. WarmQAOA / WarmPCE      programs that carry their own warm-start angles, so they warm-start
                           even when an ensemble calls `program.run()` with no arguments
                           (PartitioningProgramEnsemble has no initial_params hook).
2. BlockEnsemble           a ProgramEnsemble over caller-built, constraint-respecting blocks:
                           Divi runs the sub-programs in parallel (batched on the cloud), we
                           stitch. Divi's own decomposer path is QUBO-only and cuts through the
                           degree constraints; ours does not.
3. NodePairEncoding        PCE encoding with one qubit per *node*: arc i→j (i<j) is X_i X_j and
                           arc j→i is Y_i Y_j. Arcs sharing an endpoint share a qubit, so the
                           degree constraint of node v lives on the correlators of qubit v.
                           n(n-1) arc variables on n qubits (132 on 12, 552 on 24).
4. pce_soft_values         per-variable soft values (1 + tanh(α⟨P_v⟩))/2 after training, for
                           constraint-aware decoding (b-matching) instead of sign rounding.
5. characterize_cached     characterize_and_validate with a JSON cache of job ids; re-fetching
                           a finished report is free (notebook 00, section 1.2).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np

from divi.qprog.algorithms import PCE, QAOA, MultiBasisEncoding
from divi.qprog.ensembles._base import ProgramEnsemble

from common import RUNS


# ── 1. warm-start-carrying programs ──────────────────────────────────────
class _WarmMixin:
    warm_params: np.ndarray | None = None

    def run(self, initial_params=None, **kwargs):
        if initial_params is None and self.warm_params is not None:
            initial_params = self.warm_params
        return super().run(initial_params=initial_params, **kwargs)


class WarmQAOA(_WarmMixin, QAOA):
    pass


class WarmPCE(_WarmMixin, PCE):
    pass


def population_around(angles, pop, scale=0.05, seed=0):
    """MC-optimiser starts: the given angles in row 0, plus Gaussian jitter (notebook 00 recipe)."""
    angles = np.atleast_2d(np.asarray(angles, dtype=float))[0]
    starts = angles + np.random.default_rng(seed).normal(scale=scale, size=(pop, angles.size))
    starts[0] = angles
    return starts


# ── 2. ensemble over caller-built blocks ─────────────────────────────────
class BlockEnsemble(ProgramEnsemble):
    """Run a dict {block_id: program} in parallel through Divi's ensemble machinery.

    Usage: ens = BlockEnsemble(backend); ens.programs = {...}; ens.run(); ens.aggregate_results()
    """

    def create_programs(self, state=None):  # programs are supplied by the caller
        super().create_programs()

    def aggregate_results(self):
        return dict(self._programs)


# ── 3. node-structured PCE encoding ──────────────────────────────────────
@dataclass(frozen=True)
class NodePairEncoding(MultiBasisEncoding):
    """Variable v ↔ Pauli correlator `pairs[v] = (letters, (a, b))` on node qubits a, b."""

    pairs: tuple = ()

    def min_qubits(self, n_vars: int) -> int:
        return 1 + max(max(q) for _, q in self.pairs[:n_vars])

    def strings(self, n_vars: int, n_qubits: int):
        return list(self.pairs[:n_vars])


def node_pair_layout(n: int, arcs_index):
    """Pauli strings for arc variables `arcs_index` on n node-qubits: i<j → XX, i>j → YY."""
    pairs = []
    for i, j in arcs_index:
        a, b = min(i, j), max(i, j)
        pairs.append(("XX" if i < j else "YY", (a, b)))
    return NodePairEncoding(order=2, n_qubits=n, pairs=tuple(pairs))


# ── 4. soft values after training ────────────────────────────────────────
def pce_soft_values(pce: PCE, alpha: float) -> np.ndarray:
    """(1 + tanh(α⟨P_v⟩))/2 at the best parameters: the trained relaxation, before any rounding."""
    from threading import Event

    # An ensemble sets its shared cancellation event when it finishes; give the program a fresh
    # one so this post-run evaluation is not refused as "cancelled".
    pce._set_cancellation_event(Event())
    z = pce._variable_expectations(np.atleast_2d(pce.best_params))[0]
    return (1 + np.tanh(alpha * z)) / 2


# ── 5. cached characterization ───────────────────────────────────────────
CHAR_CACHE = RUNS / "characterization_cache.json"


def characterize_cached(key, problem, reference_states, service, layers=2, constraints=None, mixer=None):
    """Return a CharacterizationResult; the job id is cached under `key` so reruns are free."""
    from divi.backends.characterization import (
        CharacterizationOptions,
        characterize_and_validate,
        get_characterization_result,
    )

    cache = json.loads(CHAR_CACHE.read_text()) if CHAR_CACHE.exists() else {}
    if key in cache:
        return get_characterization_result(cache[key]["job_id"], service=service)
    opts = {"ansatz": {"layers": layers, **({"mixer": mixer} if mixer else {})}}
    if constraints:
        opts["constraints"] = constraints
    report = characterize_and_validate(problem, reference_states=reference_states, service=service,
                                       options=CharacterizationOptions(**opts))
    cache[key] = {"job_id": report.job_id, "recommended_layers": report.recommended_layers}
    CHAR_CACHE.write_text(json.dumps(cache, indent=1))
    return report


def hubo_problem(hubo: dict, n_q: int):
    """BinaryOptimizationProblem from a {tuple(vars): coeff} HUBO; every qubit kept as a wire."""
    from divi.qprog.problems import BinaryOptimizationProblem

    h = {k: v for k, v in hubo.items() if k}
    for v in range(n_q):
        h[(v,)] = h.get((v,), 0.0) + 1e-9
    return BinaryOptimizationProblem(h, hamiltonian_builder="native")


def all_pairs(n):
    return list(itertools.permutations(range(n), 2))
