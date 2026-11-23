"""QAOA-seeded reheat: heuristic restart distribution for the SA reheat stage."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, fields

import numpy as np

REASONS = ("always", "no_improvement", "plateau", "block_stalled")


@dataclass
class QuantumStats:
    qaoa_calls: int = 0
    qaoa_cache_hits: int = 0
    qaoa_skipped_small: int = 0
    qaoa_skipped_large: int = 0
    skipped_budget: int = 0
    qaoa_seconds: float = 0.0
    credits_used: float = 0.0            # stays 0 on the local simulator
    qaoa_direct_improvements: int = 0
    reheats_after_quantum: int = 0
    reheats_after_classical: int = 0
    improvements_after_quantum_reheat: int = 0
    improvements_after_classical_reheat: int = 0
    triggers: dict = field(default_factory=lambda: dict.fromkeys(REASONS, 0))

    def merge(self, o: "QuantumStats") -> None:
        for f in fields(self):
            if f.name == "triggers":
                for k, v in o.triggers.items():
                    self.triggers[k] = self.triggers.get(k, 0) + v
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(o, f.name))

    def as_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["triggers"] = dict(self.triggers)
        d["qaoa_seconds"] = round(self.qaoa_seconds, 2)
        return d


@dataclass
class StuckPolicy:
    mode: str = "stuck"          # off | always | stuck
    stuck_frac: float = 0.5
    stuck_eps: float = 1e-6      # relative to max(1, |e_min|)
    budget: int = 20             # max QAOA runs per solver (= per seed)

    def reason(self, best_cost, cur_cost, energies, block_stalled):
        """Return the trigger reason after stage 1, or None. Size/budget checked by caller/stage."""
        if self.mode == "off":
            return None
        if self.mode == "always":
            return "always"
        if best_cost >= cur_cost - 1e-9:
            return "no_improvement"
        e = np.asarray(energies)
        tol = self.stuck_eps * max(1.0, abs(float(e.min())))
        if np.mean(e <= e.min() + tol) > self.stuck_frac:
            return "plateau"
        if block_stalled:
            return "block_stalled"
        return None


class QAOASeedStage:
    """BlockStage returning QAOA samples (top-K bitstrings) in label order 0..n-1."""

    seeds_for_next: bool = True

    def __init__(self, layers=2, iters=15, shots=4000, top=64, n_seeds=8, init="tqa",
                 tqa_dt=1.0, backend=None, seed=0, min_qubits=24, max_qubits=26,
                 null_random=False, stats: QuantumStats | None = None):
        if iters == 0 and init != "tqa":
            raise ValueError("iters=0 requires init='tqa'")
        self.layers, self.iters, self.shots, self.top = layers, iters, shots, top
        self.n_seeds, self.init, self.tqa_dt = n_seeds, init, tqa_dt
        self.backend, self.seed = backend, seed
        self.min_qubits, self.max_qubits = min_qubits, max_qubits
        self.null_random = null_random
        self.stats = stats if stats is not None else QuantumStats()
        self.cache: dict = {}
        self._solver = None

    # -- seed selection ------------------------------------------------
    @staticmethod
    def select_seeds(bits, energies, k):
        """k distinct rows: greedy max-min Hamming among the lowest-energy 4k distinct samples."""
        bits = np.asarray(bits)
        if k <= 0 or not len(bits):
            return bits[:0].copy()
        order = np.argsort(np.asarray(energies), kind="stable")
        pool = bits[order]
        _, first = np.unique(pool, axis=0, return_index=True)
        pool = pool[np.sort(first)][: 4 * k]          # distinct, still energy-ordered
        chosen = [0]
        dmin = (pool != pool[0]).sum(axis=1)
        while len(chosen) < min(k, len(pool)):
            j = int(np.argmax(dmin))                  # ties -> lower energy
            chosen.append(j)
            dmin = np.minimum(dmin, (pool != pool[j]).sum(axis=1))
        return pool[chosen].copy()

    # -- caching ---------------------------------------------------------
    @staticmethod
    def _hash(bqm) -> str:
        n = bqm.num_variables
        lin = np.array([bqm.linear[j] for j in range(n)], dtype=float)
        quad = sorted((min(i, j), max(i, j), b) for (i, j), b in bqm.quadratic.items())
        s = max(float(np.abs(lin).max(initial=0.0)),
                max((abs(b) for _, _, b in quad), default=0.0)) or 1.0
        h = hashlib.blake2b(digest_size=16)
        h.update(np.round(lin / s, 6).tobytes())
        h.update(repr([(i, j, round(b / s, 6)) for i, j, b in quad]).encode())
        return h.hexdigest()

    # -- sampling --------------------------------------------------------
    def _helper(self):
        if self._solver is None:
            from qaoa_signed_blocks import QAOABlockSolver   # imports Divi lazily
            self._solver = QAOABlockSolver(
                layers=self.layers, iters=self.iters, shots=self.shots, top=self.top,
                init=self.init, tqa_dt=self.tqa_dt, backend=self.backend, seed=self.seed)
        return self._solver

    def _sample(self, bqm):
        s = self._helper()
        if self.iters > 0:
            return s._qaoa(bqm)
        from qaoa_signed_blocks import tqa_params
        q, bqmn = s._make_qaoa(bqm)
        theta = tqa_params(self.layers, self.tqa_dt).reshape(q.get_expected_param_shape())
        q.sample_solution(theta)                           # no optimiser (verify on your Divi)
        return s._top_bits(q, bqmn)

    def __call__(self, bqm, starts, rng):
        n = bqm.num_variables
        empty = np.zeros((0, n), dtype=np.int8)
        if n < self.min_qubits:
            self.stats.qaoa_skipped_small += 1
            return empty
        if n > self.max_qubits:
            self.stats.qaoa_skipped_large += 1
            return empty
        if self.null_random:                              # bench control, not a quantum call
            return rng.integers(0, 2, size=(self.top, n)).astype(np.int8)
        key = (self._hash(bqm), self.init, self.layers, self.iters)
        if key in self.cache:
            self.stats.qaoa_cache_hits += 1
            return self.cache[key].copy()
        t0 = time.time()
        bits = np.asarray(self._sample(bqm), dtype=np.int8)
        self.stats.qaoa_seconds += time.time() - t0
        self.stats.qaoa_calls += 1
        self.cache[key] = bits
        return bits.copy()


@dataclass
class QuantumReheat:
    stage: QAOASeedStage
    policy: StuckPolicy
    reheat_scale: float
    stats: QuantumStats

    @classmethod
    def from_args(cls, a, seed: int) -> "QuantumReheat":
        stats = QuantumStats()
        stage = QAOASeedStage(
            layers=a.quantum_layers, iters=a.quantum_iters, shots=a.quantum_shots,
            top=a.quantum_top, n_seeds=a.quantum_seeds, init=a.quantum_init,
            tqa_dt=a.quantum_tqa_dt, seed=seed, min_qubits=a.quantum_min_qubits,
            max_qubits=a.quantum_max_qubits, null_random=a.quantum_null_control, stats=stats)
        policy = StuckPolicy(a.quantum_reheat, a.stuck_frac, budget=a.quantum_budget)
        return cls(stage, policy, a.quantum_reheat_scale, stats)