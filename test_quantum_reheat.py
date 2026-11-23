import dimod
import numpy as np
import pytest
from fractions import Fraction

import qoblib_tools as qt
from quantum_reheat import QAOASeedStage, QuantumReheat, QuantumStats, StuckPolicy
from solver_partitioned_sa import AnnealStage, PartitionSASolver, numpy_anneal
from asset_groups import BlockEvaluator


def tiny_model(n=6, T=4, seed=0):
    rng = np.random.default_rng(seed)
    syms = [f"A{i}" for i in range(n)]
    prices = [[Fraction(int(rng.integers(50, 150))) for _ in range(T)] for _ in range(n)]
    A = rng.normal(size=(n, n)); C = A @ A.T / n
    cov = {(i, j, t): Fraction(float(C[i, j])).limit_denominator(10_000)
           for t in range(T) for i in range(n) for j in range(n)}
    inst = qt.PortfolioInstance("po_tiny", syms, T, prices, cov)
    return qt.PortfolioModel(inst, budget=4, lam="1e-2")


def rand_bqm(n, seed=0):
    rng = np.random.default_rng(seed)
    return dimod.BinaryQuadraticModel(
        {i: float(rng.normal()) for i in range(n)},
        {(i, j): float(rng.normal()) for i in range(n) for j in range(i + 1, n)}, 0.0, "BINARY")


def test_select_seeds_distinct_and_includes_best():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (64, 10)).astype(np.int8)
    bits = np.vstack([bits, bits[:10]])              # duplicates
    e = rng.normal(size=len(bits))
    s = QAOASeedStage.select_seeds(bits, e, 5)
    assert len(s) == 5 and len({tuple(r) for r in s}) == 5
    assert np.array_equal(s[0], bits[int(np.argmin(e))])


def test_small_block_returns_empty():
    st = QuantumStats()
    stage = QAOASeedStage(min_qubits=24, stats=st)
    out = stage(rand_bqm(12), np.zeros((1, 12), np.int8), np.random.default_rng(0))
    assert out.shape == (0, 12) and out.dtype == np.int8
    assert st.qaoa_skipped_small == 1 and st.qaoa_calls == 0


def test_more_starts_than_reads():
    bqm, n = rand_bqm(8), 8
    starts = np.random.default_rng(0).integers(0, 2, (5, n)).astype(np.int8)
    out = numpy_anneal(bqm, starts, np.random.default_rng(0), 2, 5, 1.0, True)
    assert out.shape == (2, n)
    pytest.importorskip("dwave.samplers")
    out = AnnealStage(2, 5, 1.0, True, "dwave")(bqm, starts, np.random.default_rng(0))
    assert out.shape == (2, n)


class Boom:
    n_seeds = 8
    def __call__(self, *a, **k): raise AssertionError("QAOA stage must not run when off")


def test_flag_off_equivalence():
    model = tiny_model(); ev = BlockEvaluator(model)
    L = np.zeros((4, 6), np.int64); S = L.copy()
    stages = (AnnealStage(8, 20, backend="numpy"),
              AnnealStage(8, 20, 2.0, True, backend="numpy"))
    off = QuantumReheat(Boom(), StuckPolicy("off"), 1.25, QuantumStats())
    s0 = PartitionSASolver(stages, 7)
    s1 = PartitionSASolver(stages, 7, quantum=off)
    G = np.arange(4)
    r0, r1 = s0(ev, L, S, [1], G), s1(ev, L, S, [1], G)
    assert np.array_equal(r0.V, r1.V)
    assert s0.rng.bit_generator.state == s1.rng.bit_generator.state
    assert s0.stage_calls == s1.stage_calls