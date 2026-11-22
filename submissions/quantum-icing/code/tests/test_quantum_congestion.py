from itertools import permutations
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.quantum_info import Statevector

import qoblib_tools as qt
from network_design.block_search import rewire_block
from network_design.quantum_congestion import (
    grover_probabilities,
    ordered_heads,
    screen_rewirings,
    selections,
    topology,
)


class QuantumCongestionTests(unittest.TestCase):
    def test_screen_preserves_every_valid_four_arc_rewiring_of_a_block(self):
        n = 7
        arcs = frozenset((i, (i + offset) % n) for i in range(n) for offset in (1, 2))
        start = ordered_heads(arcs)
        slots = np.array([0, 1, 4, 6])
        orders = np.array([
            p for p in permutations(range(len(slots))) if all(i != j for i, j in enumerate(p))
        ])
        demand = np.ones((n, n)) - np.eye(n)
        scores = screen_rewirings(start, demand, slots[None, :], orders, 5.)
        observed = set()
        for index in np.flatnonzero(np.isfinite(scores[0, :, 0])):
            candidate = start.copy()
            candidate[slots] = start[slots[orders[index]]]
            observed.add(frozenset(topology(candidate)))

        removed = [topology(start)[i] for i in slots]
        expected = {
            candidate for candidate in rewire_block(arcs, removed)
            if len(arcs - candidate) == len(slots)
        }
        self.assertTrue(expected)
        self.assertEqual(observed, expected)
        self.assertTrue(all(qt.is_valid_topology(n, candidate) for candidate in observed))

    def test_grover_agrees_with_a_circuit_and_ignores_padding(self):
        energy = np.array([4., 7., 1., 5., 3., 6.])
        probability, layers, marked = grover_probabilities(energy, 1 / 6)
        self.assertEqual(marked, 1)
        prepare = StatePreparation(np.r_[np.full(6, 1 / np.sqrt(6)), 0., 0.])
        oracle = DiagonalGate([1, 1, -1, 1, 1, 1, 1, 1])
        reflect_zero = DiagonalGate([1] + [-1] * 7)
        circuit = QuantumCircuit(3)
        circuit.append(prepare, circuit.qubits)
        for _ in range(layers):
            circuit.append(oracle, circuit.qubits)
            circuit.append(prepare.inverse(), circuit.qubits)
            circuit.append(reflect_zero, circuit.qubits)
            circuit.append(prepare, circuit.qubits)
        np.testing.assert_allclose(
            Statevector.from_instruction(circuit).probabilities(),
            np.r_[probability, 0., 0.], atol=1e-12,
        )
        self.assertGreater(probability[2], .9)

    def test_selectors_get_equal_unique_lp_budgets_even_with_duplicate_shots(self):
        selected, _layers, _marked = selections(
            np.arange(40, dtype=float), 9, 5, .1, np.random.default_rng(73),
            ("quantum", "greedy", "uniform"),
        )
        count = len(selected["quantum"])
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 5)
        for indices in selected.values():
            self.assertEqual(len(set(indices)), count)


if __name__ == "__main__":
    unittest.main()
