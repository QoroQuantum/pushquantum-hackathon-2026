"""Run Divi PCE on a PeriodFormulation and harvest candidates two ways.

* samples      : shots of the final circuit, decoded by the parity code (always codewords)
* expectations : every parameter set the optimiser evaluated gives the vector z_i = <P_i>;
                 x_i = [z_i > 0] (the rounding of the PCE relaxation, as in Sciorilli et al.).
                 This is NOT restricted to the code: any sign pattern of the correlators is reachable.

Candidates are mapped back through the reference offset (complemented variables) and decoded to v.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import PCE, MultiBasisEncoding, PolyEncoding, SoftObjective
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem

import codes as cd
import formulations as fm


class HarvestPCE(PCE):
    """PCE that keeps every expectation vector it evaluates."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.harvest: list[np.ndarray] = []

    def _evaluate_cost_param_sets(self, param_sets, **kwargs):
        out = super()._evaluate_cost_param_sets(param_sets, **kwargs)
        self.harvest.extend(np.array(list(self._cost_expectations.values())))
        return out


def build_problem(F: fm.PeriodFormulation, flip: np.ndarray | None):
    poly = F.full if flip is None else cd.complement(F.full, flip)
    scale = max(abs(c) for c in poly.t.values())
    d = {k: c / scale for k, c in poly.as_divi_dict(F.n_vars).items()}
    prob = BinaryOptimizationProblem(d)
    assert prob.canonical_problem.n_vars == F.n_vars
    order = list(prob.canonical_problem.variable_order)
    assert order == sorted(order), "variable order must be 0..N-1"
    return prob, scale


def run_pce(F: fm.PeriodFormulation, code: str = "poly", flip: np.ndarray | None = None, masks=None, q=None,
            n_layers: int = 3, pop: int = 40, iters: int = 8, shots: int = 4000, alpha: float = 2.0,
            seed: int = 0, top: int = 256, backend=None, optimizer: str = "mc"):
    prob, scale = build_problem(F, flip)
    if code == "poly":
        enc = PolyEncoding()
    elif code == "mask":
        enc = cd.MaskEncoding(masks=masks, q=q)
    elif code.startswith("multibasis"):
        enc = MultiBasisEncoding(order=int(code[-1]))
    else:
        raise ValueError(code)
    backend = backend or MaestroSimulator(shots=shots)
    t0 = time.time()
    opt = (MonteCarloOptimizer(population_size=pop) if optimizer == "mc"
           else ScipyOptimizer(method=ScipyMethod.L_BFGS_B) if optimizer == "lbfgs"
           else ScipyOptimizer(method=ScipyMethod.COBYLA))
    prog = HarvestPCE(prob, encoding=enc, objective=SoftObjective(alpha=alpha), n_layers=n_layers,
                      optimizer=opt, max_iterations=iters,
                      backend=backend, seed=seed)
    prog.run()
    sols = prog.get_top_solutions(n=top, min_prob=0.0, sort_by="energy")
    Xs = np.array([[int(c) for c in s.bitstring] for s in sols], dtype=np.uint8)  # already decoded
    Z = np.array(prog.harvest)
    Xe = (Z > 0).astype(np.uint8)
    if flip is not None:
        Xs = Xs ^ flip[None, :]
        Xe = Xe ^ flip[None, :]
    return dict(samples=np.asarray(Xs, dtype=np.uint8), expect=Xe, qubits=prog.n_qubits,
                circuits=prog.total_circuit_count, sec=time.time() - t0, scale=scale)


def evaluate(F: fm.PeriodFormulation, X: np.ndarray, repair: bool = False):
    """(V feasible-or-repaired, energies) for candidate assignments; infeasible -> inf unless repaired."""
    P = F.prob
    if len(X) == 0:
        return np.zeros((0, P.n), dtype=np.int64), np.zeros(0)
    V = np.unique(F.decode(X), axis=0)
    if repair:
        V = np.unique(np.array([P.repair(v) for v in V]), axis=0)
    E = P.energy(V).astype(float)
    E[~P.feasible(V)] = np.inf
    return V, E


def ideal_relaxation(F: fm.PeriodFormulation, flip=None, alpha: float = 2.0, starts: int = 20, steps: int = 300,
                     lr: float = 0.05, seed: int = 0) -> np.ndarray:
    """Classical control: the SAME SoftObjective loss, but z free in [-1,1]^N (an infinitely expressive
    ansatz). Projected Adam from random starts; returns the sign roundings of every iterate."""
    prob, _ = build_problem(F, flip)
    obj = SoftObjective(alpha=alpha)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(starts):
        z = rng.uniform(-0.1, 0.1, F.n_vars)
        m = np.zeros_like(z); v2 = np.zeros_like(z)
        for k in range(1, steps + 1):
            g = obj.gradient(z, prob)
            m = 0.9 * m + 0.1 * g; v2 = 0.999 * v2 + 0.001 * g * g
            z = np.clip(z - lr * (m / (1 - 0.9**k)) / (np.sqrt(v2 / (1 - 0.999**k)) + 1e-12), -1, 1)
            if k % 10 == 0:
                out.append((z > 0).astype(np.uint8))
    X = np.array(out)
    if flip is not None:
        X = X ^ flip[None, :]
    return X


def run_pce_slots(M, n_layers: int = 3, pop: int = 40, iters: int = 8, shots: int = 4000, alpha: float = 2.0,
                  seed: int = 0, top: int = 256, backend=None, optimizer: str = "mc", n_qubits: int | None = None,
                  code: str = "poly", ansatz=None):
    """PCE on a SlotModel with the closed-form SlotObjective. Returns sampled + expectation-rounded bits."""
    import slots as sl
    prob = sl.placeholder_problem(M.n_vars)
    enc = (PolyEncoding(n_qubits=n_qubits) if code == "poly" else
           MultiBasisEncoding(order=int(code[-1]), n_qubits=n_qubits))
    backend = backend or MaestroSimulator(shots=shots)
    opt = (MonteCarloOptimizer(population_size=pop) if optimizer == "mc"
           else ScipyOptimizer(method=ScipyMethod.L_BFGS_B) if optimizer == "lbfgs"
           else ScipyOptimizer(method=ScipyMethod.COBYLA))
    kw = dict(ansatz=ansatz) if ansatz is not None else {}
    t0 = time.time()
    prog = HarvestPCE(prob, encoding=enc, objective=sl.SlotObjective(M, alpha=alpha), n_layers=n_layers,
                      optimizer=opt, max_iterations=iters, backend=backend, seed=seed, **kw)
    prog.run()
    sols = prog.get_top_solutions(n=top, min_prob=0.0, sort_by="prob")
    Xs = np.array([[int(c) for c in s.bitstring] for s in sols], dtype=np.uint8)
    Xe = (np.array(prog.harvest) > 0).astype(np.uint8)
    return dict(samples=Xs, expect=Xe, qubits=prog.n_qubits, circuits=prog.total_circuit_count,
                sec=time.time() - t0, best_loss=float(prog.best_loss) if hasattr(prog, "best_loss") else None)


def ideal_slots(M, alpha: float = 2.0, starts: int = 20, steps: int = 300, lr: float = 0.05, seed: int = 0):
    """Classical control for the slot loss: z free in [-1,1]^N, projected Adam, sign roundings."""
    import slots as sl
    obj = sl.SlotObjective(M, alpha=alpha)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(starts):
        z = rng.uniform(-0.1, 0.1, M.n_vars)
        m = np.zeros_like(z); v2 = np.zeros_like(z)
        for k in range(1, steps + 1):
            g = obj.gradient(z, None)
            m = 0.9 * m + 0.1 * g; v2 = 0.999 * v2 + 0.001 * g * g
            z = np.clip(z - lr * (m / (1 - 0.9**k)) / (np.sqrt(v2 / (1 - 0.999**k)) + 1e-12), -1, 1)
            if k % 10 == 0:
                out.append((z > 0).astype(np.uint8))
    return np.array(out)
