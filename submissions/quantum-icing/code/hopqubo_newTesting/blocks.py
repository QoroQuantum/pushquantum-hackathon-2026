"""Constraint-respecting partitions ("bands") of a network topology, and their sub-solvers.

Why bands. Divi's decomposers cut the n(n-1) arc variables by coupling strength, so every
partition slices through degree constraints and almost no stitched sample is valid (0-15%
measured). A *band* is a rectangle of the adjacency matrix chosen so that the degree
constraints split exactly:

  row band     tails A (k nodes), heads B = heads of the incumbent's out-arcs of A.
               Variables: every arc a→b with a∈A, b∈B, a≠b. All of A's out-arcs live here.
  column band  heads B (k nodes), tails A = tails of the incumbent's in-arcs of B.

Every arc outside the band is fixed at the incumbent. Inside the band the incumbent fixes a
local row sum r_a and column sum c_b, and a band assignment is globally valid iff it meets
them. Bands of one round are disjoint in arcs, so ANY combination of locally valid band
assignments is a valid topology: stitching never breaks the constraints.

Sub-cost: the exact H2 hop polynomial (network_mixer.h2_hubo) with all other arcs fixed.

Sub-solvers (each returns a list of locally valid band assignments, best first):
  xch    QAOA with the 2-exchange mixer restricted to the band, started from the incumbent's
         band state: every state it reaches is locally valid (100% valid by construction).
         Warm start by angle transfer: angles calibrated on the first band are reused.
  xpen   QAOA, X mixer, local row/column sums as penalties, warm-started from
         characterize_and_validate angles (notebook 00 recipe).
  pce    PCE (PolyEncoding) on the penalised band, decoded by a band b-matching on the soft
         values: valid by construction.
  enum   classical: enumerate every locally valid assignment (≤ ~100 for k=3). Oracle/baseline.
  random classical: random locally valid assignments.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from common import H2, Arc

# ── band construction ────────────────────────────────────────────────────


@dataclass
class Band:
    bid: str
    kind: str                 # "row" | "col"
    tails: list[int]
    heads: list[int]
    var_arcs: list[Arc]       # variable v ↔ arc var_arcs[v]
    row_sum: dict[int, int]
    col_sum: dict[int, int]
    inc_bits: list[int]       # incumbent's band assignment
    fixed: list[Arc] = field(default_factory=list)  # all incumbent arcs outside the band

    @property
    def n_vars(self):
        return len(self.var_arcs)

    def arcs_of(self, bits) -> list[Arc]:
        return [a for a, b in zip(self.var_arcs, bits) if int(b)]

    def locally_valid(self, bits) -> bool:
        rows = {a: 0 for a in self.tails}
        cols = {b: 0 for b in self.heads}
        for (a, b), x in zip(self.var_arcs, bits):
            if int(x):
                rows[a] += 1
                cols[b] += 1
        return rows == self.row_sum and cols == self.col_sum

    def n_valid(self) -> int:
        return len(enumerate_band(self))


def make_band(bid, kind, group, inc: list[Arc]) -> Band:
    group = sorted(group)
    gs = set(group)
    if kind == "row":
        inside = [a for a in inc if a[0] in gs]
        tails, heads = group, sorted({b for _, b in inside})
    else:
        inside = [a for a in inc if a[1] in gs]
        tails, heads = sorted({a for a, _ in inside}), group
    var_arcs = [(a, b) for a in tails for b in heads if a != b]
    ins = set(inside)
    return Band(
        bid, kind, tails, heads, var_arcs,
        row_sum={a: sum(1 for x in inside if x[0] == a) for a in tails},
        col_sum={b: sum(1 for x in inside if x[1] == b) for b in heads},
        inc_bits=[int(a in ins) for a in var_arcs],
        fixed=[a for a in inc if a not in ins],
    )


def partition_round(n, inc, kind, k=3, rng=None, hot_tails=()):
    """Split all n nodes into groups of ≤k and return one band per group (all arcs covered once).

    Nodes in `hot_tails` (endpoints of LP-bottleneck arcs) are spread so each lands in a
    different band where possible, so every band sees some congestion to fix."""
    rng = rng or np.random.default_rng()
    nodes = list(rng.permutation(n))
    hot = [v for v in nodes if v in set(hot_tails)]
    cold = [v for v in nodes if v not in set(hot_tails)]
    n_groups = -(-n // k)
    groups = [[] for _ in range(n_groups)]
    for idx, v in enumerate(hot + cold):  # round-robin: hot nodes spread first
        groups[idx % n_groups].append(int(v))
    return [make_band(f"{kind[0]}{g}", kind, grp, inc) for g, grp in enumerate(groups) if grp]


# ── exact sub-cost and enumeration ───────────────────────────────────────


def band_hubo(T, band: Band) -> dict:
    """Exact H2 (normalised by T.max()) as a HUBO over the band's variables, rest fixed."""
    from network_mixer import h2_hubo

    var_of = {a: v for v, a in enumerate(band.var_arcs)}
    return h2_hubo(T, var_of, fixed={a: 1 for a in band.fixed})


def enumerate_band(band: Band) -> list[tuple[int, ...]]:
    """Every locally valid band assignment (row/col sums met), as bit tuples."""
    idx = {a: v for v, a in enumerate(band.var_arcs)}
    out = []
    cap = dict(band.col_sum)

    def rec(t, chosen):
        if t == len(band.tails):
            if all(c == 0 for c in cap.values()):
                bits = [0] * band.n_vars
                for a in chosen:
                    bits[idx[a]] = 1
                out.append(tuple(bits))
            return
        a = band.tails[t]
        opts = [b for b in band.heads if b != a and cap[b] > 0]
        for heads in itertools.combinations(opts, band.row_sum[a]):
            for b in heads:
                cap[b] -= 1
            rec(t + 1, chosen + [(a, b) for b in heads])
            for b in heads:
                cap[b] += 1

    rec(0, [])
    return out


def band_bmatch(band: Band, soft, k=10, noise=0.15, rng=None) -> list[tuple[int, ...]]:
    """Constraint-aware decode of soft band values: max-weight assignment with the band's
    row/column sums (TU LP → integral), plus Gumbel-perturbed variants for diversity."""
    from scipy.optimize import linprog

    rng = rng or np.random.default_rng(0)
    m = band.n_vars
    rows = {a: r for r, a in enumerate(band.tails)}
    cols = {b: len(rows) + c for c, b in enumerate(band.heads)}
    A = np.zeros((len(rows) + len(cols), m))
    for v, (a, b) in enumerate(band.var_arcs):
        A[rows[a], v] = 1
        A[cols[b], v] = 1
    rhs = [band.row_sum[a] for a in band.tails] + [band.col_sum[b] for b in band.heads]
    out, seen = [], set()
    for t in range(3 * k):
        w = np.asarray(soft, float) + (0 if t == 0 else noise * rng.gumbel(size=m))
        r = linprog(-w, A_eq=A, b_eq=rhs, bounds=(0, 1), method="highs")
        if r.status != 0:
            continue
        bits = tuple(int(x > 0.5) for x in r.x)
        if band.locally_valid(bits) and bits not in seen:
            seen.add(bits)
            out.append(bits)
        if len(out) >= k:
            break
    return out


# ── stitching: beam search over bands, global H2, all candidates valid ───


def stitch(T, bands: list[Band], cands: dict[str, list[tuple]], beam=32, per_band=8):
    """Combine per-band candidates into full topologies ranked by global H2.

    Each band contributes its incumbent assignment plus up to `per_band` candidates; since bands
    are arc-disjoint and each candidate is locally valid, every combination is a valid topology."""
    h2 = H2(T)
    base = set(bands[0].fixed) | set(bands[0].arcs_of(bands[0].inc_bits))  # full incumbent
    beams = [(h2(base), frozenset(base))]
    for band in bands:
        options = [band.inc_bits] + [c for c in cands.get(band.bid, [])[:per_band]]
        own = set(band.arcs_of(band.inc_bits))
        nxt = {}
        for _, arcs in beams:
            rest = arcs - own
            for bits in options:
                new = frozenset(rest | set(band.arcs_of(bits)))
                if new not in nxt:
                    nxt[new] = h2(new)
        beams = sorted(((e, a) for a, e in nxt.items()), key=lambda x: x[0])[:beam]
    return [sorted(a) for _, a in beams]


# ── sub-solvers ──────────────────────────────────────────────────────────


def solve_enum(T, band, top=8, **_):
    h2 = H2(T)
    allb = enumerate_band(band)
    scored = sorted(allb, key=lambda b: h2(band.fixed + band.arcs_of(b)))
    return scored[:top], {"n_valid": len(allb), "circuits": 0}


def solve_random(T, band, top=8, rng=None, **_):
    rng = rng or np.random.default_rng()
    allb = enumerate_band(band)
    pick = [allb[i] for i in rng.permutation(len(allb))[:top]]
    return pick, {"n_valid": len(allb), "circuits": 0}


def _qaoa_candidates(qaoa, band, top):
    sols = qaoa.get_top_solutions(n=0)  # n=0: every sampled bitstring
    out, seen, p_valid, prob = [], set(), 0.0, {}
    for s in sols:
        bits = tuple(int(c) for c in s.bitstring)
        if band.locally_valid(bits):
            p_valid += s.prob
            prob[bits] = prob.get(bits, 0.0) + s.prob
            if bits not in seen:
                seen.add(bits)
                out.append(bits)
    return out, p_valid, prob


def concentration(T, band, prob):
    """P(band's exact H2 optimum | valid) under the sampler ÷ 1/(#valid band states): >1 = the
    quantum state concentrates on the best band assignment more than uniform valid sampling."""
    h2 = H2(T)
    allb = enumerate_band(band)
    best = min(allb, key=lambda b: h2(band.fixed + band.arcs_of(b)))
    p_valid = sum(prob.values())
    return prob.get(best, 0.0) / p_valid * len(allb) if p_valid else 0.0


def build_xch_program(T, band, n_layers, backend, pop, iters, seed, warm=None):
    """Exchange-mixer QAOA on the band, from the incumbent's band state."""
    from divi.qprog.initial_states import CustomPerQubitState
    from divi.qprog.optimizers import MonteCarloOptimizer

    from divi_ext import WarmQAOA, hubo_problem, population_around
    from network_mixer import exchange_mixer

    qubit_of = {a: v for v, a in enumerate(band.var_arcs)}
    mixer = exchange_mixer(qubit_of, band.n_vars)
    if mixer is None:  # no 2-exchange inside the band: nothing to move
        return None
    q = WarmQAOA(hubo_problem(band_hubo(T, band), band.n_vars), n_layers=n_layers,
                 initial_state=CustomPerQubitState("".join(map(str, band.inc_bits))),
                 optimizer=MonteCarloOptimizer(population_size=pop), max_iterations=iters,
                 backend=backend, seed=seed)
    q.mixer_hamiltonian = mixer
    if warm is not None:
        q.warm_params = population_around(warm, pop, seed=seed)
    return q


def penalised_band_problem(T, band, penalty_weight, rows=True):
    """Band H2 HUBO + local row/column sums as Divi LinearConstraints (squared penalties).
    rows=False leaves the row sums out (an XY mixer per row conserves them exactly)."""
    from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint

    h = {k: v for k, v in band_hubo(T, band).items() if k}
    for v in range(band.n_vars):
        h[(v,)] = h.get((v,), 0.0) + 1e-9
    cons = []
    for a in band.tails if rows else ():
        cons.append(LinearConstraint({v: 1 for v, x in enumerate(band.var_arcs) if x[0] == a}, "==", band.row_sum[a]))
    for b in band.heads:
        cons.append(LinearConstraint({v: 1 for v, x in enumerate(band.var_arcs) if x[1] == b}, "==", band.col_sum[b]))
    return BinaryOptimizationProblem(h, constraints=cons, penalty_weight=penalty_weight,
                                     hamiltonian_builder="native")


def row_groups(band):
    """Qubits of each tail's out-arcs inside the band (the XY mixer's blocks)."""
    return [[v for v, x in enumerate(band.var_arcs) if x[0] == a] for a in band.tails]


def build_xy_program(T, band, n_layers, backend, pop, iters, seed, penalty_weight, warm=None):
    """QAOA with a complete XY mixer inside each row block, from the incumbent's band state:
    every out-degree stays exact; column sums are penalised (reduced phase operator, as Divi's
    CE-QAOA does for TSP)."""
    from divi.hamiltonians import xy_mixer
    from divi.qprog.initial_states import CustomPerQubitState
    from divi.qprog.optimizers import MonteCarloOptimizer

    from divi_ext import WarmQAOA, population_around

    edges = [e for g in row_groups(band) for e in itertools.combinations(g, 2)]
    if not edges:
        return None
    q = WarmQAOA(penalised_band_problem(T, band, penalty_weight, rows=False), n_layers=n_layers,
                 initial_state=CustomPerQubitState("".join(map(str, band.inc_bits))),
                 optimizer=MonteCarloOptimizer(population_size=pop), max_iterations=iters,
                 backend=backend, seed=seed)
    q.mixer_hamiltonian = xy_mixer(edges, n_qubits=band.n_vars)
    if warm is not None:
        q.warm_params = population_around(warm, pop, seed=seed)
    return q


def band_char_constraints(band, rows=True, cols=True):
    """The band's row/column sums as Characterization Service eq_cardinality constraints."""
    out = []
    for a in band.tails if rows else ():
        out.append({"type": "eq_cardinality", "bound": band.row_sum[a],
                    "qubits": [v for v, x in enumerate(band.var_arcs) if x[0] == a]})
    for b in band.heads if cols else ():
        out.append({"type": "eq_cardinality", "bound": band.col_sum[b],
                    "qubits": [v for v, x in enumerate(band.var_arcs) if x[1] == b]})
    return out


def build_xpen_program(T, band, n_layers, backend, pop, iters, seed, penalty_weight, warm=None):
    from divi.qprog.optimizers import MonteCarloOptimizer

    from divi_ext import WarmQAOA, population_around

    q = WarmQAOA(penalised_band_problem(T, band, penalty_weight), n_layers=n_layers,
                 optimizer=MonteCarloOptimizer(population_size=pop), max_iterations=iters,
                 backend=backend, seed=seed)
    if warm is not None:
        q.warm_params = population_around(warm, pop, seed=seed)
    return q


def build_pce_program(T, band, n_layers, backend, pop, iters, seed, penalty_weight, alpha):
    from divi.qprog.algorithms import PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer

    from divi_ext import WarmPCE

    return WarmPCE(penalised_band_problem(T, band, penalty_weight), encoding=PolyEncoding(),
                   objective=SoftObjective(alpha=alpha), n_layers=n_layers,
                   optimizer=MonteCarloOptimizer(population_size=pop), max_iterations=iters,
                   backend=backend, seed=seed)


def harvest(program, band, method, top, alpha=None, rng=None, T=None):
    """Locally valid candidates from a finished program + diagnostics."""
    if method in ("xch", "xpen", "xy"):
        cands, p_valid, prob = _qaoa_candidates(program, band, top)
        d = {"p_valid": round(p_valid, 4), "circuits": program.total_circuit_count,
             "n_valid": len(enumerate_band(band)), "qubits": band.n_vars}
        if T is not None:
            d["conc"] = round(concentration(T, band, prob), 2)
        return cands[:top], d
    if method == "pce":
        from divi_ext import pce_soft_values

        soft = pce_soft_values(program, alpha)
        cands = band_bmatch(band, soft, k=top, rng=rng)
        return cands, {"p_valid": 1.0, "circuits": program.total_circuit_count}
    raise ValueError(method)
