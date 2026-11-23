"""Warm-started whole-day QAOA at 150 qubits (one a050 day), simulated with Maestro MPS.

Cutting by period gives one block per day with every asset in it. With the signed
3-bit encoding (``qaoa_signed_blocks.block_qubo``) a whole a050 day is 50 x 3 = 150
qubits and ~11k couplings, far beyond a statevector. Two things make it workable:

* **Warm start** (Egger, Mareček, Woerner, Quantum 5, 479 (2021)). Qubit q starts in
  ``RY(θ_q)|0>`` with ``sin²(θ_q/2) = c_q``, where ``c_q`` is the current solution's
  bit, regularised to ``eps`` / ``1 - eps``. The mixer is ``RY(θ_q) RZ(2β) RY(-θ_q)``,
  which has that state as an eigenstate. With ``γ = 0`` the circuit samples exactly
  the classical baseline: independent bit flips of the current solution with
  probability ``eps``. Any difference at ``γ != 0`` is the cost layer's work.
* **MPS.** A near-product state carries little entanglement, so Maestro's
  matrix-product-state simulator handles 150 qubits at a small bond dimension,
  locally or on Qoro's cloud.

The QUBO is scaled so that the median |energy change of one bit flip| at the warm
start is 1, so angles carry over between block sizes. Each sample is decoded to a
day holding and scored exactly (``day_swap.day_form``); the best feasible sample
that beats the current day replaces it (re-scored with the official referee).

    python warm_day_qaoa.py tune  --group 8 ...          # local statevector, small blocks
    python warm_day_qaoa.py sweep --cluster local ...    # whole days, γ=0 baseline vs QAOA
    python warm_day_qaoa.py sweep --cluster qoro_cpu --mps-chi 32 ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DIVI_DISABLE_PROGRESS", "1")
import numpy as np
from qiskit.circuit import QuantumCircuit

import qoblib_tools as qt
from asset_groups import BlockEvaluator, _load
from day_swap import day_form
from qaoa_signed_blocks import block_qubo, decode
from divi.backends import JobConfig, MaestroConfig, MaestroSimulator, QoroService
from divi.qprog.algorithms import QAOA
from divi.qprog.ensembles import BatchConfig, ProgramEnsemble
from divi.qprog.initial_states import InitialState
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem


# ── warm-start state, mixer and fixed-angle sampling ─────────────────────────


class WarmState(InitialState):
    """Product state ``⊗ RY(θ_q)|0>``."""

    def __init__(self, thetas):
        self.thetas = np.asarray(thetas, float)

    def build(self, wires) -> QuantumCircuit:
        qc = QuantumCircuit(len(wires))
        for q, th in enumerate(self.thetas):
            if th:
                qc.ry(float(th), q)
        return qc


class WarmProblem(BinaryOptimizationProblem):
    """BinaryOptimizationProblem with the warm-start mixer ``RY(θ) RZ(2β) RY(-θ)`` per qubit."""

    def __init__(self, bqm, thetas):
        super().__init__(bqm)
        self.thetas = np.asarray(thetas, float)

    @property
    def recommended_initial_state(self):
        return WarmState(self.thetas)

    @property
    def _applies_mixer_as_gates(self) -> bool:
        return True

    def apply_mixer(self, circuit, beta, qubits) -> None:
        for q in qubits:
            th = float(self.thetas[q])
            circuit.ry(-th, q)
            circuit.rz(2 * beta, q)
            circuit.ry(th, q)

    def mixer_parameter_frequency(self):
        return 1.0, len(self.thetas)


class FixedAngleQAOA(QAOA):
    """``run()`` only samples at the given angles: one circuit per program."""

    def __init__(self, *a, angles=None, **k):
        self.angles = np.asarray(angles, float)
        super().__init__(*a, **k)

    def run(self, **kwargs):  # noqa: ARG002 - ProgramEnsemble calls run() without arguments
        return self.sample_solution(params=self.angles)


class Ensemble(ProgramEnsemble):
    def __init__(self, backend, problems, angles_list, layers):
        super().__init__(backend)
        self.problems, self.angles_list, self.layers = problems, angles_list, layers

    def create_programs(self, state=None):  # noqa: ARG002
        super().create_programs()
        self.programs = {
            f"p{i}": FixedAngleQAOA(pr, n_layers=self.layers, angles=ang, backend=self.backend, seed=i,
                                    optimizer=MonteCarloOptimizer(population_size=1, n_best_sets=1,
                                                                  keep_best_params=False),
                                    max_iterations=1)
            for i, (pr, ang) in enumerate(zip(self.problems, self.angles_list))}

    def aggregate_results(self):
        return None


# ── swap-network circuits (MPS-friendly: nearest-neighbour gates only) ────────


def swapnet_circuit(spo, thetas, angles) -> QuantumCircuit:
    """The same warm-start QAOA circuit as Divi's, laid out as a swap network.

    Divi emits ``exp(-i γ c P)`` for each term of the cost Hamiltonian ``spo`` (RZ for Z,
    RZZ for ZZ, all commuting), so their order and placement are free. With every pair
    of the n qubits coupled, Divi's RZZs span the whole chain and Maestro's MPS spends
    its time routing. An odd-even transposition network of n layers of nearest-
    neighbour RZZ+SWAP makes every pair adjacent exactly once (and reverses the order);
    the mixer then acts on whichever logical qubit sits on each wire, and each wire is
    measured into its logical qubit's classical bit, so counts read as Divi's do.
    """
    n = len(thetas)
    lin = np.zeros(n)
    J = {}
    for pauli, idx, c in spo.to_sparse_list():
        c = float(np.real(c))
        if pauli == "Z":
            lin[idx[0]] += c
        elif pauli == "ZZ":
            a, b = sorted(idx)
            J[a, b] = J.get((a, b), 0.0) + c
        elif pauli.strip("I"):
            raise ValueError(f"unexpected term {pauli}")
    qc = QuantumCircuit(n, n)
    perm = list(range(n))  # perm[wire] = logical qubit on that wire
    for w in range(n):
        if thetas[w]:
            qc.ry(float(thetas[w]), w)
    for gamma, beta in np.asarray(angles, float).reshape(-1, 2):
        for w in range(n):
            if lin[perm[w]]:
                qc.rz(2 * gamma * lin[perm[w]], w)
        for step in range(n):
            for w in range(step % 2, n - 1, 2):
                a, b = perm[w], perm[w + 1]
                c = J.get((min(a, b), max(a, b)), 0.0)
                if c:
                    qc.rzz(2 * gamma * c, w, w + 1)
                qc.swap(w, w + 1)
                perm[w], perm[w + 1] = b, a
        for w in range(n):
            th = float(thetas[perm[w]])
            qc.ry(-th, w)
            qc.rz(2 * beta, w)
            qc.ry(th, w)
    for w in range(n):
        qc.measure(w, perm[w])
    return qc


def sample_swapnet(backend, blocks, angles_list):
    """Sample every (block, angles) swap-network circuit in one submission."""
    from qiskit import qasm2  # noqa: PLC0415

    circuits = {}
    for i, (b, ang) in enumerate(zip(blocks, angles_list)):
        spo = BinaryOptimizationProblem(b["bqm"]).cost_hamiltonian
        circuits[f"c{i}"] = qasm2.dumps(swapnet_circuit(spo, b["thetas"], ang))
    t0 = time.time()
    res = backend.submit_circuits(circuits)
    if isinstance(backend, QoroService):
        backend.poll_job_status(res, loop_until_complete=True)
        res = backend.get_job_results(res)
    by_label = {r["label"]: r["results"] for r in res.results}
    out = []
    for i in range(len(blocks)):
        counts = by_label[f"c{i}"]
        bits = np.array([[int(ch) for ch in s] for s in counts], dtype=np.int8)
        out.append((bits, np.array(list(counts.values()), float)))
    return out, time.time() - t0


# ── block construction ────────────────────────────────────────────────────────


CODES = np.array([[(k >> j) & 1 for j in range(3)] for k in range(8)], np.int8)  # bit j of code k


def encode(v, W, C, bqm) -> np.ndarray:
    """Net positions (g,) -> bits (3g,): per asset the lowest-energy code that decodes to v_i."""
    g = len(C)
    bits = np.zeros(3 * g, np.int8)
    for i in range(g):
        vals = CODES @ W[i] + C[i]
        cand = np.flatnonzero(np.isclose(vals, v[i]))
        if len(cand) == 0:
            raise ValueError(f"asset {i}: value {v[i]} not representable")
        def e(k):
            b = CODES[k]
            s = sum(bqm.linear[3 * i + j] * b[j] for j in range(3))
            for a, c in ((0, 1), (0, 2), (1, 2)):
                s += bqm.quadratic.get((3 * i + a, 3 * i + c), 0.0) * b[a] * b[c]
            return s
        bits[3 * i:3 * i + 3] = CODES[min(cand, key=e)]
    return bits


def to_arrays(bqm, nq):
    """Dense (h, J) of the BINARY bqm: E(b) = h·b + Σ_{k<l} J_kl b_k b_l + offset."""
    h = np.array([bqm.linear[k] for k in range(nq)])
    J = np.zeros((nq, nq))
    for (k, l), w in bqm.quadratic.items():
        J[k, l] = J[l, k] = w
    return h, J


def flip_deltas(h, J, b):
    """Energy change of flipping each bit of b."""
    bf = b.astype(float)
    return (1 - 2 * bf) * (h + J @ bf)


def day_costs(F, V):
    """Exact day cost of holdings V (K, g) and feasibility, from ``day_form``'s ``A`` and ``f``."""
    V = np.asarray(V, np.int64)
    risk = np.einsum("ki,ij,kj->k", V, F["A"], V)
    sep = F["f"][np.arange(V.shape[1])[None, :], V + 3].sum(1)
    units, net = np.abs(V).sum(1), V.sum(1)
    ok = (units <= F["U"]) & (net >= F["lo"]) & (net <= F["hi"])
    return risk + sep, ok


def build_block(ev, L, S, t, mu_step, eps, G=None):
    """QUBO, warm-start angles and bookkeeping for day t (all assets unless G is given)."""
    n = L.shape[1]
    G = np.arange(n) if G is None else np.asarray(G)
    g = len(G)
    V1 = np.concatenate([np.eye(g, dtype=np.int64), -np.eye(g, dtype=np.int64)])
    c1, _ = ev.costs(L, S, [t], G, V1)
    c0, _ = ev.costs(L, S, [t], G, np.zeros((1, g), np.int64))
    unit = float(np.mean(np.abs(c1 - c0[0]))) or 1.0
    bqm, W, C, margins = block_qubo(ev, L, S, [t], G, mu_step * unit)
    cur = (L[t, G] - S[t, G]).astype(np.int64)
    b0 = encode(cur, W, C, bqm)
    assert (decode(b0[None], W, C)[0] == cur).all()
    nq = 3 * g
    h, J = to_arrays(bqm, nq)
    scale = float(np.median(np.abs(flip_deltas(h, J, b0)))) or 1.0
    bn = bqm.copy()
    bn.scale(1 / scale)
    c = np.where(b0 == 1, 1 - eps, eps)
    thetas = 2 * np.arcsin(np.sqrt(c))
    return dict(t=t, G=G, bqm=bn, W=W, C=C, b0=b0, cur=cur, thetas=thetas, nq=nq, scale=scale,
                couplings=len(bqm.quadratic), mu=mu_step * unit, eps=eps)


def classical_baseline(blk, shots, rng):
    """Samples of the warm-start product state itself (the γ = 0 circuit)."""
    p1 = np.sin(blk["thetas"] / 2) ** 2
    return (rng.random((shots, blk["nq"])) < p1).astype(np.int8)


def score_samples(blk, F, bits, probs=None):
    """Decode bit samples, score them exactly against the current day."""
    V = decode(bits, blk["W"], blk["C"])
    if len(blk["G"]) < F["A"].shape[0]:  # sub-block: the other assets keep their holding
        full = np.repeat(blk["full_cur"][None], len(V), 0)
        full[:, blk["G"]] = V
        V = full
    cost, ok = day_costs(F, V)
    c_cur = float(day_costs(F, blk["full_cur"][None])[0][0])
    w = np.ones(len(V)) if probs is None else np.asarray(probs, float)
    w = w / w.sum()
    impr = ok & (cost < c_cur - 1e-6)
    kb = int(np.argmin(np.where(ok, cost, np.inf))) if ok.any() else None
    return dict(p_feasible=float(w[ok].sum()), p_improve=float(w[impr].sum()),
                distinct_improving=int(len({tuple(v) for v in V[impr]})),
                mean_cost_feasible=float((w[ok] * cost[ok]).sum() / w[ok].sum()) - c_cur if ok.any() else None,
                best_delta=None if kb is None else float(cost[kb] - c_cur),
                best_v=None if kb is None else V[kb], current_cost=c_cur)


def make_backend(cluster, shots, chi, tag):
    cfg = None if not chi else MaestroConfig(simulation_type="MatrixProductState", max_bond_dimension=chi)
    if cluster == "local":
        return MaestroSimulator(shots=shots, force_sampling=True,
                                **({} if cfg is None else dict(maestro_config=cfg)))
    return QoroService(job_config=JobConfig(shots=shots, simulator_cluster=cluster, force_sampling=True, tag=tag),
                       **({} if cfg is None else dict(maestro_config=cfg)))


def run_programs(backend, blocks, angles_list, layers):
    """One merged submission: every (block, angles) program sampled once."""
    ens = Ensemble(backend, [WarmProblem(b["bqm"], b["thetas"]) for b in blocks], angles_list, layers)
    ens.create_programs()
    t0 = time.time()
    ens.run_one_round(blocking=True, batch_config=BatchConfig(max_concurrent_programs=-1))
    out = []
    for b, prog in zip(blocks, ens.programs.values()):
        tops = prog.get_top_solutions(n=0)
        bits = np.array([[int(ch) for ch in s.bitstring] for s in tops], dtype=np.int8)
        out.append((bits, np.array([s.prob for s in tops])))
    return out, time.time() - t0


# ── commands ─────────────────────────────────────────────────────────────────


def _setup(a):
    model, bkv = _load(a.instance, a.budget, a.lam)
    ev = BlockEvaluator(model)
    _, pos = qt.read_portfolio_solution(a.sol)
    L, S = (x.astype(np.int64) for x in model.to_arrays(pos))
    return model, bkv, ev, L, S


def _angles(gs, bs, layers):
    """(γ, β) grid -> list of Divi param vectors [γ0, β0, γ1, β1, ...] (same γ, β per layer)."""
    return [np.array([g, b] * layers, float) for g in gs for b in bs]


def _parse_angle_sets(spec):
    """'0.25,-0.4; 0.25,-0.4,0.1,-0.2; tqa:0.6:2' -> list of [γ0, β0, γ1, β1, ...] vectors.

    ``tqa:dt:p`` is the TQA linear ramp (``qaoa_signed_blocks.tqa_params``): γ ramps up,
    β (negative, as the warm-start mixer needs) ramps down, total time p*dt.
    """
    from qaoa_signed_blocks import tqa_params  # noqa: PLC0415

    out = []
    for item in spec.split(";"):
        item = item.strip()
        if item.startswith("tqa:"):
            _, dt, p = item.split(":")
            out.append(tqa_params(int(p), float(dt)))
        elif item:
            v = np.array([float(x) for x in item.split(",")], float)
            assert len(v) % 2 == 0, item
            out.append(v)
    return out


def cmd_sweep(a):
    """Whole days (or sub-blocks): QAOA at each (γ, β) vs the γ=0 product-state baseline."""
    model, bkv, ev, L, S = _setup(a)
    T, n = model.n_periods, len(model.symbols)
    days = list(range(T)) if a.days == "all" else [int(d) for d in a.days.split(",")]
    rng = np.random.default_rng(a.seed)
    blocks, Fs = [], {}
    for t in days:
        F = day_form(ev, L, S, t)
        Fs[t] = F
        G = None
        if a.group and a.group < n:
            G = np.sort(rng.choice(n, a.group, replace=False))
        blk = build_block(ev, L, S, t, a.mu_step, a.eps, G)
        blk["full_cur"] = (L[t] - S[t]).astype(np.int64)
        blocks.append(blk)
    if a.angle_sets:
        angles = _parse_angle_sets(a.angle_sets)
        if len({len(x) for x in angles}) > 1 and not a.swapnet:
            raise SystemExit("mixed depths need --swapnet")
    else:
        angles = _angles([float(x) for x in a.gammas.split(",")], [float(x) for x in a.betas.split(",")], a.layers)
    progs, prog_blocks = [], []
    for blk in blocks:
        for ang in angles:
            progs.append(ang)
            prog_blocks.append(blk)
    nq = blocks[0]["nq"]
    print(f"{a.instance} λ{a.lam} start {model.score_arrays(L, S).objective} (bkv {bkv}); {len(days)} days, "
          f"{nq} qubits, {blocks[0]['couplings']} couplings; {len(angles)} angle sets -> {len(progs)} programs; "
          f"eps {a.eps}, mu_step {a.mu_step}, p={a.layers}, chi {a.mps_chi}, cluster {a.cluster}", flush=True)
    backend = make_backend(a.cluster, a.shots, a.mps_chi, f"warm_day_{nq}q")
    service = QoroService() if a.cluster != "local" else None
    bal0 = float(service.get_credit_balance()["balance"]) if service else None
    if a.swapnet:
        res, wall = sample_swapnet(backend, prog_blocks, progs)
    else:
        res, wall = run_programs(backend, prog_blocks, progs, a.layers)
    bal1 = float(service.get_credit_balance()["balance"]) if service else None
    rows = []
    for blk, ang, (bits, probs) in zip(prog_blocks, progs, res):
        q = score_samples(blk, Fs[blk["t"]], bits, probs)
        base = score_samples(blk, Fs[blk["t"]], classical_baseline(blk, a.shots, rng))
        row = dict(day=blk["t"], qubits=blk["nq"], gamma=float(ang[0]), beta=float(ang[1]),
                   angles=[float(x) for x in ang], p=len(ang) // 2,
                   qaoa={k: v for k, v in q.items() if k != "best_v"},
                   base={k: v for k, v in base.items() if k != "best_v"},
                   qaoa_best_v=None if q["best_v"] is None else q["best_v"].tolist())
        rows.append(row)
        print(f"day {blk['t']:2d} p={len(ang) // 2} {np.round(ang, 3).tolist()}: "
              f"feasible {q['p_feasible']:.3f}/{base['p_feasible']:.3f}  improve {q['p_improve']:.4f}/"
              f"{base['p_improve']:.4f}  mean Δ {q['mean_cost_feasible'] if q['mean_cost_feasible'] is None else round(q['mean_cost_feasible'],1)}"
              f"/{base['mean_cost_feasible'] if base['mean_cost_feasible'] is None else round(base['mean_cost_feasible'],1)}  "
              f"best Δ {q['best_delta']}/{base['best_delta']}", flush=True)
    rec = dict({k: v for k, v in vars(a).items() if k != "fn"}, qubits=nq, couplings=blocks[0]["couplings"], programs=len(progs), wall_s=round(wall, 1),
               credits_used=None if bal0 is None else round(bal0 - bal1, 3), rows=rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "sweeps.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"wall {wall:.1f}s, credits {rec['credits_used']}", flush=True)


def _accept(model, ev, L, S, t, V):
    """Best feasible holding among samples V (K, n) for day t under the *current* L, S
    (neighbouring days may have changed since sampling); apply it if it improves."""
    F = day_form(ev, L, S, t)
    cur = (L[t] - S[t]).astype(np.int64)
    c_cur = float(day_costs(F, cur[None])[0][0])
    cost, ok = day_costs(F, V)
    if not ok.any():
        return 0.0
    k = int(np.argmin(np.where(ok, cost, np.inf)))
    if cost[k] >= c_cur - 1e-6:
        return 0.0
    L[t], S[t] = np.maximum(V[k], 0), np.maximum(-V[k], 0)
    return float(cost[k] - c_cur)


def cmd_loop(a):
    """Hybrid loop: each round warm-starts every day from the current solution, samples
    all days in one merged submission and keeps the best improving sample per day.
    A twin run does the same with the γ = 0 circuit (the warm-start product state), with
    the same shots, so the difference is the QAOA cost layer's."""
    model, bkv = _load(a.instance, a.budget, a.lam)
    ev = BlockEvaluator(model)
    T, n = model.n_periods, len(model.symbols)
    if a.sol == "empty":
        L0 = np.zeros((T, n), np.int64)
        S0 = L0.copy()
    else:
        _, pos = qt.read_portfolio_solution(a.sol)
        L0, S0 = (x.astype(np.int64) for x in model.to_arrays(pos))
    tracks = {"qaoa": [L0.copy(), S0.copy()], "base": [L0.copy(), S0.copy()]}
    rng = np.random.default_rng(a.seed)
    days = list(range(T)) if a.days == "all" else [int(d) for d in a.days.split(",")]
    if a.angle_sets:
        (ang,) = _parse_angle_sets(a.angle_sets)
        a.layers = len(ang) // 2
    else:
        ang = np.array([a.gamma, a.beta] * a.layers, float)
    backend = make_backend(a.cluster, a.shots, a.mps_chi, f"warm_loop_{3 * n}q")
    service = QoroService() if a.cluster != "local" else None
    bal0 = float(service.get_credit_balance()["balance"]) if service else None
    start = model.score_arrays(L0, S0).objective
    hist = [dict(round=0, qaoa=start, base=start)]
    print(f"{a.instance} λ{a.lam} start {start} (bkv {bkv}); {len(days)} days x {3 * n} qubits per round, "
          f"eps {a.eps}, angles {np.round(ang, 3).tolist()}, p={a.layers}, shots {a.shots}, chi {a.mps_chi}, {a.cluster}",
          flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = a.angle_sets.replace(":", "").replace(";", "") if a.angle_sets else f"g{a.gamma}_b{a.beta}"
    tag = f"{a.instance}_l{a.lam}_{'empty' if a.sol == 'empty' else 'warm'}_{sched}_e{a.eps}_{a.cluster}"
    for r in range(1, a.rounds + 1):
        L, S = tracks["qaoa"]
        blocks = [build_block(ev, L, S, t, a.mu_step, a.eps) for t in days]
        if a.swapnet:
            res, wall = sample_swapnet(backend, blocks, [ang] * len(blocks))
        else:
            res, wall = run_programs(backend, blocks, [ang] * len(blocks), a.layers)
        gains = {"qaoa": 0.0, "base": 0.0}
        for blk, (bits, probs) in zip(blocks, res):
            # QAOA: draw `shots` samples from the measured distribution (counts already are shots)
            Vq = decode(bits, blk["W"], blk["C"])
            gains["qaoa"] += _accept(model, ev, L, S, blk["t"], Vq)
        Lb, Sb = tracks["base"]
        for t in days:
            bb = build_block(ev, Lb, Sb, t, a.mu_step, a.eps)
            Vb = decode(classical_baseline(bb, a.shots, rng), bb["W"], bb["C"])
            gains["base"] += _accept(model, ev, Lb, Sb, t, Vb)
        sq, sb = model.score_arrays(L, S), model.score_arrays(Lb, Sb)
        assert sq.feasible and sb.feasible
        hist.append(dict(round=r, qaoa=int(sq.objective), base=int(sb.objective), wall_s=round(wall, 1),
                         units_qaoa=int((L + S).sum()), units_base=int((Lb + Sb).sum())))
        print(f"round {r:2d}: QAOA {sq.objective} (units {(L + S).sum()})  γ=0 baseline {sb.objective} "
              f"(units {(Lb + Sb).sum()})  [{wall:.0f}s]", flush=True)
        model.write_solution(out / f"{tag}_qaoa.sol", model.to_positions(L, S),
                             comment=f"warm_day_qaoa.py loop, round {r}")
    bal1 = float(service.get_credit_balance()["balance"]) if service else None
    rec = dict({k: v for k, v in vars(a).items() if k != "fn"}, qubits=3 * n, history=hist,
               credits_used=None if bal0 is None else round(bal0 - bal1, 3))
    with open(out / "loops.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--instance", default="po_a050_t10_orig")
    lp.add_argument("--budget", type=int, default=20)
    lp.add_argument("--lam", default="1e-3")
    lp.add_argument("--sol", default="empty")
    lp.add_argument("--days", default="all")
    lp.add_argument("--rounds", type=int, default=5)
    lp.add_argument("--mu-step", type=float, default=0.25)
    lp.add_argument("--eps", type=float, default=0.02)
    lp.add_argument("--gamma", type=float, default=0.5)
    lp.add_argument("--beta", type=float, default=-0.4)
    lp.add_argument("--layers", type=int, default=1)
    lp.add_argument("--angle-sets", default="", help="one schedule, e.g. tqa:0.5:3; overrides --gamma/--beta/--layers")
    lp.add_argument("--shots", type=int, default=2000)
    lp.add_argument("--mps-chi", type=int, default=16)
    lp.add_argument("--cluster", default="local")
    lp.add_argument("--swapnet", action="store_true", help="swap-network circuits (fast on MPS)")
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--out", default="results/warm_day_qaoa")
    lp.set_defaults(fn=cmd_loop)
    s = sub.add_parser("sweep")
    s.add_argument("--instance", default="po_a050_t10_orig")
    s.add_argument("--budget", type=int, default=20)
    s.add_argument("--lam", default="1e-3")
    s.add_argument("--sol", required=True, help="current solution; each day is warm-started from it")
    s.add_argument("--days", default="all")
    s.add_argument("--group", type=int, default=0, help="assets per block (0 = whole day)")
    s.add_argument("--mu-step", type=float, default=0.25)
    s.add_argument("--eps", type=float, default=0.05)
    s.add_argument("--gammas", default="0,0.3")
    s.add_argument("--betas", default="0.3")
    s.add_argument("--layers", type=int, default=1)
    s.add_argument("--angle-sets", default="",
                   help="';'-separated angle vectors γ0,β0[,γ1,β1...] or tqa:dt:p; overrides --gammas/--betas/--layers")
    s.add_argument("--shots", type=int, default=2000)
    s.add_argument("--mps-chi", type=int, default=0, help="MPS bond dimension (0 = statevector)")
    s.add_argument("--cluster", default="local")
    s.add_argument("--swapnet", action="store_true", help="swap-network circuits (fast on MPS)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/warm_day_qaoa")
    s.set_defaults(fn=cmd_sweep)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
