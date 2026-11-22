"""Quantum large-neighbourhood search (QLNS) for QOBLIB 06 portfolio optimisation.

Every step builds an exact move-selection QUBO (qlns_core.build_qubo) on a window of the
incumbent and hands it to a sub-solver:

  * ``round`` windows: the continuous relaxation of one day (dayrelax.solve_day, neighbours
    fixed) gives fractional positions v*.  Each fractional slot gets one qubit choosing
    between trunc(v*) and the next unit away from zero.  The day budget is saturated in the
    relaxation, so the number of units to add is fixed: the quantum solver works in the
    Hamming-weight-k subspace (Dicke state + XY mixer) with no penalty and no slack qubits.
  * ``move`` windows: unit moves (+1 / -1) on a few assets of one day or a short block of
    days around the incumbent, with budget/capital as Divi LinearConstraints.

Sub-solvers: ``qaoa`` (Divi QAOA, Maestro simulator or Qoro cloud) and ``oracle`` (exact
enumeration, the classical reference used to measure how often QAOA finds the window optimum).
The classical side only builds windows, scores candidates exactly and accepts improvements.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayrelax as dr  # noqa: E402
import qlns_core as qc  # noqa: E402
import windows as W  # noqa: E402

ps = qc.ps
qt = qc.qt
OUT = Path(os.environ.get("PQ_SOL_DIR", HERE / "solutions"))


# ════════════════════════════════════════════════════════════════════════
# Sub-solvers
# ════════════════════════════════════════════════════════════════════════


class OracleSolver:
    name = "oracle"

    def __init__(self):
        self.calls = 0

    def solve(self, s, moves, any_sign, hw=None):
        self.calls += 1
        de, x = qc.oracle(s, moves, any_sign=any_sign)
        return [] if de is None else [x]


class RandomSolver:
    """Ablation: uniform samples from the same space the QAOA circuit explores (same shots)."""
    name = "random"

    def __init__(self, shots=4000, seed=0):
        self.shots, self.rng, self.calls = shots, np.random.default_rng(seed), 0

    def solve(self, s, moves, any_sign, hw=None, flip=None):
        self.calls += 1
        M = len(moves)
        c = np.zeros(M, dtype=np.int64) if flip is None else np.asarray(flip, dtype=np.int64)
        if hw is None:
            Y = self.rng.integers(0, 2, (self.shots, M))
        else:
            Y = np.zeros((self.shots, M), dtype=np.int64)
            idx = np.argsort(self.rng.random((self.shots, M)), axis=1)[:, :hw]
            np.put_along_axis(Y, idx, 1, axis=1)
        X = np.where(c[None, :] == 1, 1 - Y, Y)
        return list(np.unique(X, axis=0))


def _distinct_feasible(s, moves, xs, a, b):
    out = {}
    for x in xs:
        key = x.tobytes()
        if key in out:
            continue
        if qc.feasible_x(s, moves, x):
            out[key] = (qc.energy(a, b, x), x)
    return sorted(out.values(), key=lambda e: e[0])


# ════════════════════════════════════════════════════════════════════════
# Steps
# ════════════════════════════════════════════════════════════════════════


def warm_seed(prior, hw):
    """Basis state with the hw moves the relaxation prefers (largest fractional parts)."""
    x = np.zeros(len(prior), dtype=np.int64)
    x[np.argsort(-prior)[:hw]] = 1
    return x


def round_seed(tmp, moves, hw, prior):
    """Warm seed and conserved groups for a fixed-weight rounding window.

    With hw moves selected, the net change is (#plus) - (#minus) = 2 k_plus - hw, so the capital
    rule fixes the number of '+' moves to a range.  If that range is narrower than what the window
    allows, the seed takes k_plus from it (closest to the relaxation's choice) and the mixer conserves
    the '+' and '-' groups separately: every sample then meets budget and capital rules exactly.
    Returns (seed, groups) or (None, None) when no count split is feasible.
    """
    t = moves[0][0]
    p = tmp.p
    lo, hi = p.C - p.maxcash - tmp.net[t], p.C - tmp.net[t]
    plus = [m for m, (_, _, d) in enumerate(moves) if d > 0]
    minus = [m for m, (_, _, d) in enumerate(moves) if d < 0]
    kmin, kmax = max(0, hw - len(minus)), min(len(plus), hw)
    flo, fhi = max(kmin, -(-(hw + lo) // 2)), min(kmax, (hw + hi) // 2)
    if flo > fhi:
        return None, None
    pref = warm_seed(prior, hw)
    kp = int(min(max(sum(pref[m] for m in plus), flo), fhi))
    x = np.zeros(len(moves), dtype=np.int64)
    x[[plus[j] for j in np.argsort(-prior[plus])[:kp]]] = 1
    if hw - kp:
        x[[minus[j] for j in np.argsort(-prior[minus])[:hw - kp]]] = 1
    groups = None if (flo == kmin and fhi == kmax) else [plus, minus]
    return x, groups


def round_window(s, t, rng, kmax, noise=0.0, drop=0, band=0.0):
    """Relaxation-guided rounding window on day t. Returns (temp_state, moves, hw) or None."""
    p = s.p
    tilt = None
    if noise > 0:
        tilt = noise * rng.normal(size=p.n) * np.maximum(np.abs(p.lin[t]), 1.0)
    mask = None
    if drop > 0:
        mask = np.ones(p.n, dtype=bool)
        cand = np.flatnonzero(s.v[t] != 0)
        if len(cand) == 0:
            cand = np.arange(p.n)
        mask[rng.choice(cand, size=min(drop, len(cand)), replace=False)] = False
    vs, _ = dr.solve_day(p, s.v, t, tilt=tilt, ub_mask=mask)
    base = np.trunc(vs).astype(np.int64)
    frac = np.abs(vs - np.trunc(vs))
    if band > 0:  # near-integral slots are rounded directly; only uncertain ones get a qubit
        up = (frac >= 1 - band)
        base2 = base + np.where(up, np.sign(vs), 0).astype(np.int64)
        if np.abs(base2).sum() <= p.budget:
            base = base2
            frac = np.where(up | (frac <= band), 0.0, frac)
        cand = [int(i) for i in np.argsort(np.abs(frac - 0.5)) if frac[i] > 1e-6]
    else:
        cand = [int(i) for i in np.argsort(-np.abs(vs)) if frac[i] > 1e-6]
    win = cand[:kmax]
    v2 = s.v.copy()
    v2[t] = base
    tmp = qc.State(p, v2)
    moves = [(t, i, int(np.sign(vs[i]))) for i in win]
    room = int(p.budget - tmp.absv[t])
    if not moves:
        return None
    round_window.prior = np.array([frac[i] for i in win])  # relaxation's preference for each move
    if room < 0:
        return None
    # saturated budget: exactly `room` units can be added (fixed Hamming weight, XY mixer);
    # otherwise the budget never binds inside the window (free weight, X mixer)
    hw = room if room < len(moves) else None
    return tmp, moves, hw


def run_window(s, tmp, moves, solver, any_sign, hw=None, flip=None, seed=None, prior=None):
    """Solve a window, return (new_v, obj, n_candidates, window_energy) or None."""
    a, b, _ = qc.build_qubo(tmp, moves)
    run_window.prior = prior
    if solver.name == "oracle":
        xs = solver.solve(tmp, moves, any_sign)
    elif solver.name == "qaoa":
        groups = None
        if seed is not None and hw is not None and flip is None:
            seed, groups = round_seed(tmp, moves, hw, run_window.prior)
        xs = solver.solve(tmp, moves, any_sign, hw=hw, flip=flip, seed=seed, groups=groups)
    else:
        xs = solver.solve(tmp, moves, any_sign, hw=hw, flip=flip)
    cands = _distinct_feasible(tmp, moves, xs, a, b)
    if not cands:
        return None
    e, x = cands[0]
    if not any_sign and e >= 0:
        return None
    v2 = tmp.v.copy()
    for (t, i, d), on in zip(moves, x):
        if on:
            v2[t, i] += d
    return v2, s.p.objective(v2), len(cands), e


# ════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════


def save(p, v, method, settings, log_rows):
    model, sc = qc.check_with_referee(p, v)
    if not sc.feasible:
        raise RuntimeError(f"referee says infeasible: {sc.violations}")
    d = OUT / p.stem
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = d / f"{p.stem}__{method}__{stamp}.sol"
    comment = f"Quantum LNS ({method}); {settings}"
    model.write_solution(path, p.positions(v), comment=comment)
    return path, sc.objective


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--solver", choices=["oracle", "qaoa", "random"], default="oracle")
    ap.add_argument("--start", default="empty", help="empty | best (best earlier classical .sol) | <path>")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=10**9)
    ap.add_argument("--kround", type=int, default=18, help="max qubits of a rounding window")
    ap.add_argument("--kmove", type=int, default=8, help="assets of a move window (2 qubits each)")
    ap.add_argument("--p-round", type=float, default=0.5)
    ap.add_argument("--p-block", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    # QAOA settings
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--retrain", type=int, default=300, help="re-train the QAOA angles every N windows")
    ap.add_argument("--train-iters", type=int, default=120)
    ap.add_argument("--angles", default="", help="JSON file of transferred QAOA angles (read and updated)")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--warm", type=float, default=0.0,
                    help="probability that a rounding window starts QAOA from the relaxation's rounding "
                         "(CustomPerQubitState + XY rings conserving budget and capital) instead of Dicke")
    ap.add_argument("--check-oracle", type=float, default=0.0, help="fraction of windows also solved exactly (hit-rate stats)")
    a = ap.parse_args(argv)

    p = ps.Problem(a.inst, a.lam)
    rng = np.random.default_rng(a.seed)
    if a.start == "empty":
        v0 = np.zeros((p.T, p.n), dtype=np.int64)
    elif a.start == "best":
        v0 = qc.load_v(p, qc.best_known_file(p.stem))
    elif a.start == "qbest":  # best of the earlier classical files and this pipeline's own results
        cands = [qc.best_known_file(p.stem)] + list((OUT / p.stem).glob("*.sol"))
        cands = [c for c in cands if c is not None and c.exists()]
        v0 = qc.load_v(p, min(cands, key=lambda c: ps.read_objective(c)))
    else:
        v0 = qc.load_v(p, Path(a.start))
    s = qc.State(p, v0)
    if a.solver == "oracle":
        solver = OracleSolver()
    elif a.solver == "random":
        solver = RandomSolver(shots=a.shots, seed=a.seed)
    else:
        import qaoa_solver
        solver = qaoa_solver.QAOASolver(layers=a.layers, shots=a.shots, cloud=a.cloud, seed=a.seed,
                                        retrain_every=a.retrain, train_iters=a.train_iters,
                                        angles_file=a.angles or None)
    oracle = OracleSolver()
    corr = W.corr_matrix(p)
    start_obj = s.obj
    bkv = ps.bkv_table().get(p.bkv_key)
    print(f"{p.stem}: start {start_obj}  BKV {bkv}  solver {solver.name}", flush=True)
    t_end = time.time() + 60 * a.minutes
    log_rows, hits, checks = [], 0, 0
    stats = {}
    it = 0
    while time.time() < t_end and it < a.iters:
        it += 1
        r = rng.random()
        if r < a.p_round:
            t = int(rng.integers(0, p.T - 1))
            noise = a.noise * rng.random() if rng.random() < 0.7 else 0.0
            drop = int(rng.integers(0, 3)) if rng.random() < 0.3 else 0
            w = round_window(s, t, rng, a.kround, noise=noise, drop=drop)
            if w is None:
                continue
            tmp, moves, hw = w
            kind, any_sign = "round", True
            prior = round_window.prior
        else:
            if r < a.p_round + a.p_block:
                moves, kind = W.block_window(s, rng, K=max(3, a.kmove // 2), D=int(rng.integers(2, 4)), corr=corr)
            else:
                moves, kind = W.day_window(s, rng, a.kmove, corr=corr)
            if not moves:
                continue
            tmp, hw, any_sign = s, None, False
        flip, seed = None, None
        if kind.startswith("day") and solver.name in ("qaoa", "random"):
            import qaoa_solver
            hw, flip = qaoa_solver.fixed_weight(s, moves)
        if kind == "round" and hw is not None and solver.name == "qaoa" and rng.random() < a.warm:
            seed = warm_seed(prior, hw)
            kind = "round-warm"
        res = run_window(s, tmp, moves, solver, any_sign, hw=hw, flip=flip, seed=seed,
                         prior=prior if kind.startswith("round") else None)
        if a.check_oracle > 0 and solver.name != "oracle" and rng.random() < a.check_oracle:
            ores = run_window(s, tmp, moves, oracle, any_sign)
            checks += 1
            if ores is not None and res is not None and res[3] <= ores[3]:
                hits += 1
            elif ores is None and res is None:
                hits += 1
        if res is None:
            continue
        v2, obj, ncand, e = res
        if obj < s.obj:
            gain = obj - s.obj
            s = qc.State(p, v2)
            stats[kind] = stats.get(kind, 0) + 1
            log_rows.append(dict(it=it, kind=kind, qubits=len(moves), gain=gain, obj=obj, t=round(time.time(), 1)))
            print(f"it {it:5d} {kind:10s} q={len(moves):2d} cand={ncand:4d} {gain:+7d} -> {obj}", flush=True)
    method = f"qlns-{solver.name}"
    settings = (f"start={a.start} seed={a.seed} kround={a.kround} kmove={a.kmove} p_round={a.p_round} "
                f"p_block={a.p_block} noise={a.noise} iters={it}")
    if solver.name == "qaoa":
        settings += (f" layers={a.layers} shots={a.shots} retrain={a.retrain} cloud={a.cloud} qaoa_calls={solver.calls} "
                     f"circuits={solver.circuits} train_circuits={solver.train_circuits} max_qubits={solver.max_qubits}")
        print(f"QAOA: {solver.calls} windows, {solver.circuits} circuits ({solver.train_circuits} for angle training), "
              f"largest program {solver.max_qubits} qubits")
    print(f"final {s.obj} (start {start_obj}, BKV {bkv}) its={it} accepted={stats}", flush=True)
    if checks:
        print(f"window-optimum hit rate: {hits}/{checks} = {hits / checks:.1%}")
    if s.obj < start_obj:
        path, obj = save(p, s.v, method + (f"-{a.tag}" if a.tag else ""), settings, log_rows)
        print("saved", path, obj)
        with open(path.with_suffix(".log.json"), "w") as fh:
            json.dump(dict(settings=settings, start=start_obj, final=s.obj, bkv=bkv, accepted=log_rows,
                           hit_checks=checks, hits=hits, max_qubits=max([r["qubits"] for r in log_rows] or [0]) + 0), fh)
    return s


if __name__ == "__main__":
    main()
