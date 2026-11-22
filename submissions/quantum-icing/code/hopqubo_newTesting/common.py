"""Shared helpers for the hopqubo_newTesting pipeline (QOBLIB 08, network design).

Instance + referee (exact routing LP), the H2 hop surrogate, the 2-exchange polish, a
constraint-aware decoder (degree-2 b-matching), backend factory, and run logging.

Every topology is a list of 0-indexed arcs (i, j); valid = every node has 2 out and 2 in.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=".*constant offset.*")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # qoblib_tools caches instance data relative to the repo root

import qoblib_tools as qt  # noqa: E402
from network_design.pipeline import BKV  # noqa: E402

Arc = tuple[int, int]


# ── instance & referee ───────────────────────────────────────────────────
def load(n: int) -> np.ndarray:
    return qt.load_network_demand(n)


def lp(T, arcs) -> float:
    """QOBLIB's official objective (max link load × 1000, integer)."""
    return qt.route_min_congestion(T, list(arcs)).objective


def gap(n: int, score: float) -> float:
    return (score - BKV[n]) / BKV[n]


def gap_str(n: int, score: float) -> str:
    return f"{gap(n, score):+.2%} vs BKV {BKV[n]}"


def is_valid(n: int, arcs) -> bool:
    return qt.is_valid_topology(n, list(arcs))


# ── H2 hop surrogate (fast, numpy) ───────────────────────────────────────
class H2:
    """H2(arcs) = Σ_{i≠k} t_ik · min(hops(i→k), 3) with hops ∈ {1, 2, 3}. Lower is better."""

    def __init__(self, T):
        self.T = np.asarray(T, dtype=float)
        self.n = len(T)
        self.off = ~np.eye(self.n, dtype=bool)

    def __call__(self, arcs) -> float:
        A = np.zeros((self.n, self.n))
        for i, j in arcs:
            A[i, j] = 1
        A2 = A @ A
        h = np.where(A > 0, 1, np.where(A2 > 0, 2, 3))
        return float((self.T * h)[self.off].sum())


# ── constraint-aware decoding: soft scores → valid topology ──────────────
def bmatch_decode(n: int, score: np.ndarray, forbid=()) -> list[Arc] | None:
    """Max-weight 2-out/2-in simple digraph for an n×n score matrix (higher = prefer arc).

    The bipartite b-matching LP (tails × heads, capacity 1 per arc) is totally unimodular,
    so the LP optimum is integral. Self-loops and `forbid` arcs are excluded.
    """
    from scipy.optimize import linprog

    cand = [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in set(forbid)]
    m = len(cand)
    c = -np.array([score[i, j] for i, j in cand], dtype=float)
    A = np.zeros((2 * n, m))
    for v, (i, j) in enumerate(cand):
        A[i, v] = 1
        A[n + j, v] = 1
    r = linprog(c, A_eq=A, b_eq=np.full(2 * n, 2.0), bounds=(0, 1), method="highs")
    if r.status != 0:
        return None
    arcs = [cand[v] for v in np.flatnonzero(r.x > 0.5)]
    return sorted(arcs) if is_valid(n, arcs) else None


def soft_to_topologies(n, arcs_index, soft, k=20, noise=0.15, rng=None, forbid=()):
    """Decode a soft assignment (one value per arc variable, ~P(x=1)) into up to k distinct valid
    topologies: the noiseless b-matching, then b-matchings of Gumbel-perturbed scores."""
    rng = rng or np.random.default_rng(0)
    S = np.full((n, n), -1e3)
    for v, (i, j) in enumerate(arcs_index):
        S[i, j] = soft[v]
    out, seen = [], set()
    for t in range(k * 3):
        pert = S if t == 0 else S + noise * rng.gumbel(size=S.shape)
        top = bmatch_decode(n, pert, forbid)
        if top is not None and tuple(top) not in seen:
            seen.add(tuple(top))
            out.append(top)
        if len(out) >= k:
            break
    return out


# ── 2-exchange polish (LP-scored, H2-ranked) ─────────────────────────────
def polish(T, arcs, lp_budget=60, k=5):
    """Wraps network_moves.polish with the H2 ranker. Returns (arcs, score, lp_calls)."""
    from network_moves import polish as _polish

    return _polish(T, arcs, rank=H2(T), lp_budget=lp_budget, k=k)


def rank_and_score(T, tops, h2=None, top_k=10):
    """Rank distinct valid topologies by H2, LP-score the best top_k. Returns sorted [(lp, arcs)]."""
    h2 = h2 or H2(T)
    uniq = {tuple(sorted(t)) for t in tops if t is not None}
    ranked = sorted(uniq, key=h2)[:top_k]
    return sorted(((lp(T, a), list(a)) for a in ranked), key=lambda x: x[0])


# ── incumbent start ──────────────────────────────────────────────────────
def start_topology(n, T, how="random", seed=0):
    rng = np.random.default_rng(seed)
    if how == "random":
        return sorted(qt.random_topology(n, rng))
    if how == "h2greedy":  # best H2 of 200 random topologies: cheap, classical, not QOBLIB's
        h2 = H2(T)
        return min((sorted(qt.random_topology(n, rng)) for _ in range(200)), key=h2)
    if how.endswith(".json") or how.endswith(".sol"):
        if how.endswith(".sol"):
            _, arcs, _ = qt.read_network_solution(how)
            return sorted(map(tuple, arcs))
        return sorted(map(tuple, json.load(open(how))["arcs"]))
    raise ValueError(how)


# ── backend ──────────────────────────────────────────────────────────────
def make_backend(cloud: str | None = None, shots: int = 10_000, force_sampling=False):
    """cloud=None → local Maestro; 'qoro_cpu' / 'qoro_gpu' → Qoro cloud (credits)."""
    from divi.backends import JobConfig, MaestroSimulator, QoroService

    if cloud:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster=cloud,
                                                force_sampling=force_sampling))
    return MaestroSimulator(shots=shots, force_sampling=force_sampling)


def service_or_none():
    try:
        from divi.backends import QoroService

        s = QoroService()
        s.get_credit_balance()
        return s
    except Exception as err:  # no key / no network
        print(f"[no Qoro service: {type(err).__name__}] cloud steps skipped")
        return None


# ── logging ──────────────────────────────────────────────────────────────
class RunLog:
    """Line log to stdout + runs/<name>.txt, JSON summary to runs/<name>.json."""

    def __init__(self, name: str):
        self.name = name
        self.txt = open(RUNS / f"{name}.txt", "a")
        self.t0 = time.time()
        self.data: dict = {"name": name, "events": []}

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.txt.write(line + "\n")
        self.txt.flush()

    def event(self, **kw):
        kw["t"] = round(time.time() - self.t0, 1)
        self.data["events"].append(kw)
        self.save()

    def save(self):
        with open(RUNS / f"{self.name}.json", "w") as f:
            json.dump(self.data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
