"""Shared helpers for quantum_focus/ experiments (imports the parent folder's modules)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for p in (str(PARENT), str(PARENT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dimod  # noqa: E402
import numpy as np  # noqa: E402

import lns  # noqa: E402  (FastModel, stage_cost, chain_dp)
import pf_core as pc  # noqa: E402

qt = pc.qt


def load(instance: str, lam: str):
    model = pc.make_model(instance, lam)
    fm = lns.FastModel(model)
    bkv = pc.load_bkv().get((instance, model.budget, lam), (None, "none"))
    return model, fm, bkv


def gap(obj, bkv):
    return f"{(obj - bkv) / abs(bkv):+.2%}" if bkv else "n/a"


def best_known_sol(instance: str, budget: int, lam: str, method_filter=None):
    """Our best recorded .sol for an instance (optionally filtered by method substring)."""
    d = pc.SOL_DIR / f"{instance}_b{budget:03d}_l{lam}"
    best = None
    for f in d.glob("*.sol"):
        text = f.read_text()
        if method_filter and method_filter not in text:
            continue
        h, pos = qt.read_portfolio_solution(f)
        obj = int(h.get("objective", 0))
        if best is None or obj < best[0]:
            best = (obj, f, pos)
    return best


# ── per-period ±1 QUBO (constructor) ─────────────────────────────────────


def period_qubo(fm, t: int, K: int, extra: float = 1.0, penalty: float | None = None,
                cash_bits: int = 4):
    """QUBO for one period: x = [l_0..l_{n-1}, s_0..s_{n-1}, y_0..y_{cash_bits-1}], v = l - s in {-1,0,1}.

    Exact stage cost of period t restricted to |v| <= 1, plus extra * tc[t] per held unit
    (emulates entry/exit costs), plus penalties: exactly K units held, and the capital
    rule net + cash = C with cash in 0..2^cash_bits-1, plus l_i s_i forbidden.
    """
    n = fm.n
    R = fm.R[t].astype(float)
    lin_v = fm.lin[t].astype(float)
    absw = fm.absw[t].astype(float) + extra * fm.model.tc[t]
    negw = fm.negw[t].astype(float)
    N = 2 * n + cash_bits
    Q = np.zeros((N, N))  # upper-triangular incl. diag (linear)
    sign = np.concatenate([np.ones(n), -np.ones(n)])
    asset = np.concatenate([np.arange(n), np.arange(n)])
    M = np.outer(sign, sign) * R[np.ix_(asset, asset)]  # risk over x-part (2n)
    diag = np.diag(M).copy()
    Qx = np.triu(M + M.T, k=1)
    Q[: 2 * n, : 2 * n] += Qx
    lin = np.zeros(N)
    lin[: 2 * n] += diag + sign * np.concatenate([lin_v, lin_v]) + np.concatenate([absw, absw + negw])
    if penalty is None:
        penalty = float(np.median(np.abs(np.diag(R)))) + 2 * float(np.abs(lin_v).max()) + 1.0
    # forbid l_i & s_i
    for i in range(n):
        Q[i, n + i] += penalty
    # budget: (sum x - K)^2
    ones = np.zeros(N)
    ones[: 2 * n] = 1
    # capital: (sum sign x + sum 2^k y - C)^2
    cap = np.zeros(N)
    cap[: 2 * n] = sign
    cap[2 * n:] = [2**k for k in range(cash_bits)]
    const = float(fm.const[t]) + penalty * (K**2 + fm.C**2)
    for a_vec, target in ((ones, K), (cap, fm.C)):
        lin += penalty * (a_vec**2 - 2 * target * a_vec)
        Q += penalty * 2 * np.triu(np.outer(a_vec, a_vec), k=1)
    quad = {(int(a), int(b)): float(Q[a, b]) for a, b in zip(*np.nonzero(Q))}
    bqm = dimod.BinaryQuadraticModel({i: float(lin[i]) for i in range(N)}, quad, const, dimod.BINARY)
    return bqm


def decode_period(bits, n: int) -> np.ndarray:
    bits = np.asarray([int(b) for b in bits])
    return bits[:n] - bits[n: 2 * n]


def repair_period(fm, v: np.ndarray) -> np.ndarray:
    """Clamp a period vector into the feasible set by dropping the worst units."""
    v = v.copy()
    t = 0
    while np.abs(v).sum() > fm.B or v.sum() > fm.C or v.sum() < fm.netmin:
        if v.sum() > fm.C:
            cand = np.flatnonzero(v > 0)
        elif v.sum() < fm.netmin:
            cand = np.flatnonzero(v < 0)
        else:
            cand = np.flatnonzero(v != 0)
        i = int(cand[t % len(cand)])
        v[i] -= np.sign(v[i])
        t += 1
    return v
