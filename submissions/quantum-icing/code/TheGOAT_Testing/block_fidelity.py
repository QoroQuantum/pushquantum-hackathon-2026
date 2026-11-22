"""Does a block surrogate rank the block's rewirings like the exact LP?

A block is r arcs of the incumbent whose tails stay and whose heads are permuted, so every node
keeps 2 in / 2 out whatever the permutation. This script enumerates every valid rewiring of a
block, scores each one with the exact LP (the referee's model), and asks
``diagnostics.rank_agreement`` how well each surrogate reproduces that ranking:

* ``team``      the sub-QUBO of ``network_qlns.build_subqubo`` (single-arc LP probes + two-hop
                reward), the model the existing quantum LNS samples;
* ``bound``     the LP weak-duality bound  sum_ij d_ij dist_w(i,j) / sum_a w_a  with the
                incumbent's duals carried by each arc (a moved arc keeps its weight);
* ``hops``      the same bound with unit weights;
* ``fit_lin``   a linear model in the r*r assignment bits, fitted on half the block. Its
                optimum is an assignment problem that the Hungarian method solves exactly, so
                if it ranks well no quantum sampler is needed;
* ``fit_quad``  linear + pairwise terms fitted on the same half: a learned QUBO, the kind of
                model a QAOA/PCE sampler would be worth running on.

Fitted models are scored on the held-out half only. One JSON per block under
``runs/goat/fidelity/``: a rerun skips the blocks already done.

    venv/bin/python TheGOAT_Testing/block_fidelity.py --n 24 --r 6 --blocks 4 --workers 2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from itertools import permutations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import csr_array  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
warnings.filterwarnings("ignore")

import qoblib_tools as qt  # noqa: E402
from diagnostics import rank_agreement  # noqa: E402
from network_design import pipeline as npl  # noqa: E402
from network_design.sa_search import lp_eval  # noqa: E402
from network_lns import load_start, pick_free  # noqa: E402
from network_qlns import build_subqubo  # noqa: E402

OUT = ROOT / "runs" / "goat" / "fidelity"

_D: np.ndarray | None = None


def _init(demand: np.ndarray) -> None:
    global _D
    _D = demand


def _lp(arcs) -> float:
    return lp_eval(_D, arcs)[0]


# ── the block ──────────────────────────────────────────────────────────────────────────


def rewirings(arcs, free, n):
    """Every distinct valid topology reachable by permuting the heads of arcs[free].

    Returns (perms, topologies). Heads may repeat, so several permutations can give the same
    topology: only the first is kept.
    """
    tails = [arcs[f][0] for f in free]
    slots = [arcs[f][1] for f in free]
    seen, perms, tops = set(), [], []
    for p in permutations(range(len(free))):
        new = list(arcs)
        for f, s in enumerate(p):
            new[free[f]] = (tails[f], slots[s])
        key = tuple(sorted(new))
        if key in seen or len(set(new)) != len(new) or not qt.is_valid_topology(n, new):
            continue
        seen.add(key)
        perms.append(p)
        tops.append(new)
    return perms, tops


def onehot(perms, r):
    X = np.zeros((len(perms), r * r))
    for k, p in enumerate(perms):
        X[k, [f * r + s for f, s in enumerate(p)]] = 1.0
    return X


def pair_features(X, r):
    """Products of assignment bits in different rows and different columns (the only pairs a
    permutation can switch on together)."""
    pairs = [(i, j) for i in range(r * r) for j in range(i + 1, r * r)
             if i // r != j // r and i % r != j % r]
    a, b = np.array(pairs).T
    return X[:, a] * X[:, b]


def flow_bound(demand, tops, weights):
    """Weak-duality LP bound sum_ij d_ij dist_w(i,j) / sum_a w_a, one value per topology."""
    n = len(demand)
    out = np.empty(len(tops))
    for k, (arcs, w) in enumerate(zip(tops, weights)):
        tail, head = np.array(arcs).T
        # Sparse, so a zero-dual arc is still an edge (dense csgraph input reads 0 as "no edge").
        G = csr_array((np.asarray(w, float) + 1e-9, (tail, head)), shape=(n, n))
        dist = shortest_path(G, method="D", directed=True)
        load = np.where(demand > 0, demand * dist, 0.0)   # 0 * inf (no demand, no path) is 0
        out[k] = float(load.sum() / (np.sum(w) + 1e-9 * len(w)))
    return out


# ── fitted surrogates ──────────────────────────────────────────────────────────────────


def ridge(X, y, alpha):
    mu_x, mu_y = X.mean(axis=0), y.mean()
    Xc = X - mu_x
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - mu_y))
    return lambda Z: (Z - mu_x) @ beta + mu_y


def fit_cv(X, y, rng, alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3), folds=5):
    """Ridge with the penalty chosen by k-fold cross-validated Spearman, refitted on all of X."""
    from scipy.stats import spearmanr

    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    best = (-np.inf, alphas[0])
    for alpha in alphas:
        scores = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            pred = ridge(X[tr], y[tr], alpha)(X[te])
            scores.append(spearmanr(pred, y[te]).statistic)
        best = max(best, (float(np.nanmean(scores)), alpha))
    return ridge(X, y, best[1]), best[1]


# ── one block ──────────────────────────────────────────────────────────────────────────


def run_block(n, r, b, seed, start, pool, demand):
    path = OUT / f"n{n:02d}_r{r}_s{seed}_b{b}.json"
    if path.exists():
        return json.loads(path.read_text())

    t0 = time.time()
    cur = load_start(n, start, 0)
    z0, hot = lp_eval(demand, cur)
    _, duals = npl.score_topology(cur, demand)
    dual_of = dict(zip(sorted(cur), duals))
    rng = np.random.default_rng([seed, b])
    free = pick_free(cur, hot, r, rng)

    perms, tops = rewirings(cur, free, n)
    z = np.array(pool.map(_lp, tops, chunksize=8))

    # surrogates that need no LP beyond the incumbent's
    Q, _, _ = build_subqubo(demand, cur, free, z0)
    X = onehot(perms, r)
    team = np.einsum("ki,ij,kj->k", X, Q, X)
    w_cur = [dual_of[a] for a in cur]          # weight follows the arc index (its tail)
    bound = flow_bound(demand, tops, [w_cur] * len(tops))
    hops = flow_bound(demand, tops, [np.ones(len(cur))] * len(tops))

    def agree(model, zz):
        a = rank_agreement(model, zz, model_lower_is_better=True, referee_lower_is_better=True,
                           tolerance=1e-9)
        return {"rho": a.rho, "rank_of_best": a.rank_of_best, "n_tied_with_best": a.n_tied_with_best,
                "ties_at_model_opt": a.n_tied_at_model_optimum,
                "referee_mean_at_model_opt": a.referee_mean_at_model_optimum,
                "referee_best": a.referee_best, "verdict": a.verdict}

    surrogates = {"team": agree(team, z), "bound": agree(bound, z), "hops": agree(hops, z)}

    # fitted surrogates, scored on the held-out half (too few rewirings: nothing to learn from)
    if len(z) >= 40:
        split = rng.permutation(len(z))
        tr, te = split[: len(z) // 2], split[len(z) // 2:]
        Xq = np.hstack([X, pair_features(X, r)])
        lin, a_lin = fit_cv(X[tr], z[tr], rng)
        quad, a_quad = fit_cv(Xq[tr], z[tr], rng)
        surrogates["fit_lin"] = agree(lin(X[te]), z[te]) | {"alpha": a_lin, "held_out": len(te)}
        surrogates["fit_quad"] = agree(quad(Xq[te]), z[te]) | {"alpha": a_quad, "held_out": len(te)}

    rec = {
        "n": n, "r": r, "block": b, "seed": seed, "start": start, "free": free,
        "z_incumbent": z0, "n_hot": len(hot), "n_valid": len(z),
        "n_improving": int((z < z0 - 1e-6).sum()), "n_sideways": int((np.abs(z - z0) <= 1e-6).sum()),
        "z_best": float(z.min()), "z_median": float(np.median(z)),
        "surrogates": surrogates,
        "z": z.tolist(), "team": team.tolist(), "bound": bound.tolist(), "hops": hops.tolist(),
        "perms": [list(p) for p in perms], "seconds": round(time.time() - t0, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--r", type=int, default=6)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", default="solutions", choices=["solutions", "runs", "published", "greedy"])
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    demand = qt.load_network_demand(a.n)
    names = ["team", "bound", "hops", "fit_lin", "fit_quad"]
    print(f"{'block':>5} {'valid':>6} {'better':>6} {'same':>5} {'best gain':>10}  "
          + "  ".join(f"{s:>9}" for s in names) + "   (Spearman rho vs LP)", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(demand,)) as pool:
        for b in range(a.blocks):
            rec = run_block(a.n, a.r, b, a.seed, a.start, pool, demand)
            gain = rec["z_best"] - rec["z_incumbent"]
            print(f"{b:>5} {rec['n_valid']:>6} {rec['n_improving']:>6} {rec['n_sideways']:>5} {gain:>10.0f}  "
                  + "  ".join(f"{rec['surrogates'][s]['rho']:>+9.3f}" if s in rec["surrogates"]
                              else f"{'-':>9}" for s in names)
                  + f"   {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
