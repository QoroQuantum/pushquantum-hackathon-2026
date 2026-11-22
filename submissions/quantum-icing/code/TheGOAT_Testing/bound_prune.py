"""Can a cheap lower bound replace the LP inside the radius certificate? Measured: no.

The radius certificate (`certify_radius.py`, `certify_descent.py`) spends all its time on one
LP per candidate topology: 12 ms warm, against 50 ms cold. Radius 4 on network18 is 370,545 of
them, radius 5 would be sixteen million. The obvious accelerator is a *valid lower bound* on a
candidate's objective: if the bound already reaches the incumbent, the candidate cannot be
better and the LP can be skipped. The certificate stays a proof, because skipping only happens
where the answer is forced.

Such a bound comes free from LP duality. Writing `w_a >= 0` for the dual of the capacity row of
arc `a`, weak duality gives, for **any** nonnegative weights,

    z_LP(arcs)  >=  1000 * sum_ij d_ij * dist_w(i, j) / sum_a w_a

with `dist_w` the shortest path under arc costs `w`. Unit weights give the demand-weighted hop
bound already used in `block_fidelity.py`. The incumbent's own dual solution gives the weights
that are *optimal there* -- strong duality makes the inequality an equality, which this script
checks to six digits as a self-test of the dual extraction.

The measurement is what matters. A candidate differing in k arcs has k arcs the incumbent's
duals say nothing about, so they are filled with 0, the mean, or the max of the dual vector.
Result on network18 (800 sampled candidates, radii 3 and 4):

    weights      ms   valid  bound/lp   pruned at z0-0.5
    zero       1.46    True    0.8494              33.5%
    mean       1.22    True    0.8526               3.9%
    max        1.15    True    0.8097               1.4%
    unit       1.10    True    0.8965              20.0%

Every bound is valid on every sample, and every one is useless: the candidates near a polished
incumbent sit a fraction of a percent above it, so pruning needs a bound within a fraction of
a percent of the truth, and the best of these is 15% below. Zero-fill prunes a third of the
candidates at 1.5 ms -- against 12 ms for the warm LP that answers exactly, so the ceiling on
the whole idea is a 1.27x speedup. Radius 5 needs 20x.

The reason is structural, and it is the same reason the hop count fails as a surrogate: the
objective is a *max* over arcs, the dual weights concentrate on the few congested ones (13 of
36 arcs carry the whole dual on network18), and a radius-k move is chosen precisely to disturb
those arcs. The weights that certify the incumbent are the ones a good candidate invalidates.
Tightening them back per candidate is the dual simplex -- which is what the warm LP already is.

    venv/bin/python TheGOAT_Testing/bound_prune.py --n 18 --samples 400
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import permutations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sps  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import qoblib_tools as qt  # noqa: E402
from qoblib_tools import INTSCALE, _routing_lp  # noqa: E402

from certify_radius import apply_perm, derangements  # noqa: E402

OUT = Path(os.environ.get("GOAT_RUN_DIR", ROOT / "runs/goat")) / "boundprune"


def dual_weights(demand: np.ndarray, arcs: list[tuple[int, int]]) -> tuple[float, np.ndarray]:
    """Solve the routing LP and return ``(z_lp, w)`` with ``w`` the capacity-row duals.

    ``w`` is nonnegative and sums to 1, so it is feasible for the bound below and optimal for
    this topology. Any other nonnegative vector is feasible too, just weaker.
    """
    var, c, A_ub, A_eq, b_eq = _routing_lp(demand, arcs)
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return math.inf, np.zeros(len(arcs))
    return float(res.fun), np.abs(np.asarray(res.ineqlin.marginals, dtype=float))


def flow_bound(demand: np.ndarray, arcs: list[tuple[int, int]], w: np.ndarray) -> float:
    """Weak-duality lower bound on the QOBLIB objective of ``arcs``, for any ``w >= 0``.

    ``inf`` when some demand pair is unreachable, which is correct: the topology is infeasible.
    """
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0
    n = len(demand)
    # A zero cost would let shortest_path treat the arc as absent, so nudge every weight.
    g = sps.csr_matrix((np.asarray(w, dtype=float) + 1e-15,
                        ([i for i, _ in arcs], [j for _, j in arcs])), shape=(n, n))
    dist = shortest_path(g, method="D", directed=True)
    with np.errstate(invalid="ignore"):               # no demand, no path: 0 * inf is 0 here
        load = np.where(demand > 0, demand * dist, 0.0)
    if not np.isfinite(load).all():
        return math.inf
    return INTSCALE * float(load.sum()) / sw


def sample_candidates(arcs, radii, per_radius, seed=0):
    """Random distinct valid topologies at each radius, via the derangement enumeration."""
    rng = random.Random(seed)
    out = []
    for k in radii:
        d = derangements(k)
        seen = 0
        while seen < per_radius:
            idx = tuple(sorted(rng.sample(range(len(arcs)), k)))
            new = apply_perm(arcs, idx, rng.choice(d))
            if new is None:
                continue
            out.append((k, sorted(new)))
            seen += 1
    return out


def measure(n: int, per_radius: int, radii=(3, 4), seed: int = 0) -> dict:
    demand = qt.load_network_demand(n)
    sol = ROOT / f"deliverables/network{n:02d}/network{n:02d}.sol"
    _, arcs, _ = qt.read_network_solution(sol)
    arcs = sorted({tuple(map(int, a)) for a in arcs})
    index = {a: i for i, a in enumerate(arcs)}

    z_lp, w_star = dual_weights(demand, arcs)
    z0 = math.ceil(z_lp - 1e-6)
    self_check = flow_bound(demand, arcs, w_star)
    unit_check = flow_bound(demand, arcs, np.ones(len(arcs)))
    print(f"network{n:02d}: {len(arcs)} arcs, incumbent {z0} (lp {z_lp:.3f})")
    print(f"duals: sum {w_star.sum():.6f}, nonzero {int((w_star > 1e-9).sum())}/{len(arcs)}")
    print(f"strong duality at the incumbent: bound {self_check:.1f} vs lp {z_lp:.1f} "
          f"(ratio {self_check / z_lp:.6f}); unit weights {unit_check / z_lp:.3f} of lp")
    assert abs(self_check / z_lp - 1.0) < 1e-6, "dual extraction is wrong: no strong duality"

    cands = sample_candidates(arcs, radii, per_radius, seed)
    fills = {"zero": 0.0,
             "mean": float(w_star[w_star > 1e-9].mean()),
             "max": float(w_star.max()),
             "unit": None}
    cut = z0 - 0.5
    bounds = {name: [] for name in fills}
    secs = {name: 0.0 for name in fills}
    lps, t_lp = [], 0.0

    for _, new in cands:
        for name, fill in fills.items():
            w = (np.ones(len(new)) if name == "unit"
                 else np.array([w_star[index[a]] if a in index else fill for a in new]))
            t = time.perf_counter()
            b = flow_bound(demand, new, w)
            secs[name] += time.perf_counter() - t
            bounds[name].append(b)
        t = time.perf_counter()
        lps.append(qt.route_min_congestion(demand, new).z_lp)
        t_lp += time.perf_counter() - t

    lp = np.array(lps, dtype=float)
    rec = {"n": n, "arcs": len(arcs), "incumbent": z0, "lp_ms": t_lp / len(cands) * 1e3,
           "samples": len(cands), "radii": list(radii), "fills": {}}
    print(f"\n{len(cands)} candidates, exact LP {rec['lp_ms']:.1f} ms each (cold)")
    print(f"{'weights':8} {'ms':>6} {'valid':>7} {'bound/lp':>9} {'pruned':>8}")
    for name in fills:
        b = np.array(bounds[name], dtype=float)
        fin = np.isfinite(lp) & np.isfinite(b)
        valid = bool(np.all(b[fin] <= lp[fin] + 1e-6))
        row = {"ms": secs[name] / len(cands) * 1e3, "valid": valid,
               "ratio": float((b[fin] / lp[fin]).mean()), "pruned": float((b >= cut).mean())}
        rec["fills"][name] = row
        print(f"{name:8} {row['ms']:6.2f} {str(valid):>7} {row['ratio']:9.4f} "
              f"{row['pruned'] * 100:7.1f}%")
        assert valid, f"the {name} bound exceeded the LP: it is not a lower bound"

    rec["improving_in_sample"] = int((lp <= cut).sum())
    best = max(rec["fills"].values(), key=lambda r: r["pruned"])
    warm_ms = 12.0           # measured warm LP with the incumbent as objective_bound
    rec["ceiling_speedup"] = warm_ms / (best["ms"] + (1 - best["pruned"]) * warm_ms)
    print(f"\nimproving candidates in the sample: {rec['improving_in_sample']} of {len(cands)}")
    print(f"ceiling on the whole idea: {rec['ceiling_speedup']:.2f}x "
          f"(best prune rate {best['pruned'] * 100:.1f}% at {best['ms']:.2f} ms, "
          f"warm LP {warm_ms:.0f} ms). Radius 5 needs about 20x.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"n{n:02d}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--samples", type=int, default=400, help="candidates per radius")
    p.add_argument("--radii", type=int, nargs="+", default=[3, 4])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    measure(a.n, a.samples, tuple(a.radii), a.seed)


if __name__ == "__main__":
    main()
