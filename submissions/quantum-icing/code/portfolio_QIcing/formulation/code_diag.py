"""Code diagnostic (no quantum): what can a PCE parity code reach at all?

For every single-period problem of an a010 instance (state space small enough to enumerate exactly),
every formulation and every code, enumerate all 2^q codewords and report
  feasible share, best raw codeword (rank among all feasible states), best after greedy repair.
A sampler restricted to the code can never beat "best raw"; repair can only reach "best repaired".

    python code_diag.py --instance po_a010_t10_orig --lam 1e-4
"""
from __future__ import annotations

import argparse

import numpy as np

import codes as cd
import formulations as fm

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a010_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--periods", default="0,3,6")
args = ap.parse_args()

model = fm.pc.make_model(args.instance, args.lam)
probs = fm.period_problems(model)
V_all = fm.pc.enumerate_states(len(model.symbols), model.ub, model.budget, model.capital,
                               2**model.cash_slack_bits - 1)


def priority(F: fm.PeriodFormulation) -> np.ndarray:
    """Per variable: how attractive is setting it to 1 alone (single-unit gain from empty)."""
    P = F.prob
    pr = np.zeros(F.n_vars)
    e0 = P.energy(np.zeros(P.n, dtype=np.int64))[0]
    for ia, i in enumerate(F.assets):
        g = {}
        for s in (1, -1):
            v = np.zeros(P.n, dtype=np.int64)
            v[i] = s
            g[s] = e0 - P.energy(v)[0]  # > 0: unit is profitable
        b = ia * F.k
        best = max(g.values())
        if F.cell in ("unary", "ternary"):
            for m in range(F.U):
                pr[b + m] = g[1] - m * 1e-3
                pr[b + F.U + m] = g[-1] - m * 1e-3
        else:
            pr[b] = g[-1] - g[1]  # sign bit: 1 = short
            pr[b + 1] = best
            for j in range(2, F.k):
                pr[b + j] = best - 1e9  # higher magnitude bits last
    pr[F.n_cell_vars:] = -1e12  # slack: pairs
    return pr


rows = []
for t in [int(s) for s in args.periods.split(",")]:
    P = probs[t]
    E_all = np.sort(P.energy(V_all))
    for cell, cons in [("unary", "slack"), ("signmag", "slack"), ("signmag", "none"), ("ternary", "slack"),
                       ("ternsm", "none")]:
        F = fm.PeriodFormulation(P, cell=cell, constraints=cons)
        x_empty = F.encode(np.zeros(P.n, dtype=np.int64))
        variants = {}
        m, q = cd.poly_masks(F.n_vars)
        variants["poly (Divi default)"] = (m, q, None)
        variants["poly, re-centred on empty"] = (m, q, cd.reference_flip(x_empty))
        ms, q = cd.structured_poly_masks(priority(F))
        variants["poly, structured+re-centred"] = (ms, q, cd.reference_flip(x_empty))
        m2, q2 = cd.poly_masks(F.n_vars, q + 3)
        ms2, _ = cd.structured_poly_masks(priority(F), q + 3)
        variants[f"poly q+3, structured+re-centred"] = (ms2, q2, cd.reference_flip(x_empty))
        md, qd = cd.dense_masks(F.n_vars)
        variants["dense, re-centred"] = (md, qd, cd.reference_flip(x_empty))
        for name, (masks, q, flip) in variants.items():
            X = cd.codewords(masks, q, flip)
            V = F.decode(X)
            feas = P.feasible(V)
            E = P.energy(V)
            best_raw = E[feas].min() if feas.any() else np.inf
            uniq = np.unique(V, axis=0)
            Vr = np.array([P.repair(v) for v in uniq])
            best_rep = P.energy(Vr).min()
            rank = lambda e: int(np.searchsorted(E_all, e)) if np.isfinite(e) else -1
            rows.append((t, F.summary().split(",")[0], name, q, len(X), feas.mean(), best_raw, rank(best_raw),
                         best_rep, rank(best_rep), E_all[0]))
            print(f"t={t} {cell}/{cons:5} {F.n_vars:3d}v {name:34s} q={q:2d} feas {feas.mean():6.1%} "
                  f"best raw {best_raw:>10.0f} (rank {rank(best_raw):5d})  repaired {best_rep:>10.0f} "
                  f"(rank {rank(best_rep):4d})  opt {E_all[0]:.0f}", flush=True)
