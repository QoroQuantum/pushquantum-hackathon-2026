"""PCE on a whole day in the signed 3-bit encoding (a050: 50 assets -> 150 binaries).

The unary encoding failed for PCE because its empty position is all-zeros and PCE's
parity readout gives dense bit patterns. In the signed encoding (see
``qaoa_signed_blocks.py``) every code is a valid holding and zero has two codes, so we
retest. Budget: Lagrangian ``mu * units`` (units are quadratic in this encoding), with
``mu`` picked by SA on the same QUBO; then exact repair + official score.

    python experiments/pce_signed_day.py --days 0 3 6
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import argparse
import json
import time

import numpy as np

from asset_groups import BlockEvaluator, _load, repair_fast, sa_start
from qaoa_signed_blocks import block_qubo, decode


def day_score(model, L, S, t, v):
    L2, S2 = L.copy(), S.copy()
    L2[t], S2[t] = np.maximum(v, 0), np.maximum(-v, 0)
    sc = model.score_arrays(L2, S2)
    if not sc.feasible:
        L2, S2 = repair_fast(model, L2, S2)
        sc = model.score_arrays(L2, S2)
    return sc.objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a050_t10_orig")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--days", type=int, nargs="+", default=[0, 3, 6])
    ap.add_argument("--encoding", default="poly", choices=["poly", "multibasis"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dwave.samplers import SimulatedAnnealingSampler
    from divi.backends import MaestroSimulator
    from divi.qprog.algorithms import PCE, MultiBasisEncoding, PolyEncoding, SoftObjective
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    model, bkv = _load(args.instance, args.budget, args.lam)
    ev = BlockEvaluator(model)
    L, S, _ = sa_start(model, num_reads=10, num_sweeps=1000, seed=args.seed)
    base = model.score_arrays(L, S).objective
    G = np.arange(len(model.symbols))
    rows = []
    for t in args.days:
        # mu: smallest on a grid whose SA optimum on the signed QUBO is feasible
        chosen = None
        for mu in [0, 100, 200, 400, 800, 1600, 3200, 6400]:
            bqm, W, C, margins = block_qubo(ev, L, S, t, G, mu)
            ss = SimulatedAnnealingSampler().sample(bqm, num_reads=20, num_sweeps=2000, seed=1)
            bits = ss.record.sample[:, [ss.variables.index(k) for k in range(bqm.num_variables)]]
            V = decode(bits, W, C)
            c, ok = ev.costs(L, S, [t], G, V)
            if ok.any():
                chosen = (mu, bqm, W, C, float(np.min(np.where(ok, c, np.inf))), V[np.argmin(np.where(ok, c, np.inf))])
                break
        if chosen is None:
            print(f"day {t}: no mu on the grid gives a feasible SA optimum; skipped", flush=True)
            continue
        mu, bqm, W, C, sa_cost, sa_v = chosen
        cur_cost = float(ev.costs(L, S, [t], G, (L[t] - S[t])[None])[0][0])
        empty_cost = float(ev.costs(L, S, [t], G, np.zeros((1, len(G)), int))[0][0])
        scale = max(np.abs(list(bqm.linear.values())).max(), np.abs(list(bqm.quadratic.values())).max())
        bqm_n = bqm.copy()
        bqm_n.scale(1 / scale)
        enc = PolyEncoding() if args.encoding == "poly" else MultiBasisEncoding(order=2)
        t0 = time.time()
        pce = PCE(BinaryOptimizationProblem(bqm_n), encoding=enc, n_layers=3,
                  objective=SoftObjective(alpha=2.0),
                  optimizer=MonteCarloOptimizer(population_size=50), max_iterations=args.iters,
                  backend=MaestroSimulator(shots=10_000), seed=args.seed + 1)
        pce.run()
        secs = time.time() - t0
        tops = pce.get_top_solutions(n=200, sort_by="energy")
        qb = np.array([[int(ch) for ch in s.bitstring[:bqm.num_variables]] for s in tops])
        Vq = decode(qb, W, C)
        cq, okq = ev.costs(L, S, [t], G, Vq)
        units = np.abs(Vq).sum(1)
        best_feas = float(np.min(np.where(okq, cq, np.inf)))
        # repaired official score of the best-energy PCE samples (top 20)
        rep = min(day_score(model, L, S, t, v) for v in Vq[:20])
        row = dict(day=t, mu=mu, qubits=getattr(pce, "n_qubits", None), vars=bqm.num_variables,
                   seconds=round(secs, 1), min_units_sampled=int(units.min()),
                   feasible_in_top=int(okq.sum()),
                   pce_best_feasible_minus_current=best_feas - cur_cost,
                   empty_minus_current=empty_cost - cur_cost,
                   sa_signed_minus_current=sa_cost - cur_cost,
                   pce_repaired_score=rep, start_score=base,
                   sa_signed_score=day_score(model, L, S, t, sa_v))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = f"results/pce_signed_day_{args.instance}_B{args.budget}_lam{args.lam}_{args.encoding}.json"
    json.dump(dict(vars(args), rows=rows), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
