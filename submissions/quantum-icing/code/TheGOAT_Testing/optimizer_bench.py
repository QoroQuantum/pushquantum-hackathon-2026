"""Which classical optimizer makes the quantum proposer a better optimizer of the block QUBO?

Same network24 blocks, same learned QUBO (goat_lns.learn_qubo on the hop count), one PCE run
per optimizer. Two measures per run:

* ``qubo_gap``: (best QUBO energy among the decoded permutations - SA's best) / spread of the
  energies of random permutations. 0 means the quantum proposer found what SA found; 1 means it
  did no better than a typical random permutation. This isolates the optimizer from the LP.
* ``z``: exact LP of the best of the top-K candidates by hop count (same K for every row,
  including the SA and Hungarian rows), i.e. what the walk would actually get.

One JSON per (block, optimizer) in runs/goat/optbench/, so a rerun skips what is done.

    venv/bin/python TheGOAT_Testing/optimizer_bench.py --blocks 5 --optimizers mc cobyla qng qnspsa
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import qoblib_tools as qt  # noqa: E402
from block_fidelity import flow_bound  # noqa: E402
from goat_lns import (OPTIMIZERS, Block, learn_qubo, lp_top, perm_energy, perturbed_assignments,  # noqa: E402
                      propose_quantum, propose_sa, qubo_matrix)
from network_design.sa_search import lp_eval  # noqa: E402
from network_lns import load_start, pick_free  # noqa: E402

OUT = HERE.parent / "runs" / "goat" / "optbench"


def setup_block(demand, cur, hot, r, b, seed, samples):
    rng = np.random.default_rng([seed, b])
    block = Block(cur, pick_free(cur, hot, r, rng), len(demand))
    train = block.random_perms(samples, rng)
    w = np.ones(len(cur))
    y = flow_bound(demand, [t for _, t in train], [w] * len(train))
    h, J, fit = learn_qubo([p for p, _ in train], y, r, rng)
    return block, h, J, fit, w, rng


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--start", default="greedy")
    p.add_argument("--r", type=int, default=10)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--optimizers", nargs="+", default=["mc", "cobyla", "qng", "qnspsa"],
                   choices=OPTIMIZERS + ("qnspsa_warm",),
                   help="qnspsa_warm: QN-SPSA started from the previous block's best angles")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--cands", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    demand = qt.load_network_demand(a.n)
    cur = load_start(a.n, a.start, 0)
    _, hot = lp_eval(demand, cur)
    print(f"{'block':>5} {'optimizer':>9} {'qubo_gap':>9} {'z':>9} {'SA z':>9} {'circuits':>8} {'s':>6}", flush=True)
    for b in range(a.blocks):
        block, h, J, fit, w, rng = setup_block(demand, cur, hot, a.r, b, a.seed, a.samples)
        Jsym = J + J.T
        hh = h + 1e3 * block.forbidden.ravel()
        energy = lambda perms: [perm_energy(q, hh, Jsym, a.r) for q in perms]  # noqa: E731
        sa = propose_sa(block, h, J, a.cands, rng)
        e_sa = min(energy([q for q, _ in sa]))
        e_rand = energy([q for q, _ in block.random_perms(200, rng)])
        spread = float(np.median(e_rand) - e_sa) or 1e-9
        z_sa = lp_top(demand, sa, w, a.K)[0]
        hung = perturbed_assignments(block, -h, a.cands, rng)
        z_hung = lp_top(demand, hung, w, a.K)[0]
        # the warm variant runs first in each block, so the angles it inherits are the previous
        # block's, never this block's
        order = sorted(a.optimizers, key=lambda o: o != "qnspsa_warm")
        for opt in order:
            path = OUT / f"n{a.n:02d}_r{a.r}_s{a.seed}_b{b}_{opt}.json"
            if path.exists():
                rec = json.loads(path.read_text())
            else:
                warm = opt == "qnspsa_warm"
                args = SimpleNamespace(solver="pce", optimizer="qnspsa" if warm else opt, pop=20, iters=a.iters,
                                       layers=a.layers, shots=5000, top=200, alpha=4.0, soft=1, cloud=None,
                                       warm_angles=int(warm))
                Q = qubo_matrix(h, J, block, 2.0)
                t0 = time.time()
                try:
                    cands, info = propose_quantum(block, Q, h, a.cands, args, a.seed * 100 + b, rng)
                    err = None
                except Exception as exc:                       # an optimizer that fails is a result too
                    cands, info, err = [], {"circuits": 0}, f"{type(exc).__name__}: {exc}"[:300]
                secs = time.time() - t0
                e_q = min(energy([q for q, _ in cands])) if cands else float("inf")
                rec = {"block": b, "optimizer": opt, "qubo_gap": (e_q - e_sa) / spread,
                       "z": lp_top(demand, cands, w, a.K)[0], "z_sa": z_sa, "z_hungarian": z_hung,
                       "circuits": info.get("circuits", 0), "seconds": round(secs, 1), "fit_rho": fit,
                       "iters": a.iters, "error": err}
                path.write_text(json.dumps(rec))
            print(f"{b:>5} {opt:>9} {rec['qubo_gap']:>9.3f} {rec['z']:>9.0f} {rec['z_sa']:>9.0f} "
                  f"{rec['circuits']:>8} {rec['seconds']:>6.0f}" + (f"  ERROR {rec['error']}" if rec["error"] else ""),
                  flush=True)


if __name__ == "__main__":
    main()
