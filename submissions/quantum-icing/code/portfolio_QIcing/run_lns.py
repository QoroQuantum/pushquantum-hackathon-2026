"""Production runner (classical): per-period candidate pools (QP relaxation + rounding + pair-move
descent + ILS) -> exact chain DP over periods -> asset-subset LNS polish (exact sub-DP).
Parallel over instances; the parent process records every final solution (referee-checked).

Usage: python run_lns.py <assets> <T> <time_per_instance_s> [lam_tags...]
"""
import os
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multiprocessing as mp

import numpy as np

import pf_core as pc
import lns

LOG = pc.HERE / "logs"


def solve(task):
    inst, lam, tlim, seed = task
    t0 = time.time()
    model = pc.make_model(inst, lam)
    fm = lns.FastModel(model)
    pool = [np.zeros(fm.n, dtype=np.int64)]
    light = int(os.environ.get("LIGHT", 0))  # 1: a200 (QP once per period), 2: a400 (no QP)
    extras = (0.0, 1.0, 2.0) if not light else ((1.0,) if light == 1 else ())
    for t in range(fm.T - 1):
        for e in extras:
            ea = np.round(e * model.tc[t]).astype(np.int64)
            c = lns.period_candidates_qp(fm, t, extra=(e,), scales=(1, 3, 8, 20) if light else (1, 2, 3, 5, 8, 12, 20, 40))
            pool += c + lns.period_ils(fm, t, c, ea, n_kicks=8 if light else 25, seed=seed + t)
        sa_c = lns.period_candidates(fm, t, n_seeds=1, extra=(0.0, 1.0), n_steps=800_000 * (1 + light), seed=seed)
        pool += sa_c + lns.period_ils(fm, t, sa_c, None, n_kicks=15, seed=seed + 7 * t)
    v, obj = lns.chain_dp(fm, pool)
    t_pool = time.time() - t0
    obj_pool = obj
    v, obj, it = lns.lns(fm, v0=v, time_limit=max(10.0, tlim - (time.time() - t0)), seed=seed)
    assert fm.feasible(v) and fm.objective(v) == obj
    return inst, lam, v, obj, obj_pool, t_pool, time.time() - t0, it, len(pool)


def main():
    assets, T, tlim = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    lams = sys.argv[4:] or pc.LAM_TAGS
    seeds = os.environ.get("SEEDS", "orig,s00,s01,s02").split(",")
    tasks = [(f"po_a{assets:03d}_t{T}_{s}", lam, tlim, 0) for lam in lams for s in seeds]
    bkv = pc.load_bkv()
    logf = open(LOG / f"lns_a{assets:03d}_t{T}.txt", "a")
    with mp.get_context("fork").Pool(int(os.environ.get("LNS_PROCS", 8))) as p:
        for inst, lam, v, obj, obj_pool, t_pool, dt, it, npool in p.imap_unordered(solve, tasks):
            model = pc.make_model(inst, lam)
            pos = pc.v_to_positions(model, v)
            score = model.score(pos)
            assert score.feasible and score.objective == obj
            ref = bkv.get((inst, model.budget, lam), (None, ""))[0]
            gap = f"{(obj - ref) / abs(ref):+.3%}" if ref else "no BKV"
            verdict = "" if ref is None else ("BEATS BKV" if obj < ref else ("= BKV" if obj == ref else ""))
            pc.record_solution(model, pos, method="pool-DP + LNS-exactDP (classical)",
                               settings=f"QP-relax+round+pair-descent+ILS pool ({npool}), chain DP, "
                                        f"LNS k<=4 {dt:.0f}s seed 0", tag="lns")
            line = (f"{inst} λ={lam:>5}: {obj:>9} (pool-DP {obj_pool}, {t_pool:.0f}s; LNS its {it}) "
                    f"BKV {ref} gap {gap} {verdict} [{dt:.0f}s]")
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()


if __name__ == "__main__":
    main()
