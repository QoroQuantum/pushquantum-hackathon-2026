"""PT / baseline runner for QOBLIB 06 portfolio (B=20, a050 high-lambda instances).

  python run_pt.py --instance po_a050_t10_orig --lam 1e-2 --algo pt --start empty --time 150 --polish 45
  algo: pt | sa (lns.sa restarts, same move-free baseline) | pooldp (run_lns.solve pipeline)
  start: empty | overfull (our lambda=1e-3 best re-scored at this lambda) | best (our best here) | mix

Writes pt/results/<label>.json and records (pf_core.record_solution) any feasible result that
matches or beats our previous best for that instance. Referee = model.score always.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quantum_focus"))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import qf_common as qf  # noqa: E402
import lns  # noqa: E402
import pf_core as pc  # noqa: E402

RES = HERE / "results"
RES.mkdir(exist_ok=True)


def our_best(inst, budget, lam):
    b = qf.best_known_sol(inst, budget, lam)
    return (None, None, None) if b is None else b


def load_start(model, fm, inst, lam, kind):
    if kind == "empty":
        return np.zeros((fm.T, fm.n), dtype=np.int64), "empty"
    if kind == "overfull":
        f = HERE.parent / "quantum_focus" / "starts" / f"{inst}_l{lam}_from_l1e-3.sol"
        if f.exists():
            _, pos = pc.qt.read_portfolio_solution(f)
            src = str(f.relative_to(HERE.parent))
        else:
            obj, f, pos = our_best(inst, model.budget, "1e-3")
            src = str(f.relative_to(HERE.parent))
        v = pc.positions_to_v(model, pos)
        return v, f"overfull ({src})"
    if kind == "best":
        obj, f, pos = our_best(inst, model.budget, lam)
        return pc.positions_to_v(model, pos), f"best ({f.relative_to(HERE.parent)})"
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--lam", default="1e-2")
    ap.add_argument("--algo", default="pt")
    ap.add_argument("--start", default="empty")
    ap.add_argument("--time", type=float, default=150)
    ap.add_argument("--polish", type=float, default=45)
    ap.add_argument("--replicas", type=int, default=24)
    ap.add_argument("--tmin", type=float, default=20.0)
    ap.add_argument("--tmax", type=float, default=20000.0)
    ap.add_argument("--adapt", type=float, default=0.4)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--reseed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    label = a.label or f"{a.algo}_{a.instance}_l{a.lam}_{a.start}_s{a.seed}"
    logf = open(HERE / "logs" / f"{label}.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    model, fm, (bkv, bkv_status) = qf.load(a.instance, a.lam)
    prev_obj, prev_file, _ = our_best(a.instance, model.budget, a.lam)
    log(f"{label}: BKV {bkv} ({bkv_status}), our previous best {prev_obj} ({prev_file and prev_file.name})")
    t0 = time.time()
    out = dict(label=label, instance=a.instance, lam=a.lam, algo=a.algo, start=a.start,
               bkv=bkv, prev_best=prev_obj, seed=a.seed)
    if a.algo == "pt":
        import pt_core

        if a.start == "mix":
            starts, descs = [], []
            for k in ("empty", "overfull", "best"):
                try:
                    v, d = load_start(model, fm, a.instance, a.lam, k)
                    starts.append(v)
                    descs.append(d)
                except Exception as e:  # noqa: BLE001
                    log(f"  no {k} start: {e}")
            desc = " + ".join(descs)
        else:
            v0, desc = load_start(model, fm, a.instance, a.lam, a.start)
            starts = [v0]
        out["start_desc"] = desc
        out["start_obj"] = [fm.objective(s) for s in starts]
        log(f"  start {desc}: obj {out['start_obj']}")
        res = pt_core.parallel_tempering(fm, starts, n_replicas=a.replicas, t_min=a.tmin,
                                         t_max=a.tmax, s_min=a.smin, time_limit=a.time, steps_per_round=a.steps,
                                         reseed_every=a.reseed, adapt_frac=a.adapt, seed=a.seed, log=log)
        v, raw = res["v"], res["E"]
        out.update(raw=raw, temps=[round(x, 1) for x in res["temps"]], ss=[round(x, 4) for x in res["ss"]],
                   swap_rate=[round(x, 3) for x in res["swap_rate"]], rounds=res["rounds"],
                   steps=res["steps"], trace=res["trace"],
                   move_stats=res["move_stats"].tolist())
        kinds = ["single", "transfer", "hedge", "time-asset", "time-day", "shift"]
        log("  moves (proposed/accepted): " + ", ".join(
            f"{k} {p}/{c} ({c / max(p, 1):.1%})" for k, (p, c) in zip(kinds, res["move_stats"])))
        log(f"  ladder T: {np.round(res['temps'], 0).tolist()}")
        log(f"  ladder s: {np.round(res['ss'], 3).tolist()}")
        log(f"  swap rates: {np.round(res['swap_rate'], 2).tolist()}")
        kindpt = "lambda-tempering PT" if a.smin != 1.0 else "PT"
        method = f"{kindpt} ({a.replicas} replicas, {a.start} start) + LNS-exactDP polish (classical)"
        settings = (f"PT {a.time:.0f}s T=[{res['temps'][0]:.0f}..{res['temps'][-1]:.0f}] s=[1..{a.smin}] adaptive ladder, "
                    f"6 move kinds, steps/round {a.steps}; LNS polish {a.polish:.0f}s; seed {a.seed}")
    elif a.algo == "sa":
        v0, desc = load_start(model, fm, a.instance, a.lam, a.start)
        out["start_desc"] = desc
        best_v, raw, k = v0.copy(), fm.objective(v0), 0
        while time.time() - t0 < a.time:
            bv, obj = lns.sa(fm, v0=v0, n_steps=2_000_000, seed=a.seed * 1000 + k)
            k += 1
            if obj < raw:
                best_v, raw = bv, obj
        v = best_v
        out.update(raw=raw, sa_restarts=k)
        method = f"SA baseline (lns.sa, {a.start} start, restarts) + LNS-exactDP polish (classical)"
        settings = f"SA {k}x2M steps in {a.time:.0f}s; LNS polish {a.polish:.0f}s; seed {a.seed}"
    elif a.algo == "pooldp":
        import run_lns

        inst, lam, v, raw, obj_pool, t_pool, dt, it, npool = run_lns.solve(
            (a.instance, a.lam, a.time + a.polish, a.seed))
        out.update(raw=obj_pool, start_desc="pool-DP", pool=npool, t_pool=t_pool)
        a.polish = 0  # run_lns.solve already includes its LNS for the remaining time
        method = "pool-DP + LNS-exactDP (classical) [PT baseline, equal wall time]"
        settings = f"run_lns.solve, total {a.time:.0f}s, pool {npool}, LNS its {it}; seed {a.seed}"
    else:
        raise ValueError(a.algo)
    t_main = time.time() - t0
    if a.polish > 0:
        v, polished, it = lns.lns(fm, v0=v, time_limit=a.polish, seed=a.seed)
    else:
        polished = fm.objective(v)
    pos = pc.v_to_positions(model, v)
    score = model.score(pos)
    assert score.feasible and score.objective == polished, (score, polished)
    out.update(polished=int(polished), t_main=round(t_main, 1), t_total=round(time.time() - t0, 1),
               gap_raw=qf.gap(out["raw"], bkv), gap=qf.gap(polished, bkv),
               held_days=int((np.abs(v).sum(1) > 0).sum()), held_assets=int((np.abs(v).sum(0) > 0).sum()),
               units_per_day=np.abs(v).sum(1).tolist())
    rec = False
    if not a.no_record and (prev_obj is None or polished <= prev_obj):
        _, fname, _ = pc.record_solution(model, pos, method=method, settings=settings, quantum=False,
                                         tag=a.algo)
        out["file"] = str(fname.relative_to(HERE.parent))
        rec = True
    log(f"RESULT {label}: raw {out['raw']} ({out['gap_raw']}) -> polished {polished} ({out['gap']}) "
        f"BKV {bkv} prev {prev_obj} recorded={rec} [{out['t_total']}s]")
    (RES / f"{label}.json").write_text(json.dumps(out, indent=1, default=int))


if __name__ == "__main__":
    main()
