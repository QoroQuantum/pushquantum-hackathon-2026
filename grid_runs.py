"""Run day_swap over every portfolio instance that has a QOBLIB best-known value.

Each task is one (instance, λ, seed) day_swap run from the empty portfolio (or from a
given start file); tasks run in a process pool and each writes its own .sol + .json
under ``--out``. ``python grid_runs.py report`` collects the best file per instance and
its gap to the best-known value (parsed from QOBLIB's solutions README).

    python grid_runs.py run --sizes a050_t15 a050_t10 --seeds 0 1 --time 180 --workers 6
    python grid_runs.py report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import qoblib_tools as qt

BKV_README = Path("qoblib_data/06-portfolio/solutions/README.md")
LAM_TAGS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]
BUDGET = {"a010": 4, "a050": 20, "a200": 50, "a400": 100}
SEEDS = ["orig", "s00", "s01", "s02"]


def lam_tag(x: str) -> str:
    """'1e-02' / '0.001' / '1e-3' -> file-name spelling ('1e-2', '1e-3', '5e-4', '0')."""
    f = Fraction(str(float(x)))
    for t in LAM_TAGS:
        if Fraction(str(float(t))) == f:
            return t
    raise ValueError(x)


def load_bkv() -> dict[str, float]:
    """{'po_a050_t15_s02_b020_l1e-2': -40622, ...} from the BKV table of QOBLIB's README."""
    if not BKV_README.exists():
        qt.fetch_qoblib_file("06-portfolio/solutions/README.md")
    out = {}
    for line in BKV_README.read_text().splitlines():
        m = re.match(r"\|\s*(a\d{3}_t\d{2}_\w+?)_b(\d{3})_l([\de.\-+]+)\s*\|\s*(-?[\d.]+)\s*\|", line)
        if m:
            inst, b, lam, val = m.groups()
            out[f"po_{inst}_b{b}_l{lam_tag(lam)}"] = float(val)
    return out


def _task(args):
    inst, budget, lam, seed, tlim, init, out, stall, *rest = args
    window = rest[0] if rest else 2
    import day_swap  # noqa: PLC0415 - import in the worker (numba cache)

    stem = f"{inst}_b{budget:03d}_l{lam}"
    tag = "empty" if init == "empty" else "warm"
    rec_path = Path(out) / f"{stem}_{tag}_seed{seed}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    model = qt.PortfolioModel(qt.load_portfolio_instance(inst), budget=budget, lam=lam)
    if init == "empty":
        L = np.zeros((model.n_periods, len(model.symbols)), np.int64)
        S = L.copy()
    else:
        _, pos = qt.read_portfolio_solution(init)
        L, S = (a.astype(np.int64) for a in model.to_arrays(pos))
    start = model.score_arrays(L, S).objective
    t0 = time.time()
    r = day_swap.run(model, L, S, tlim, seed, verbose=False, stall=stall or None, window=window)
    final = model.score_arrays(r["L"], r["S"])
    assert final.feasible and final.objective == r["score"]
    sol = Path(out) / f"{stem}_{tag}_seed{seed}.sol"
    model.write_solution(sol, model.to_positions(r["L"], r["S"]),
                         comment=f"day_swap.py run(seed={seed}, {tlim:.0f} s) from {init}")
    rec = dict(instance=inst, budget=budget, lam=lam, seed=seed, init=init, start=start,
               score=int(final.objective), seconds=round(time.time() - t0, 1),
               accepted=r["accepted"], accepted2=r["accepted2"],
               last_gain=r["trace"][-1][0], solution_file=str(sol))
    rec_path.write_text(json.dumps(rec, indent=1))
    print(f"{stem:32s} {tag} seed{seed}: {start} -> {rec['score']} "
          f"(last gain {rec['last_gain']}s of {tlim:.0f})", flush=True)
    return rec


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    bkv = load_bkv()
    tasks = []
    for size in a.sizes:
        for sd in a.inst_seeds:
            inst = f"po_{size}_{sd}"
            b = BUDGET[size[:4]]
            for lam in a.lams:
                key = f"{inst}_b{b:03d}_l{lam}"
                if a.only_bkv and key not in bkv:
                    continue
                if a.keys and key not in a.keys:
                    continue
                for s in a.seeds:
                    tasks.append((inst, b, lam, s, a.time, "empty", a.out, a.stall, a.window))
                if a.warm:
                    warm = Path("handin") / f"{key}.sol"
                    if warm.exists():
                        tasks.append((inst, b, lam, 0, a.time, str(warm), a.out, a.stall, a.window))
    print(f"{len(tasks)} tasks on {a.workers} workers", flush=True)
    # load every instance once in the parent so workers find the data on disk
    for inst in sorted({t[0] for t in tasks}):
        qt.load_portfolio_instance(inst)
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_task, tasks):
            pass


def best_per_instance(dirs) -> dict[str, dict]:
    best = {}
    for d in dirs:
        for p in Path(d).glob("*.json"):
            r = json.loads(p.read_text())
            if "score" not in r or "instance" not in r:
                continue
            key = f"{r['instance']}_b{int(r['budget']):03d}_l{lam_tag(str(r['lam']))}"
            if key not in best or r["score"] < best[key]["score"]:
                best[key] = r
    return best


def cmd_report(a):
    bkv = load_bkv()
    best = best_per_instance(a.dirs)
    # include the current hand-in files too
    for p in Path("handin").glob("*.sol"):
        h, pos = qt.read_portfolio_solution(p)
        key = p.stem
        if key in best and not a.rescore_handin:
            pass
        model = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]),
                                  lam=h["lambda"])
        sc = model.score(pos)
        assert sc.feasible, p
        if key not in best or sc.objective < best[key]["score"]:
            best[key] = dict(score=int(sc.objective), solution_file=str(p), init="handin")
    rows = []
    for key in sorted(best, key=lambda k: (k.split("_")[1], k.split("_")[2], k)):
        r = best[key]
        bk = bkv.get(key)
        gap = (r["score"] - bk) / abs(bk) * 100 if bk is not None else None
        rows.append((key, r["score"], bk, gap, r["solution_file"]))
        g = f"{gap:+8.2f}%" if gap is not None else "       —"
        print(f"{key:34s} {r['score']:>12} {bk if bk is not None else '—':>12} {g}  {r['solution_file']}")
    n = sum(1 for r in rows if r[3] is not None)
    print(f"\n{len(rows)} instances; {n} with a BKV; "
          f"{sum(1 for r in rows if r[3] is not None and r[3] < -1e-9)} beat it, "
          f"{sum(1 for r in rows if r[3] is not None and abs(r[3]) <= 1e-9)} match, "
          f"{sum(1 for r in rows if r[3] is not None and r[3] > 1e-9)} worse")
    missing = sorted(set(bkv) - set(best))
    print(f"BKV instances without a file: {len(missing)}")
    if a.json:
        Path(a.json).write_text(json.dumps([dict(key=k, score=s, bkv=b, gap=g, file=f)
                                            for k, s, b, g, f in rows], indent=1))


def cmd_polish(a):
    """Warm-start every instance whose best file is above its BKV (or all, --all) from that file."""
    os.makedirs(a.out, exist_ok=True)
    bkv = load_bkv()
    best = best_per_instance(a.dirs)
    for p in Path("handin").glob("*.sol"):
        if p.stem in bkv and p.stem not in best:
            h, pos = qt.read_portfolio_solution(p)
            m = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]), lam=h["lambda"])
            best[p.stem] = dict(score=int(m.score(pos).objective), solution_file=str(p))
    tasks = []
    for key, r in sorted(best.items()):
        if key not in bkv or (not a.all and r["score"] <= bkv[key]):
            continue
        if a.match and not any(m in key for m in a.match):
            continue
        inst, rest = key.rsplit("_b", 1)
        b, lam = rest.split("_l")
        for s in a.seeds:
            tasks.append((inst, int(b), lam, s, a.time, r["solution_file"], a.out, a.stall, a.window))
    print(f"{len(tasks)} polish tasks on {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_task, tasks):
            pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("polish")
    q.add_argument("--dirs", nargs="+", default=["results/grid"])
    q.add_argument("--out", default="results/grid_polish")
    q.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    q.add_argument("--time", type=float, default=300)
    q.add_argument("--stall", type=float, default=120)
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--match", nargs="*", default=None, help="only keys containing one of these")
    q.add_argument("--all", action="store_true")
    q.add_argument("--window", type=int, default=2, help="longest multi-day move (days)")
    q.set_defaults(fn=cmd_polish)
    r = sub.add_parser("run")
    r.add_argument("--sizes", nargs="+", default=["a050_t15"])
    r.add_argument("--inst-seeds", nargs="+", default=SEEDS)
    r.add_argument("--lams", nargs="+", default=LAM_TAGS)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--time", type=float, default=180)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--out", default="results/grid")
    r.add_argument("--stall", type=float, default=45, help="stop a run after this long without a gain (0 = off)")
    r.add_argument("--window", type=int, default=2, help="longest multi-day move (days)")
    r.add_argument("--keys", nargs="*", default=None, help="only these instance keys (po_..._bBBB_lLAM)")
    r.add_argument("--warm", action="store_true", help="also run once from the hand-in file")
    r.add_argument("--only-bkv", action="store_true", default=True)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--dirs", nargs="+", default=["results/grid"])
    p.add_argument("--json", default=None)
    p.add_argument("--rescore-handin", action="store_true")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
