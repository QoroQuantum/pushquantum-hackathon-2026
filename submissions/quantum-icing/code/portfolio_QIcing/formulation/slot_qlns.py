"""slot-QLNS: large-neighbourhood search whose move is a joint re-choice of k units in one period, by PCE.

Move: pick period t, remove r <= k held units (random or a correlated cluster), keep the rest as a fixed
base; k free slots (log-encoded asset/sign or nothing) choose up to k new units jointly.  Neighbourhood
size ~ (2n+1)^k  (a050, k=6: ~1e12) on k*b = 42 variables -> MultiBasis-2 PCE on 6 qubits.
Candidates (shots + expectation roundings) are scored on the EXACT multi-period objective (transaction
costs included); optional period descent on each candidate (--inner-descent); accept if better.

Sub-solver controls with the same loop / candidate counts: --solver rand (uniform bitstrings),
--solver ideal (same loss, z unconstrained, classical).

    python slot_qlns.py --instance po_a050_t10_s00 --lam 1e-4 --start <sol> --time-limit 1200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np

import formulations as fm
import pce_tools as pt
import slots as sl

sys.path.insert(0, str(fm.PF))
import lns as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a050_t10_s00")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--start", default="best", help="'best' (best held .sol), 'empty', or a .sol path")
ap.add_argument("--solver", default="pce", choices=["pce", "rand", "ideal"])
ap.add_argument("--k", type=int, default=6, help="free slots per move")
ap.add_argument("--code", default="multibasis2")
ap.add_argument("--n-layers", type=int, default=6)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--alpha", type=float, default=6.0)
ap.add_argument("--restarts", type=int, default=1)
ap.add_argument("--inner-descent", action="store_true")
ap.add_argument("--p-cluster", type=float, default=0.5)
ap.add_argument("--time-limit", type=float, default=1200)
ap.add_argument("--max-moves", type=int, default=100000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--record", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
model = fm.pc.make_model(args.instance, args.lam)
probs = fm.period_problems(model)
fmod = L.FastModel(model)
T, n = model.n_periods, len(model.symbols)
bkv = fm.pc.load_bkv().get((args.instance, model.budget, args.lam), (None, ""))[0]

if args.start == "empty":
    v = np.zeros((T, n), dtype=np.int64)
elif args.start == "best":
    best_o = None
    for f in glob.glob(str(fm.PF / "solutions" / f"{args.instance}_b{model.budget:03d}_l{args.lam}" / "*.sol")):
        vv = fm.pc.positions_to_v(model, fm.pc.qt.read_portfolio_solution(f)[1])
        o = fm.pc.score_v(model, vv).objective
        if best_o is None or o < best_o:
            v, best_o, start_file = vv, o, f
    args.start = start_file
else:
    v = fm.pc.positions_to_v(model, fm.pc.qt.read_portfolio_solution(args.start)[1])
cur = fmod.objective(v)
start_obj = cur
assert cur == fm.pc.score_v(model, v).objective
gap = lambda o: f"{(o - bkv) / abs(bkv):+.3%}" if bkv else "n/a"
print(f"{args.instance} λ={args.lam} B={model.budget}: start {cur} (BKV {bkv}, {gap(cur)}) | solver={args.solver} "
      f"k={args.k} inner_descent={args.inner_descent} | start={args.start}", flush=True)

R_all = model.risk.astype(float)


def choose_removal(t, x):
    held = np.flatnonzero(x)
    units = [i for i in held for _ in range(abs(int(x[i])))]
    if not units:
        return x.copy()
    r = int(rng.integers(1, min(args.k, len(units)) + 1))
    if rng.random() < args.p_cluster:  # correlated cluster around a seed asset
        seed = int(rng.choice(held))
        d = np.sqrt(np.maximum(np.diag(R_all[t]), 1e-12))
        corr = np.abs(R_all[t][seed]) / (d[seed] * d)
        order = sorted(units, key=lambda i: -corr[i] + 1e-6 * rng.random())
        rem = order[:r]
    else:
        rem = list(rng.choice(units, size=r, replace=False))
    base = x.copy()
    for i in rem:
        base[i] -= np.sign(base[i])
    return base


log, t0, circuits, n_acc, it = [], time.time(), 0, 0, 0
max_q = 0
while time.time() - t0 < args.time_limit and it < args.max_moves:
    it += 1
    t = int(rng.integers(0, T - 1))
    base = choose_removal(t, v[t])
    K = min(args.k, model.budget - int(np.abs(base).sum()))
    if K <= 0:
        continue
    M = sl.SlotModel(probs[t], n_slots=K, base=base, rng=rng)
    tq = time.time()
    if args.solver == "pce":
        X = []
        for _ in range(args.restarts):
            r = pt.run_pce_slots(M, n_layers=args.n_layers, iters=args.iters, alpha=args.alpha, optimizer="lbfgs",
                                 code=args.code, seed=int(rng.integers(1 << 30)))
            X += [r["samples"], r["expect"]]
            circuits += r["circuits"]
            max_q = max(max_q, r["qubits"])
        X = np.vstack(X)
    elif args.solver == "ideal":
        X = pt.ideal_slots(M, alpha=args.alpha, starts=4, steps=150, seed=int(rng.integers(1 << 30)))
    else:
        X = rng.integers(0, 2, size=(300, M.n_vars)).astype(np.uint8)
    Vt = np.unique(M.decode(X), axis=0)
    if args.inner_descent:
        Vt = np.unique(np.array([L.period_descent(fmod, t, x) for x in Vt]), axis=0)
    Vt = Vt[probs[t].feasible(Vt)]
    best_here, best_x = None, None
    for x in Vt:
        old = v[t].copy()
        v[t] = x
        o = fmod.objective(v)
        v[t] = old
        if best_here is None or o < best_here:
            best_here, best_x = o, x
    acc = best_here is not None and best_here < cur
    if acc:
        v[t] = best_x
        cur = best_here
        n_acc += 1
    log.append(dict(it=it, t=t, K=K, removed=int(np.abs(v[t]).sum() - np.abs(base).sum()) if not acc else None,
                    n_cand=len(Vt), cand=best_here, cur=cur, sec=round(time.time() - tq, 2)))
    if acc or it % 25 == 0:
        print(f"it {it:4d} t={t} K={K} cand {len(Vt):4d} best {best_here} {'ACCEPT' if acc else '      '} "
              f"cur {cur} ({gap(cur)}) [{time.time() - t0:.0f}s]", flush=True)

sc = fm.pc.score_v(model, v)
assert sc.feasible and sc.objective == cur
print(f"FINAL {cur} (start {start_obj}, BKV {bkv}, {gap(cur)}), accepted {n_acc}/{it}, circuits {circuits}, "
      f"≤{max_q} qubits, {time.time() - t0:.0f}s", flush=True)
stem = f"slotqlns_{args.instance}_l{args.lam}_{args.solver}_k{args.k}{'_ld' if args.inner_descent else ''}_s{args.seed}{args.tag}"
(fm.HERE / "logs" / f"{stem}.json").write_text(json.dumps(dict(args=vars(args), start=start_obj, final=cur, bkv=bkv,
                                                               accepted=n_acc, moves=it, circuits=circuits, log=log),
                                                          indent=1, default=float))
if args.record and cur < start_obj:
    settings = (f"slot-QLNS solver={args.solver} k={args.k} code={args.code} n_layers={args.n_layers} "
                f"alpha={args.alpha} iters={args.iters} restarts={args.restarts} inner_descent={args.inner_descent} "
                f"moves={it} accepted={n_acc} circuits={circuits} start={args.start} ({start_obj})")
    _, path, _ = fm.pc.record_solution(model, fm.pc.v_to_positions(model, v),
                                       method=f"slot-QLNS ({args.solver} sub-solver" +
                                              (", + period descent)" if args.inner_descent else ")"),
                                       settings=settings, n_qubo_vars=f"{args.k * 7}/move", max_qubits=max_q,
                                       quantum=args.solver == "pce", tag=f"slotqlns-{args.solver}")
    print("recorded", path, flush=True)
