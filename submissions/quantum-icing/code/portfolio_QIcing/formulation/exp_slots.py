"""Experiment: slot formulation (constraint-free, log-encoded) + PCE per period + exact chain DP.

    python exp_slots.py --instance po_a010_t10_orig --lam 1e-4 --order spectral --optimizer mc
Candidate sources per period (no repair anywhere; slot bitstrings are always budget-feasible):
  samples (final-circuit shots), expect (sign rounding of every evaluated expectation vector),
  rand (same number of uniform random bitstrings), ideal (classical: same loss, z unconstrained).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import formulations as fm
import pce_tools as pt
import slots as sl

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a010_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--order", default="spectral")
ap.add_argument("--slots", type=int, default=0, help="0 = budget B")
ap.add_argument("--optimizer", default="mc")
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--pop", type=int, default=40)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--qubits", type=int, default=0)
ap.add_argument("--code", default="poly")
ap.add_argument("--cap-weight", type=float, default=1.0)
ap.add_argument("--dup-weight", type=float, default=0.0)
ap.add_argument("--pool", type=int, default=20)
ap.add_argument("--repeats", type=int, default=1, help="independent PCE runs per period (pooled)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--periods", default="")
ap.add_argument("--no-ideal", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--polish", action="store_true", help="L1 period descent + L2 multi-period LNS, every source")
ap.add_argument("--polish-pool", type=int, default=60, help="raw states per period sent to period descent")
ap.add_argument("--lns-time", type=float, default=60.0, help="L2 LNS seconds per source (equal for all)")
ap.add_argument("--record", action="store_true", help="record the best PCE-source .sol (hybrid label)")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
model = fm.pc.make_model(args.instance, args.lam)
probs = fm.period_problems(model)
T, n = model.n_periods, len(model.symbols)
bkv = fm.pc.load_bkv().get((args.instance, model.budget, args.lam), (None, ""))[0]
exact = n <= 10
if exact:
    _, ref = fm.pc.exact_dp(model)
    V_all = fm.pc.enumerate_states(n, model.ub, model.budget, model.capital, 2**model.cash_slack_bits - 1)
    E_sorted = [np.sort(probs[t].energy(V_all)) for t in range(T)]
else:
    ref = bkv
# per-period reference: the stage energies of the best .sol we hold for this instance (classical LNS)
import glob
best_v, best_o = None, None
for f in glob.glob(str(fm.PF / "solutions" / f"{args.instance}_b{model.budget:03d}_l{args.lam}" / "*.sol")):
    vv = fm.pc.positions_to_v(model, fm.pc.qt.read_portfolio_solution(f)[1])
    o = fm.pc.score_v(model, vv).objective
    if best_o is None or o < best_o:
        best_v, best_o = vv, o
ref_stage = [float(probs[t].energy(best_v[t])[0]) for t in range(T)] if best_v is not None else None
if best_v is not None:
    print(f"best held solution {best_o}; per-period stage energies {[round(e) for e in ref_stage]}")
print(f"{args.instance} λ={args.lam} n={n} B={model.budget}: reference {ref} ({'DP optimum' if exact else 'BKV'}) "
      f"| {vars(args)}", flush=True)
periods = [int(s) for s in args.periods.split(",")] if args.periods else list(range(T - 1))
empty = np.zeros((1, n), dtype=np.int64)
keys = ["samples", "expect", "both", "rand"] + ([] if args.no_ideal else ["ideal"])
pools = {k: [] for k in keys}
raw = {k: [] for k in keys}  # up to polish-pool best raw feasible states per period, for L1
per, circuits, t_run, max_q = [], 0, time.time(), 0


def cand(P, M, X):
    if len(X) == 0:
        return empty, np.array([P.energy(empty)[0]])
    V = np.unique(M.decode(X), axis=0)
    E = P.energy(V).astype(float)
    E[~P.feasible(V)] = np.inf  # only the capital rule can fail (a050)
    return V, E


def topk(V, E, k):
    ok = np.isfinite(E)
    V, E = V[ok], E[ok]
    o = np.argsort(E)[:k]
    return np.vstack([empty, V[o]]), (E[o] if len(o) else np.array([np.inf]))


for t in range(T):
    if t not in periods or t == T - 1:
        for k in keys:
            pools[k].append(empty)
            raw[k].append(empty)
        continue
    P = probs[t]
    M = sl.SlotModel(P, n_slots=args.slots or None, order=args.order, cap_weight=args.cap_weight,
                     dup_weight=args.dup_weight, rng=rng)
    Xs, Xe, sec, q = [], [], 0.0, 0
    for r_ in range(args.repeats):
        r = pt.run_pce_slots(M, n_layers=args.n_layers, pop=args.pop, iters=args.iters, alpha=args.alpha,
                             seed=int(rng.integers(1 << 30)), optimizer=args.optimizer,
                             n_qubits=args.qubits or None, code=args.code)
        Xs.append(r["samples"]); Xe.append(r["expect"]); sec += r["sec"]; q = r["qubits"]
        circuits += r["circuits"]
    max_q = max(max_q, q)
    Xs, Xe = np.vstack(Xs), np.vstack(Xe)
    src = dict(samples=cand(P, M, Xs), expect=cand(P, M, Xe),
               rand=cand(P, M, rng.integers(0, 2, size=(len(Xs) + len(Xe), M.n_vars))))
    if not args.no_ideal:
        src["ideal"] = cand(P, M, pt.ideal_slots(M, alpha=args.alpha, seed=int(rng.integers(1 << 30))))
    src["both"] = (np.vstack([src["samples"][0], src["expect"][0]]), np.concatenate([src["samples"][1], src["expect"][1]]))
    rec = dict(t=t, vars=M.n_vars, qubits=q, sec=round(sec, 1))
    line = []
    for k in keys:
        Vk, Ek = topk(*src[k], args.pool)
        pools[k].append(Vk)
        raw[k].append(topk(*src[k], args.polish_pool)[0])
        rk = int(np.searchsorted(E_sorted[t], Ek[0])) if exact and np.isfinite(Ek[0]) else None
        rec[k] = dict(best=float(Ek[0]), rank=rk, n_unique=len(src[k][0]))
        line.append(f"{k} {Ek[0]:>9.0f}" + (f" r{rk}" if exact else "") + f" ({len(src[k][0])}u)")
    per.append(rec)
    print(f"  t={t} {M.n_vars}v q={q} {sec:5.1f}s | " + " | ".join(line) +
          (f" | opt {E_sorted[t][0]:.0f}" if exact else (f" | best-held {ref_stage[t]:.0f}" if ref_stage else "")), flush=True)

finals = {}
for k in keys:
    v, o = fm.chain_dp(model, pools[k])
    sc = fm.pc.score_v(model, v)
    assert sc.feasible and sc.objective == o
    finals[k] = (sc.objective, v)
gap = {k: (f"{(o - ref) / abs(ref):+.3%}" if ref else "") for k, (o, _) in finals.items()}
print("== chained objective: " + "  ".join(f"{k} {finals[k][0]} ({gap[k]})" for k in keys) +
      f"   [{time.time() - t_run:.0f}s, {circuits} circuits, ≤{max_q} qubits]", flush=True)
# ── local search (classical), applied identically to every source ────────
polished = {}
if args.polish:
    import sys as _s
    _s.path.insert(0, str(fm.PF))
    import lns as L
    fmod = L.FastModel(model)
    l1 = {}
    for k in keys + ["empty"]:
        src_pools = raw[k] if k != "empty" else [empty] * T
        pl = []
        for t in range(T):
            if t == T - 1:
                pl.append(empty)
                continue
            D = np.unique(np.array([L.period_descent(fmod, t, x) for x in src_pools[t]]), axis=0)
            E = probs[t].energy(D).astype(float)
            E[~probs[t].feasible(D)] = np.inf
            pl.append(topk(D, E, args.pool)[0])
        v1, o1 = fm.chain_dp(model, pl)
        assert fm.pc.score_v(model, v1).objective == o1
        l1[k] = (o1, v1)
    print("== L1 (period descent on each pool state, then chain DP): " + "  ".join(
        f"{k} {o} ({(o - ref) / abs(ref):+.3%})" for k, (o, _) in l1.items()), flush=True)
    l2 = {}
    for k in ["both", "rand"] + (["ideal"] if "ideal" in keys else []) + ["empty"]:
        v2, o2, its = L.lns(fmod, v0=l1[k][1], time_limit=args.lns_time, seed=args.seed)
        sc = fm.pc.score_v(model, v2)
        assert sc.feasible and sc.objective == o2
        l2[k] = (o2, v2)
    print(f"== L2 (+{args.lns_time:.0f}s multi-period LNS from L1): " + "  ".join(
        f"{k} {o} ({(o - ref) / abs(ref):+.3%})" for k, (o, _) in l2.items()), flush=True)
    polished = dict(L1={k: o for k, (o, _) in l1.items()}, L2={k: o for k, (o, _) in l2.items()})
    if args.record:
        o2, v2 = l2["both"]
        settings = (f"slot formulation K={args.slots or model.budget} order={args.order}, PCE {args.code} "
                    f"n_layers={args.n_layers} alpha={args.alpha} optimizer={args.optimizer} iters={args.iters} "
                    f"repeats={args.repeats} pool={args.pool}; L1 period descent on {args.polish_pool} states/period; "
                    f"L2 lns {args.lns_time:.0f}s; circuits={circuits}; raw={finals['both'][0]} L1={l1['both'][0]}; "
                    f"controls rand L2={l2['rand'][0]} empty L2={l2['empty'][0]}")
        _, path, _ = fm.pc.record_solution(model, fm.pc.v_to_positions(model, v2),
                                           method="slot-PCE pools + period descent + LNS polish (hybrid)",
                                           settings=settings, n_qubo_vars=f"{M.n_vars}/period", max_qubits=max_q,
                                           quantum=True, tag="slotpce-ls")
        print("recorded", path, flush=True)

np.savez_compressed(fm.HERE / "logs" / f"pools_{args.instance}_l{args.lam}_s{args.seed}{args.tag}.npz",
                    **{f"{k}_t{t}": raw[k][t] for k in keys for t in range(T)})
stem = f"slots_{args.instance}_l{args.lam}_{args.order}_{args.optimizer}_{args.code}_s{args.seed}{args.tag}"
(fm.HERE / "logs" / f"{stem}.json").write_text(json.dumps(dict(
    args=vars(args), reference=ref, per_period=per, final={k: o for k, (o, _) in finals.items()}, gap=gap,
    circuits=circuits, max_qubits=max_q, polished=polished, v={k: v.tolist() for k, (_, v) in finals.items()}), indent=1, default=float))
