"""Experiment: period-decoupled PCE pools + exact chain DP, per formulation / code / decoder.

For each period t < T-1 (the last period is always empty) and each variant:
  PCE on the single-period formulation -> candidates (samples, expectation roundings)
  -> raw-feasible states only (no repair) -> pool_t (always contains the empty state)
Then the exact chain DP over the pools gives a full multi-period portfolio, scored by the referee.
Controls with the same candidate counts: random codewords (samples) and random sign vectors (expect.).

    python exp_period_pce.py --instance po_a010_t10_orig --lam 1e-4 --variants all
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import codes as cd
import formulations as fm
import pce_tools as pt

VARIANTS = {
    # name: (cell, constraints, recentre-on-empty)
    "unary/slack": ("unary", "slack", False),
    "unary/slack/rc": ("unary", "slack", True),
    "signmag/slack": ("signmag", "slack", False),
    "signmag/slack/rc": ("signmag", "slack", True),
    "signmag/unbal/rc": ("signmag", "unbalanced", True),
    "ternary/slack/rc": ("ternary", "slack", True),
    "ternsm/unbal/rc": ("ternsm", "unbalanced", True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a010_t10_orig")
ap.add_argument("--lam", default="1e-4")
ap.add_argument("--variants", default="all")
ap.add_argument("--code", default="poly", help="poly | multibasis2 | multibasis3")
ap.add_argument("--n-layers", type=int, default=3)
ap.add_argument("--pop", type=int, default=40)
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--optimizer", default="mc", help="mc | lbfgs | cobyla")
ap.add_argument("--penalty-scale", type=float, default=1.0, help="multiplies the auto penalty weight")
ap.add_argument("--pool", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--periods", default="")
ap.add_argument("--tag", default="")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
model = fm.pc.make_model(args.instance, args.lam)
probs = fm.period_problems(model)
T, n = model.n_periods, len(model.symbols)
bkv = fm.pc.load_bkv().get((args.instance, model.budget, args.lam), (None, ""))[0]
exact = n <= 10
if exact:
    v_opt, opt = fm.pc.exact_dp(model)
    V_all = fm.pc.enumerate_states(n, model.ub, model.budget, model.capital, 2**model.cash_slack_bits - 1)
    E_sorted = [np.sort(probs[t].energy(V_all)) for t in range(T)]
else:
    opt = bkv
print(f"{args.instance} λ={args.lam} n={n} B={model.budget}: reference {opt} ({'DP optimum' if exact else 'BKV'})")
names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
periods = [int(s) for s in args.periods.split(",")] if args.periods else list(range(T - 1))


def topk(V, E, k):
    ok = np.isfinite(E)
    V, E = V[ok], E[ok]
    o = np.argsort(E)[:k]
    return V[o], E[o]


def rank(t, e):
    return int(np.searchsorted(E_sorted[t], e)) if exact and np.isfinite(e) else None


results = {}
empty = np.zeros((1, n), dtype=np.int64)
for name in names:
    cell, cons, rc = VARIANTS[name]
    pools = {k: [] for k in ("samples", "expect", "both", "rand_code", "rand_sign", "ideal")}
    per = []
    t_run = time.time()
    circuits = 0
    for t in range(T):
        if t not in periods or t == T - 1:
            for k in pools:
                pools[k].append(empty)
            continue
        F0 = fm.PeriodFormulation(probs[t], cell=cell, constraints=cons)
        F = fm.PeriodFormulation(probs[t], cell=cell, constraints=cons,
                                 penalty_weight=F0.penalty_weight * args.penalty_scale) if args.penalty_scale != 1 else F0
        flip = cd.reference_flip(F.encode(empty[0])) if rc else None
        r = pt.run_pce(F, code=args.code, flip=flip, n_layers=args.n_layers, pop=args.pop, iters=args.iters,
                       alpha=args.alpha, seed=int(rng.integers(1 << 30)), optimizer=args.optimizer)
        circuits += r["circuits"]
        Vs, Es = pt.evaluate(F, r["samples"])
        Ve, Ee = pt.evaluate(F, r["expect"])
        # controls: same counts, uniform random over the code / over all sign vectors
        if args.code == "poly":
            masks, q = cd.poly_masks(F.n_vars)
            Zr = rng.integers(0, 2, size=(len(r["samples"]), q))
            M = cd.mask_matrix(masks, q)
            Xr = (1 - ((Zr @ M.T) & 1)).astype(np.uint8)
            if flip is not None:
                Xr ^= flip[None, :]
        else:
            Xr = rng.integers(0, 2, size=(len(r["samples"]), F.n_vars)).astype(np.uint8)
        Vrc, Erc = pt.evaluate(F, Xr)
        Vrs, Ers = pt.evaluate(F, rng.integers(0, 2, size=(len(r["expect"]), F.n_vars)).astype(np.uint8))
        Vid, Eid = pt.evaluate(F, pt.ideal_relaxation(F, flip, alpha=args.alpha, seed=int(rng.integers(1 << 30))))
        rec = dict(t=t, qubits=r["qubits"], vars=F.n_vars, sec=round(r["sec"], 1))
        for key, (V, E) in dict(samples=(Vs, Es), expect=(Ve, Ee), rand_code=(Vrc, Erc), rand_sign=(Vrs, Ers), ideal=(Vid, Eid)).items():
            Vk, Ek = topk(V, E, args.pool)
            pools[key].append(np.vstack([empty, Vk]))
            b = float(Ek[0]) if len(Ek) else float("inf")
            rec[key] = dict(best=b, rank=rank(t, b), n_feas=int(np.isfinite(E).sum()), n=len(E))
        Vb = np.vstack([Vs, Ve])
        Eb = np.concatenate([Es, Ee])
        Vk, Ek = topk(Vb, Eb, args.pool)
        pools["both"].append(np.vstack([empty, Vk]))
        per.append(rec)
        print(f"  {name:18s} t={t} {F.n_vars}v q={r['qubits']} {r['sec']:4.1f}s | " + " | ".join(
            f"{k}: best {rec[k]['best']:>8.0f} r{rec[k]['rank']} feas {rec[k]['n_feas']}/{rec[k]['n']}"
            for k in ("samples", "expect", "rand_code", "rand_sign", "ideal")) +
            (f" | opt {E_sorted[t][0]:.0f}" if exact else ""), flush=True)
    finals = {}
    for key, pl in pools.items():
        v, obj_chain = fm.chain_dp(model, pl)
        sc = fm.pc.score_v(model, v)
        assert sc.feasible and sc.objective == obj_chain, (sc, obj_chain)
        finals[key] = sc.objective
    gap = {k: (f"{(o - opt) / abs(opt):+.2%}" if opt else "") for k, o in finals.items()}
    print(f"== {name}: chained objective " + "  ".join(f"{k} {finals[k]} ({gap[k]})" for k in finals) +
          f"   [{time.time() - t_run:.0f}s, {circuits} circuits]", flush=True)
    results[name] = dict(per_period=per, final=finals, gap=gap, circuits=circuits)

out = fm.HERE / "logs" / f"period_pce_{args.instance}_l{args.lam}_{args.code}_s{args.seed}{args.tag}.json"
out.write_text(json.dumps(dict(args=vars(args), reference=opt, results=results), indent=1, default=float))
print("wrote", out)
