"""Stronger per-day pool (randomised rounding of the relaxation + pair descent + ILS, entry-cost
emulation e=0,1) + our current best's day vectors -> chain DP -> LNS polish. Not recorded."""
import sys, glob, time, numpy as np
sys.path.insert(0, '.')
import pf_core as pc, lns as L
inst, lam, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = pc.make_model(inst, lam); fm = L.FastModel(m)
best = None
for f in glob.glob(f"solutions/{inst}_b{m.budget:03d}_l{lam}/*.sol"):
    v = pc.positions_to_v(m, pc.qt.read_portfolio_solution(f)[1]); o = fm.objective(v)
    if best is None or o < best[0]: best = (o, v)
print("current best", best[0], flush=True)
pool = [np.zeros(fm.n, dtype=np.int64)] + [x for x in best[1]]
t0 = time.time()
for t in range(fm.T - 1):
    rng = np.random.default_rng(100 + t)
    for e in (0.0, 1.0):
        ea = np.round(e * m.tc[t]).astype(np.int64)
        xc = L.period_relaxation(fm, t, ea)
        cost = lambda x: int(L.stage_cost(fm, t, x)[0] + ea @ np.abs(x))
        starts = [L.period_descent(fm, t, L._repair_period(fm, np.floor(xc + rng.random(fm.n)).astype(np.int64)), ea) for _ in range(150)]
        cand = starts + L.period_ils(fm, t, starts, extra_abs=ea, n_kicks=600, seed=t)
        uniq = {x.tobytes(): x for x in cand}
        top = sorted(uniq.values(), key=cost)[:40]
        pool += top
        print(f"  t={t:2d} e={e:.0f} best stage {int(L.stage_cost(fm,t,top[0])[0]):8d}  [{time.time()-t0:.0f}s]", flush=True)
v, obj = L.chain_dp(fm, pool)
print(f"chain DP over {len(pool)} vectors: {obj}  [{time.time()-t0:.0f}s]", flush=True)
v2, o2, its = L.lns(fm, v0=v, time_limit=120, seed=0)
print(f"after LNS 120s: {o2}  referee {pc.score_v(m, v2).objective} feasible {pc.score_v(m, v2).feasible}", flush=True)
np.save(out, v2)
