import sys, time, numpy as np
sys.path.insert(0, '.')
import pf_core as pc, lns as L
inst, lam = sys.argv[1], sys.argv[2]; days = [int(d) for d in sys.argv[3].split(',')]
m = pc.make_model(inst, lam); fm = L.FastModel(m)
for t in days:
    t0=time.time()
    cands = L.period_candidates_qp(fm, t, extra=(0.0,))
    base = min(int(L.stage_cost(fm,t,x)[0]) for x in cands)
    best = base; bx=None
    rng = np.random.default_rng(t)
    xc = L.period_relaxation(fm, t)
    # randomized rounding of relaxation + descent, then ILS
    starts=[]
    for k in range(300):
        x = np.floor(xc + rng.random(fm.n)).astype(np.int64)
        x = L.period_descent(fm, t, L._repair_period(fm, x))
        c = int(L.stage_cost(fm,t,x)[0]); starts.append(x)
        if c < best: best, bx = c, x
    ils = L.period_ils(fm, t, starts, n_kicks=1500, seed=t)
    for x in ils:
        c=int(L.stage_cost(fm,t,x)[0])
        if c<best: best,bx=c,x
    print(f"t={t:2d} QP-pool(e=0) {base:8d}   rand-round+descent+ILS {best:8d}  held {int(np.abs(bx).sum()) if bx is not None else '-'}u  {time.time()-t0:.0f}s", flush=True)
