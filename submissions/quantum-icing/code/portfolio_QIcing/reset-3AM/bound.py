import sys, glob, numpy as np
sys.path.insert(0, '.')
import pf_core as pc, lns as L
for inst, lam, jules in [("po_a050_t15_orig","1e-2",-77193),("po_a050_t10_s00","1e-2",-24353),("po_a050_t10_s02","1e-2",-52358),("po_a050_t15_s02","1e-2",-74860)]:
    m = pc.make_model(inst, lam); fm = L.FastModel(m)
    best=None
    for f in glob.glob(f"solutions/{inst}_b{m.budget:03d}_l{lam}/*.sol"):
        v = pc.positions_to_v(m, pc.qt.read_portfolio_solution(f)[1]); o = fm.objective(v)
        if best is None or o < best[0]: best=(o,v,f)
    o,v,f = best
    ev = [np.linalg.eigvalsh(fm.R[t].astype(float)) for t in range(fm.T)]
    print(f"\n{inst} λ{lam}: ours {o}  Jules {jules}  minEig {min(e.min() for e in ev):.1f} maxEig {max(e.max() for e in ev):.0f}")
    tot_lb=0; tot_ours=0
    for t in range(fm.T):
        x = L.period_relaxation(fm, t)
        lb = float(L.stage_cost(fm, t, x[None,:].round(6)) [0]) if False else None
        # continuous objective value
        R=fm.R[t].astype(float); p=np.maximum(x,0); q=np.maximum(-x,0)
        val = x@R@x + fm.lin[t]@x + fm.absw[t]@(p+q) + fm.negw[t]@q + fm.const[t]
        s = int(L.stage_cost(fm, t, v[t])[0])
        tot_lb += val; tot_ours += s
        print(f"  t={t:2d} relaxLB {val:9.0f}  ours {s:8d}  held {int(np.abs(v[t]).sum()):2d}u/{int((v[t]!=0).sum()):2d}a   relax mass {np.abs(x).sum():5.1f} on {(np.abs(x)>0.05).sum()} assets")
    print(f"  SUM relaxLB {tot_lb:.0f}  ours-stage {tot_ours}  ours-trans {o-tot_ours}")
