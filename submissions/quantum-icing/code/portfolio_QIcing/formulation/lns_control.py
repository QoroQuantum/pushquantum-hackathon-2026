"""Control: plain classical LNS (lns.lns) from the same start, same wall time."""
import argparse, glob, sys, time
import numpy as np
import formulations as fm
sys.path.insert(0, str(fm.PF))
import lns as L
ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="po_a050_t10_s00"); ap.add_argument("--lam", default="1e-3")
ap.add_argument("--time-limit", type=float, default=1200); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
m = fm.pc.make_model(a.instance, a.lam); f_ = L.FastModel(m)
best = None
for f in glob.glob(str(fm.PF / "solutions" / f"{a.instance}_b{m.budget:03d}_l{a.lam}" / "*.sol")):
    vv = fm.pc.positions_to_v(m, fm.pc.qt.read_portfolio_solution(f)[1]); o = fm.pc.score_v(m, vv).objective
    if best is None or o < best[0]: best = (o, vv, f)
print("start", best[0], best[2], flush=True)
v, o, its = L.lns(f_, v0=best[1], time_limit=a.time_limit, seed=a.seed, log=lambda s: print(s, flush=True))
print(f"FINAL {o} (start {best[0]}) its {its}", flush=True)
