"""Exact DP (classical) on every 10-asset QOBLIB portfolio instance; compare with BKV.

Gives the proven optimum of the official model for all a010 instances: the ground
truth our quantum pipeline is measured against. Writes logs/dp_all.csv and records
every optimum in solutions/ + results.csv.
"""
import csv
import time

import pf_core as pc

bkv = pc.load_bkv()
rows = []
for T in (10, 15):
    for seed in ("orig", "s00", "s01", "s02"):
        inst = f"po_a010_t{T}_{seed}"
        for lam in pc.LAM_TAGS:
            model = pc.make_model(inst, lam)
            t0 = time.time()
            v, obj = pc.exact_dp(model)
            score = pc.score_v(model, v)
            assert score.feasible and score.objective == obj, (inst, lam, obj, score)
            ref, status = bkv[(inst, 4, lam)]
            verdict = "BEATS BKV" if obj < ref else ("= BKV" if obj == ref else "WORSE?!")
            print(f"{inst} λ={lam:>5}: DP optimum {obj:>8}  BKV {ref:>8} ({status})  {verdict}  "
                  f"[{time.time() - t0:.1f}s]", flush=True)
            pc.record_solution(model, pc.v_to_positions(model, v), method="exact-DP (classical)",
                               settings="reduced-state DP over periods, v=L-S", tag="dp")
            rows.append([inst, 4, lam, obj, ref, status, obj - ref, verdict])
with open(pc.HERE / "logs" / "dp_all.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["instance", "budget", "lambda", "dp_optimum", "bkv", "bkv_status", "diff", "verdict"])
    w.writerows(rows)
