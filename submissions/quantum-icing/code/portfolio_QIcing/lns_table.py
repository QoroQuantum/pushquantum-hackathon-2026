"""Markdown table of LNS results from logs/lns_*.txt (best per instance/λ)."""
import re
import sys
from pathlib import Path

best = {}
for f in sorted(Path("logs").glob("lns_a*.txt")):
    for line in f.read_text().splitlines():
        m = re.match(r"(\S+) λ=\s*(\S+):\s+(-?\d+) \(pool-DP (-?\d+).*BKV (\S+) gap (\S+(?: BKV)?)\s*(.*?)\s*\[(\d+)s\]", line)
        if not m:
            continue
        inst, lam, obj, pool, bkv, gap, verdict, sec = m.groups()
        key = (inst, lam)
        if key not in best or int(obj) < best[key][0]:
            best[key] = (int(obj), int(pool), bkv, gap, verdict, sec)
print("| instance | λ | ours | pool-DP only | BKV | gap vs BKV | |")
print("|---|---|---:|---:|---:|---:|---|")
order = {"1e-2": 0, "1e-3": 1, "5e-4": 2, "1e-4": 3, "5e-5": 4, "1e-5": 5, "1e-6": 6, "0": 7}
for (inst, lam), (obj, pool, bkv, gap, verdict, sec) in sorted(best.items(), key=lambda kv: (kv[0][0][:11], order.get(kv[0][1], 9), kv[0][0])):
    mark = "**BEATS BKV**" if "BEATS" in verdict else ("= BKV" if "=" in verdict else "")
    print(f"| {inst} | {lam} | {obj} | {pool} | {bkv} | {gap} | {mark} |")
