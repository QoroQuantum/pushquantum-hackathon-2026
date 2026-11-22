"""Collect pt/results/*.json into a markdown table (gap vs QOBLIB BKV)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gap(x, b):
    return f"{(x - b) / abs(b):+.2%}" if b else "n/a"


rows = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("*.json"))
        if not p.name.startswith(("pilot", "sanity"))]
rows.sort(key=lambda r: (r["instance"], r["algo"], r["start"]))
print("| instance | λ | algo | start | raw (gap) | after LNS polish (gap) | prev best (gap) | BKV | wall s | swap rate min–med–max | held days/assets |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
for r in rows:
    b = r["bkv"]
    sw = sorted(r.get("swap_rate", []))
    swt = f"{sw[0]:.2f}–{sw[len(sw) // 2]:.2f}–{sw[-1]:.2f}" if sw else "—"
    algo = r["algo"] + (" λ-temp" if r.get("ss") and min(r["ss"]) < 1 else "")
    mark = " **≤prev**" if r["prev_best"] is not None and r["polished"] <= r["prev_best"] else ""
    if r["prev_best"] is not None and r["polished"] < r["prev_best"]:
        mark = " **NEW BEST**"
    print(f"| {r['instance']} | {r['lam']} | {algo} | {r['start']} | {r['raw']} ({gap(r['raw'], b)}) | "
          f"{r['polished']} ({gap(r['polished'], b)}){mark} | {r['prev_best']} ({gap(r['prev_best'], b)}) | {b} | "
          f"{r['t_total']:.0f} | {swt} | {r['held_days']}/{r['held_assets']} |")
