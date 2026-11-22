"""Re-read every .sol in solutions/, re-score it with the official referee, and write
best_solutions.md: best quantum and best overall solution per instance vs QOBLIB BKV."""
from collections import defaultdict
from pathlib import Path

import pf_core as pc

qt = pc.qt
bkv = pc.load_bkv()
best = defaultdict(dict)  # stem -> {"quantum": (obj, path, method), "any": ...}
n_files = n_bad = 0
for path in sorted(pc.SOL_DIR.rglob("*.sol")):
    header, positions = qt.read_portfolio_solution(path)
    lam_tag = path.parent.name.split("_l")[-1]
    budget = int(header["budget"])
    model = pc.make_model(header["instance"], lam_tag, budget)
    score = model.score(positions)
    n_files += 1
    if not score.feasible or int(header.get("objective", score.objective)) != score.objective:
        n_bad += 1
        print("MISMATCH/INFEASIBLE", path, score)
        continue
    text = path.read_text()
    quantum = "quantum: True" in text
    method = next((ln.split("method:", 1)[1].strip() for ln in text.splitlines() if "method:" in ln), "?")
    key = (header["instance"], budget, lam_tag)
    for kind in ("any", "quantum") if quantum else ("any",):
        cur = best[key].get(kind)
        if cur is None or score.objective < cur[0]:
            best[key][kind] = (score.objective, path.relative_to(pc.HERE), method)

lines = ["# Best solutions per instance (re-scored by the official referee)", "",
         f"{n_files} .sol files checked, {n_bad} infeasible or with a wrong objective line.", "",
         "| instance | B | λ | BKV (status) | best quantum-pipeline | gap | file | best overall | gap |",
         "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
for key in sorted(best):
    inst, b, lam = key
    ref = bkv.get(key)
    def fmt(entry):
        if entry is None:
            return "—", "—", ""
        obj = entry[0]
        gap = f"{(obj - ref[0]) / abs(ref[0]):+.3%}" if ref else "no BKV"
        return str(obj), gap, f"`{entry[1]}`"
    q, qg, qf = fmt(best[key].get("quantum"))
    a, ag, _ = fmt(best[key].get("any"))
    lines.append(f"| {inst} | {b} | {lam} | {ref[0] if ref else '—'} ({ref[1] if ref else 'none'}) | {q} | {qg} | {qf} | {a} | {ag} |")
(pc.HERE / "best_solutions.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
