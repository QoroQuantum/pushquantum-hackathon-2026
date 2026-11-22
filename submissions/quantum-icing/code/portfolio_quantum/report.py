"""Verify every quantum .sol with the official referee and tabulate it against QOBLIB and the
repo's earlier classical results.

    python portfolio_quantum/report.py            # prints a markdown table
    python portfolio_quantum/report.py --copy D   # also copies the best file per config to D/<stem>.sol
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qlns_core as qc  # noqa: E402

ps, qt = qc.ps, qc.qt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "solutions"))
    ap.add_argument("--copy", default="")
    a = ap.parse_args()
    bkv = ps.bkv_table()
    baseline = json.loads((HERE / "classical_baseline.json").read_text())  # repo's classical bests (snapshot)
    rows = []
    for d in sorted(Path(a.dir).iterdir()):
        if not d.is_dir():
            continue
        best = None
        for f in sorted(d.glob("*.sol")):
            header, pos = qt.read_portfolio_solution(f)
            model = qt.PortfolioModel(qt.load_portfolio_instance(header["instance"]),
                                      budget=int(header["budget"]), lam=header["lambda"])
            sc = model.score(pos)
            written = int(header.get("objective", sc.objective))
            if not sc.feasible or written != sc.objective:
                print(f"!! {f.name}: feasible={sc.feasible} written={written} referee={sc.objective}")
                continue
            meta = {}
            mj = f.with_suffix(".log.json")
            if mj.exists():
                meta = json.loads(mj.read_text())
            if best is None or sc.objective < best[0]:
                best = (sc.objective, f, meta, header)
        if best is None:
            continue
        obj, f, meta, header = best
        stem = d.name
        inst = header["instance"]
        p = ps.Problem(inst, ps.LAMS[[float(x) for x in ps.LAMS].index(float(header["lambda"]))])
        ref = bkv.get(p.bkv_key)
        prev = baseline.get(stem)
        settings = meta.get("settings", "")
        qubits = next((int(t.split("=")[1]) for t in settings.split() if t.startswith("max_qubits=")), None)
        start = next((t.split("=")[1] for t in settings.split() if t.startswith("start=")), "?")
        if "paired-transfer" in settings:  # paired_transfer_run.py: <= 16 transfer qubits, start = a repo file
            qubits, start = 16, "best"
        rows.append(dict(stem=stem, inst=inst, B=p.budget, lam=header["lambda"], vars=p.qubo_vars, obj=obj,
                         bkv=ref[0] if ref else None, status=ref[1] if ref else "none", prev=prev, file=f,
                         qubits=qubits, start=start))
        if a.copy:
            Path(a.copy).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, Path(a.copy) / f"{stem}.sol")
            if f.with_suffix(".log.json").exists():
                shutil.copy(f.with_suffix(".log.json"), Path(a.copy) / f"{stem}.log.json")

    def gap(x, r):
        return f"{(x - r) / abs(r):+.2%}" if r else "—"

    print("| Instance | B | λ | Binary variables | Largest QAOA (qubits) | Start | Quantum LNS | QOBLIB best known | Gap | Earlier classical (repo) | vs classical |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['inst']}` | {r['B']} | {r['lam']} | {r['vars']:,} | {r['qubits'] or '—'} | {r['start']} | {r['obj']:,} | "
              f"{(f'{int(r['bkv']):,}' if r['bkv'] else '—')} | {gap(r['obj'], r['bkv'])} | "
              f"{(f'{r['prev']:,}' if r['prev'] is not None else '—')} | {(f'{r['obj'] - r['prev']:+,}' if r['prev'] is not None else '—')} |")


if __name__ == "__main__":
    main()
