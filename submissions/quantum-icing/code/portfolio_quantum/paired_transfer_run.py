"""Run the paired-transfer QAOA search (paired_trajectory_qaoa.py) on any portfolio configuration.

One qubit = one transfer of up to 3 units between two assets on the same side of one day, so every
bitstring keeps the day budget and the net position.  The QAOA (p = 3, local statevector simulation in
paired_trajectory_qaoa.evolve) samples combinations of disjoint transfers; the best sample is kept if
it lowers the official objective.  This driver only generalises the instance (the original is fixed to
a400_t15 l1e-3), starts from the best .sol already in the repo and adds a time limit.

    python portfolio_quantum/paired_transfer_run.py --inst po_a050_t10_s01 --lam 1e-3 --minutes 35
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_trajectory_qaoa as pq  # noqa: E402
import qlns_core as qc  # noqa: E402

ps = qc.ps
ROOT = qc.ROOT


def best_start(p):
    stem = p.stem
    cands = [ROOT / "deliverables" / "portfolio" / f"{stem}.sol"]
    cands += list((ROOT / "portfolio_QIcing" / "solutions").glob(f"{stem}/*.sol"))
    cands += list((ROOT / "portfolio_quantum" / "solutions").glob(f"{stem}/*.sol"))
    cands += list((ROOT / "deliverables" / "portfolio_quantum").glob(f"{stem}*.sol"))
    best = None
    for c in cands:
        if not c.exists():
            continue
        v = qc.load_v(p, c)
        if p.feasible(v) and (best is None or p.objective(v) < best[0]):
            best = (p.objective(v), v, c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    p = ps.Problem(a.inst, a.lam)
    obj0, v0, src = best_start(p)
    scale = p.budget / 100.0  # the original kick/noise sizes were tuned for B = 100
    rng = np.random.default_rng(a.seed)
    elite = qc.State(p, v0)
    print(f"{p.stem}: start {obj0} from {src.relative_to(ROOT)}", flush=True)
    work = HERE / "solutions" / p.stem
    work.mkdir(parents=True, exist_ok=True)
    npy = work / f"_paired_running_seed{a.seed}.npy"
    t_end = time.time() + 60 * a.minutes
    circuits, k, log = 0, 0, []
    while time.time() < t_end:
        for t in rng.permutation(p.T - 1):
            if time.time() >= t_end:
                break
            s = elite
            for j in range(4):
                pert = 0 if (k % 5 < 2 or j) else rng.choice([750, 1800, 3000, 7000]) * scale
                noise = rng.choice([70, 220, 540, 1100, 2300]) * scale if not pert else np.sqrt(pert) * 14
                tilt = rng.normal(size=p.n) @ p.RS[t]
                tilt = tilt * (rng.choice([0, 0, 200, 550, 1000, 1750]) * scale / max(tilt.std(), 1e-9))
                width = int(rng.choice([12, 13, 14, 15, 16]))
                groups = pq.pack_wide(s, int(t), rng, width, noise, tilt)
                s, ex = pq.choose_step(s, int(t), groups, rng, perturb=pert)
                circuits += ex
                if not ex and j >= 2:
                    break
            if s.obj < elite.obj:
                log.append(dict(step=k, day=int(t), obj=int(s.obj), gain=int(s.obj - elite.obj), circuits=circuits))
                elite = s
                np.save(npy, elite.v)
                print(f"step {k} day {t}: {elite.obj} ({log[-1]['gain']:+d}), circuits {circuits}", flush=True)
        k += 1
    print(f"final {elite.obj} (start {obj0}), {k} steps, {circuits} QAOA circuits", flush=True)
    if elite.obj < obj0:
        model, sc = qc.check_with_referee(p, elite.v)
        assert sc.feasible and sc.objective == elite.obj, sc
        path = work / f"{p.stem}__paired-transfer-qaoa__{time.strftime('%Y%m%d-%H%M%S')}.sol"
        settings = (f"paired-transfer QAOA (p=3, local statevector, <=16 qubits, 448 shots); start={src.relative_to(ROOT)} "
                    f"({obj0}); seed={a.seed} minutes={a.minutes} steps={k} circuits={circuits}")
        model.write_solution(path, p.positions(elite.v), comment=settings)
        path.with_suffix(".log.json").write_text(json.dumps(dict(settings=settings, start=obj0, final=int(elite.obj),
                                                                 accepted=log), indent=0))
        print("saved", path, sc.objective, flush=True)
    npy.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
