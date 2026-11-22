"""Batched quantum LNS on Qoro's cloud: large rounding windows, one per day, run in parallel.

Days interact only with their neighbours (rebalancing), so rounding windows on days of the
same parity (0, 2, 4, ... or 1, 3, 5, ...) are independent: their exact energy changes add
up.  Each round builds one window per day of one parity, sends every QAOA program to the
Qoro cloud at once (threads; each program = 2 circuits at transferred angles), and applies
every improving window together.  Same QUBOs and QAOA family as qlns.py (Dicke + ring XY,
p layers, angles transferred from the local runs), only wider.

    python portfolio_quantum/cloud_batch.py --inst po_a400_t15_orig --lam 1e-3 --start best \
        --kround 24 --rounds 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import qlns
import qaoa_solver as qs

qc, ps = qlns.qc, qlns.ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--start", default="best")
    ap.add_argument("--kround", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angles", default=str(Path(__file__).resolve().parent / "angles" / "a050_l1e-3.json"))
    ap.add_argument("--local", action="store_true", help="Maestro simulator instead of the cloud (testing)")
    ap.add_argument("--warm", type=float, default=0.0, help="share of windows seeded from the relaxation's rounding")
    ap.add_argument("--variants", type=int, default=1, help="windows per day and round (different noise/band)")
    ap.add_argument("--band", type=float, default=0.0, help="slots with frac within band of 0/1 get no qubit")
    ap.add_argument("--grouped", action="store_true", help="grouped Dicke start (budget + capital exact)")
    a = ap.parse_args()

    p = ps.Problem(a.inst, a.lam)
    rng = np.random.default_rng(a.seed)
    if a.start == "best":
        v0 = qc.load_v(p, qc.best_known_file(p.stem))
    elif a.start == "qbest":
        cands = [qc.best_known_file(p.stem)] + list((qlns.OUT / p.stem).glob("*.sol"))
        cands = [c for c in cands if c is not None and c.exists()]
        v0 = qc.load_v(p, min(cands, key=lambda c: ps.read_objective(c)))
    elif a.start == "empty":
        v0 = np.zeros((p.T, p.n), dtype=np.int64)
    else:
        v0 = qc.load_v(p, Path(a.start))
    s = qc.State(p, v0)
    start = s.obj
    svc = None
    if not a.local:
        from divi.backends import QoroService
        svc = QoroService()
        credits0 = float(svc.get_credit_balance()["balance"])
    # one solver per thread (programs keep their own state); angles read-only (no retraining)
    solvers = [qs.QAOASolver(layers=a.layers, shots=a.shots, cloud=not a.local, seed=a.seed + k,
                             retrain_every=10**9, angles_file=None, grouped_dicke=True)
               for k in range(p.T * a.variants)]
    lib = json.loads(Path(a.angles).read_text())
    for sv in solvers:
        sv.library["xy"] = np.array(lib[f"xy_p{a.layers}"])
        if lib.get(f"xyw_p{a.layers}"):
            sv.library["xyw"] = np.array(lib[f"xyw_p{a.layers}"])
        if lib.get(f"x_p{a.layers}"):
            sv.library["x"] = np.array(lib[f"x_p{a.layers}"])
    print(f"{p.stem}: start {start}, {p.qubo_vars} QUBO variables, cloud={not a.local}", flush=True)
    log = []
    for r in range(a.rounds):
        parity = r % 2
        days = [t for t in range(p.T - 1) if t % 2 == parity]
        windows = []
        for t in days:
            for var in range(a.variants):
                noise = 0.0 if var == 0 else a.noise * rng.random()
                band = a.band if var % 2 == 0 else a.band / 2
                w = qlns.round_window(s, t, rng, a.kround, noise=noise, band=band)
                if w is not None:
                    windows.append((t, w, qlns.round_window.prior.copy(), rng.random() < a.warm))
        t0 = time.time()

        def job(item):
            k, (t, (tmp, moves, hw), prior, warm) = item
            if hw is None and len(moves) > 22:  # free-weight windows would add slack: keep them small
                moves, prior = moves[:22], prior[:22]
            seed = qlns.warm_seed(prior, hw) if ((warm or a.grouped) and hw is not None) else None
            res = qlns.run_window(s, tmp, moves, solvers[k], True, hw=hw, seed=seed, prior=prior)
            return t, moves, res

        with ThreadPoolExecutor(max_workers=len(windows)) as ex:
            results = list(ex.map(job, enumerate(windows)))
        v_new = s.v.copy()
        best_day = {}
        for t, moves, res in results:
            if res is None:
                continue
            v2, obj, ncand, e = res
            if obj < s.obj and (t not in best_day or obj < best_day[t][0]):  # best variant per day
                best_day[t] = (obj, v2[t].copy(), len(moves))
        gains = []
        for t, (obj, row, nq) in best_day.items():
            v_new[t] = row
            gains.append((t, obj - s.obj, nq))
        new_obj = p.objective(v_new)
        expect = s.obj + sum(g for _, g, _ in gains)
        assert new_obj == expect, (new_obj, expect)  # independence of same-parity days
        qubits = max(len(m) for _, m, _ in results)
        print(f"round {r} parity {parity}: {len(windows)} programs (<= {qubits} qubits) in {time.time() - t0:.0f}s, "
              f"{len(gains)} improved, {s.obj} -> {new_obj}", flush=True)
        log.append(dict(round=r, programs=len(windows), max_qubits=qubits, improved=len(gains),
                        obj=new_obj, seconds=round(time.time() - t0)))
        improved_round = new_obj < s.obj
        s = qc.State(p, v_new)
        if improved_round:  # never lose progress: save after every improving round
            circ = sum(sv.circuits for sv in solvers)
            path, obj = qlns.save(p, s.v, "qlns-qaoa-cloudbatch", f"start={a.start} kround={a.kround} band={a.band} "
                                  f"variants={a.variants} grouped={a.grouped} round={r} cloud={not a.local} "
                                  f"circuits={circ} max_qubits={max(sv.max_qubits for sv in solvers)}", log)
            with open(path.with_suffix(".log.json"), "w") as fh:
                json.dump(dict(settings=f"start={a.start} kround={a.kround} band={a.band} variants={a.variants} "
                               f"grouped={a.grouped} cloud={not a.local} circuits={circ} "
                               f"max_qubits={max(sv.max_qubits for sv in solvers)}",
                               start=start, final=s.obj, rounds=log), fh)
            print("  saved", path.name, obj, flush=True)
    circuits = sum(sv.circuits for sv in solvers)
    maxq = max(sv.max_qubits for sv in solvers)
    used = None
    if svc is not None:
        time.sleep(5)
        used = credits0 - float(svc.get_credit_balance()["balance"])
    print(f"final {s.obj} (start {start}); {circuits} circuits, largest program {maxq} qubits, credits used {used}")
    if s.obj < start:
        settings = (f"start={a.start} kround={a.kround} rounds={a.rounds} layers={a.layers} shots={a.shots} "
                    f"cloud={not a.local} circuits={circuits} max_qubits={maxq} credits={used}")
        path, obj = qlns.save(p, s.v, "qlns-qaoa-cloudbatch", settings, log)
        with open(path.with_suffix(".log.json"), "w") as fh:
            json.dump(dict(settings=settings, start=start, final=s.obj, rounds=log), fh)
        print("saved", path, obj)


if __name__ == "__main__":
    main()
