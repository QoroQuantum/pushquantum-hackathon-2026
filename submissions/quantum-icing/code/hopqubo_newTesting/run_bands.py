"""Constraint-respecting partitioned QAOA / PCE as a large-neighbourhood search (LNS).

Each round: split the incumbent into arc-disjoint bands (alternating row/column bands, LP
bottleneck nodes spread over the bands), solve every band with the chosen sub-solver in one
Divi ensemble (parallel; batched on the cloud), stitch the band candidates by beam search on
global H2 (every stitched topology is valid by construction), LP-score the best few, accept
if better. Finish with the 2-exchange polish. Gap vs QOBLIB best known is always reported.

    .venv/bin/python hopqubo_newTesting/run_bands.py --n 12 --methods enum random xch --rounds 6
    .venv/bin/python hopqubo_newTesting/run_bands.py --n 12 --methods xpen --char      # service warm start
    .venv/bin/python hopqubo_newTesting/run_bands.py --n 24 --methods xch --cloud qoro_cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DIVI_DISABLE_PROGRESS", "1")

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402
from common import H2, RunLog, gap_str, is_valid, load, lp, make_backend, polish, rank_and_score, \
    service_or_none, start_topology  # noqa: E402

QUANTUM = ("xch", "xpen", "xy", "pce")


def hot_nodes(T, inc, kind):
    from network_design.sa_search import lp_eval

    _, hot = lp_eval(T, inc)
    return {inc[a][0] if kind == "row" else inc[a][1] for a in hot}


def service_warm_angles(args, T, band, log, method="xpen"):
    """Angles from characterize_and_validate on a representative band (cached; rerun is free).
    xpen: X mixer, all sums penalised. xy: service told mixer="xy" + row eq_cardinality blocks,
    column sums penalised; reference state = our incumbent's band bits (row sums exact)."""
    service = service_or_none()
    if service is None:
        return None
    import hashlib

    ref = "".join(map(str, band.inc_bits))  # our own incumbent, never QOBLIB's published solution
    sig = hashlib.sha1(repr((T.tolist(), band.var_arcs, band.fixed, ref)).encode()).hexdigest()[:10]
    key = f"n{args.n}_{method}_L{args.layers}_pw{args.penalty_weight}_{sig}"
    xy = method == "xy"
    try:
        rep = __import__("divi_ext").characterize_cached(
            key, B.penalised_band_problem(T, band, args.penalty_weight, rows=not xy), [ref], service,
            layers=args.layers, mixer="xy" if xy else None,
            constraints=B.band_char_constraints(band, rows=True, cols=False) if xy else None)
    except Exception as err:
        log(f"characterization failed ({type(err).__name__}: {err}); runs blind")
        return None
    pen = getattr(rep, "penalty", None)
    log(f"characterization {rep.job_id}: recommended layers {rep.recommended_layers}, "
        f"penalty {pen.get('recommended_lambda') if isinstance(pen, dict) else pen}")
    try:
        return rep.qaoa_initial_params(layers=args.layers)
    except Exception as err:
        log(f"no angles at {args.layers} layers ({err}); runs blind")
        return None


def run_method(args, T, start, method, log):
    from divi_ext import BlockEnsemble

    n, rng = args.n, np.random.default_rng(args.seed)
    h2 = H2(T)
    inc, z = list(start), lp(T, start)
    lp_calls, circuits = 1, 0
    backend = make_backend(args.cloud, shots=args.shots)
    warm = {}  # method -> transferred angles (xch / pce) or service angles (xpen)
    log(f"[{method}] start LP {z:.0f} ({gap_str(n, z)})")
    for r in range(args.rounds):
        kind = "row" if r % 2 == 0 else "col"
        bands = B.partition_round(n, inc, kind, args.k, rng, hot_nodes(T, inc, kind))
        bands = [b for b in bands if len(B.enumerate_band(b)) > 1]
        lp_calls += 1  # hot_nodes() solved the routing LP once
        if not bands:
            log(f"  round {r} {kind}: every band is rigid, skipped")
            continue
        cands, diag = {}, {}
        t0 = time.time()
        if method in ("enum", "random"):
            for b in bands:
                fn = B.solve_enum if method == "enum" else B.solve_random
                cands[b.bid], diag[b.bid] = fn(T, b, top=args.per_band, rng=rng)
        else:
            def build(b, w):
                kw = dict(n_layers=args.layers, backend=backend, pop=args.pop, seed=args.seed + r)
                if method == "xch":
                    return B.build_xch_program(T, b, iters=args.iters if w is None else args.warm_iters, warm=w, **kw)
                if method == "xpen":
                    return B.build_xpen_program(T, b, iters=args.iters, penalty_weight=args.penalty_weight, warm=w, **kw)
                if method == "xy":
                    return B.build_xy_program(T, b, iters=args.iters, penalty_weight=args.penalty_weight, warm=w, **kw)
                return B.build_pce_program(T, b, iters=args.iters, penalty_weight=args.penalty_weight,
                                           alpha=args.alpha, **kw)

            if method in ("xpen", "xy") and method not in warm and args.char:
                warm[method] = service_warm_angles(args, T, bands[0], log, method)
            if method == "xch" and "xch" not in warm and not args.blind:
                # transfer warm start: calibrate on the first band, reuse on all later bands
                for cb in bands:  # first band that has a 2-exchange inside it
                    calib = build(cb, None)
                    if calib is not None:
                        break
                if calib is not None:
                    calib.run()
                    warm["xch"] = np.atleast_2d(calib.best_params)[0]
                    cands[cb.bid], diag[cb.bid] = B.harvest(calib, cb, method, args.per_band * 4, T=T)
                    circuits += calib.total_circuit_count
                    log(f"  calibrated xch angles on band {cb.bid} ({cb.n_vars} qubits): "
                        f"{np.round(warm['xch'], 3).tolist()}")
            progs = {}
            for b in bands:
                if b.bid in cands:
                    continue
                p = build(b, warm.get(method))
                if p is not None:
                    progs[b.bid] = p
            if progs:
                ens = BlockEnsemble(backend)
                ens.programs = progs
                ens.run()
                for bid, p in progs.items():
                    band = next(b for b in bands if b.bid == bid)
                    cands[bid], diag[bid] = B.harvest(p, band, method, args.per_band * 4, alpha=args.alpha, rng=rng, T=T)
                    circuits += p.total_circuit_count
        # rank each band's candidates by global H2, keep the best few
        for b in bands:
            cands[b.bid] = sorted(cands.get(b.bid, []), key=lambda bits: h2(b.fixed + b.arcs_of(bits)))[:args.per_band]
        tops = B.stitch(T, bands, cands, beam=args.beam, per_band=args.per_band)
        assert all(is_valid(n, t) for t in tops), "stitching produced an invalid topology"
        scored = rank_and_score(T, [t for t in tops if t != sorted(inc)], h2, top_k=args.lp_topk)
        lp_calls += len(scored)
        moved = False
        if scored and scored[0][0] < z:
            z, inc, moved = scored[0][0], scored[0][1], True
        qubits = max((b.n_vars for b in bands), default=0)
        pv = [d.get("p_valid") for d in diag.values() if d.get("p_valid") is not None]
        cc = [d.get("conc") for d in diag.values() if d.get("conc") is not None]
        log(f"  round {r} {kind:3s}: {len(bands)} bands ≤{qubits} qubits, "
            f"{sum(len(c) for c in cands.values())} cands"
            + (f", P(valid) mean {np.mean(pv):.2f}" if pv else "")
            + (f", P(band opt)/uniform mean {np.mean(cc):.1f}×" if cc else "")
            + f" | best stitched LP {scored[0][0] if scored else float('nan'):.0f} → incumbent {z:.0f} "
            f"({gap_str(n, z)}){' *' if moved else ''} [{time.time() - t0:.0f}s]")
        log.event(method=method, round=r, kind=kind, z=z, lp_calls=lp_calls, circuits=circuits,
                  n_bands=len(bands), max_qubits=qubits, diag=diag)
    pre = z
    inc, z, used = polish(T, inc, lp_budget=args.polish_budget)
    lp_calls += used + 1  # polish re-scores its start once
    log(f"[{method}] LNS {pre:.0f} → polish {z:.0f} ({gap_str(n, z)}) | LP calls {lp_calls}, circuits {circuits}")
    log.data.setdefault("final", {})[method] = {"z": z, "pre_polish": pre, "gap": gap_str(n, z),
                                                "lp_calls": lp_calls, "circuits": circuits, "arcs": inc}
    log.save()
    return z, inc, lp_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--methods", nargs="+", default=["enum", "random", "xch"],
                    choices=["enum", "random", "xch", "xpen", "xy", "pce"])
    ap.add_argument("--start", default="random", help="random | h2greedy | path.json | path.sol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=3, help="nodes per band (k=3 → ≤18 qubits)")
    ap.add_argument("--layers", type=int, default=2, help="QAOA n_layers / PCE n_layers")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warm-iters", type=int, default=2, help="MC iterations after a warm start")
    ap.add_argument("--blind", action="store_true", help="xch: no angle transfer")
    ap.add_argument("--char", action="store_true", help="xpen/xy: warm start from the Characterization Service")
    ap.add_argument("--penalty-weight", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--cloud", default=None, help="qoro_cpu | qoro_gpu (default: local Maestro)")
    ap.add_argument("--per-band", type=int, default=6)
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--lp-topk", type=int, default=6)
    ap.add_argument("--polish-budget", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    T = load(args.n)
    start = start_topology(args.n, T, args.start, args.seed)
    log = RunLog(f"bands_n{args.n:02d}_s{args.seed}{('_' + args.tag) if args.tag else ''}")
    log.data["args"] = vars(args)
    log(f"network{args.n:02d}: {args.n * (args.n - 1)} arc variables, k={args.k}, methods {args.methods}")
    results = {m: run_method(args, T, start, m, log) for m in args.methods}
    log("summary: " + " | ".join(f"{m} {z:.0f} ({gap_str(args.n, z)}, {c} LP)" for m, (z, _, c) in results.items()))


if __name__ == "__main__":
    main()
