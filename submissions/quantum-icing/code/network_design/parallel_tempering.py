"""Parallel tempering (replica exchange) for QOBLIB network design (08), e.g. network24.

    NETWORK_SOL_DIR=solutions/agent_pt .venv/bin/python network_pt.py 24 --minutes 10

Each replica runs Metropolis at its own temperature on degree-preserving moves:
  * 2-exchange      (a,b),(c,d)            -> (a,d),(c,b)          (first arc biased by LP duals)
  * 3-rotation      (a1,b1),(a2,b2),(a3,b3) -> (a1,b2),(a2,b3),(a3,b1)
  * triangle flip   a->b->c->a              -> a->c->b->a           (needed for connectivity)
Walk energy E = LP congestion + w * (#arcs with positive dual), a small tie-breaker that makes
plateaus of equal max load non-flat.  Every proposal is pre-screened with an exact lower bound
(sum of demand * hop distance / #arcs): Metropolis draws its threshold first and the LP is only
solved if the bound does not already exceed it, so rejections by the bound are exact.
Neighbouring temperatures swap configurations after every chunk of steps.  Improvements are
banked through ``network_pipeline.bank`` (integer routing + official checker).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse.csgraph import shortest_path

import qoblib_tools as qt
from network_design import pipeline as npl

_D: np.ndarray | None = None  # demand (int)
_DF: np.ndarray | None = None  # demand as float, for the bound
_CACHE: dict = {}  # per-worker LP cache: topology -> (z, duals)
_CACHE_MAX = 60000


def score_cached(arcs):
    key = tuple(arcs)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], True
    z, d = npl.score_topology(arcs, _D)
    if len(_CACHE) < _CACHE_MAX:
        _CACHE[key] = (z, d)
    return z, d, False


def _init(demand: np.ndarray) -> None:
    global _D, _DF
    _D, _DF = demand, demand.astype(float)


def lower_bound(arcs, n: int) -> float:
    """z >= sum_{k,i} d[k,i] * hops(k,i) * 1000 / m   (total load spread over all arcs)."""
    A = np.zeros((n, n))
    for i, j in arcs:
        A[i, j] = 1.0
    D = shortest_path(A, unweighted=True)
    if np.isinf(D).any():
        return math.inf
    return float((_DF * D).sum()) * qt.INTSCALE / len(arcs)


def energy(z: float, duals, w: float) -> float:
    if duals is None or math.isinf(z):
        return math.inf
    return z + w * int((duals > 1e-9).sum())


# ── Moves ─────────────────────────────────────────────────────────────────────────────

def two_exchange(arcs, p, q):
    return npl.swap(arcs, p, q)


def three_rotation(arcs, p, q, r):
    if len({p, q, r}) < 3:
        return None
    (a1, b1), (a2, b2), (a3, b3) = arcs[p], arcs[q], arcs[r]
    new_arcs = [(a1, b2), (a2, b3), (a3, b1)]
    if any(i == j for i, j in new_arcs):
        return None
    rest = set(arcs) - {arcs[p], arcs[q], arcs[r]}
    if len(set(new_arcs)) < 3 or any(a in rest for a in new_arcs):
        return None
    return sorted(rest | set(new_arcs))


def triangle_flip(arcs, rng):
    """Reverse a random directed triangle a->b->c->a, if any (and the result is simple)."""
    present = set(arcs)
    out = {}
    for i, j in arcs:
        out.setdefault(i, []).append(j)
    tris = []
    for a, b in arcs:
        for c in out[b]:
            if c != a and (c, a) in present and a < b and a < c:
                tris.append((a, b, c))
    rng.shuffle(tris)
    for a, b, c in tris:
        new = [(a, c), (c, b), (b, a)]
        if any(x in present for x in new):
            continue
        return sorted((present - {(a, b), (b, c), (c, a)}) | set(new))
    return None


def propose(arcs, duals, rng, p_tri=0.1, p_rot=0.2, bias=0.5):
    m = len(arcs)
    w = np.clip(duals, 0, None) + 1e-12
    w = w / w.sum()
    for _ in range(50):
        u = rng.random()
        if u < p_tri:
            new = triangle_flip(arcs, rng)
            kind = "tri"
        else:
            p = int(rng.choice(m, p=w)) if rng.random() < bias else int(rng.integers(m))
            if u < p_tri + p_rot:
                new = three_rotation(arcs, p, int(rng.integers(m)), int(rng.integers(m)))
                kind = "rot"
            else:
                new = two_exchange(arcs, p, int(rng.integers(m)))
                kind = "x2"
        if new is not None:
            return new, kind
    return None, None


def guided(arcs, duals, rng, n, K, p_double, p_tri, p_rot, skip_above=math.inf):
    """Best-of-K candidates by the cheap hop-distance bound (single or double moves).
    Candidates already LP-scored (cache) above ``skip_above`` are ignored."""
    best, best_lb = None, math.inf
    for _ in range(K):
        new, _ = propose(arcs, duals, rng, p_tri, p_rot)
        if new is not None and rng.random() < p_double:
            new2, _ = propose(new, duals, rng, p_tri, p_rot)
            new = new2 if new2 is not None and new2 != arcs else new
        if new is None:
            continue
        hit = _CACHE.get(tuple(new))
        if hit is not None and hit[0] > skip_above:
            continue
        lb = lower_bound(new, n)
        if lb < best_lb:
            best, best_lb = new, lb
    return best, best_lb


# ── One replica's chunk of Metropolis steps (runs in a worker) ────────────────────────

def run_chunk(st: dict) -> dict:
    n = len(_D)
    rng = np.random.default_rng(st["seed"])
    arcs = [tuple(a) for a in st["arcs"]]
    z, duals, T, w = st["z"], np.asarray(st["duals"]), st["T"], st["w"]
    E = energy(z, duals, w)
    lb = lower_bound(arcs, n)
    margin = st["margin"]
    best_z, best_arcs = st["best_z"], st["best_arcs"]
    stats = {"prop": 0, "lp": 0, "acc": 0, "skip": 0}
    for _ in range(st["steps"]):
        if st["K"] > 1:
            new, lb_new = guided(arcs, duals, rng, n, st["K"], st["p_double"], st["p_tri"], st["p_rot"],
                                 skip_above=E + 3 * T)
        else:
            new, kind = propose(arcs, duals, rng, st["p_tri"], st["p_rot"])
            lb_new = lower_bound(new, n) if new is not None else math.inf
        if new is None:
            continue
        stats["prop"] += 1
        thr = E - T * math.log(max(rng.random(), 1e-300))  # accept iff E_new <= thr
        if lb_new + w > thr:  # exact: E_new >= z_new + w >= LB + w
            continue
        if lb_new - lb > (thr - E) + margin:  # heuristic: excess z-LB rarely shrinks by > margin
            stats["skip"] += 1
            continue
        z_new, d_new, hit = score_cached(new)
        stats["lp"] += 0 if hit else 1
        E_new = energy(z_new, d_new, w)
        if E_new <= thr:
            arcs, z, duals, E, lb = new, z_new, d_new, E_new, lb_new
            stats["acc"] += 1
            if z < best_z - 1e-6:
                best_z, best_arcs = z, list(arcs)
    st.update(arcs=arcs, z=z, duals=duals, E=E, best_z=best_z, best_arcs=best_arcs,
              stats=stats, seed=int(rng.integers(2**62)))
    return st


# ── Driver ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--tmin", type=float, default=150.0)
    ap.add_argument("--tmax", type=float, default=4000.0)
    ap.add_argument("--w", type=float, default=0.0, help="energy weight per positive-dual arc")
    ap.add_argument("--steps", type=int, default=25, help="Metropolis steps per replica per round")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--banked", type=int, default=2, help="replicas started from the banked file "
                    "(coldest ones); the rest start from random topologies")
    ap.add_argument("--start-json", default=None, help="optional JSON topology to seed replicas")
    ap.add_argument("--p-tri", type=float, default=0.1)
    ap.add_argument("--p-rot", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=math.inf,
                    help="skip the LP when the bound rises by more than (threshold slack + margin)")
    ap.add_argument("--K", type=int, default=1, help="candidates per proposal, best by bound (1 = plain)")
    ap.add_argument("--p-double", type=float, default=0.0, help="prob. a candidate is two moves")
    ap.add_argument("--tag", default="pt")
    args = ap.parse_args()
    n = args.n
    demand = qt.load_network_demand(n)
    _init(demand)
    rng = np.random.default_rng(args.seed)
    R = args.replicas
    temps = [args.tmin * (args.tmax / args.tmin) ** (r / max(1, R - 1)) for r in range(R)]

    seeds_topo = []
    if args.start_json:
        seeds_topo.append([tuple(a) for a in json.load(open(args.start_json))["arcs"]])
    if npl.banked_path(n).exists():
        _, b, _ = qt.read_network_solution(npl.banked_path(n))
        seeds_topo.append(sorted(b))
    states = []
    for r in range(R):
        if r < args.banked and seeds_topo:
            arcs = seeds_topo[r % len(seeds_topo)]
        else:
            arcs = sorted(qt.random_topology(n, rng))
        z, d = npl.score_topology(arcs, demand)
        while math.isinf(z):
            arcs = sorted(qt.random_topology(n, rng))
            z, d = npl.score_topology(arcs, demand)
        states.append(dict(arcs=arcs, z=z, duals=d, T=temps[r], w=args.w, steps=args.steps,
                           p_tri=args.p_tri, p_rot=args.p_rot, margin=args.margin, K=args.K, p_double=args.p_double, best_z=z, best_arcs=arcs,
                           seed=int(rng.integers(2**62)), rid=r))
    banked = npl.banked_score(n, demand)
    best_z = min(s["z"] for s in states)
    best_arcs = min(states, key=lambda s: s["z"])["arcs"]
    print(f"network{n:02d} PT: R={R} temps={[round(t) for t in temps]} w={args.w} banked={banked}",
          flush=True)
    print("start z:", [round(s["z"]) for s in states], flush=True)

    sw_try, sw_acc = np.zeros(R - 1), np.zeros(R - 1)
    tot = {"prop": 0, "lp": 0, "acc": 0, "skip": 0}
    acc_slot = np.zeros(R)
    prop_slot = np.zeros(R)
    t0 = time.time()
    deadline = t0 + 60 * args.minutes
    rnd = 0
    best_json = npl.SOL_DIR / f"{args.tag}_best{n:02d}.json"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(demand,)) as pool:
        while time.time() < deadline:
            states = list(pool.map(run_chunk, states))
            rnd += 1
            for r, s in enumerate(states):
                for k in tot:
                    tot[k] += s["stats"][k]
                acc_slot[r] += s["stats"]["acc"]
                prop_slot[r] += s["stats"]["prop"]
                if s["best_z"] < best_z - 1e-6:
                    best_z, best_arcs = s["best_z"], s["best_arcs"]
                    json.dump({"z": best_z, "arcs": best_arcs}, open(best_json, "w"))
                    if banked is None or math.ceil(best_z - 1e-6) < banked:
                        ok, sc, msg = npl.bank(n, demand, best_arcs,
                                               f"network_pt {args.tag} seed={args.seed} round={rnd} rid={s['rid']}")
                        if ok:
                            banked = sc
                        print(f"  [{time.time()-t0:6.0f}s] round {rnd} slot {r}: LP {best_z:.1f} -> {msg}",
                              flush=True)
            # replica exchange between neighbours (alternate even / odd pairs)
            for r in range(rnd % 2, R - 1, 2):
                a, b = states[r], states[r + 1]
                delta = (a["E"] - b["E"]) * (1 / a["T"] - 1 / b["T"])
                sw_try[r] += 1
                if delta >= 0 or rng.random() < math.exp(delta):
                    sw_acc[r] += 1
                    for key in ("arcs", "z", "duals", "E", "best_z", "best_arcs", "rid"):
                        a[key], b[key] = b[key], a[key]
            if rnd % 10 == 0:
                el = time.time() - t0
                rates = np.divide(sw_acc, np.maximum(sw_try, 1))
                print(f"[{el:6.0f}s] rnd {rnd} best {best_z:.1f} banked {banked} "
                      f"lp/s {tot['lp']/el:.0f} lp-frac {tot['lp']/max(1,tot['prop']):.2f} prop/s {tot['prop']/el:.0f} | "
                      f"z {[round(s['z']/1000) for s in states]} | "
                      f"acc {[round(float(x), 3) for x in acc_slot / np.maximum(prop_slot, 1)]} | "
                      f"swap {[round(float(x), 2) for x in rates]}", flush=True)
    el = time.time() - t0
    rates = np.divide(sw_acc, np.maximum(sw_try, 1))
    print(f"done {rnd} rounds in {el:.0f}s: best LP {best_z:.1f} (ceil {math.ceil(best_z - 1e-6)}), "
          f"banked {banked}, gap {npl.gap(n, banked):+.3%}" if banked else "done", flush=True)
    print("temps", [round(t) for t in temps], "swap rates", [round(float(x), 3) for x in rates],
          "acc", [round(float(x), 3) for x in acc_slot / np.maximum(prop_slot, 1)], flush=True)


if __name__ == "__main__":
    main()
