#!/usr/bin/env python3
"""Paired ablation: classical reheat vs random seeds vs QAOA seeds, equal SA budget."""
import argparse, itertools, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from asset_groups import BlockEvaluator, _net_configs, _load as load, sa_start   # signatures: check
from qaoa_signed_blocks import block_qubo, decode
from quantum_reheat import QAOASeedStage, QuantumStats
from solver_partitioned_sa import AnnealStage, PartitionSASolver, encode_holding

SEEDS = [s for s in range(1, 11) if s != 3]
ARMS = {  # name -> (kind, init, iters)
    "A_classical": ("classical", None, None),
    "B_random_bits": ("random", None, None),
    "C_qaoa_random": ("qaoa", "random", 15),
    "D_qaoa_tqa_iters0": ("qaoa", "tqa", 0),
    "E_qaoa_tqa_iters15": ("qaoa", "tqa", 15),
}


def exact_optimum(ev, L, S, t, G):
    g = len(G)
    inner = _net_configs(g - 2, 3)
    best = np.inf
    for a in itertools.product(range(-3, 4), repeat=2):    # chunked: g=8 has 5.7M states
        V = np.hstack([np.tile(a, (len(inner), 1)), inner])
        c, ok = ev.costs(L, S, [t], G, V)
        if ok.any():
            best = min(best, float(c[ok].min()))
    return best


def best_feasible(ev, L, S, t, G, bits, W, C):
    if not len(bits):
        return np.inf
    c, ok = ev.costs(L, S, [t], G, decode(bits, W, C))
    return float(np.min(np.where(ok, c, np.inf)))


def make_blocks(model, L, S, g, n, rng):
    T, na = model.n_periods, len(model.symbols)
    out = []
    while len(out) < n:
        t = int(rng.integers(1, T - 1))
        held = np.flatnonzero(L[t] - S[t])
        k = min(len(held), g // 2)
        pick = list(rng.choice(held, k, replace=False)) if k else []
        pick += list(rng.choice(np.setdiff1d(np.arange(na), held), g - k, replace=False))
        out.append((t, np.sort(np.array(pick, dtype=int))))
    return out


def bootstrap(x, rng, n=10_000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def block_study(a):
    model, _ = load(a.instance, a.budget, a.lam)
    ev = BlockEvaluator(model)
    L, S, _ = sa_start(model, num_reads=5, num_sweeps=500, seed=3, verbose=False)
    rng = np.random.default_rng(0)
    backend = "numpy" if a.backend == "numpy" else "dwave"
    rows = []
    for g in a.groups:
        for bi, (t, G) in enumerate(make_blocks(model, L, S, g, a.blocks, rng)):
            opt = exact_optimum(ev, L, S, t, G)
            cur = L[t, G] - S[t, G]
            unit = PartitionSASolver._unit_scale(ev, L, S, [t], G)
            bqm, W, C, _ = block_qubo(ev, L, S, [t], G, mu=a.mu * unit)
            labels = list(range(bqm.num_variables))
            cur_cost = float(ev.costs(L, S, [t], G, cur[None])[0][0])
            start = encode_holding(cur, W, C)
            for seed in a.seed_list:
                r1 = np.random.default_rng([seed, bi, g])
                b1 = AnnealStage(a.reads, a.sweeps, 1.0, False, backend)(bqm, start[None], r1)
                c1 = best_feasible(ev, L, S, t, G, b1, W, C)
                e1 = bqm.energies((b1, labels))
                cost1 = min(c1, cur_cost)
                best1 = b1[int(np.argmin(np.where(                # best feasible stage-1 bits
                    ev.costs(L, S, [t], G, decode(b1, W, C))[1],
                    ev.costs(L, S, [t], G, decode(b1, W, C))[0], np.inf)))]
                seeds_by_arm = {}
                for arm, (kind, init, iters) in ARMS.items():
                    t0 = time.time()
                    qb = np.zeros((0, bqm.num_variables), np.int8)
                    if kind == "classical":
                        low = b1[int(np.argmin(e1))]
                        st = np.stack([best1, low])
                    elif kind == "random":
                        sd = np.random.default_rng([seed, bi, g, 99]).integers(
                            0, 2, (a.seeds, bqm.num_variables)).astype(np.int8)
                        st = np.vstack([best1[None], sd])
                    else:
                        stg = QAOASeedStage(init=init, iters=iters, shots=a.shots, top=64,
                                            n_seeds=a.seeds, seed=seed, min_qubits=0,
                                            max_qubits=64, stats=QuantumStats())
                        qb = stg(bqm, start[None], r1)
                        sd = stg.select_seeds(qb, bqm.energies((qb, labels)), a.seeds)
                        st = np.vstack([best1[None], sd])
                    prep = time.time() - t0
                    cq = best_feasible(ev, L, S, t, G, qb, W, C)
                    for sc in a.scales:
                        t1 = time.time()
                        rr = np.random.default_rng([seed, bi, g, int(sc * 100)])
                        b2 = AnnealStage(a.reads, a.sweeps, sc, True, backend)(bqm, st, rr)
                        best = min(cost1, cq, best_feasible(ev, L, S, t, G, b2, W, C))
                        rows.append(dict(g=g, block=bi, seed=seed, arm=arm, scale=sc,
                                         gap=best - opt, hit=float(best - opt <= 1e-6),
                                         seconds=prep + time.time() - t1,
                                         qaoa_runs=int(kind == "qaoa")))
            print(f"g={g} block {bi} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "blocks_raw.csv", index=False)
    return df


def summarise(df, out):
    rng = np.random.default_rng(1)
    base = df[df.arm == "A_classical"].set_index(["g", "block", "seed", "scale"])
    lines = []
    for (arm, sc), d in df[df.arm != "A_classical"].groupby(["arm", "scale"]):
        d = d.set_index(["g", "block", "seed", "scale"])
        dg = (d.gap - base.gap.reindex(d.index)).values
        dh = (d.hit - base.hit.reindex(d.index)).values
        (mg, lg, hg), (mh, lh, hh) = bootstrap(dg, rng), bootstrap(dh, rng)
        lines.append(dict(arm=arm, scale=sc, n=len(dg), d_gap=mg, d_gap_lo=lg, d_gap_hi=hg,
                          d_hit=mh, d_hit_lo=lh, d_hit_hi=hh, d_seconds=float(
                              (d.seconds - base.seconds.reindex(d.index)).mean())))
    s = pd.DataFrame(lines)
    s.to_csv(out / "blocks_paired_vs_A.csv", index=False)
    print(s.to_string(index=False))
    print("\nDecide only if the 95% CI of d_gap excludes 0 (negative = better than A).")


def full_instance(a):
    flags = {
        "A_classical": [],
        "B_random_bits": ["--quantum-reheat", "always", "--quantum-null-control"],
        "C_qaoa_random": ["--quantum-reheat", "always", "--quantum-init", "random"],
        "D_qaoa_tqa_iters0": ["--quantum-reheat", "always", "--quantum-iters", "0"],
        "E_qaoa_tqa_iters15": ["--quantum-reheat", "always"],
    }
    rows = []
    for arm, f in flags.items():
        for seed in a.full_seeds:
            SOLVER = str(Path(__file__).resolve().with_name("solver_partitioned_sa.py"))
            cmd = [sys.executable, SOLVER, "--instance", "po_a050_t10_orig",
                   "--budget", str(a.budget), "--lambda", a.lam, "--group-size", "8",
                   "--sa-reads", str(a.reads), "--sa-sweeps", str(a.sweeps), "--reheats", "1",
                   "--quantum-reheat-scale", "1.25", "--block-sweeps", "10",
                   "--solver-seed", str(seed), "--sa-backend", a.backend, *f]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=a.out)
            m = re.search(r"AUTORUNNER_RESULT: (.*)", out.stdout)
            if not m:
                print(arm, seed, "FAILED\n", out.stderr[-500:]); continue
            rows.append(dict(arm=arm, seed=seed, objective=json.loads(m[1])["score"]))
    df = pd.DataFrame(rows); df.to_csv(a.out / "full_instance.csv", index=False)
    piv = df.pivot(index="seed", columns="arm", values="objective")
    rng = np.random.default_rng(2)
    for arm in piv.columns[1:]:
        d = (piv[arm] - piv["A_classical"]).dropna().values
        if len(d):
            print(arm, "paired objective diff (negative=better): %.1f [%.1f, %.1f]" % bootstrap(d, rng))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a050_t10_orig")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--lam", default="1e-2")
    ap.add_argument("--groups", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--reads", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--seeds", type=int, default=8, help="seed rows per QAOA reheat")
    ap.add_argument("--seed-list", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--mu", type=float, default=1.0, help="unit-price multiplier for the block QUBO")
    ap.add_argument("--backend", choices=("numpy", "dwave"), default="dwave")
    ap.add_argument("--full-seeds", type=int, nargs="*", default=[], help="enable full-instance runs")
    ap.add_argument("--out", type=Path, default=Path("bench_out"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    df = block_study(a)
    summarise(df, a.out)
    if a.full_seeds:
        full_instance(a)