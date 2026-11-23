#!/usr/bin/env python3
"""runs_to_data.py - .sol files (+ optional AutoRunner JSON) -> arrays/tables -> PCA + 3D surfaces.

    python runs_to_data.py --sol-dir handin/            # JSON-free: only .sol files
    python runs_to_data.py --json paste.txt --rundir AutoData/run_0134   # with JSON metadata
    python runs_to_data.py --demo
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

OUT = Path("figures"); OUT.mkdir(exist_ok=True)


# ---------- .sol -> arrays -------------------------------------------------
def read_sol(path):
    """Header lines 'key value'; rows 'period symbol long short'. Same grammar as qoblib_tools.read_portfolio_solution."""
    header, pos = {}, {}
    for line in Path(path).read_text().splitlines():
        toks = line.split("#", 1)[0].split()
        if len(toks) == 4 and toks[0].isdigit():
            pos[int(toks[0]), toks[1]] = (int(toks[2]), int(toks[3]))
        elif len(toks) == 2:
            header[toks[0]] = toks[1]
    return header, pos


def sol_to_arrays(path, symbols, T):
    _, pos = read_sol(path)
    idx = {s: i for i, s in enumerate(symbols)}
    L = np.zeros((T, len(symbols)), int); S = np.zeros_like(L)
    for (t, s), (lo, sh) in pos.items():
        L[t, idx[s]], S[t, idx[s]] = lo, sh
    return L, S


# ---------- JSON-free table: everything the .sol header knows ---------------
def sol_dir_to_df(folder):
    rows = []
    for f in sorted(Path(folder).rglob("*.sol")):
        h, pos = read_sol(f)
        if "instance" not in h or "objective" not in h: continue
        rows.append(dict(sol=str(f), name=f.stem, instance=h["instance"], budget=int(h.get("budget", 0)),
                         lam=float(h.get("lambda", 0)), objective=int(h["objective"]), n_pos=len(pos)))
    return pd.DataFrame(rows)


def symbols_for(instance, sol_paths):
    """Official order via qoblib_tools if available, else sorted union of symbols in the files (fine for PCA)."""
    try:
        from qoblib_tools import load_portfolio_instance
        inst = load_portfolio_instance(instance); return inst.symbols, inst.n_periods
    except Exception as e:
        print(f"[warn] {instance}: symbols taken from .sol files ({type(e).__name__})")
        ps = [read_sol(p)[1] for p in sol_paths]
        return sorted({s for p in ps for _, s in p}), max(t for p in ps for t, _ in p) + 1


# ---------- optional JSON table ---------------------------------------------
def json_to_df(path, rundir="."):
    d = json.loads(Path(path).read_text()); rows = []
    for inst, r in d["results"].items():
        p = r["parameters"]
        for s in r.get("seed_runs", []):
            ex, sa = s["extra"], s["extra"].get("sa", {})
            rows.append(dict(
                instance=r["extra"]["base_instance"], run=inst, assets=p["assets"], days=p["days"], budget=p["budget"],
                lam=p["lambda"], seed=s["solver_seed"], objective=s["benchmark_score"], best_known=r.get("best_known_score"),
                gap=s.get("gap_to_best_known"), seconds=s["duration_seconds"],
                sol=str(Path(rundir) / s["solution_file"].replace("\\", "/")), name=f"{inst}_s{s['solver_seed']}",
                **{f"bd_{k}": v for k, v in s["solution_checker"]["breakdown"].items()},
                **{f"sa_{k}": v for k, v in sa.items() if not isinstance(v, (dict, list))}))
    return pd.DataFrame(rows)

# ---------- patch snippet ---------------------------------------------
def order_assets(V_TN):
    """Reorder assets so similar holding patterns are adjacent (greedy nearest-neighbour on correlation distance)."""
    X = V_TN.T.astype(float); n = len(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2); order = [int(np.argmax(np.abs(X).sum(1)))]
    left = set(range(n)) - set(order)
    while left:
        j = min(left, key=lambda k: D[order[-1], k]); order.append(j); left.remove(j)
    return np.array(order)


def fig_position_heatmap(V_TN, name="positions_heatmap"):
    """Honest view of V[t,i]: day x asset image, assets reordered; plus 3D bars for the slide."""
    o = order_assets(V_TN); W = V_TN[:, o]
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121); im = ax.imshow(W, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    ax.set(xlabel="asset (reordered)", ylabel="day", title="Net position V[t,i] (units, -3..3)"); fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(122, projection="3d"); T, n = W.shape; t, i = np.nonzero(W)
    ax.bar3d(i, t, np.zeros(len(i)), 0.8, 0.8, W[t, i], color=plt.cm.coolwarm((W[t, i] + 3) / 6), shade=True)
    ax.set(xlabel="asset (reordered)", ylabel="day", zlabel="net units", title="Non-zero positions only")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def descriptors(L, S):
    """Size-independent features, so runs with different assets/days can share one PCA."""
    V = L - S; T, n = V.shape; g = np.abs(V).sum(1)
    turn = np.abs(np.diff(L, axis=0)).sum(1) + np.abs(np.diff(S, axis=0)).sum(1) if T > 1 else np.zeros(1)
    return np.array([(V != 0).mean(), np.abs(V).mean(), g.mean() / n, g.std() / (g.mean() + 1e-9),
                     S.sum() / (L.sum() + S.sum() + 1e-9), turn.mean() / n, (np.abs(V) == 3).mean() / ((V != 0).mean() + 1e-9)])


def fig_mixed_map(df, descs, name="pca_mixed_descriptors"):
    """df needs objective, lam, assets, days, name. Colour = log10 lambda, size = assets, z = objective / per-(instance) best."""
    X = np.array(descs); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P, _, _, ev = pca(X)
    rel = df.objective / df.groupby("instance").objective.transform("min")      # 1.0 = best of its instance (objectives < 0)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], rel, c=np.log10(df.lam.clip(lower=1e-9)), s=20 + 2 * df.assets, cmap="plasma", depthshade=False)
    fig.colorbar(sc, label="log10 lambda", shrink=0.6)
    ax.set(xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})", zlabel="objective / best of instance")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- PCA + plots -----------------------------------------------------
def pca(X, k=2):
    mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return (X - mu) @ Vt[:k].T, Vt[:k], mu, (s ** 2) / (s ** 2).sum()


def fig_solution_space(V, objectives, labels, name="pca_solution_space"):
    P, _, _, ev = pca(V)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], objectives, c=objectives, cmap="viridis_r", s=60, depthshade=False)
    if len(P) >= 8:
        g = [np.linspace(P[:, i].min(), P[:, i].max(), 50) for i in (0, 1)]
        A, B = np.meshgrid(*g)
        Z = RBFInterpolator(P, objectives, smoothing=1e-3 * np.ptp(objectives) ** 2, kernel="thin_plate_spline")(
            np.c_[A.ravel(), B.ravel()]).reshape(A.shape)
        ax.plot_surface(A, B, Z, cmap="viridis_r", alpha=0.45, edgecolor="none")
    for p, o, l in zip(P, objectives, labels): ax.text(p[0], p[1], o, str(l), fontsize=6)
    ax.set(xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})", zlabel="objective",
           title="Final solutions in PCA space (z = QOBLIB objective)")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_time_modes(V_TN, name="pca_time_modes"):
    X = V_TN - V_TN.mean(0); s = np.linalg.svd(X, compute_uv=False); ev = s ** 2 / (s ** 2).sum()
    fig = plt.figure(figsize=(12, 5)); ax = fig.add_subplot(121, projection="3d")
    T, n = V_TN.shape; A, B = np.meshgrid(np.arange(n), np.arange(T))
    ax.plot_surface(A, B, V_TN, cmap="coolwarm", edgecolor="none")
    ax.set(xlabel="asset", ylabel="day", zlabel="net position", title="Position surface V[t, i]")
    ax2 = fig.add_subplot(122); k = np.arange(1, len(ev) + 1)
    ax2.bar(k, ev); ax2.plot(k, np.cumsum(ev), "k.-"); ax2.set(xlabel="mode", ylabel="explained variance", title="SVD spectrum")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def run_table(df):
    for inst, g in df.groupby("instance"):
        if len(g) < 3: print(f"skip {inst}: {len(g)} solution(s)"); continue
        syms, T = symbols_for(inst, g.sol)
        Vs = [(lambda L, S: L - S)(*sol_to_arrays(f, syms, T)) for f in g.sol]
        fig_solution_space(np.array([v.ravel() for v in Vs]), g.objective.values, g.name.values, f"pca_{inst}")
        fig_time_modes(Vs[int(np.argmin(g.objective.values))], f"modes_{inst}")


def demo():
    r = np.random.default_rng(0); T, n = 15, 50
    base = r.integers(-3, 4, (T, n)) * (r.random((T, n)) < 0.1)
    runs = [np.clip(base + r.integers(-1, 2, (T, n)) * (r.random((T, n)) < 0.05), -3, 3) for _ in range(24)]
    V = np.array([x.ravel() for x in runs]); obj = -3e5 + 2e3 * np.linalg.norm(V - V.mean(0), axis=1)
    fig_solution_space(V, obj, range(24)); fig_time_modes(runs[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-dir"); ap.add_argument("--json"); ap.add_argument("--rundir", default=".")
    ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
    if a.demo: demo()
    elif a.sol_dir:
        df = sol_dir_to_df(a.sol_dir); df.to_csv("runs_sol_only.csv", index=False); print(df); run_table(df)
    elif a.json:
        df = json_to_df(a.json, a.rundir); df.to_csv("runs.csv", index=False); print(df[["name", "objective", "gap"]]); run_table(df)
    else: ap.error("give --sol-dir, --json or --demo")
