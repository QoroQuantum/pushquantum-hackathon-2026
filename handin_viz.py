#!/usr/bin/env python3
"""handin_viz.py - presentation figures straight from a folder of .sol files (no JSON, no QOBLIB download).

    python handin_viz.py handin/            # -> figures/handin_*.png

Needs only numpy, scipy, pandas, matplotlib. Instance size / budget / lambda come from the file name + header.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RBFInterpolator

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.facecolor": "white", "axes.labelsize": 11, "axes.titlesize": 13})
NAME = re.compile(r"po_a(\d+)_t(\d+)_(orig|s\d+)")


def read_sol(path):
    h, pos = {}, {}
    for line in Path(path).read_text().splitlines():
        t = line.split("#", 1)[0].split()
        if len(t) == 4 and t[0].isdigit(): pos[int(t[0]), t[1]] = (int(t[2]), int(t[3]))
        elif len(t) == 2: h[t[0]] = t[1]
    return h, pos


def load_folder(folder):
    rows = []
    for f in sorted(Path(folder).glob("*.sol")):
        h, pos = read_sol(f); m = NAME.match(h.get("instance", f.stem))
        if not m or "objective" not in h: continue
        A, T, var = int(m[1]), int(m[2]), m[3]
        syms = sorted({s for _, s in pos}); idx = {s: i for i, s in enumerate(syms)}
        L = np.zeros((T, A), int); S = np.zeros_like(L)   # asset order = first-seen order, only used for shape stats
        full = {}
        for (t, s), (lo, sh) in pos.items():
            full.setdefault(s, len(full)); L[t, full[s]], S[t, full[s]] = lo, sh
        rows.append(dict(name=f.stem, inst=h["instance"], assets=A, days=T, variant=var, budget=int(h["budget"]),
                         lam=float(h["lambda"]), objective=int(h["objective"]), L=L, S=S, symbols=frozenset(full)))
    return pd.DataFrame(rows)


def feats(r):
    L, S, B = r.L, r.S, r.budget; V = L - S; absV = np.abs(V).astype(float)
    w = absV / np.maximum(absV.sum(1, keepdims=True), 1e-9)
    held = V != 0; T = len(V)
    jac = np.mean([(held[t] & held[t + 1]).sum() / max((held[t] | held[t + 1]).sum(), 1) for t in range(T - 1)])
    return dict(eff_assets=np.mean(1 / np.maximum((w ** 2).sum(1), 1e-9)), utilisation=np.mean((L + S).sum(1)) / B,
                short_share=S.sum() / max(L.sum() + S.sum(), 1), persistence=jac,
                cap_share=(absV == 3).sum() / max(held.sum(), 1), assets_held=held.sum(1).mean() / r.assets)


# A: lambda morph ----------------------------------------------------------
def fig_lambda_morph(df, key="po_a050_t10_orig"):
    g = df[df.inst.str.startswith(key)].sort_values("lam", ascending=False)
    if len(g) < 2: return
    fig = plt.figure(figsize=(5.2 * len(g), 5))
    for k, (_, r) in enumerate(g.iterrows()):
        V = r.L - r.S; ax = fig.add_subplot(1, len(g), k + 1, projection="3d")
        t, i = np.nonzero(V); c = cm.coolwarm((V[t, i] + 3) / 6)
        ax.bar3d(i, t, np.zeros(len(i)), 0.9, 0.9, V[t, i], color=c, shade=True, edgecolor="none")
        ax.set(xlabel="asset", ylabel="day", zlabel="net units", zlim=(-3, 3), title=f"$\\lambda$ = {r.lam:g}")
        ax.view_init(28, -60)
    fig.suptitle(f"{key}: how risk aversion reshapes the book", y=0.98, fontsize=15)
    fig.savefig(OUT / "handin_A_lambda_morph.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# B: lambda x price-sample surface ---------------------------------------------
def fig_lambda_variant_surface(df, z="eff_assets"):
    g = df[(df.assets == 50) & (df.days == 10)].copy()
    if len(g) < 6: return
    g[z] = [feats(r)[z] for _, r in g.iterrows()]
    vars_ = sorted(g.variant.unique(), key=lambda v: (v != "orig", v)); lams = np.sort(g.lam.unique())
    X, Y = np.meshgrid(np.log10(lams), np.arange(len(vars_)))
    pts = np.array([[np.log10(r.lam), vars_.index(r.variant)] for _, r in g.iterrows()])
    rbf = RBFInterpolator(pts, g[z].values, kernel="thin_plate_spline", smoothing=1e-2)
    xs, ys = np.linspace(X.min(), X.max(), 60), np.linspace(0, len(vars_) - 1, 60); XX, YY = np.meshgrid(xs, ys)
    ZZ = rbf(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    fig = plt.figure(figsize=(9, 6.5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.85, edgecolor="none", antialiased=True)
    ax.contourf(XX, YY, ZZ, zdir="z", offset=ZZ.min() - 0.1 * np.ptp(ZZ), cmap="viridis", alpha=0.6)
    ax.scatter(pts[:, 0], pts[:, 1], g[z], c="k", s=45, depthshade=False, label="actual solutions")
    ax.set(xlabel=r"log$_{10}\,\lambda$", ylabel="price sample", zlabel=z.replace("_", " "),
           yticks=range(len(vars_)), yticklabels=vars_, title=f"{z.replace('_', ' ')} over risk aversion and price sample (a050, t10)")
    ax.legend(); ax.view_init(28, -120)
    fig.savefig(OUT / "handin_B_lambda_variant_surface.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# C: all instances, descriptor PCA map -------------------------------------------
def fig_descriptor_map(df, z="persistence"):
    F = pd.DataFrame([feats(r) for _, r in df.iterrows()]); X = ((F - F.mean()) / (F.std() + 1e-9)).values
    mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False); P = (X - mu) @ Vt[:2].T; ev = s ** 2 / (s ** 2).sum()
    a = [np.linspace(P[:, i].min(), P[:, i].max(), 60) for i in (0, 1)]; A, B = np.meshgrid(*a)
    pad = 0.15
    Z = RBFInterpolator(P, F[z].values, kernel="thin_plate_spline", smoothing=0.5)(np.c_[A.ravel(), B.ravel()]).reshape(A.shape)
    fig = plt.figure(figsize=(9.5, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, cmap="magma", alpha=0.55, edgecolor="none")
    sc = ax.scatter(P[:, 0], P[:, 1], F[z], c=np.log10(df.lam), s=25 + df.assets * 0.9, cmap="cool", edgecolor="k", depthshade=False)
    for p, zz, (_, r) in zip(P, F[z], df.iterrows()): ax.text(p[0], p[1], zz + 0.02, f"a{r.assets}", fontsize=7)
    fig.colorbar(sc, shrink=0.55, pad=0.08, label=r"log$_{10}\,\lambda$")
    ax.set(xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})", zlabel=z, title="Every handin solution on one map (size-independent descriptors)")
    ax.view_init(26, -55); fig.savefig(OUT / "handin_C_descriptor_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# D: scaling bars -----------------------------------------------------------------
def fig_scaling(df, lam=1e-3):
    g = df[(df.lam == lam) & (df.variant == "orig")].sort_values("assets")
    if len(g) < 2: return
    F = pd.DataFrame([feats(r) for _, r in g.iterrows()]); cols = ["assets_held", "utilisation", "persistence", "short_share", "cap_share"]
    fig = plt.figure(figsize=(9, 6.5)); ax = fig.add_subplot(111, projection="3d")
    for j, c in enumerate(cols):
        ax.bar(np.arange(len(g)), F[c].values, zs=j, zdir="y", width=0.6, color=cm.viridis(j / len(cols)), alpha=0.9)
    ax.set(xticks=range(len(g)), xticklabels=[f"a{a}\nt{t}" for a, t in zip(g.assets, g.days)], yticks=range(len(cols)),
           yticklabels=cols, title=rf"Solution structure vs instance size ($\lambda$={lam:g})")
    ax.view_init(25, -65); fig.savefig(OUT / "handin_D_scaling.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    df = load_folder(sys.argv[1] if len(sys.argv) > 1 else "handin")
    print(df[["name", "assets", "days", "budget", "lam", "objective"]].to_string())
    for grp, g in df.groupby(["assets", "days"]):
        sets = list(g.symbols)
        if len(sets) > 1: print(f"a{grp[0]} t{grp[1]}: asset-set overlap between files (Jaccard of symbols held): "
                                f"{np.mean([len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]):.2f}")
    fig_lambda_morph(df); fig_lambda_variant_surface(df); fig_descriptor_map(df); fig_scaling(df)
    print("figures:", sorted(p.name for p in OUT.glob("handin_*.png")))
