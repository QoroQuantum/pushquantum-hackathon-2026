#!/usr/bin/env python3
"""viz_surfaces.py - 3D surface visualisations for the QOBLIB QUBO hackathon.

Pure NumPy + Matplotlib: no Divi / Maestro needed (partitioned SA never builds circuits).

    python viz_surfaces.py --demo                       # synthetic data, all figures
    python viz_surfaces.py --sweep results/sweep.csv    # SA hyper-parameter surface
    python viz_surfaces.py --qubo block.npy             # QAOA/CVaR landscape of a saved block Q

Figures (saved to figures/):
  1. sa_param_surface   : final objective vs (sa_sweeps, group_size)      [from CSV or demo]
  2. qubo_pca_landscape : relaxed QUBO energy over the plane spanned by the top-2 PCA
                          directions of an SA trajectory (orqviz-style PCA scan), trajectory on top
  3. qaoa_cvar_surface  : p=1 QAOA <H> and CVaR_alpha over (gamma, beta), exact statevector (<=14 qubits)
CSV columns for --sweep: group_size, sa_sweeps, objective   (extra columns ignored; mean over duplicates)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = Path("figures"); OUT.mkdir(exist_ok=True)


def energies(Q, X):
    """x^T Q x for rows of X (binary or relaxed)."""
    return np.einsum("ki,ij,kj->k", X, Q, X)


def random_qubo(n, seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n)); Q = (A + A.T) / 2
    return Q / np.abs(Q).max()


def sa_trajectory(Q, sweeps=300, seed=0):
    """Plain Metropolis SA; returns visited states (sweeps*n, n). Swap in your solver's history if you log it."""
    r = np.random.default_rng(seed); n = len(Q)
    x = r.integers(0, 2, n).astype(float); traj = []
    for T in np.geomspace(2.0, 0.02, sweeps):
        for i in r.permutation(n):
            d = (1 - 2 * x[i]) * (Q[i, i] + 2 * Q[i] @ x - 2 * Q[i, i] * x[i])
            if d <= 0 or r.random() < np.exp(-d / T):
                x[i] = 1 - x[i]
        traj.append(x.copy())
    return np.array(traj)


def surface(ax, X, Y, Z, title, xl, yl, zl, cmap=cm.viridis, elev=30, azim=-125):
    s = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.95, antialiased=True)
    ax.contour(X, Y, Z, zdir="z", offset=np.nanmin(Z), cmap=cmap, levels=12, alpha=0.6)
    ax.set(xlabel=xl, ylabel=yl, zlabel=zl, title=title); ax.view_init(elev, azim)
    return s


# 1 ------------------------------------------------------------------------
def fig_param_surface(csv=None):
    if csv:
        import pandas as pd
        df = pd.read_csv(csv).groupby(["group_size", "sa_sweeps"], as_index=False)["objective"].mean()
        P = df.pivot(index="group_size", columns="sa_sweeps", values="objective")
        gs, sw, Z = P.index.values, P.columns.values, P.values
    else:  # demo: diminishing returns in sweeps, sweet spot in block size
        gs, sw = np.arange(2, 13), np.array([10, 20, 40, 80, 160, 320, 640])
        G, S = np.meshgrid(gs, sw, indexing="ij")
        Z = -1000 - 120 * np.exp(-((G - 7) / 3.5) ** 2) - 80 * (1 - np.exp(-S / 120)) + 8 * np.sin(G)
    S, G = np.meshgrid(np.log2(sw), gs)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    s = surface(ax, S, G, Z, "Partitioned SA: objective vs. effort and block size",
                "log2(SA sweeps)", "assets per block", "objective (lower = better)", cmap=cm.magma_r)
    i, j = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.scatter(S[i, j], G[i, j], Z[i, j], c="cyan", s=80, marker="*", label="best")
    ax.legend(); fig.colorbar(s, shrink=0.6, pad=0.1)
    fig.savefig(OUT / "sa_param_surface.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# 2 ------------------------------------------------------------------------
def fig_pca_landscape(Q, n_grid=70):
    traj = sa_trajectory(Q)
    mu = traj.mean(0); U, sv, Vt = np.linalg.svd(traj - mu, full_matrices=False)
    var = (sv ** 2)[:2].sum() / (sv ** 2).sum()
    P = (traj - mu) @ Vt[:2].T
    pad = 0.3 * np.ptp(P, axis=0)
    a = np.linspace(P[:, 0].min() - pad[0], P[:, 0].max() + pad[0], n_grid)
    b = np.linspace(P[:, 1].min() - pad[1], P[:, 1].max() + pad[1], n_grid)
    A, B = np.meshgrid(a, b)
    pts = mu + A.reshape(-1, 1) * Vt[0] + B.reshape(-1, 1) * Vt[1]
    Z = energies(Q, np.clip(pts, 0, 1)).reshape(A.shape)  # clip = box-relaxed QUBO
    ez = energies(Q, traj)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    surface(ax, A, B, Z, f"Relaxed QUBO energy on SA-trajectory PCA plane ({var:.0%} var.)",
            "PC1", "PC2", "x$^T$Qx", cmap=cm.coolwarm)
    ax.plot(P[:, 0], P[:, 1], ez, c="k", lw=1.2, label="SA trajectory")
    ax.scatter(*P[0], ez[0], c="lime", s=50, label="start"); ax.scatter(*P[-1], ez[-1], c="red", s=70, label="end")
    ax.legend(); fig.savefig(OUT / "qubo_pca_landscape.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# 3 ------------------------------------------------------------------------
def cvar(E, p, alpha):
    o = np.argsort(E); cp = np.cumsum(p[o]); k = np.searchsorted(cp, alpha)
    w = p[o].copy(); w[k] -= cp[k] - alpha; w[k + 1:] = 0
    return (w * E[o]).sum() / alpha


def qaoa_p1(Ediag, n, gamma, beta):
    """|psi> = e^{-i beta sum X} e^{-i gamma H} |+>^n ; returns probabilities. (Divi uses +sum X -> sign of beta flips.)"""
    psi = np.exp(-1j * gamma * Ediag) / np.sqrt(2 ** n)
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a, b = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return np.abs(psi.reshape(-1)) ** 2


def fig_qaoa_cvar(Q, alpha=0.1, n_grid=36):
    n = len(Q)
    assert n <= 14, "exact statevector: use a <=14-qubit block"
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[::-1]) & 1).astype(float)
    E = energies(Q, X); E = (E - E.mean()) / E.std()
    g = np.linspace(0, np.pi, n_grid); b = np.linspace(0, np.pi / 2, n_grid)
    mean = np.zeros((n_grid, n_grid)); cv = np.zeros_like(mean)
    for i, gi in enumerate(g):
        for j, bj in enumerate(b):
            p = qaoa_p1(E, n, gi, bj); mean[i, j] = p @ E; cv[i, j] = cvar(E, p, alpha)
    B, G = np.meshgrid(b, g)
    fig = plt.figure(figsize=(14, 6))
    for k, (Z, t) in enumerate([(mean, r"$\langle H\rangle$"), (cv, rf"CVaR$_{{\alpha={alpha}}}$")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        surface(ax, G, B, Z, f"p=1 QAOA, {n} qubits: {t}", r"$\gamma$", r"$\beta$", "normalised energy", cmap=cm.plasma_r)
        i, j = np.unravel_index(Z.argmin(), Z.shape)
        ax.scatter(G[i, j], B[i, j], Z[i, j], c="cyan", s=80, marker="*")
    fig.savefig(OUT / "qaoa_cvar_surface.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--sweep"); ap.add_argument("--qubo")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    Q = np.load(a.qubo) if a.qubo else random_qubo(12)
    if a.sweep or a.demo: fig_param_surface(a.sweep)
    fig_pca_landscape(random_qubo(30) if a.demo else Q)
    fig_qaoa_cvar(Q, a.alpha)
    print("saved to", OUT.resolve())
