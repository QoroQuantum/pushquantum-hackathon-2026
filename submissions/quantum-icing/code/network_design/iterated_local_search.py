"""Iterated local search / VNS for QOBLIB network design (network24 by default).

    NETWORK_SOL_DIR=solutions/agent_ils python network_ils.py run --minutes 10 --chains 5

Each chain is an independent sequential ILS (one CPU each):
  * neighbourhoods: 2-exchange (all pairs), node relabelling (swap the roles of two nodes,
    i.e. a QAP move on the fixed graph shape), directed 3/4-cycle reversal, sampled 3-rotations
    of arc heads;
  * every candidate gets a cheap *valid lower bound* on its LP value first,
    LB(w) = sum_ij d_ij dist_w(i,j) / sum_a w_a for any w >= 0 (LP weak duality), using unit
    weights and the parent's LP duals carried over to the child. Candidates are evaluated in
    increasing order of the bound and the scan stops as soon as LB >= incumbent;
  * the exact LP (same model as the referee) decides; kicks of k random moves perturb a local
    optimum; worse local optima are accepted within a small threshold; tabu on recent arcs.
Banking goes through ``network_pipeline.bank`` (route_integral -> .sol -> referee checker).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import fcntl
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import scipy.optimize._highspy._core as _h

import qoblib_tools as qt
from network_design import pipeline as npl

N_DEFAULT = 24
INF = 1e18


# ── Graph helpers ──────────────────────────────────────────────────────────────────────

def kautz(n: int = 24) -> list[tuple[int, int]]:
    """Kautz graph K(2, L): strings over {0,1,2}, no two equal consecutive symbols."""
    L = {6: 2, 12: 3, 24: 4, 48: 5}[n]
    nodes = []

    def rec(s):
        if len(s) == L:
            nodes.append(tuple(s))
            return
        for x in range(3):
            if not s or s[-1] != x:
                rec(s + [x])

    rec([])
    idx = {s: i for i, s in enumerate(nodes)}
    arcs = []
    for s in nodes:
        for x in range(3):
            if x != s[-1]:
                arcs.append((idx[s], idx[s[1:] + (x,)]))
    return sorted(arcs)


def imase_itoh(n: int, a: int = 1) -> list[tuple[int, int]] | None:
    """Imase-Itoh style digraph i -> (-2i - r) mod n, r in {1, 2} (shifted by a)."""
    arcs = set()
    for i in range(n):
        for r in (a, a + 1):
            j = (-2 * i - r) % n
            if j == i or (i, j) in arcs:
                return None
            arcs.add((i, j))
    return sorted(arcs) if qt.is_valid_topology(n, arcs) else None


def valid(arcs, n) -> bool:
    s = set(arcs)
    return len(s) == 2 * n and all(i != j for i, j in s) and qt.is_valid_topology(n, s)


# ── Scoring ────────────────────────────────────────────────────────────────────────────

class WarmLP:
    def __init__(self, demand, arcs, eps=0.0):
        self.n = n = len(demand); self.m = m = len(arcs)
        self.slots = [tuple(a) for a in arcs]
        nr_eq = n * (n - 1)
        self.nr_eq = nr_eq
        rows, cols, vals = [], [], []
        ub = np.full(n * m + 1, np.inf)
        for k in range(n):
            for a, (i, j) in enumerate(self.slots):
                col = k * m + a
                if j != k: rows.append(self.r(k, j)); cols.append(col); vals.append(1.0)
                else: ub[col] = 0.0
                if i != k: rows.append(self.r(k, i)); cols.append(col); vals.append(-1.0)
                rows.append(nr_eq + a); cols.append(col); vals.append(1.0)
        for a in range(m):
            rows.append(nr_eq + a); cols.append(n * m); vals.append(-1.0)
        A = sps.csc_array((vals, (rows, cols)), shape=(nr_eq + m, n * m + 1))
        A.sort_indices()
        b = np.array([demand[k, v] * qt.INTSCALE for k in range(n) for v in range(n) if v != k], float)
        lhs = np.concatenate([b, np.full(m, -np.inf)]); rhs = np.concatenate([b, np.zeros(m)])
        c = np.full(n * m + 1, float(eps)); c[-1] = 1.0  # eps: tie-break on total flow
        lp = _h.HighsLp()
        lp.num_col_ = n * m + 1; lp.num_row_ = nr_eq + m
        lp.a_matrix_.num_col_ = n * m + 1; lp.a_matrix_.num_row_ = nr_eq + m
        lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
        lp.col_cost_ = c; lp.col_lower_ = np.zeros(n * m + 1); lp.col_upper_ = ub
        lp.row_lower_ = lhs; lp.row_upper_ = rhs
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        h = _h._Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        self.h = h
        self.iters = 0

    def r(self, k, v):
        return k * (self.n - 1) + (v if v < k else v - 1)

    def set_slot(self, a, new):
        """Re-point slot ``a`` (a block of n columns, one per commodity) to arc ``new``.
        All old entries are cleared before the new ones are written (an arc may reverse)."""
        i, j = self.slots[a]; i2, j2 = new
        n, m, h = self.n, self.m, self.h
        for k in range(n):
            col = k * m + a
            if j != k: h.changeCoeff(self.r(k, j), col, 0.0)
            if i != k: h.changeCoeff(self.r(k, i), col, 0.0)
        for k in range(n):
            col = k * m + a
            if j2 != k: h.changeCoeff(self.r(k, j2), col, 1.0)
            if i2 != k: h.changeCoeff(self.r(k, i2), col, -1.0)
            if j == k or j2 == k:
                h.changeColBounds(col, 0.0, 0.0 if j2 == k else np.inf)
        self.slots[a] = (i2, j2)

    def solve(self, arcs, cutoff=None, basis=None):
        """LP optimum (or None if the dual simplex proves it is >= cutoff) and duals."""
        for a, new in enumerate(arcs):
            if tuple(new) != self.slots[a]:
                self.set_slot(a, tuple(new))
        if basis is not None:
            self.h.setBasis(basis)
        self.h.setOptionValue("objective_bound", np.inf if cutoff is None else float(cutoff))
        self.h.run()
        st = self.h.getModelStatus()
        if st not in (_h.HighsModelStatus.kOptimal, _h.HighsModelStatus.kObjectiveBound):
            self.h.clearSolver()  # warm start went wrong numerically: solve cold
            self.h.run()
            st = self.h.getModelStatus()
        info = self.h.getInfo()
        self.iters += info.simplex_iteration_count
        if st == _h.HighsModelStatus.kObjectiveBound:
            return None, None
        if st != _h.HighsModelStatus.kOptimal:
            return float("inf"), None
        sol = self.h.getSolution()
        du = -np.asarray(sol.row_dual)[self.nr_eq:]
        self.last_z = float(sol.col_value[-1])
        return float(info.objective_function_value), du


class Scorer:
    """Warm-started LP (scipy's vendored HiGHS, same model as the referee) + cheap bounds."""

    def __init__(self, n: int):
        self.n = n
        self.demand = qt.load_network_demand(n)
        self.D = self.demand.astype(float) * qt.INTSCALE
        self.cache: dict[tuple, float] = {}  # key -> LP value or proven lower bound
        self.n_lp = 0
        self.n_cut = 0
        self.warm: WarmLP | None = None
        self.parent_basis = None
        self.eps = 0.0
        self.true_z: dict[tuple, float] = {}

    def lp(self, arcs, cutoff=None, use_parent=True):
        """Returns (z, duals); z None if proven >= cutoff."""
        key = tuple(sorted(arcs))
        if self.warm is None:
            self.warm = WarmLP(self.demand, list(arcs), self.eps)
        basis = self.parent_basis if use_parent else None
        z, du = self.warm.solve(arcs, cutoff, basis)
        self.n_lp += 1
        if z is None:
            self.n_cut += 1
            self.cache[key] = max(self.cache.get(key, 0.0), float(cutoff))
        else:
            self.cache[key] = z
            self.true_z[key] = self.warm.last_z
        return z, du

    def set_parent(self):
        """Remember the basis of the LP just solved as the warm start for its neighbours."""
        self.parent_basis = self.warm.h.getBasis()

    def bounds(self, cand_arcs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Batch lower bounds. cand_arcs: (B, m, 2) ints, weights: (B, K, m) >= 0.
        Returns (B,) max over the K weightings of sum d*dist_w / sum w."""
        B, m, _ = cand_arcs.shape
        K = weights.shape[1]
        n = self.n
        d = np.full((B, K, n, n), INF)
        bi = np.repeat(np.arange(B), m)
        src = cand_arcs[:, :, 0].ravel()
        dst = cand_arcs[:, :, 1].ravel()
        for k in range(K):
            d[bi, k, src, dst] = weights[:, k, :].ravel() + 1e-9
        idx = np.arange(n)
        d[:, :, idx, idx] = 0.0
        for v in range(n):
            np.minimum(d, d[:, :, :, v:v + 1] + d[:, :, v:v + 1, :], out=d)
        num = np.einsum("bkij,ij->bk", np.minimum(d, 1e12), self.D)
        den = weights.sum(axis=2)
        return (num / den).max(axis=1)


# ── Moves ──────────────────────────────────────────────────────────────────────────────
# A move returns (new_arc_list, mapping) where mapping[i] = index of the parent arc whose
# dual is carried to new arc i (or -1 for genuinely new arcs).

def move_2ex(arcs, p, q):
    (a, b), (c, d) = arcs[p], arcs[q]
    if a == c or b == d or a == d or c == b:
        return None
    new = list(arcs)
    new[p], new[q] = (a, d), (c, b)
    return new


def move_relabel(arcs, u, v):
    def f(x):
        return v if x == u else (u if x == v else x)
    return [(f(i), f(j)) for i, j in arcs]


def move_rot3(arcs, p, q, r, direction):
    (a, b), (c, d), (e, f) = arcs[p], arcs[q], arcs[r]
    new = list(arcs)
    if direction == 0:
        new[p], new[q], new[r] = (a, d), (c, f), (e, b)
    else:
        new[p], new[q], new[r] = (a, f), (c, b), (e, d)
    return new


def directed_cycles(arcs, n, max_len=4):
    out = [[] for _ in range(n)]
    for idx, (i, j) in enumerate(arcs):
        out[i].append((j, idx))
    cycles = set()
    for s in range(n):
        stack = [(s, [s], [])]
        while stack:
            v, path, aidx = stack.pop()
            for w, ai in out[v]:
                if w == s and len(path) >= 3:
                    cycles.add(tuple(sorted(aidx + [ai])))
                elif w > s and w not in path and len(path) < max_len:
                    stack.append((w, path + [w], aidx + [ai]))
    return [list(c) for c in cycles]


def move_reverse(arcs, idxs):
    new = list(arcs)
    for t in idxs:
        i, j = arcs[t]
        new[t] = (j, i)
    return new


def is_ok(new, n):
    s = set(new)
    return len(s) == len(new) and all(i != j for i, j in new)


# ── Chain ──────────────────────────────────────────────────────────────────────────────

class Chain:
    def __init__(self, n, seed, args, log):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sc = Scorer(n)
        self.sc.eps = args.eps
        self.args = args
        self.seed = seed
        self.log = log
        self.banked = npl.banked_score(n, self.sc.demand)
        self.best_z = INF
        self.best = None
        self.tabu: dict[tuple, int] = {}
        self.it = 0
        self.stats = {"2ex": [0, 0], "relabel": [0, 0], "rev": [0, 0], "rot3": [0, 0]}

    # candidates around `arcs` with duals `du`
    def neighbourhood(self, arcs, kinds, n_rot=600):
        n, m = self.n, len(arcs)
        present = set(arcs)
        cands = []
        if "2ex" in kinds:
            for p in range(m):
                for q in range(p + 1, m):
                    new = move_2ex(arcs, p, q)
                    if new is not None and new[p] not in present and new[q] not in present:
                        cands.append(("2ex", new, (p, q)))
        if "relabel" in kinds:
            for u in range(n):
                for v in range(u + 1, n):
                    new = move_relabel(arcs, u, v)
                    if set(new) != present:
                        cands.append(("relabel", new, None))
        if "rev" in kinds:
            for cyc in directed_cycles(arcs, n, 4):
                new = move_reverse(arcs, cyc)
                if is_ok(new, n):
                    cands.append(("rev", new, None))
        if "rot3" in kinds:
            seen = set()
            for _ in range(n_rot * 3):
                if len(seen) >= n_rot:
                    break
                p, q, r = self.rng.choice(m, 3, replace=False)
                dr = int(self.rng.integers(2))
                key = (min(p, q, r), dr, tuple(sorted((p, q, r))))
                if key in seen:
                    continue
                new = move_rot3(arcs, p, q, r, dr)
                if is_ok(new, n):
                    seen.add(key)
                    cands.append(("rot3", new, None))
        return cands

    def weights_for(self, parent, du, new):
        """K=3 weightings: positional duals, carried duals (new arcs 0), carried duals (new arcs = mean)."""
        dd = {a: max(float(x), 0.0) for a, x in zip(parent, du)}
        mean = np.mean(list(dd.values()))
        w1 = np.array([dd.get(a, 0.0) for a in new])
        w2 = np.array([dd.get(a, mean) for a in new])
        if w1.sum() <= 0:
            w1 = np.ones(len(new))
        w0 = np.clip(np.asarray(du, float), 0, None)  # positional: slot t inherits dual of parent slot t
        return np.stack([w0, w1, w2])

    def descent(self, cur, z, du, kinds):
        """First-improvement descent; candidates evaluated in increasing LB order, each LP
        warm-started from the parent's basis and cut off once it provably cannot improve.
        Precondition: the last LP solved was ``cur`` (its basis becomes the parent basis)."""
        cur = list(cur)
        self.sc.set_parent()
        while True:
            cands = self.neighbourhood(cur, kinds)
            cache = self.sc.cache
            cands = [c for c in cands if cache.get(tuple(sorted(c[1])), -1.0) < z - 1e-6]
            if not cands:
                return cur, z, du
            arr = np.array([c[1] for c in cands])
            W = np.stack([self.weights_for(cur, du, c[1]) for c in cands])
            lbs = np.concatenate([self.sc.bounds(arr[i:i + 256], W[i:i + 256])
                                  for i in range(0, len(cands), 256)])
            order = np.argsort(lbs)
            improved = False
            target = z - 1e-3
            for t in order:
                if lbs[t] >= target:
                    break
                kind, new, _ = cands[t]
                nz, ndu = self.sc.lp(new, cutoff=target)
                self.stats[kind][0] += 1
                if nz is not None and nz < target:
                    self.stats[kind][1] += 1
                    cur, z, du = list(new), nz, ndu
                    self.sc.set_parent()
                    improved = True
                    self.update_best(cur, z, f"descent {kind}")
                    break
            if not improved:
                return cur, z, du

    def deep(self, cur, z, du, kinds, deadline):
        """Depth-2 search: score every neighbour of ``cur`` (cut off at z*(1+deep_tol)), then
        run a full descent from each of the best non-improving ones (this chain's share)."""
        self.sc.set_parent()
        cands = self.neighbourhood(cur, kinds)
        cut = z * (1 + self.args.deep_tol)
        scored = []
        for kind, new, _ in cands:
            nz, _ = self.sc.lp(new, cutoff=cut)
            if nz is not None and not math.isinf(nz):
                scored.append((nz, kind, new))
        scored.sort(key=lambda x: x[0])
        idx, nch = self.seed - self.args.seed, self.args.chains
        mine = scored[idx::nch][: self.args.deep]
        self.log(f"[seed {self.seed}] deep: {len(scored)} neighbours within {self.args.deep_tol:.1%}, "
                 f"exploring {len(mine)} (z {mine[0][0]:.0f}..{mine[-1][0]:.0f})" if mine else "deep: none")
        best = (z, cur, du)
        for nz, kind, new in mine:
            if time.time() > deadline:
                break
            cz, cdu = self.sc.lp(new)
            if cdu is None:
                continue
            c2, cz, cdu = self.descent(new, cz, cdu, kinds)
            self.update_best(c2, cz, f"deep via {kind}")
            if cz < best[0] - 1e-3:
                best = (cz, c2, cdu)
                self.log(f"[seed {self.seed}] deep: escaped to {cz:.1f}")
        z, cur, du = best
        self.sc.lp(cur)
        return cur, z, du

    def kick(self, cur, k):
        cur = list(cur)
        n, m = self.n, len(cur)
        done = 0
        tries = 0
        while done < k and tries < 200:
            tries += 1
            r = self.rng.random()
            if r < self.args.p_relabel:
                u, v = self.rng.choice(n, 2, replace=False)
                new = move_relabel(cur, u, v)
            elif r < self.args.p_relabel + 0.15:
                cyc = directed_cycles(cur, n, 4)
                if not cyc:
                    continue
                new = move_reverse(cur, cyc[self.rng.integers(len(cyc))])
            else:
                p, q = self.rng.choice(m, 2, replace=False)
                new = move_2ex(cur, p, q)
            if new is None or not is_ok(new, n):
                continue
            cur = new
            done += 1
        return cur

    def update_best(self, arcs, z, how):
        z = self.sc.true_z.get(tuple(sorted(arcs)), z)
        if z < self.best_z - 1e-6:
            self.best_z, self.best = z, sorted(arcs)
            self.log(f"[seed {self.seed}] it {self.it} new chain best LP {z:.1f} ({how}) lps={self.sc.n_lp}")
            if self.banked is None or math.ceil(z - 1e-6) < self.banked:
                self.try_bank(arcs, how)

    def try_bank(self, arcs, how):
        lock = npl.SOL_DIR / ".ils.lock"
        npl.SOL_DIR.mkdir(exist_ok=True, parents=True)
        with open(lock, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            ok, score, msg = npl.bank(self.n, self.sc.demand, sorted(arcs),
                                      f"network_ils seed={self.seed} {how}")
            self.banked = npl.banked_score(self.n, self.sc.demand)
        self.log(f"[seed {self.seed}] bank: {msg}")

    def run(self, start, minutes):
        n = self.n
        deadline = time.time() + 60 * minutes
        cur = start_topology(start, n, self.rng, self.sc.demand)
        z, du = self.sc.lp(cur, use_parent=False)
        self.log(f"[seed {self.seed}] start={start} LP {z:.1f}")
        kinds = self.args.kinds.split(",")
        if self.args.deep > 0:
            cur, z, du = self.deep(cur, z, du, kinds, deadline)
        cur, z, du = self.descent(cur, z, du, kinds)
        self.update_best(cur, z, "initial descent")
        home, home_z, home_du = cur, z, du
        k = self.args.kmin
        since_best = 0
        while time.time() < deadline:
            self.it += 1
            cand = self.kick(home, k)
            cz, cdu = self.sc.lp(cand)
            if cdu is None:
                continue
            cand, cz, cdu = self.descent(cand, cz, cdu, kinds)
            if self.args.rot3 and time.time() < deadline:
                self.sc.lp(cand)
                cand, cz, cdu = self.descent(cand, cz, cdu, ["rot3"])
                self.sc.lp(cand)
                cand, cz, cdu = self.descent(cand, cz, cdu, kinds)
            prev_best = self.best_z
            self.update_best(cand, cz, f"ils k={k}")
            if cz < home_z - 1e-6:
                home, home_z, home_du = cand, cz, cdu
                k = self.args.kmin
            else:
                k = min(k + 1, self.args.kmax) if self.rng.random() < 0.5 else k
                if cz < home_z + self.args.accept * home_z and sorted(cand) != sorted(home):
                    home, home_z, home_du = cand, cz, cdu
            if self.best_z < prev_best - 1e-6:
                since_best = 0
            else:
                since_best += 1
            if since_best >= self.args.restart_best and self.best is not None:
                home, home_z = list(self.best), self.best_z
                since_best = 0
                if self.args.sync:  # adopt the folder's banked topology if another chain beat us
                    b = npl.banked_score(n, self.sc.demand)
                    if b is not None and b < math.ceil(self.best_z - 1e-6):
                        _, arcs, _ = qt.read_network_solution(npl.banked_path(n))
                        hz, _ = self.sc.lp(arcs, use_parent=False)
                        home, home_z = list(arcs), hz
                        self.log(f"[seed {self.seed}] sync: adopted banked {b}")
            if self.it % 10 == 0:
                self.log(f"[seed {self.seed}] it {self.it} home {home_z:.0f} best {self.best_z:.0f} "
                         f"k={k} lps={self.sc.n_lp} cut={self.sc.n_cut} stats={self.stats}")
        return self.best_z, self.best


def start_topology(start, n, rng, demand):
    if start == "banked" and npl.banked_path(n).exists():
        _, arcs, _ = qt.read_network_solution(npl.banked_path(n))
        return sorted(arcs)
    if start == "kautz":
        arcs = kautz(n)
        perm = rng.permutation(n)
        return sorted((int(perm[i]), int(perm[j])) for i, j in arcs)
    if start == "greedy":
        return sorted(qt.repair_topology(n, [], demand))
    return sorted(qt.random_topology(n, rng))


def _chain_main(n, seed, start, minutes, args, logfile):
    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(logfile, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    ch = Chain(n, seed, args, log)
    z, arcs = ch.run(start, minutes)
    log(f"[seed {seed}] DONE best LP {z:.1f} lps={ch.sc.n_lp} stats={ch.stats}")
    return z, arcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", default="banked,kautz,kautz,random,greedy",
                    help="comma list, cycled over chains")
    ap.add_argument("--kinds", default="2ex,relabel,rev")
    ap.add_argument("--rot3", action="store_true")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--p_relabel", type=float, default=0.3)
    ap.add_argument("--accept", type=float, default=0.003, help="relative worsening accepted")
    ap.add_argument("--restart_best", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="LP objective z + eps*total_flow (lexicographic tie-break, plateau moves)")
    ap.add_argument("--deep", type=int, default=0,
                    help="depth-2 search from the start: descend from this many best neighbours per chain")
    ap.add_argument("--deep_tol", type=float, default=0.01)
    ap.add_argument("--sync", action="store_true",
                    help="on restart, adopt the banked topology if another chain found better")
    ap.add_argument("--log", default="solutions/agent_ils/ils_log.txt")
    args = ap.parse_args()
    assert args.chains <= 5, "at most 5 worker processes"
    starts = args.starts.split(",")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    jobs = [(args.n, args.seed + c, starts[c % len(starts)], args.minutes, args, args.log)
            for c in range(args.chains)]
    if args.chains == 1:
        _chain_main(*jobs[0])
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(args.chains) as pool:
        res = pool.starmap(_chain_main, jobs)
    for (z, _), j in zip(res, jobs):
        print(f"seed {j[1]} start {j[2]}: LP {z:.1f}")


if __name__ == "__main__":
    main()
