"""QAOA block solver on a signed 3-bit encoding (3 qubits per asset instead of 6).

A day x asset-group block (rest of the portfolio fixed) has cost
``V A V + sum_i f_i(v_i)`` (pairwise risk + per-asset terms; see
``asset_groups.BlockEvaluator.costs``). Encode each net position with 3 bits,
``v = w . b + c`` with ``w = s*(1, 2, -3), c = 0`` or ``w = s*(1, 2, 3), c = -3*s``
(``s = +-1``). Every one of the 8 codes is a valid position (one value appears twice),
so QAOA never samples an invalid holding, and ``V A V`` stays exactly quadratic.

Each ``f_i`` (+ Lagrangian ``mu*|v_i|`` for the budget) is a function of 3 bits; it is
quadratic iff its cubic Moebius coefficient is 0. We make the QUBO exact on 7 codes and
give the *duplicate* code whatever energy cancels the cubic term; if that energy is
>= its true cost (margin >= 0) the QUBO ground state equals the Lagrangian optimum.
Per asset we pick, among 8 variants (2 encodings x 2 signs x 2 duplicate codes), the
one with the largest margin.

    python qaoa_signed_blocks.py --check                        # exactness, brute force
    python qaoa_signed_blocks.py --blocks 4 --group 6          # local Maestro
    python qaoa_signed_blocks.py --blocks 1 --group 8 --cloud  # Qoro cloud (credits)

QAOA starts from TQA angles by default (``--init random`` for Divi's uniform start).
On a050 λ1e-3, 8 blocks of 12 qubits, p=2: TQA found the block optimum in its top
64 samples on 6/8 blocks vs 4/8 for random (one seed).

On the cloud each COBYLA step is one round trip (~20 s): 40 steps at 24 qubits took
14 min and 16.9 credits. Use a batched optimiser there (e.g. MonteCarloOptimizer).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import dimod
import numpy as np

import qoblib_tools as qt
from asset_groups import BlockEvaluator, BlockResult, _load, _net_configs, sa_start

CODES = np.array(list(itertools.product([0, 1], repeat=3)))  # (8, 3)
SUBSETS = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def _moebius(g: np.ndarray) -> dict:
    """Multilinear coefficients of g over CODES (dict subset -> coef)."""
    idx = {tuple(c): k for k, c in enumerate(CODES)}
    out = {}
    for S in SUBSETS:
        tot = 0.0
        for r in range(len(S) + 1):
            for T in itertools.combinations(S, r):
                code = tuple(1 if j in T else 0 for j in range(3))
                tot += (-1) ** (len(S) - len(T)) * g[idx[code]]
        out[S] = tot
    return out


def _variants():
    for base, c0 in (((1, 2, -3), 0), ((1, 2, 3), -3)):
        for s in (1, -1):
            w, c = s * np.array(base), s * c0
            v = CODES @ w + c
            for val in np.unique(v):
                dup = np.flatnonzero(v == val)
                if len(dup) == 2:
                    for d in dup:
                        yield w, c, v, int(d)


def block_qubo(ev: BlockEvaluator, L, S, t, G: np.ndarray, mu: float):
    """BQM over 3|G| bits (bit 3i+j = bit j of asset G[i]) and the decoding (w, c).

    ``t`` is a day or a run of consecutive days (the same holding on each, as in
    ``BlockEvaluator.costs``).
    """
    g = len(G)
    days = [t] if np.isscalar(t) else list(t)
    A = sum(ev.Rs[d][np.ix_(G, G)] for d in days)
    vals = np.arange(-3, 4)
    c0, _ = ev.costs(L, S, days, G, np.zeros((1, g), int))
    f = np.zeros((g, 7))
    for i in range(g):
        V = np.zeros((7, g), int)
        V[:, i] = vals
        c, _ = ev.costs(L, S, days, G, V)
        f[i] = c - c0[0] - A[i, i] * vals**2 + mu * np.abs(vals)
    W, C, margins, coefs = np.zeros((g, 3)), np.zeros(g), np.zeros(g), []
    for i in range(g):
        best = None
        for w, c, v, d in _variants():
            gt = f[i][v + 3].astype(float)
            # choose g(d) so the cubic coefficient vanishes
            sign_d = (-1) ** (3 - CODES[d].sum())
            co = _moebius(np.where(np.arange(8) == d, 0.0, gt))
            gd = -co[(0, 1, 2)] / sign_d
            margin = gd - gt[d]
            if best is None or margin > best[0]:
                best = (margin, w, c, _moebius(np.where(np.arange(8) == d, gd, gt)))
        margins[i], W[i], C[i] = best[0], best[1], best[2]
        coefs.append(best[3])
    # v = M b + C,  M (g, 3g)
    M = np.zeros((g, 3 * g))
    for i in range(g):
        M[i, 3 * i:3 * i + 3] = W[i]
    Q = M.T @ A @ M
    lin = np.diag(Q) + 2 * C @ A @ M
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for k in range(3 * g):
        bqm.add_linear(k, lin[k])
    for k in range(3 * g):
        for l in range(k + 1, 3 * g):
            if Q[k, l] != 0:
                bqm.add_quadratic(k, l, 2 * Q[k, l])
    for i in range(g):
        co = coefs[i]
        assert abs(co[(0, 1, 2)]) < 1e-6
        for j in range(3):
            bqm.add_linear(3 * i + j, co[(j,)])
        for a, b in ((0, 1), (0, 2), (1, 2)):
            bqm.add_quadratic(3 * i + a, 3 * i + b, co[(a, b)])
    return bqm, W, C, margins


def decode(bits: np.ndarray, W, C) -> np.ndarray:
    """bits (K, 3g) -> net positions (K, g)."""
    g = len(C)
    return np.einsum("kgj,gj->kg", bits.reshape(len(bits), g, 3), W).astype(int) + C.astype(int)


def tqa_params(p: int, dt: float = 1.0) -> np.ndarray:
    """TQA start (linear ramp, total time ``p*dt``) in Divi's layout [γ0, β0, γ1, β1, …].

    γ ramps up, β ramps down; β is negative because Divi's mixer is +ΣX. Same
    construction as ``init_test.py`` on the initialization-comparisons branch.
    """
    s = (np.arange(p) + 0.5) / p
    row = np.empty(2 * p)
    row[0::2] = s * dt
    row[1::2] = -(1 - s) * dt
    return row


def lagrange_mu(Vall, cost, ok):
    """Smallest mu >= 0 whose unconstrained (units-priced) optimum is feasible.

    If no price on units restores feasibility (the cash/net-position rule binds, not the
    budget), returns ``mu = 0`` and its infeasible optimum, rather than a huge ``mu``
    that would swamp the QUBO.
    """
    units = np.abs(Vall).sum(1)
    def opt(mu):
        return int(np.argmin(cost + mu * units))
    if ok[opt(0.0)]:
        return 0.0, opt(0.0)
    lo, hi = 0.0, 1.0
    while not ok[opt(hi)] and hi < 1e9:
        lo, hi = hi, hi * 4
    if not ok[opt(hi)]:
        return 0.0, opt(0.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok[opt(mid)] else (mid, hi)
    # Step just past the breakpoint so ties with infeasible (more-units) states are
    # broken toward fewer units; costs are integers, so 1e-3 * units never reorders others.
    hi += 1e-3
    return hi, opt(hi)


class QAOABlockSolver:
    """QAOA (TQA start by default) as the pipeline's block solver (``asset_groups.BlockSolver``).

    Per block: build the signed 3-bit QUBO, run QAOA, decode the ``top`` most probable
    samples and propose the best feasible one by true block cost; if none beats the
    current holding, propose the current holding (the driver skips it). Nothing here
    enumerates the block: the budget/cash price ``mu`` starts at 0 and is raised
    (``mu_ladder`` x a per-block unit-cost scale) only when the lowest-energy sample
    breaks a rule.

    Results are cached on the block's surroundings (its days' full holdings and the
    group's holdings on the neighbouring days), so the repeated sweeps of
    clamp-and-iterate don't re-run QAOA on unchanged blocks; QAOA is seeded, so a cache
    hit returns what a rerun would. ``compare_exact`` (diagnostic only, it enumerates
    ``7**g`` holdings) records how often the proposal is the block optimum
    (``exact_hits``) and how often QAOA's own samples contain it (``sample_hits``).
    """

    def __init__(self, layers: int = 2, iters: int = 30, shots: int = 10000, top: int = 64,
                 init: str = "tqa", tqa_dt: float = 1.0, mu_ladder=(0.25, 1.0, 4.0),
                 backend=None, seed: int = 0, compare_exact: bool = False,
                 optimizer: str = "cobyla", population: int = 50, tqa_dt_range=(0.05, 8.0),
                 tqa_dt_sampling: str = "geom"):
        """``optimizer="montecarlo"``: MonteCarloOptimizer(population) for ``iters`` iterations;
        with ``init="tqa"`` its starting population is TQA ramps with dt in
        geomspace(*tqa_dt_range, population), so the first iteration searches over dt.
        ``tqa_dt_sampling="random"``: instead, ``population`` uniform draws in tqa_dt_range;
        ``"stratified"``: one uniform draw in each of ``population`` equal slices of the range
        (random, but always covering it). Both are seeded by ``seed``: same values for every
        block of a run."""
        self.layers, self.iters, self.shots, self.top = layers, iters, shots, top
        self.init, self.tqa_dt, self.mu_ladder, self.seed = init, tqa_dt, tuple(mu_ladder), seed
        self.compare_exact = compare_exact
        self.optimizer, self.population, self.tqa_dt_range = optimizer, population, tuple(tqa_dt_range)
        self.tqa_dt_sampling = tqa_dt_sampling
        lo, hi = self.tqa_dt_range
        rng = np.random.default_rng(seed + 7919)
        if tqa_dt_sampling == "random":
            self.dts = np.sort(rng.uniform(lo, hi, population))
        elif tqa_dt_sampling == "stratified":  # one uniform draw in each of `population` equal slices
            self.dts = lo + (np.arange(population) + rng.uniform(0, 1, population)) * (hi - lo) / population
        else:
            self.dts = np.geomspace(lo, hi, population)
        if backend is None:
            from divi.backends import MaestroSimulator
            backend = MaestroSimulator(shots=shots)
        if hasattr(backend, "set_seed"):
            backend.set_seed(seed)
        self.backend = backend
        self.cache: dict = {}
        self.dt_chosen: list[float] = []
        self.dt_curves: list[list[float]] = []
        self.stats = dict(calls=0, cache_hits=0, qaoa_runs=0, mu_raised=0, proposals=0,
                          no_feasible=0, qaoa_seconds=0.0, exact_checked=0, exact_hits=0,
                          sample_hits=0, exact_gap=0.0,
                          mu_stage=[0] * (1 + len(self.mu_ladder)))
        # mu_stage[k]: fresh calls that stopped at ladder stage k (k = 0: mu = 0);
        # a call that never got a feasible lowest-energy sample counts in the last stage

    def _key(self, L, S, days, G):
        lo, hi, T = days[0], days[-1], len(L)
        nb = [d for d in (lo - 1, hi + 1) if 0 <= d < T]
        return (tuple(days), tuple(G.tolist()), L[days].tobytes(), S[days].tobytes(),
                L[nb][:, G].tobytes(), S[nb][:, G].tobytes())

    def _make_qaoa(self, bqm):
        from divi.qprog.algorithms import QAOA
        from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
        from divi.qprog.problems import BinaryOptimizationProblem
        # QAOA needs O(1) energies: rescale (argmin unchanged)
        scale = max(np.abs(list(bqm.linear.values())).max(),
                    np.abs(list(bqm.quadratic.values()) or [0.0]).max())
        bqm_n = bqm.copy()
        bqm_n.scale(1.0 / scale)
        mc = self.optimizer == "montecarlo"
        opt = (MonteCarloOptimizer(population_size=self.population) if mc
               else ScipyOptimizer(method=ScipyMethod.COBYLA))
        q = QAOA(BinaryOptimizationProblem(bqm_n), n_layers=self.layers, optimizer=opt,
                 max_iterations=self.iters, backend=self.backend, seed=self.seed)
        return q, bqm_n
    
    def _top_bits(self, q, bqm_n):
        tops = q.get_top_solutions(n=self.top)
        # Divi orders bitstring characters like bqm_n.variables (0..nq-1 here)
        order = list(bqm_n.variables)
        bits = np.array([[int(ch) for ch in s.bitstring] for s in tops], dtype=np.int8)
        out = np.zeros_like(bits)
        out[:, order] = bits        
        return out

    def _qaoa(self, bqm) -> np.ndarray:
        """Top-``self.top`` bitstrings (K, nq) of QAOA on ``bqm``."""
        q, bqmn = self._make_qaoa(bqm)
        t0 = time.time()
        if self.init == "tqa" and mc:  # Monte Carlo over the TQA time step
            dts = self.dts
            q.run(initial_params=np.array([tqa_params(self.layers, dt) for dt in dts]))
            first = q.losses_history[0]
            curve = [float(first[str(i)]) for i in range(len(dts))]
            self.dt_chosen.append(float(dts[int(np.argmin(curve))]))
            self.dt_curves.append(curve)  # energy (normalised QUBO) vs dt, one per QAOA call
        elif self.init == "tqa":
            q.run(initial_params=tqa_params(self.layers, self.tqa_dt).reshape(q.get_expected_param_shape()))
        else:  # Divi's default: uniform random angles
            q.run()
        self.stats["qaoa_seconds"] += time.time() - t0
        self.stats["qaoa_runs"] += 1
        return self._top_bits(q, bqmn)

    def __call__(self, ev: BlockEvaluator, L, S, days, G: np.ndarray) -> BlockResult:
        assert ev.model.ub == 3, "signed 3-bit encoding needs ub = 3"
        days = [days] if np.isscalar(days) else list(days)
        G = np.asarray(G)
        self.stats["calls"] += 1
        key = self._key(L, S, days, G)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return BlockResult(self.cache[key].copy())
        cur = (L[days[0], G] - S[days[0], G]).astype(np.int64)
        cur_cost = float(ev.costs(L, S, days, G, cur[None])[0][0])
        best_V, best_c = cur, cur_cost
        unit = None
        any_feasible = False
        sample_best = np.inf  # best feasible QAOA sample, ignoring the current holding
        for k, step in enumerate((0.0,) + self.mu_ladder):
            if k == 1:
                # per-unit cost scale of the block: mean |cost change| of one unit on one asset
                g = len(G)
                V1 = np.concatenate([np.eye(g, dtype=np.int64), -np.eye(g, dtype=np.int64)])
                c1, _ = ev.costs(L, S, days, G, V1)
                c0, _ = ev.costs(L, S, days, G, np.zeros((1, g), np.int64))
                unit = float(np.mean(np.abs(c1 - c0[0]))) or 1.0
                self.stats["mu_raised"] += 1
            mu = step * (unit or 0.0)
            bqm, W, C, _ = block_qubo(ev, L, S, days, G, mu)
            bits = self._qaoa(bqm)
            Vq = decode(bits, W, C)
            cq, okq = ev.costs(L, S, days, G, Vq)
            if okq.any():
                any_feasible = True
                j = int(np.argmin(np.where(okq, cq, np.inf)))
                sample_best = min(sample_best, float(cq[j]))
                if cq[j] < best_c - 1e-9:
                    best_V, best_c = Vq[j].copy(), float(cq[j])
            # stop raising mu once the QUBO's own best sample obeys the rules
            e = bqm.energies((bits, list(range(bits.shape[1]))))
            if okq[int(np.argmin(e))]:
                break
        self.stats["mu_stage"][k] += 1
        self.stats["no_feasible"] += int(not any_feasible)
        self.stats["proposals"] += int(best_c < cur_cost - 1e-9)
        if self.compare_exact:
            Vall = _net_configs(len(G), ev.model.ub)
            cost, ok = ev.costs(L, S, days, G, Vall)
            ex = float(np.min(np.where(ok, cost, np.inf)))
            self.stats["exact_checked"] += 1
            self.stats["exact_hits"] += int(np.isclose(best_c, ex))
            self.stats["sample_hits"] += int(np.isclose(sample_best, ex))
            self.stats["exact_gap"] += best_c - ex
        self.cache[key] = best_V
        return BlockResult(best_V.copy())

    def summary(self) -> dict:
        st = dict(self.stats, layers=self.layers, iters=self.iters, shots=self.shots, top=self.top,
                  init=self.init, tqa_dt=self.tqa_dt, mu_ladder=list(self.mu_ladder),
                  optimizer=self.optimizer, population=self.population, tqa_dt_sampling=self.tqa_dt_sampling)
        st["qaoa_seconds"] = round(st["qaoa_seconds"], 1)
        st.update(dt_summary(self.dt_chosen, self.dt_curves, self.tqa_dt_range, self.dts))
        return st


def dt_summary(chosen: list[float], curves: list[list[float]], dt_range, dts) -> dict:
    """Which TQA time steps won, and the mean energy-vs-dt curve (each curve rescaled to 0..1)."""
    if not chosen:
        return {}
    dts = np.asarray(dts, dtype=float)
    lo, hi = dt_range
    hist, edges = np.histogram(chosen, bins=np.geomspace(lo, hi, 9) if lo > 0 else np.linspace(lo, hi, 9))
    out = dict(tqa_dt_range=list(dt_range), dt_chosen_median=float(np.median(chosen)),
               dt_chosen_hist=dict(counts=hist.tolist(), edges=np.round(edges, 3).tolist()))
    if curves:
        C = np.array(curves, dtype=float)
        span = C.max(1, keepdims=True) - C.min(1, keepdims=True)
        C = (C - C.min(1, keepdims=True)) / np.where(span > 0, span, 1.0)
        out.update(dt_grid=np.round(dts, 4).tolist(), dt_curve_mean=np.round(C.mean(0), 4).tolist(),
                   dt_curve_n=len(curves))
    return out


def run(args):
    model, bkv = _load(args.instance, args.budget, args.lam)
    ev = BlockEvaluator(model)
    rng = np.random.default_rng(args.seed)
    L, S, _ = sa_start(model, num_reads=10, num_sweeps=1000, seed=args.seed)
    print(f"start score {model.score_arrays(L, S).objective}  (bkv {bkv})", flush=True)

    from divi.backends import JobConfig, MaestroSimulator, QoroService
    from divi.qprog.algorithms import QAOA
    from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    if args.cloud:
        service = QoroService()
        bal0 = service.get_credit_balance()
        print("credits before:", bal0, flush=True)
        backend = QoroService(job_config=JobConfig(shots=args.shots, simulator_cluster="qoro_maestro"))
    else:
        backend = MaestroSimulator(shots=args.shots)

    T, n = model.n_periods, len(model.symbols)
    rows = []
    for b in range(args.blocks):
        t = int(rng.integers(0, T - 1))
        held = np.flatnonzero((L[t] + S[t]) > 0)
        k_held = min(len(held), args.group // 2)
        G = np.concatenate([rng.choice(held, k_held, replace=False) if k_held else np.array([], int),
                            rng.choice(np.setdiff1d(np.arange(n), held), args.group - k_held, replace=False)])
        G = np.sort(G.astype(int))
        Vall = _net_configs(len(G), model.ub)
        cost, ok = ev.costs(L, S, [t], G, Vall)
        exact = float(np.min(np.where(ok, cost, np.inf)))
        cur = float(ev.costs(L, S, [t], G, (L[t, G] - S[t, G])[None])[0][0])
        mu, k_mu = lagrange_mu(Vall, cost, ok)
        bqm, W, C, margins = block_qubo(ev, L, S, t, G, mu)
        # sanity: QUBO energy - (cost + mu*units) is >= a constant, with equality off the
        # duplicate codes; and (brute force, <= 18 qubits) the QUBO ground state decodes
        # to the Lagrangian optimum.
        nq = 3 * len(G)
        exact_qubo = None
        if nq <= 18:
            allb = ((np.arange(2**nq)[:, None] >> np.arange(nq)[::-1]) & 1).astype(np.int8)
            e = bqm.energies((allb, list(range(nq))))
            Vb = decode(allb, W, C)
            cb, _ = ev.costs(L, S, [t], G, Vb)
            diff = e - (cb + mu * np.abs(Vb).sum(1))
            gs = int(np.argmin(e))
            exact_qubo = bool(diff.min() > diff[gs] - 1e-6 and
                              np.isclose(cb[gs] + mu * np.abs(Vb[gs]).sum(), (cost + mu * np.abs(Vall).sum(1))[k_mu]))
        # QAOA needs O(1) energies: rescale (argmin unchanged)
        scale = max(np.abs(list(bqm.linear.values())).max(), np.abs(list(bqm.quadratic.values())).max())
        bqm_n = bqm.copy()
        bqm_n.scale(1.0 / scale)

        t0 = time.time()
        q = QAOA(BinaryOptimizationProblem(bqm_n), n_layers=args.layers,
                 optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
                 max_iterations=args.iters, backend=backend, seed=args.seed)
        if args.init == "tqa":
            q.run(initial_params=tqa_params(args.layers, args.tqa_dt).reshape(q.get_expected_param_shape()))
        else:  # Divi's default: uniform random angles
            q.run()
        secs = time.time() - t0
        tops = q.get_top_solutions(n=args.top)
        qb = np.array([[int(ch) for ch in s.bitstring] for s in tops])
        probs = np.array([s.prob for s in tops])
        Vq = decode(qb, W, C)
        cq, okq = ev.costs(L, S, [t], G, Vq)
        best_q = float(np.min(np.where(okq, cq, np.inf)))
        p_opt = float(probs[okq & np.isclose(cq, exact)].sum())
        # how many distinct optimal bitstrings exist (duplicate zero codes) -> uniform prob
        n_opt_codes = int(np.prod([2 if v == 0 else 1 for v in Vall[int(np.argmin(np.where(ok, cost, np.inf)))]]))
        p_uniform = n_opt_codes / 2**nq
        # baseline: same number of uniformly random bitstrings
        rb = rng.integers(0, 2, (args.top, 3 * len(G)))
        cr, okr = ev.costs(L, S, [t], G, decode(rb, W, C))
        best_r = float(np.min(np.where(okr, cr, np.inf)))
        row = dict(block=b, day=t, group=G.tolist(), qubits=3 * len(G), mu=mu,
                   min_margin=float(margins.min()), qubo_exact=None if exact_qubo is None else bool(exact_qubo),
                   lagrange_hits_opt=bool(np.isclose(cost[k_mu], exact)),
                   current=cur - exact, qaoa_best=best_q - exact, random_best=best_r - exact,
                   p_opt_in_top=p_opt, p_opt_uniform=p_uniform, feasible_top=int(okq.sum()), p_feasible_top=float(probs[okq].sum()), seconds=round(secs, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.cloud:
        bal1 = service.get_credit_balance()
        print("credits after:", bal1, flush=True)
    out = f"results/qaoa_signed_{args.instance}_B{args.budget}_lam{args.lam}_g{args.group}_p{args.layers}_{'cloud' if args.cloud else 'local'}{'' if args.init == 'random' else '_' + args.init}.json"
    json.dump(dict(vars(args), rows=rows), open(out, "w"), indent=1)
    print("wrote", out)


def check(blocks: int = 40, group: int = 4) -> None:
    """Brute force: QUBO == cost + mu*units off the duplicate codes (margin >= 0), and
    the QUBO ground state vs the exact constrained block optimum."""
    nq = 3 * group
    allb = ((np.arange(2**nq)[:, None] >> np.arange(nq)[::-1]) & 1).astype(np.int8)
    for inst, B, lam in [("po_a050_t10_orig", 20, "1e-3"), ("po_a050_t10_orig", 20, "1e-4"),
                         ("po_a010_t10_orig", 4, "1e-4")]:
        model, _ = _load(inst, B, lam)
        ev = BlockEvaluator(model)
        L, S, _ = sa_start(model, num_reads=5, num_sweeps=500, seed=3, verbose=False)
        rng = np.random.default_rng(1)
        T, n = model.n_periods, len(model.symbols)
        neg = gs_ok = lagr_ok = 0
        for _ in range(blocks):
            t = int(rng.integers(0, T - 1))
            G = np.sort(rng.choice(n, group, replace=False))
            Vall = _net_configs(group, model.ub)
            cost, ok = ev.costs(L, S, [t], G, Vall)
            exact = np.min(np.where(ok, cost, np.inf))
            mu, k = lagrange_mu(Vall, cost, ok)
            bqm, W, C, margins = block_qubo(ev, L, S, t, G, mu)
            e = bqm.energies((allb, list(range(nq))))
            Vb = decode(allb, W, C)
            cb, okb = ev.costs(L, S, [t], G, Vb)
            diff = e - (cb + mu * np.abs(Vb).sum(1))
            # Duplicate codes always decode to v = 0, so states without a zero position
            # use none: there the QUBO must equal cost + mu*units up to one constant c,
            # and no state may lie below c (margin >= 0).
            canon = (Vb != 0).all(axis=1)
            c = diff[canon]
            tol = 1e-9 * max(1.0, np.abs(e).max())  # float rounding on large energies
            assert np.ptp(c) < tol and diff.min() > c[0] - tol, "QUBO not exact"
            gs = int(np.argmin(e))
            neg += int((margins < -1e-9).any())
            gs_ok += int(okb[gs] and np.isclose(cb[gs], exact))
            lagr_ok += int(np.isclose(cost[k], exact))
        print(f"{inst} B={B} λ={lam}: {blocks} blocks of {group} ({nq} qubits) | negative margin "
              f"{neg} | ground state = block optimum {gs_ok} | Lagrangian optimum = block optimum "
              f"{lagr_ok}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="po_a050_t10_orig")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["tqa", "random"], default="tqa",
                    help="QAOA starting angles: TQA ramp (default) or Divi's uniform random")
    ap.add_argument("--tqa-dt", type=float, default=1.0, help="TQA time step per layer")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--check", action="store_true", help="brute-force exactness check, no QAOA")
    a = ap.parse_args()
    check() if a.check else run(a)
