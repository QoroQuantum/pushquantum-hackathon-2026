"""Checks that CompactPortfolioQUBO is exact (keeps the official score and optimum).

Run: python experiments/check_portfolio_compact.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root: qoblib_tools & co.

import numpy as np

import qoblib_tools as qt
from portfolio_compact import CompactPortfolioQUBO

rng = np.random.default_rng(0)


def energies(bqm, bits: np.ndarray, order) -> np.ndarray:
    """Energies of a batch of bit rows (columns in ``order``)."""
    lin, (r, c, q), off = bqm.to_numpy_vectors(variable_order=order)
    return bits @ lin + np.einsum("kp,p->k", bits[:, r] * bits[:, c], q) + off


def energy(bqm, bits) -> float:
    return float(bqm.energy(dict(enumerate(bits))))


def sliced(inst: qt.PortfolioInstance, n_assets: int, n_periods: int) -> qt.PortfolioInstance:
    cov = {k: v for k, v in inst.cov.items() if k[0] < n_assets and k[1] < n_assets and k[2] < n_periods}
    return qt.PortfolioInstance(
        f"{inst.name}[:{n_assets},:{n_periods}]", inst.symbols[:n_assets], n_periods,
        [row[:n_periods] for row in inst.prices[:n_assets]], cov,
    )


def random_feasible(model: qt.PortfolioModel) -> tuple[np.ndarray, np.ndarray]:
    """Random feasible (L, S): units dropped at random until the period's rules hold."""
    T, n, ub = model.n_periods, len(model.symbols), model.ub
    L = rng.integers(0, ub + 1, (T, n))
    S = rng.integers(0, ub + 1, (T, n))
    L[rng.random((T, n)) < 0.6] = 0
    S[rng.random((T, n)) < 0.7] = 0
    while not model.score_arrays(L, S).feasible:
        arr = L if rng.random() < 0.5 else S
        nz = np.argwhere(arr > 0)
        if len(nz) == 0:
            continue
        t, i = nz[rng.integers(len(nz))]
        arr[t, i] -= 1
    return L, S


def with_last(model, q, L, S):
    """(L, S) with the final period replaced by the compact QUBO's fixed one."""
    L, S = L.copy(), S.copy()
    if q.last_period is not None:
        L[-1], S[-1] = q.last_period
    return L, S


def check_identity(model: qt.PortfolioModel, q: CompactPortfolioQUBO, trials: int) -> None:
    """For random feasible portfolios: energy == official score, penalty == 0; and
    for random *copy placements* (not prefix-filled), energy >= score of the decode."""
    orig = qt.PortfolioQUBO(model)
    for _ in range(trials):
        L, S = with_last(model, q, *random_feasible(model))
        pos = model.to_positions(L, S)
        score = model.score(pos).objective
        bits = q.encode(pos)
        assert energy(q.penalty, bits) == 0, "feasible portfolio is penalised"
        assert energy(q.objective, bits) == score, (energy(q.objective, bits), score)
        assert energy(orig.objective, orig.encode(pos)) == score  # referee sanity

        # Same unit counts, copies placed at random: rebalancing can only cost more.
        shuffled = bits.copy()
        for (i, sym) in enumerate(model.symbols):
            for tau in (1, -1):
                for t in q.periods:
                    idx = [q._x(i, m, tau, t) for m in range(q.ub)]
                    shuffled[idx] = rng.permutation(shuffled[idx])
        assert energy(q.penalty, shuffled) == 0
        assert q.decode(shuffled) == pos
        assert energy(q.objective, shuffled) >= score


def check_exhaustive(model: qt.PortfolioModel, q: CompactPortfolioQUBO) -> None:
    """Minimise over *every* bitstring with zero penalty; compare with the exact optimum."""
    assert q.n_vars <= 24, "too large for exhaustive search"
    opt_pos, opt = model.brute_force()
    order = list(range(q.n_vars))
    best, best_bits = np.inf, None
    for start in range(0, 2**q.n_vars, 2**16):
        k = np.arange(start, min(start + 2**16, 2**q.n_vars))
        bits = ((k[:, None] >> np.arange(q.n_vars)) & 1).astype(float)
        e = energies(q.objective, bits, order)
        e[energies(q.penalty, bits, order) != 0] = np.inf
        j = int(np.argmin(e))
        if e[j] < best:
            best, best_bits = e[j], bits[j].astype(int)
    decoded = q.decode(best_bits)
    assert best == opt, (best, opt)
    assert model.score(decoded).objective == opt and model.score(decoded).feasible
    print(f"  exhaustive over 2^{q.n_vars} bitstrings: QUBO min {best:g} == exact optimum {opt}")


def check_domain_wall_terms(model: qt.PortfolioModel, weight: float, trials: int) -> None:
    """Domain-wall terms: zero on canonical encodings, positive on every invalid slot."""
    q = CompactPortfolioQUBO(model, domain_wall=weight)
    base = CompactPortfolioQUBO(model)
    for _ in range(trials):
        L, S = with_last(model, q, *random_feasible(model))
        L, S = np.where(L > S, L - S, 0), np.where(S > L, S - L, 0)  # one direction per slot
        bits = q.encode(model.to_positions(L, S))
        assert energy(q.wall, bits) == 0
        assert energy(q.objective, bits) == energy(base.objective, bits)
        # Random position bits: the wall energy counts the slots out of domain-wall form.
        rnd = bits.copy()
        rnd[: q.n_x] = rng.integers(0, 2, q.n_x)
        bad = 0
        for i in range(len(model.symbols)):
            for t in q.periods:
                lo = [rnd[q._x(i, m, 1, t)] for m in range(q.ub)]
                sh = [rnd[q._x(i, m, -1, t)] for m in range(q.ub)]
                ordered = lambda c: all(c[m] >= c[m + 1] for m in range(q.ub - 1))
                bad += (not ordered(lo)) or (not ordered(sh)) or (lo[0] and sh[0])
        assert (energy(q.wall, rnd) > 0) == (bad > 0)
        diff = energy(q.objective, rnd) - energy(base.objective, rnd)
        assert np.isclose(diff, weight * energy(q.wall, rnd))


def check_netting(model: qt.PortfolioModel, trials: int) -> None:
    """Holding long and short at once never beats holding the net position."""
    tested = 0
    for _ in range(trials):
        L, S = random_feasible(model)
        for t in range(model.n_periods):  # one hedged pair per period where it fits
            i = rng.integers(len(model.symbols))
            if (L[t] + S[t]).sum() + 2 <= model.budget and max(L[t, i], S[t, i]) < model.ub:
                L[t, i] += 1
                S[t, i] += 1
        if not ((L > 0) & (S > 0)).any() or not model.score_arrays(L, S).feasible:
            continue
        tested += 1
        v = L - S
        assert model.score_arrays(np.maximum(v, 0), np.maximum(-v, 0)).objective <= \
            model.score_arrays(L, S).objective
    assert tested >= trials // 2, f"only {tested}/{trials} trials tested"


def main() -> None:
    print("1. Tiny notebook instance (po_a003_t02_orig, B=3, λ=1e-5)")
    tiny = qt.PortfolioModel(qt.load_portfolio_instance("po_a003_t02_orig"), budget=3, lam="1e-5")
    q = CompactPortfolioQUBO(tiny)
    print("  ", q.reductions)
    check_identity(tiny, q, trials=300)
    check_exhaustive(tiny, q)
    # With the final period kept in the QUBO (40 variables: too many to enumerate).
    check_identity(tiny, CompactPortfolioQUBO(tiny, decouple_last_period=False), trials=300)

    print("\n2. Slices of po_a010_t10_orig (3 assets, 4 periods): exact optimum by DP")
    base = qt.load_portfolio_instance("po_a010_t10_orig")
    inst = sliced(base, 3, 4)
    for B, lam in [(2, "1e-4"), (3, "1e-3"), (4, "1e-4"), (5, "1e-2"), (6, "1e-4"), (8, "1e-3")]:
        model = qt.PortfolioModel(inst, budget=B, lam=lam)
        for kw in ({}, {"decouple_last_period": False}):
            q = CompactPortfolioQUBO(model, **kw)
            check_identity(model, q, trials=200)
            opt_pos, opt = model.brute_force()
            L, S = with_last(model, q, *model.to_arrays(opt_pos))
            # The fixed final period is optimal, so the optimum is still reachable.
            assert energy(q.objective, q.encode(model.to_positions(L, S))) == opt
            assert energy(q.penalty, q.encode(model.to_positions(L, S))) == 0
        print(f"   B={B} λ={lam}: ok  (capital rule kept: {q.keep_capital}, "
              f"vars {q.reductions['variables']})")

    print("\n3. po_a010_t10_orig, published reference solutions")
    for B, lam in [(4, "1e-4"), (4, "1e-3")]:
        model = qt.PortfolioModel(base, budget=B, lam=lam)
        header, ref = qt.read_portfolio_solution(qt.fetch_portfolio_reference(base.name, budget=B, lam_tag=lam))
        q = CompactPortfolioQUBO(model)
        L, S = model.to_arrays(ref)
        pos = model.to_positions(*with_last(model, q, L, S))
        assert model.score(pos).objective <= model.score(ref).objective  # fixed final period is optimal
        e, p = energy(q.objective, q.encode(pos)), energy(q.penalty, q.encode(pos))
        assert p == 0 and e == model.score(pos).objective == int(header["objective"])
        check_identity(model, q, trials=20)
        print(f"   B={B} λ={lam}: reference {header['objective']}, compact energy {e:g}, "
              f"penalty {p:g}, vars {q.reductions['variables']}")

        # The heuristic restriction: report, don't assert.
        qe = CompactPortfolioQUBO(model, budget_equality=True)
        pe = energy(qe.penalty, qe.encode(pos))
        print(f"      budget_equality=True ({qe.n_vars} vars): reference "
              f"{'still feasible' if pe == 0 else 'EXCLUDED (not every period uses the full budget)'}")

    print("\n4. Domain-wall ordering (exact for any weight)")
    model = qt.PortfolioModel(base, budget=4, lam="1e-4")
    for w in (30.0, 300.0):
        check_domain_wall_terms(model, w, trials=50)
    check_netting(model, trials=300)
    check_netting(qt.PortfolioModel(base, budget=4, lam="1e-3"), trials=300)
    print("   terms zero on canonical encodings, positive exactly on invalid slots; netting never worse")
    small = sliced(base, 2, 3)  # 2 periods in the QUBO, so rebalancing is exercised
    for B, lam, ub in [(2, "1e-4", 2), (3, "1e-3", 2), (3, "1e-4", 2)]:
        m = qt.PortfolioModel(small, budget=B, lam=lam, ub=ub)
        for w in (0.0, 30.0, 300.0, 1e4):
            check_exhaustive(m, CompactPortfolioQUBO(m, domain_wall=w))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
