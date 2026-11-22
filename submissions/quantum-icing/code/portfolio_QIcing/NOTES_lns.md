# Large-instance hunt (a050 / a200 / a400) — classical LNS notes

Owner: LNS fork. Code: `lns.py` (solvers), `run_lns.py` (parallel runner), `prefetch.py` (data download).
All final solutions recorded via `pf_core.record_solution` → `solutions/<inst>/…__lns__*.sol` + `results.csv`.
Logs: `logs/lns_a050_t10.txt|.out`, `logs/lns_a050_t15.*`, `logs/lns_a200_t10.*`, `logs/lns_a400_t10.*`.

## Method (all classical, referee-checked)
State = net positions v[t, i] ∈ {−3..3} (long+short of one asset is dominated). Exact integer evaluator `FastModel` (verified = referee on 107 random feasible points).
Constraints per period: Σ|v| ≤ B, C−15 ≤ Σv ≤ C (capital binds at B = 20).

1. **Per-period candidate pool.** Periods are only weakly coupled (transaction cost 0.1 %/unit), so each period t is first solved as an integer QP alone, with entry/exit cost emulated by adding e·tc[t]·|v| (e = 0, 1, 2):
   - continuous convex relaxation (v = p − q, SLSQP) → rounding at scales 1…40 → repair → exact best-improvement descent over single and **pair** unit moves (vectorised 2n×2n delta matrix);
   - iterated local search (kick 2–6 assets, descend);
   - numba SA with compound moves (period blocks × partner-biased hedge pairs).
2. **Chain DP.** Union pool (~1000 vectors) → exact DP choosing one pool vector per period with exact transaction costs (a pool vector can be held over several periods).
3. **LNS polish.** Pick k ≤ 4 assets (correlated partners of held assets / held / random), fix all others, exact DP over all periods for the k assets jointly (7^k states, per-period budget/capital feasibility). Monotone. Sub-solver pluggable: `lns.lns(..., solve_sub=f)` with `f(stage[T,K], trans_w[T,k], states[K,k]) -> chain` — drop-in point for a QUBO/QAOA sub-solver.

Key insight: at high λ the BKV portfolios are **diversified ±1 positions on ~15 assets with the budget saturated**, held only in a few periods. Single-unit moves (SA, notebook local search) never reach them (one unit alone is too risky) → −1000 (empty). Per-period QP relaxation + pool DP gets there.

## What did not work
- Plain LNS from empty (k ≤ 4): a050 λ=1e-3 gap 5.4 %, λ=1e-2 stuck near empty (−1520 vs −37107).
- Full-horizon SA (single/pair, then compound block moves): λ=1e-3 gap 1.4–1.7 %, λ=1e-2 −9038 (gap 76 %).
- (sanity) LNS alone reaches the DP-proven optimum on a010 λ=1e-3 and 5e-4 within 20 s.

## Results a050, T = 10 (150 s per instance, 1 core, seed 0)
| instance | λ | ours | pool-DP only | BKV | gap vs BKV | |
|---|---|---:|---:|---:|---:|---|
| po_a050_t10_orig | 1e-2 | -34520 | -34520 | -37107 | +6.972% |  |
| po_a050_t10_s00 | 1e-2 | -17675 | -17675 | -12340 | -43.233% | **BEATS BKV** |
| po_a050_t10_s01 | 1e-2 | -1000 | -1000 | -2248 | +55.516% |  |
| po_a050_t10_s02 | 1e-2 | -40149 | -40149 | -43565 | +7.841% |  |
| po_a050_t10_orig | 1e-3 | -205278 | -205144 | -206239 | +0.466% |  |
| po_a050_t10_s00 | 1e-3 | -190685 | -190641 | -191879 | +0.622% |  |
| po_a050_t10_s01 | 1e-3 | -139048 | -139048 | -141376 | +1.647% |  |
| po_a050_t10_s02 | 1e-3 | -232490 | -232308 | -233766 | +0.546% |  |
| po_a050_t10_orig | 5e-4 | -254640 | -254561 | -254972 | +0.130% |  |
| po_a050_t10_s00 | 5e-4 | -234587 | -234234 | -234259 | -0.140% | **BEATS BKV** |
| po_a050_t10_s01 | 5e-4 | -183279 | -183279 | -183574 | +0.161% |  |
| po_a050_t10_s02 | 5e-4 | -286932 | -286475 | -287872 | +0.327% |  |
| po_a050_t10_orig | 1e-4 | -370843 | -370778 | -370869 | +0.007% |  |
| po_a050_t10_s00 | 1e-4 | -355530 | -355321 | -355530 | +0.000% | = BKV |
| po_a050_t10_s01 | 1e-4 | -301297 | -300765 | -301400 | +0.034% |  |
| po_a050_t10_s02 | 1e-4 | -443357 | -442351 | -443357 | +0.000% | = BKV |
| po_a050_t10_orig | 5e-5 | -417142 | -416754 | -417218 | +0.018% |  |
| po_a050_t10_s00 | 5e-5 | -408936 | -408886 | -408937 | +0.000% |  |
| po_a050_t10_s01 | 5e-5 | -342074 | -341697 | -342074 | +0.000% | = BKV |
| po_a050_t10_s02 | 5e-5 | -507544 | -506817 | -507544 | +0.000% | = BKV |
| po_a050_t10_orig | 1e-5 | -478956 | -478802 | -478956 | +0.000% | = BKV |
| po_a050_t10_s00 | 1e-5 | -484396 | -483457 | -484396 | +0.000% | = BKV |
| po_a050_t10_s01 | 1e-5 | -402742 | -402324 | -402742 | +0.000% | = BKV |
| po_a050_t10_s02 | 1e-5 | -594866 | -594719 | -594866 | +0.000% | = BKV |
| po_a050_t10_orig | 1e-6 | -499339 | -499173 | -499339 | +0.000% | = BKV |
| po_a050_t10_s00 | 1e-6 | -513409 | -511881 | -513409 | +0.000% | = BKV |
| po_a050_t10_s01 | 1e-6 | -428609 | -428036 | -428609 | +0.000% | = BKV |
| po_a050_t10_s02 | 1e-6 | -622388 | -621862 | -622388 | +0.000% | = BKV |
| po_a050_t10_orig | 0 | -501737 | -501515 | -501737 | +0.000% | = BKV |

λ = 0 for s00/s01/s02 not run (worker crashed on an LNS-kick feasibility bug, fixed afterwards; those BKVs are proven optimal anyway).
Summary T = 10: **2 BKVs beaten** (s00 λ=1e-2, s00 λ=5e-4), 13 matched (s00 λ=5e-5 is 1 unit off), rest within 0.0–1.7 % except λ=1e-2 on orig/s01/s02 (+7 %, +56 %, +8 %).


## Results a050, T = 15 (high λ only; 200 s per instance, 1 core, seed 0)
| instance | λ | ours | pool-DP only | BKV | gap vs BKV | |
|---|---|---:|---:|---:|---:|---|
| po_a050_t15_orig | 1e-2 | -45031 | -44651 | -31504 | -42.937% | **BEATS BKV** |
| po_a050_t15_s00 | 1e-2 | -39280 | -39280 | -43792 | +10.303% |  |
| po_a050_t15_s01 | 1e-2 | -12983 | -12846 | -13706 | +5.275% |  |
| po_a050_t15_s02 | 1e-2 | -41517 | -41517 | -40622 | -2.203% | **BEATS BKV** |
| po_a050_t15_orig | 1e-3 | -321070 | -320562 | -321907 | +0.260% |  |
| po_a050_t15_s00 | 1e-3 | -312312 | -312201 | -314736 | +0.770% |  |
| po_a050_t15_s01 | 1e-3 | -241610 | -241610 | -243245 | +0.672% |  |
| po_a050_t15_s02 | 1e-3 | -315176 | -314906 | -316799 | +0.512% |  |
| po_a050_t15_orig | 5e-4 | -397236 | -397007 | -397497 | +0.066% |  |
| po_a050_t15_s00 | 5e-4 | -386499 | -386146 | -386990 | +0.127% |  |
| po_a050_t15_s01 | 5e-4 | -303436 | -303363 | -304289 | +0.280% |  |
| po_a050_t15_s02 | 5e-4 | -391667 | -391210 | -392668 | +0.255% |  |

Summary T = 15: **2 more BKVs beaten** (orig λ=1e-2, s02 λ=1e-2); λ=5e-4 / 1e-3 within 0.07–0.77 %; s00/s01 λ=1e-2 +10.3 % / +5.3 %.

## a200 / a400 (no published values — first solutions)
LIGHT=2 mode (no QP relaxation, SA candidates + pool DP + LNS), 200 s, λ = 1e-3, seed orig only:
| instance | B | λ | ours | BKV |
|---|---|---|---:|---|
| po_a200_t10_orig | 50 | 1e-3 | −818 362 | none published |
| po_a400_t10_orig | 100 | 1e-3 | −1 896 257 | none published |
(a400: only 1 876 LNS iterations in the time box — far from converged.)

## BKV improvements (all referee re-checked from the written .sol files)
| instance | B | λ | ours | QOBLIB BKV | improvement | file |
|---|---|---|---:|---:|---:|---|
| po_a050_t15_orig | 20 | 1e-2 | **−45 031** | −31 504 | −42.9 % | solutions/po_a050_t15_orig_b020_l1e-2/…__lns__20261121-222308.sol |
| po_a050_t10_s00 | 20 | 1e-2 | **−17 675** | −12 340 | −43.2 % | solutions/po_a050_t10_s00_b020_l1e-2/…__lns__20261121-221132.sol |
| po_a050_t15_s02 | 20 | 1e-2 | **−41 517** | −40 622 | −2.2 % | solutions/po_a050_t15_s02_b020_l1e-2/…__lns__20261121-222307.sol |
| po_a050_t10_s00 | 20 | 5e-4 | **−234 587** | −234 259 | −0.14 % | solutions/po_a050_t10_s00_b020_l5e-4/…__lns__20261121-221402.sol |
All classical (no quantum step). Not using published solutions as starts (only one BKV file was *inspected* for structure, a050_t10_orig λ=1e-2/1e-3).

## Next ideas
- More time / seeds on λ=1e-2 (still +5…+56 % on 5 instances; s01 T=10 stuck at empty −1000 vs −2248) and on 1e-3.
- Quantum hook: pass a QAOA/PCE `solve_sub` to `lns.lns` (k = 2 assets × T periods → 7^2 states/period; as QUBO ≈ 6 bits × 2 assets × T) or use QAOA as the per-period candidate generator (integer QP with ~50 assets on a subset).
- a400: QP-relaxation candidates were skipped for speed; enable LIGHT=1 with more time.

