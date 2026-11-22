# QOBLIB 06 portfolio: QIcing report

Everything is scored with QOBLIB's official objective (`qoblib_tools.PortfolioModel.score`). Every `.sol` file is re-checked by `verify_solutions.py`: 127+ files, all feasible, and every objective line matches the referee. All 127+ also pass QOBLIB's official Rust checker (`06-portfolio/check`, exact arithmetic), including the 4 a050 BKV improvements. The best file for each instance is listed in `best_solutions.md`. Every run is logged in `results.csv`. The full log is in `NOTES.md` (quantum) and `NOTES_lns.md` (classical, large instances).

## Headline results

1. **Quantum: the pipeline reaches the proven optimum on 3 leaderboard instances.** On `po_a010_t10_orig` (B = 4), *PCE quantum large-neighbourhood search* starts from an **empty portfolio**, uses **at most 9 qubits** and runs no classical local search. It reaches:

   | λ | result | BKV (= DP-proven optimum) | gap | notebook starter (SA+LS) |
   |---|---:|---:|---:|---:|
   | 1e-4 | **−69 482** | −69 482 | **0.000 %** | −69 482 (partitioned QAOA in the notebook: −69 169, +0.45 %) |
   | 5e-4 | **−27 044** | −27 044 | **0.000 %** | −26 548 (+1.8 %) |
   | 1e-3 | **−8 397** | −8 397 | **0.000 %** | −3 703 (+56 %) |
   | `po_a010_t10_s02` λ = 1e-4 | −53 220 | −53 255 | +0.066 % | −51 805 (+2.7 %) |

2. **Classical: every 10-asset BKV is proven optimal.** All 64 a010 instances were solved exactly by DP over periods, using a reduced state of net positions. 54 of these are labelled only "best known" in QOBLIB. Nothing can be beaten at 10 assets, so they serve as ground truth.

3. **Classical: 4 a050 BKVs beaten.** These come from pool-DP + exact-DP LNS, done by the parallel sub-agent (see `NOTES_lns.md`). No quantum step is involved.

   | instance | λ | ours | QOBLIB BKV | improvement |
   |---|---|---:|---:|---:|
   | po_a050_t10_s00 | 1e-2 | −17 675 | −12 340 | −43.2 % |
   | po_a050_t15_orig | 1e-2 | −45 031 | −31 504 | −42.9 % |
   | po_a050_t15_s02 | 1e-2 | −41 517 | −40 622 | −2.2 % |
   | po_a050_t10_s00 | 5e-4 | −234 587 | −234 259 | −0.14 % |

   We also have the first solutions for instances with no published value, at λ = 1e-3: `po_a200_t10_orig` −818 362 (B = 50) and `po_a400_t10_orig` −1 896 257 (B = 100, not converged).

## All quantum results, including the weak ones

This table lists every quantum configuration we ran, with its best official score. On `a010` the BKV is also the DP-proven optimum. A † means the run was stopped early: the score comes from its log and no `.sol` file was written. All circuits ran on the local Maestro simulator.

| instance, λ (BKV) | quantum configuration | qubits | circuits | best score | gap vs BKV |
|---|---|---:|---:|---:|---:|
| a010_t10_orig, 1e-4 (−69 482) | **PCE-QLNS, 2 chained runs from empty** | ≤9 | 0.22 M | **−69 482** | **0.000%** |
| | PCE-QLNS, single run from empty | 8 | 96 k | −69 397 | +0.12% |
| | PCE-QLNS, single run, larger windows | 9 | 0.13 M | −68 785 | +1.00% |
| | QAOA-QLNS, warm start, from route-B start | ≤18 | 30 k | −68 249 | +1.77% |
| | QAOA-QLNS, warm start, from empty† (59 windows) | ≤18 | – | −54 272 | +21.9% |
| | partitioned QAOA + repair + referee local search | 12 | 31 k | −54 923 | +21.0% |
| | partitioned QAOA + repair only | 12 | 31 k | −3 484 | +95.0% |
| a010_t10_orig, 5e-4 (−27 044) | **PCE-QLNS, 2 chained runs from empty** | ≤9 | 0.29 M | **−27 044** | **0.000%** |
| | PCE-QLNS, single run from empty | 8 | 0.13 M | −26 882 | +0.60% |
| a010_t10_orig, 1e-3 (−8 397) | **PCE-QLNS, 3 chained runs from empty** | ≤9 | 0.91 M | **−8 397** | **0.000%** |
| | after 2 chained runs | ≤9 | 0.29 M | −7 759 | +7.6% |
| | PCE-QLNS, single run from empty | 8 | 0.13 M | −6 962 | +17.1% |
| a010_t10_s02, 1e-4 (−53 255) | **PCE-QLNS, 2 chained runs from empty** | ≤9 | 0.73 M | **−53 220** | **+0.066%** |
| | PCE-QLNS, single run | 8 | 0.13 M | −53 136 | +0.22% |
| a050_t10_s00, 1e-2 (−12 340) | **PCE-QLNS from empty†**: 3 runs, 359 windows (random, correlation-guided and hedge windows) | ≤9 | – | **−1 000** (empty portfolio) | **+91.9%** |
| | PCE-QLNS starting from the classical −17 675 | ≤9 | 0.19 M | no change, so this is not a quantum result | – |

Our best overall for comparison: the `a010` DP optimum equals the BKV, and on `a050_t10_s00` λ=1e-2 the classical result is −17 675 (−43.2% vs BKV).

## Methods

**Model facts used.** These follow from the referee code.
- The interest term is −10·(C − net), which is linear, so no cash slack bits are needed.
- Holding long and short of the same asset is dominated, so the state can be v = L − S ∈ {−3..3}.
- With B = 4 the capital rule never binds.
- The last period is always empty.

**Exact DP** (`pf_core.exact_dp`). The state is a vector v with Σ|v| ≤ B, which gives 8 341 states for n = 10. Transitions are weighted L1 rebalancing costs, computed with a numba min-plus step with pruning. Each instance takes about 2 s.

**Compact QUBO** (`pf_quantum.CompactQUBO`). It keeps the official unary copies, folds the interest into the objective and uses a ⌈log₂(B+1)⌉-bit budget slack. A cash register is added only when B > 5. Sizes: 630 vars for a010_t10, against 710 for the official QUBO, and 3 090 vars for a050_t10. Its energy equals the referee's objective (verified).

**Quantum LNS** (`run_qlns.py`). It follows the notebook-00 toolchain inside a large-neighbourhood search loop:
1. Choose a window: 1 period × 4–6 assets, or 2–3 assets × 2 periods. A window holds 27–42 binary variables, including the slack bits.
2. Build the exact conditional sub-QUBO, with all other bits fixed to the incumbent. The penalty weight is 5000.
3. Solve the sub-QUBO with **Divi PCE**: `PolyEncoding`, 7–9 qubits, 3 layers, `SoftObjective(α = 2)`, MonteCarloOptimizer with 40 × 8, on the local Maestro simulator.
4. Splice every sampled sub-bitstring into the incumbent. Score it with the referee and apply the referee's repair if it is infeasible.
5. Accept the best candidate if it improves.

Runs are chained: a second run starts from the first run's `.sol`, with larger windows. A full chain to the optimum takes 10–30 min on a laptop and 0.2–0.9 M circuits.
- *QAOA variant:* 15–18-qubit windows, warm-started from **Characterization Service** angles (job `c3a51b2e-ad3f-4cae-924b-b4ce2c013196`, 4 layers). The exact window optimum appeared among QAOA's samples in 145 of 200 windows (73 %). The run reached −68 249 (+1.8 %) in 20 min. It is slower, about 6 s per window against 1 s for PCE, and its windows are smaller.
- *Notebook route B* (partitioned QAOA over the full compact QUBO, 62 sub-programs ≤ 12 qubits) gave −3 484 after stitching and repair, and −54 923 after the referee's local search. It is weak because partitions cut through the budget constraints.

**Classical large-instance solver** (`lns.py`, sub-agent). It builds a per-period candidate pool: a convex relaxation rounded at several scales, then pair-move descent, iterated local search and SA. An exact DP chains the pool over the periods, then an exact DP re-optimises k ≤ 4 assets over all periods. `solve_sub` is the hook for a quantum sub-solver.

## Limits (stated plainly)
- **At 50 assets the quantum windows do not help yet.** Starting from empty at λ = 1e-2, PCE-QLNS stays at the empty portfolio (−1000). Random, correlation-guided and multi-period "hedge" windows were all tried. At high λ the optima hold about 15–37 assets at ±1 unit with the budget used up, and a small window sees only moves that cost more than they gain. Starting from the classical −17 675, 600 PCE windows found no improvement. **All 4 BKV improvements are classical.**
- The quantum results are hybrid. The classical parts are the referee's repair on infeasible splices and the accept-if-better rule. There is no classical optimiser in the loop. All circuits ran on the local Maestro simulator, not the cloud (all windows are under 25 qubits). The only cloud call was the characterization.
- The optimality proof for a010 rests on the dominance argument (v = L − S). The DP matches all 10 BKVs labelled optimal and `brute_force` on a003.

## Handing in (notebook-00 checklist)
| result | file | size | method |
|---|---|---|---|
| a010_t10_orig λ1e-4 −69 482 (quantum) | `solutions/po_a010_t10_orig_b004_l1e-4/…__qlns-pce__20261121-221811.sol` | 630 QUBO vars, ≤ 9 qubits | PCE-QLNS chain seeds 3 → 7 (settings in the `.sol` header) |
| a010_t10_orig λ5e-4 −27 044 (quantum) | `solutions/po_a010_t10_orig_b004_l5e-4/…__qlns-pce__20261121-222829.sol` | 630 / ≤ 9 | PCE-QLNS seeds 5 → 10 |
| a010_t10_orig λ1e-3 −8 397 (quantum) | `solutions/po_a010_t10_orig_b004_l1e-3/…__qlns-pce__20261121-223909.sol` | 630 / ≤ 9 | PCE-QLNS seeds 6 → 8 → 15 |
| 4 a050 BKV improvements, a200, a400 (classical) | see the table in `best_solutions.md` | 3 090+ vars | pool-DP + LNS |

The `.gitignore` now has an exception, so `portfolio_QIcing/solutions/**/*.sol` can be committed. QOBLIB's Rust checker has been run on every file (all pass). QOBLIB does not allow published solutions as warm starts. None were used as starts; the sub-agent read two published a050 solutions only to understand their structure.

## Next steps
- Plug PCE into `lns.solve_sub`, the asset-subset DP. Then the a050 polish step becomes quantum and starts from a strong classical pool.
- Use quantum windows to *build* the per-period candidate pool, which is where high-λ structure is found.
- Run larger PCE windows (MultiBasisEncoding) on the Qoro cloud for a200/a400.
