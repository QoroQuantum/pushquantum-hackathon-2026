# Portfolio (QOBLIB 06): every method and idea tried so far

Snapshot taken 2026-11-22 at 03:12. It covers everything under `portfolio_QIcing/` and `deliverables/portfolio/`.
Sources: `NOTES.md`, `NOTES_lns.md`, `REPORT.md`, `pt/NOTES_pt.md`, `quantum_focus/NOTES_quantum.md`,
`sparse/NOTES.md`, `formulation/README.md`, `results.csv`, `best_solutions.md` and the logs in each `logs/` folder.

The official referee scores everything: `qoblib_tools.PortfolioModel.score`. Lower is better. Gap = (ours − BKV)/|BKV|,
so a negative gap beats the BKV. 148+ `.sol` files have been verified: all are feasible, and all pass QOBLIB's Rust checker.

---

## 0. Status in one screen

| area | best we have | quantum? |
|---|---|---|
| a010 (all 64) | exact DP = BKV on every instance, so all are **proven optimal** (54 were only "best known") | no |
| a010_t10_orig λ 1e-4 / 5e-4 / 1e-3 | PCE-QLNS from empty reaches the **proven optimum** (≤9 qubits) | **yes** |
| a010_t10_s02 λ 1e-4 | PCE-QLNS −53 220 (+0.066 %) | yes |
| a050 BKVs beaten | t10_s00 1e-2 (−43.2 %), t15_orig 1e-2 (−42.9 %), t10_s02 1e-2 (−3.45 %), t15_s02 1e-2 (−2.2 %), t10_s00 5e-4 (−0.14 %) | no (pool-DP + LNS) |
| a050 BKV beaten, new at 03:00 | t10_s00 1e-3: **−191 991 (−0.058 %)** from slot-QLNS with the *random* sub-solver | no (classical control) |
| a050 best quantum | t10_s00 1e-4: PCE-QLNS from empty −340 222 (+4.31 %); t10_s00 1e-3: slot-QLNS-PCE −191 446 (+0.23 %) from the LNS start | yes |
| a200 / a400 (no BKV published) | classical λ1e-3: −818 362 / −1 896 257; quantum (sparse PCE-QLNS) λ1e-4: −794 168 / −1 397 396 | mixed |
| **a050 λ = 1e-2, new at 03:45 (§6)** | stronger per-day pool + chain DP: t15_orig **−78 299** (Jules02 −77 193), t10_s00 **−39 665** (Jules02 −24 353). Not recorded yet | no |
| a050 λ = 1e-2, still bad (before §6) | t10_orig +4.5 %, t10_s01 +55.5 % (stuck at empty), t15_s00 +5.2 %, t15_s01 +5.3 % | — |

**Running at 03:12** (`formulation/`, 1500 s each, recorded when they finish):
- `slot_qlns.py` with `--solver rand --inner-descent` on a050_t10_orig at λ 1e-2, 1e-3 and 5e-4 (seed 2). The λ = 1e-2 log already shows **−43 255 at 39 s, against BKV −37 107 (−16.6 %)**. This is provisional and not yet recorded.
- `slot_qlns.py` on a050_t10_s00 λ 1e-3, two continuations from −191 991: PCE with Poly code (seed 3) and rand (seed 4).
- `exp_slots.py --polish` on a050_t10_s00 at λ 1e-2, 1e-3 and 1e-4 (the 8-repeat slot-PCE pools plus L1/L2 polish). The 5e-4 job is queued.

---

## 1. Model facts everything else builds on

These are derived from the referee code (`NOTES.md`).

- **Interest is linear.** It equals −10·(C − net), so no cash-slack bits are needed for the objective.
- **Long and short in the same asset is dominated.** So the reduced state is the net position v[t,i] = L − S ∈ {−3..3}, with cost |v|·tc.
- **B = 4 (a010): the capital constraint never binds**, so only Σ|v| ≤ 4 matters. **B = 20 (a050): capital binds**, with C−15 ≤ Σv ≤ C.
- **The last period is always empty.** It has no return, there is no transaction cost from T−2 to T−1, and liquidation is charged at T−1.
- **Periods are weakly coupled.** The transaction cost is 0.1 % per unit. On a010, every optimal period state is among the top 4 of its *single-period* problem (`formulation/`, finding 1).
- **The structure at high λ (1e-2) is the hard part.** The risk matrix has a few huge eigenvalues (6.7e5, 2.6e5, 1.6e5, against a median of about 1.5e3). Good portfolios are factor-neutral baskets of about 15–45 assets at ±1 unit, with the budget saturated, held over a few days. Any single unit or hedge pair costs more than it earns (the best pair costs +3 137 per day). As a result, every small-move method falls into the empty portfolio (−1000 / −1500).
- λ = 1e-2 on a010: the optimum is the empty portfolio.

---

## 2. Classical methods

### 2.1 Exact DP over periods: `pf_core.exact_dp`, `run_dp_all.py`
- State = the v vector with Σ|v| ≤ B (8 341 states for n = 10). Transitions are weighted L1 rebalancing costs, computed by a numba min-plus step with pruning. About 2 s per instance.
- Validated against `brute_force` on a003 and against the 10 BKVs labelled "optimal".
- **Result: all 64 a010 instances have DP optimum = BKV.** This is the ground truth for every quantum experiment. Table in `logs/dp_all.csv`.
- Does not scale past a010, since the state count explodes at n = 50.

### 2.2 Notebook baseline: SA → repair → local search (notebook 1.4)
- The starter's classical bar on a010_t10_orig: −69 482 at 1e-4, −26 548 (+1.8 %) at 5e-4, −3 703 (+56 %) at 1e-3, −51 805 (+2.7 %) on s02 at 1e-4.

### 2.3 Pool-DP + exact-DP LNS: `lns.py`, `run_lns.py` (the main classical engine)
1. **Per-period candidate pool.** Each period is solved alone as an integer QP. Entry/exit cost is emulated with e·tc·|v| for e = 0, 1, 2. Candidate sources:
   - a convex relaxation (v = p − q, SLSQP), rounded at scales 1…40, then repaired, then best-improvement descent over single and **pair** unit moves (vectorised delta matrix);
   - ILS (kick 2–6 assets, then descend);
   - numba SA with compound moves.
2. **Chain DP.** Exactly one pool vector is chosen per period, with true transaction costs. The pool has about 1 000–1 600 vectors.
3. **LNS polish.** k ≤ 4 assets are chosen (correlated partners, held or random). Everything else is fixed, and an exact DP over all periods runs on 7^k joint states. The polish is monotone. The `solve_sub` hook is the plug-in point for a quantum sub-solver and **has never been used with a quantum solver**.
- Results: all 5 classical BKV beats; a050 λ ≤ 1e-4 mostly = BKV; λ 5e-4 / 1e-3 within 0.1–1.7 %; a200/a400 first values (LIGHT = 2 mode, 200–232 s, a400 not converged).
- **Seed diversity worked.** With seed 1 and the same 150 s, the pipeline improved 3 of 5 λ = 1e-2 targets, including the t10_s02 BKV beat (−45 069).
- What failed: plain LNS from empty (a050 1e-3: +5.4 %; 1e-2: stuck near empty); full-horizon SA (1e-2: −9 038, +76 %).
- **Control at 03:00:** `formulation/lns_control.py` ran plain `lns.lns` from −190 685 (a050_t10_s00, 1e-3) for 1 080 s with 285 k iterations and found **no improvement**. It is window-locally optimal for the k ≤ 4 asset DP.

### 2.4 Parallel tempering: `pt/` (failed)
- Numba PT kernel with an exact int64 delta evaluator and 6 move kinds: single/block, transfer to a correlated asset, hedge pair, time-asset run extend/shrink, copy the whole day, and shift a window. 16–24 replicas, polished by a 30 s LNS.
- Variants: narrow adaptive ladder (T 10…3000), wide fixed ladder (10…1e5, cold replica reseeded), and **λ-tempering / Hamiltonian replica exchange** (risk scaled by s ∈ [0.1, 1]). Starts: empty / over-full (the λ1e-3 best re-scored) / best / mix.
- **Verdict: beat nothing in 13 runs.** From empty it fell into the empty basin. From a good start it returned the start unchanged.
- Diagnosis: a hard **swap wall at T ≈ 22–50** (swap rate 0.00) between the hedged basin and the empty basin. The adaptive ladder degenerates on the bimodal energy. λ-tempering is the right idea, but 24 replicas are far too few (swap rate about 0 near s = 1); its best was −17 145 on t10_s02 (+60 %).
- SA baseline at equal wall time also returned its start unchanged.

---

## 3. Quantum methods

### 3.1 Notebook route A: PCE on the whole QUBO (notebook 1.5)
- Only for the tiny teaching instance (58 vars → PolyEncoding on 11 qubits). Not scaled.

### 3.2 Notebook route B: partition → QAOA → stitch → polish (`run_partitioned.py`)
- Compact QUBO (630 vars), CommunityDecomposer with cluster 12, QAOA with 2 layers, MC 50×10, 62 sub-programs of ≤ 12 qubits, 31 k circuits.
- Result: −3 484 (+95 %) after stitching and repair; −54 923 (+21 %) after the referee's local search.
- **Why weak:** the partitions cut through the budget constraints, so stitching loses feasibility.

### 3.3 Compact QUBO: `pf_quantum.CompactQUBO` / `DenseQUBO`
- Keeps the official unary copies, folds the interest into the objective (linear), drops the cash slack, uses a ⌈log₂(B+1)⌉-bit budget slack, and adds a 4-bit cash register only when B > 5.
- Sizes: 630 vars for a010_t10 (official: 710) and 3 090 vars for a050_t10. Energy = referee objective (verified).
- `DenseQUBO` slices conditional sub-QUBOs out of the matrix. `dimod` copy+fix was too slow at 450 k couplings.

### 3.4 Quantum LNS with QAOA (`run_qlns.py --routine qaoa`)
- Window: 1 period × 2 assets (15 qubits) or 1 asset × 2 periods (18 qubits). Exact conditional sub-QUBO. QAOA with 4 layers on local Maestro, MC 30×5, 4 000 shots.
- **Warm-started from Characterization Service angles** (job `c3a51b2e-ad3f-4cae-924b-b4ce2c013196`; it recommends 5 layers, we used 4). It recommended a penalty weight of about 0.1 on an empty window, but we kept 5000 because windows next to held positions do bind.
- Every sample is spliced into the incumbent and scored by the referee (repaired if infeasible). The best improving candidate is accepted. There is no classical local search in the loop.
- Control: the exact window optimum (by enumeration) was among the samples in 73 % of windows (145/200).
- Result on a010_t10_orig 1e-4: −68 249 (+1.8 %) from the route-B start; −54 272 (+22 %) from empty (stopped).
- About 6–10 s per window, against about 1 s for PCE. Windows are small. **Superseded by PCE.**

### 3.5 Quantum LNS with PCE (`run_qlns.py --routine pce`): **the main quantum result**
- Windows: 1 period × 4–6 assets, or 2–3 assets × 2 periods. That is 27–42 binary vars, which Divi `PolyEncoding` packs into **7–9 qubits**. 3 layers, `SoftObjective(α = 2)`, MC 40×8, 4 000 shots, local Maestro, penalty weight 5000.
- Window types: random, correlation-guided (`p_corr`), multi-period "hedge" windows (`p_hedge`).
- **Chaining:** each run starts from the previous run's `.sol` with bigger windows.
- **a010 results (from empty):**
  - 1e-4: −69 397 after one run, then **optimum** after a 2-run chain (0.22 M circuits);
  - 5e-4: **optimum** after a 2-run chain (0.29 M);
  - 1e-3: **optimum** after a 3-run chain (0.91 M, 31 min);
  - s02 1e-4: −53 220 (+0.066 %, 0.73 M).
- **a050_t10_s00 (from empty, 30 min, about 0.9 M circuits):** 1e-4: **−340 222 (+4.31 %)**, 42 assets with the budget full; 1e-3: −157 229 (+18.1 %).
- **a050 at λ = 1e-2 fails:**
  - from empty, it stays at −1 000 with random, correlation-guided (s13) and hedge (s14) windows;
  - from the rounded-relaxation start it ends at +37 234 (+402 %), and from the λ1e-3 portfolio re-scored at 1e-2 it ends at +69 194;
  - from the classical −17 675, 600 windows found nothing (s12, relabelled `quantum: False`).
- **Caveat** (see formulation finding 4): part of the success comes from **referee repair** of infeasible splices, plus accept-if-better.

### 3.6 Block-sparse QUBO for a200/a400: `sparse/` (works for scale, quality unproven)
- Every coupling is either inside one day (risk, budget and capital penalties) or links the same unit bit on adjacent days (rebalancing XOR). So the QUBO is stored as T dense float32 day-blocks plus one adjacent-day weight vector.
- Memory: a200 takes 56 MB (dense: 2.3 GB) and a400 takes 222 MB (dense: 9 GB). Validated exactly against the dense path on a050.
- `run_qlns_sparse.py`: the same PCE-QLNS (1 day × 5 assets, or 2 assets × 2 days, ≤ 9–10 qubits, about 0.3–0.5 s per window). Infeasible splices get a cheap per-day repair.
- 60 min from empty: a200 1e-4 **−794 168**, a400 1e-4 **−1 397 396** (first values, no BKV; both still improving). a200 1e-3: −469 837, **43 % worse** than the classical −818 362.
- Lesson: **memory, not qubits, limits scale**. Window size stays constant.

### 3.7 `quantum_focus/`: four attempts to fix a050 λ = 1e-2 (all failed)
| method | idea | result |
|---|---|---|
| **Q-ROUND** (`q_round.py`) | per day: convex relaxation w*, then a QUBO over rounding bits (v = ⌊s·w*⌋ + b, cost exactly quadratic), then PCE samples it, then a pool, then chain DP | dead end. The naive control gives −1000, and even the *optimal* rounding (SA) is +8 293 against a best day of −6 900 |
| **Q-EXCHANGE** (`q_exchange.py`) | XY-mixer QAOA initialised at the incumbent (`CustomPerQubitState`, `xy_mixer(edges)`), which conserves the unit count (100 % of samples) | no change from −34 520 (a050_t10_orig, smoke test). Windows over 16 qubits went to qoro_cpu |
| **Q-MOVES** (`q_moves.py`) | a QUBO that selects a conflict-free set of moves (drop, extend, shrink, shift, flip, transfer). Energy exact (0/880 mismatches) | PCE and QAOA samples are near-uniform. QAOA 18 qubits on qoro_cpu took 115–185 s per call. No useful change (starts are +409 k). SA on the same QUBO does improve it |
| **λ-homotopy** (`homotopy.sh`) | PCE-QLNS follows the optimum from λ1e-3 through 1.5–7.5e-3 up to 1e-2 | fails. The score turns positive by 7e-3 (+5 547, +22 464). Both chains stopped. Final 1e-2 incumbents +45 037 / +59 879 |

### 3.8 `formulation/`: new QUBO/HUBO encodings built for PCE (in progress)
Findings:
1. **Exact period decoupling through a DP.** A pool of the top 5 states per period plus the exact chain DP gives the proven optimum on a010. So the quantum task becomes T independent single-period problems: 20–33 vars on a010 (instead of 630) and 140–159 on a050 (instead of 3 090).
2. **Cell encodings.** `formulations.py` has unary, sign-magnitude HUBO (3 bits per asset instead of 6; HUBO terms cost nothing extra in PCE's classical loss), ternary and ternary-SM. Each comes with 3 constraint modes: slack, unbalanced, none. All are validated against the referee (0 mismatches).
3. **Divi's Poly code is a poor fit for sparse portfolios.** It decodes x = 1 ⊕ Mz, so about N/2 variables are 1 in every shot. With sign-magnitude cells, 0 % of codewords are feasible (1 % after re-centring on empty). `codes.py` views the encodings as linear codes and has `MaskEncoding` and a complement trick. `code_diag.py` enumerates all 2^q codewords.
4. **Repair was doing much of the work.** 32 random codewords plus greedy repair already reach the per-period optimum. **From here on, results are reported without repair or local search.**
5. **Slack/penalty formulations give a bad relaxed landscape.** Even the ideal relaxation rounds to states ranked 100–2 000.
6. **Slot formulation** (`slots.py`). A period is B slots. Each slot is a ⌈log₂(n+1)⌉+1-bit choice of (asset, sign) or none. The budget holds for *every* bitstring and every fractional point (no penalty). The energy is pairwise between slots. The rank bits are a Gray code over assets ordered by their leading factor, so correlated assets are one bit flip apart. A closed-form mean-field PCE loss is in `SlotObjective`.

**Slot-PCE + chain DP, no repair** (MultiBasis-2, 5 qubits per period, L-BFGS, α = 6, 12 restarts):

| a010 instance, λ | slot-PCE | random, same count | circuits |
|---|---:|---:|---:|
| t10_orig 1e-4 | −68 448 (+1.49 %) | +19.97 % | 45 k |
| t10_s02 1e-4 | −50 888 (+4.45 %) | +13.67 % | 44 k |
| t10_orig 5e-4 | −24 206 (+10.5 %) | +15.7 % | 37 k |
| t10_orig 1e-3 | −6 244 (+25.6 %) | **+20.1 % (random wins)** | 27 k |

- Sweeps (`slots_*_sweep_l1e-4.txt`): **MultiBasis-2 beats Poly.** L-BFGS beats COBYLA and MC. MC alone is no better than random. α = 6 beats α = 15. More layers (10) and more qubits did not help.
- **a050 slot-PCE pools** (140 vars, 11 qubits, 8 repeats, about 30 k circuits): the raw chained result is bad (1e-4: +39.6 %; 1e-3: +88 %; 1e-2: empty). After **L1 polish** (period descent on each pool state, then chain DP), 1e-4 reaches −355 321 (+0.059 %), but *every* sampler, random and empty included, gives the same number. So the polish does the work. At 1e-3, random + L1 (+3.6 %) beats PCE + L1 (+6.3–7.8 %).

### 3.9 Slot-QLNS (`formulation/slot_qlns.py`): newest, 02:00–03:00
- Move: pick a period, remove r ≤ k held units (random or a correlated cluster), then let k = 6 free slots jointly re-choose up to 6 units. That is 42 vars, which MultiBasis-2 PCE packs into **6 qubits**. The neighbourhood holds about 1e12 states.
- Candidates are scored on the **exact multi-period objective**. With `--inner-descent`, a period descent runs on each candidate. A move is accepted only if it improves.
- Sub-solver controls: `--solver rand` (uniform bitstrings, same count) and `--solver ideal`.
- **a050_t10_s00 λ 1e-3, start −190 685 (LNS best), about 2 800 s:**

  | sub-solver | inner descent | final | gap | accepted |
  |---|---|---:|---:|---|
  | PCE | no | −190 685 | +0.62 % | 0/667 |
  | PCE | yes | −191 446 | +0.23 % | 4/647 (126 k circuits) |
  | **rand** | yes | **−191 991** | **−0.058 % (beats BKV)** | 6/1137 |
  | plain LNS control (`lns_control.py`) | — | −190 685 | +0.62 % | 0 (285 k iterations) |

- **Reading:** the *slot move plus period descent* is a genuinely new and strong neighbourhood. It escapes the k ≤ 4 asset-DP local optimum. PCE as the sampler is **worse than uniform random** here, so this BKV beat is classical.

---

## 4. Lessons that cut across everything

1. **Small-move neighbourhoods die at high λ.** This covers SA, PT, single/pair moves, small QLNS windows, Q-MOVES and homotopy. Good states need about 20 coordinated units. What works is a *per-period* solve over many assets (QP relaxation pool, slot move with descent) combined with an exact chain DP across periods.
2. **The exact chain DP over per-period pools is the backbone.** It is exact, cheap, and decouples time. Every successful method (DP, pool-DP, slot pools, slot-QLNS) uses period decoupling.
3. **Always run a random control at equal candidate count.** At a050, random ≥ PCE (slot pools with L1, slot-QLNS). At a010 with low λ, PCE beats random (slot-PCE +1.5 % against +20 %). The "quantum advantage" claim is therefore only defensible on a010 at low λ.
4. **Repair and polish can hide the sampler.** Report numbers both with and without them (formulation finding 4; a050 L1 polish gives −355 321 for every sampler).
5. **PCE beats QAOA for windows here.** It handles 27–42 vars on 6–9 qubits at about 1 s per window, while QAOA needs 15–18 qubits at 6–10 s per window. Cloud QAOA (qoro_cpu, 18 qubits) takes 115–185 s per call: batch it or skip it.
6. **Within PCE: MultiBasis-2 beats Poly** (feasibility and decoded-weight bias). L-BFGS beats MC and COBYLA. α ≈ 6.
7. **Seeds matter more than fancy samplers** for pool-DP (seed 1 improved 3 of 5).
8. Operations: keep to ≤ 4 Python processes on a050 dense (WSL OOM). Use `pkill -f`/`pgrep -f` only by PID (the pattern matches the waiting shell). Keep `NUMBA_NUM_THREADS=2` per worker.

---

## 5. Ideas not yet done (from all the notes)

- **A quantum `solve_sub` in `lns.lns`.** Put PCE into the k-asset exact DP so the a050 polish becomes quantum (k = 2 assets × T periods ≈ 6 bits × 2 × T).
- **Quantum per-period pool construction** at a050, where the high-λ structure lives. Slot-PCE does this, but it does not yet beat random there.
- **Slot formulation at high λ:** use fewer slots (K < B), weight the "none" code, tune α and restarts per λ. a050: K = 20 × 7 bits = 140 vars on MultiBasis-2 with 11 qubits, plus an unbalanced capital penalty.
- **Slot-QLNS scale-up:** run the rand variant on every a050 instance and λ (the λ1e-2 orig run already looks like −16.6 % vs BKV). Find a PCE configuration that beats rand on this move (code, α, more restarts, warm start).
- **More pool-DP seeds** on the λ = 1e-2 laggards (t10_orig, t10_s01, t15_s00, t15_s01).
- **λ-tempering PT with many more replicas** concentrated near s = 1. It was the only PT variant to leave the empty basin.
- PT replicas seeded with PCE-QLNS samples, or a `move_qubo` large move (never run).
- **a200/a400:** run longer, chain runs as on a010, add more λ values and seeds, enable LIGHT = 1 (QP-relaxation pool) for a400, and batch disjoint windows per `qoro_cpu` job.
- Larger PCE windows (MultiBasisEncoding) on the Qoro cloud for a200/a400.

---

## 6. Leaderboard check (03:30): why Jules02 is far ahead at λ = 1e-2

**Leaderboard** (we are LorenzoLaCorte). Jules02's method on every a050 row is "Block decomposition + local search", with 0 qubits:

| instance, λ | Jules02 | us | BKV |
|---|---:|---:|---:|
| a050_t15_orig 1e-2 | **−77 193** | −50 971 | −31 504 |
| a050_t15_s02 1e-2 | **−74 860** | −41 517 | −40 622 |
| a050_t10_s00 1e-2 | **−24 353** | −19 785 | −12 340 |
| a050_t10_s02 1e-2 | **−52 358** | −45 069 | −43 565 |
| a050_t15_s02 5e-4 | −392 793 | −392 723 | −392 668 |
| a050_t10_s00 5e-4 | −234 783 | −234 696 | −234 259 |
| a200_t10_orig 1e-3 | −840 155 | (−818 362) | none |
| a400_t15_orig 1e-3 | −3 162 386 (9 qubits) | — | none |

On a010, Jules02 lists "QAOA block solver (p = 2, secret-sauce init) + local search" on 9 qubits and also reaches the optimum.

**No loophole in the referee.** `qoblib_tools.PortfolioModel.score_arrays` charges risk and return on v = L − S. The budget is L + S ≤ B and capital is on the net position. Long and short in the same asset stays dominated. Their solutions are genuinely better optima.

**Diagnosis: our per-day integer solver is the weak link, not the DP.**
- **Our best λ = 1e-2 solutions leave many trading days empty.**

  | instance | empty trading days | stage total | transaction total |
  |---|---:|---:|---:|
  | t15_orig | 5 of 14 | −73 388 | +28 357 |
  | t10_s00 | 6 of 9 | −31 483 | +13 808 |
  | t10_s02 | 3 of 9 | −61 697 | +16 628 |
  | t15_s02 | 6 of 14 (2 more hold just 2 units) | −60 180 | +18 663 |

  A full day is worth −5k…−19k. Rotating 20 units in or out costs only about 2k each way, so empty days are a big loss.
- **There is plenty of headroom.** The per-day convex relaxation lower bound (`lns.period_relaxation`, summed over days, transactions ignored) is −287 555 on t15_orig, −171 560 on t10_s00, −182 392 on t10_s02 and −260 590 on t15_s02. It is loose, since the relaxation spreads 20 units fractionally over about 26–28 assets, but the Jules02 values sit well inside it.
- **Our pool generator (`period_candidates_qp`) misses good day portfolios.** It rounds the relaxation at fixed scales, then runs pair descent. *Randomised* rounding of the same relaxation (floor(x + U[0,1)), 300 draws), then pair descent and 1 500-kick ILS, finds much better day portfolios, about 20 s per day. On t15_orig:

  | day | `period_candidates_qp` (e = 0) | randomised rounding + descent + ILS |
  |---|---:|---:|
  | 0 | −100 (empty) | −1 075 |
  | 1 | −1 434 | **−8 767** |
  | 2 | −9 514 | −12 697 |
  | 5 | −100 (empty) | −3 194 |
  | 7 | −100 (empty) | **−6 732** |
  | 11 | −1 324 | **−10 478** |

- **Probable Jules02 recipe.** Solve each day, or a block of days, close to integer optimality, then chain and polish. 50 variables in {−3..3} with a convex quadratic is easy for an exact MIQP solver (Gurobi or SCIP). Our chain DP is exact, but it can only choose from the pool we give it.

**Experiment running** (not recorded; scratchpad `strongpool.py`): the stronger per-day pool (randomised rounding + descent + ILS, e = 0, 1, top 40 per day, plus our current best's day vectors), then `chain_dp`, then 120 s of `lns.lns`, on t15_orig and t10_s00 at λ = 1e-2.

**Result (03:45): the diagnosis holds, and both instances now beat the leaderboard leader.**

| instance λ 1e-2 | previous best (folder) | stronger pool + chain DP | Jules02 (#1) | BKV | pool size | wall time |
|---|---:|---:|---:|---:|---:|---:|
| a050_t15_orig | −45 031 | **−78 299** | −77 193 | −31 504 (−148.5 %) | 1 136 | 6 min |
| a050_t10_s00 | −17 675 | **−39 665** | −24 353 | −12 340 (−221 %) | 731 | 4 min |

- Both results are referee-checked and feasible. They are **not yet recorded** and the QOBLIB Rust checker has not been run on them yet.
- Files in `reset-3AM/`:
  - `strongpool.py`, `dayopt.py` (per-day test), `bound.py` (relaxation bound);
  - solution vectors `t15orig_l1e-2.npy`, `t10s00_l1e-2.npy`, shape [T, n], net positions v;
  - logs in `logs/strongpool_*.txt`.
- Run the scripts from `portfolio_QIcing/`, e.g. `NUMBA_NUM_THREADS=2 ../.venv/bin/python reset-3AM/strongpool.py po_a050_t15_orig 1e-2 out.npy`.
- To record a result: `pf_core.record_solution` with `pf_core.v_to_positions(model, np.load(...))`.
- The 120 s LNS polish found **nothing** on top of the chain DP. Every gain came from better per-day candidates. The k ≤ 4 asset DP cannot build these portfolios.
- The best day stage values now are −2k…−16k on *every* trading day, so no day stays empty.
- The pool is classical: randomised rounding of the relaxation, then descent, then ILS.

**Next steps from this:**
1. Replace or extend `period_candidates_qp` with randomised rounding + ILS, or with an exact per-day MIQP (`pip install pyscipopt`, or Gurobi if a licence is available). Re-run pool-DP + LNS on every a050 instance at λ 1e-2, 1e-3 and 5e-4.
2. Add multi-day blocks: optimise one shared portfolio held over 2–3 consecutive days (sum of stage costs), so the chain DP sees "hold" candidates with no rotation cost.
3. **Quantum angle:** per-day pool generation is exactly where a quantum sampler would add value (slot formulation, §3.8). Benchmark it against randomised rounding at equal candidate count.

---

## 7. File map

| path | what |
|---|---|
| `pf_core.py` | BKV table, reduced state, `exact_dp`, `record_solution` |
| `pf_quantum.py` | `CompactQUBO`, `DenseQUBO` |
| `run_dp_all.py` | exact DP on all a010 |
| `run_partitioned.py` | notebook route B on the compact QUBO |
| `run_qlns.py` | QAOA/PCE quantum LNS (main quantum runner) |
| `lns.py`, `run_lns.py`, `prefetch.py` | pool-DP + exact-DP LNS (main classical runner) |
| `verify_solutions.py` → `best_solutions.md`, `logs/verify.txt` | re-score every `.sol` |
| `pt/` | parallel tempering (+ SA and pool-DP baselines) |
| `quantum_focus/` | Q-ROUND, Q-EXCHANGE, Q-MOVES, homotopy; `starts/` are NOT submissions |
| `sparse/` | block-sparse QUBO + a200/a400 PCE-QLNS |
| `formulation/` | period decoupling, cell encodings, PCE codes, slot formulation, slot-QLNS, LNS control |
| `report/portfolio_qicing.pdf` | PDF write-up (source `.tex.in` + `plotdata.json`) |
| `deliverables/portfolio/` | 13 hand-in `.sol` files + README |
