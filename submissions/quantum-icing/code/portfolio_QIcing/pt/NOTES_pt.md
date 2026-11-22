# Parallel tempering (replica exchange) on QOBLIB 06 portfolio: high-λ a050 instances

**Verdict: PT beat nothing.** In 13 PT runs on 5 target instances at λ = 1e-2 (B = 20), PT never
went below its start. From a good start it returned that start unchanged. From the empty or over-full
start it fell into the empty basin (−1000 / −1520). The new best results in this session all came
from the **baseline** pool-DP + LNS pipeline, run with a different seed (seed 1) and the same
150 s of wall time:

| instance | new best | gap vs QOBLIB best known | previous best | file |
|---|---:|---:|---:|---|
| po_a050_t10_s02 λ=1e-2 | **−45 069** | **−3.45 % (beats best known −43 565)** | −40 149 (+7.84 %) | `solutions/po_a050_t10_s02_b020_l1e-2/po_a050_t10_s02_b020_l1e-2__pooldp__20261122-002024.sol` |
| po_a050_t10_orig λ=1e-2 | −35 445 | +4.48 % | −34 520 (+6.97 %) | `solutions/po_a050_t10_orig_b020_l1e-2/po_a050_t10_orig_b020_l1e-2__pooldp__20261122-001745.sol` |
| po_a050_t15_s00 λ=1e-2 | −41 504 | +5.22 % | −39 280 (+10.30 %) | `solutions/po_a050_t15_s00_b020_l1e-2/po_a050_t15_s00_b020_l1e-2__pooldp__20261122-002311.sol` |

All three solutions are classical (`quantum: False`). `verify_solutions.py` re-scored all 144 .sol
files with the official referee: 0 were infeasible or had a wrong objective line.

## Files

- `pt_core.py`: numba PT kernel (exact int64 delta evaluator, 6 move kinds, swap pass, ladder logic).
- `run_pt.py`: runner for `--algo pt | sa | pooldp`. It polishes with `lns.lns`, checks the result with the referee and records it.
- `run_queue.sh`, `jobs.txt`, `jobs2.txt`: the job queues (3 workers, `NUMBA_NUM_THREADS=2`).
- `summarize.py` writes `table.md`. Per-run logs are in `logs/` and per-run JSON (ladder, swap rates, move stats, trace) is in `results/`.

## Method

**State.** v[t, i] = L − S ∈ {−3..3}, the reduced state. Every replica stays feasible: a move that
breaks |v| ≤ 3, Σ|v| ≤ 20 or C−15 ≤ Σv ≤ C (C = 10) on any day it touches is reverted. Day T−1 is
never touched, because holding there is pure cost and the transition into it is free.

**Delta evaluator.** A move is a list of (t, i, d) changes. Each change updates g[t] = R_t v_t
incrementally (O(n)) and adds the exact integer change in risk, linear, |v|, short and transaction
cost terms. After every run, each replica's tracked energy equals `FastModel.objective` (asserted).
Throughput is about 1.5–2 M moves/s per process with 2 numba threads.

**Moves** (proposal probabilities 0.15 / 0.2 / 0.3 / 0.15 / 0.1 / 0.1):

1. single ±1 on (t, i), optionally as a block over consecutive days (block_prob 0.6);
2. transfer: one unit of a held (t, i) moves to one of the 8 assets with the largest |corr|, with the same factor-exposure sign, over a block of days;
3. hedge pair: +d on a and −sign(corr_ab)·d on a correlated b, over a block of days;
4. time-asset: one asset copies a neighbouring day's value (extends or shrinks a run), or clears a run;
5. time-day: the whole portfolio copies day t ± 1 into day t;
6. shift: one asset's holding window, or the whole portfolio's, moves by one day.

**Ladder.** 16–24 replicas. Each sweep round is 400 Metropolis steps per replica (prange over
replicas), followed by an even/odd neighbour swap pass. Each level sits at a ladder position
x ∈ [0, 1], with T = T_min (T_max/T_min)^x and risk weight s = s_min^x.

- **Temperature PT** (s_min = 1). I tried two settings:
  - T = 10…3000 with the adaptive ladder: fixed endpoints, interior gaps nudged toward 30 % swap acceptance during the first 40 % of the run, after a 10 % burn-in.
  - T = 10…1e5, fixed geometric ladder, 24 replicas, the coldest replica reseeded from the best found every 200 rounds, and a mixed start that cycles empty / over-full / best over the replicas.
- **λ-tempering / Hamiltonian replica exchange** (s_min = 0.1, T = 10…300). The risk matrix is
  scaled by s per level, so s = 0.1 is roughly the λ = 1e-3 landscape, where over-full portfolios
  are good. Swaps use the full two-Hamiltonian Metropolis criterion. Only s = 1 levels can record a best.

**Starts.**

- `empty`: the empty portfolio.
- `overfull`: our λ = 1e-3 best re-scored at λ = 1e-2, from `quantum_focus/starts/` where present, otherwise `solutions/*_l1e-3`. These score +357 k … +508 k at λ = 1e-2.
- `best`: our best λ = 1e-2 solution.
- `mix`: all three starts, cycled over the replicas.

No run started from a published QOBLIB solution.

**Polish.** `lns.lns` (exact-DP LNS) runs for 30 s on the PT result. **Baselines at equal wall time**
(150 s per job):

- `sa`: `lns.sa` restarts from the `best` start for 120 s, then the same 30 s LNS polish.
- `pooldp`: `run_lns.solve` (the full pool-DP + LNS pipeline) for 150 s, seed 1.

## Sanity check

On po_a010_t10_orig at λ = 1e-3 (best known −8 397, our DP optimum −8 397), PT from empty (16 replicas, T = 5…5000, 30 s)
reached **−8 322 (+0.89 %)**. The 10 s LNS polish then reached **−8 397 (+0.00 %)**. The swap rates were
0.26–0.85. On this small instance the kernel and the scoring are correct.

## Results (λ = 1e-2, B = 20; gaps vs QOBLIB best known; 150 s per job)

The pooldp row shows the start as "empty", but the pipeline actually starts from scratch.
"≤prev" means the result matched the previous best and was recorded, as the brief requires. Those
are duplicates, not improvements.

| instance | λ | algo | start | raw (gap) | after LNS polish (gap) | prev best (gap) | BKV | wall s | swap rate min–med–max | held days/assets |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| po_a050_t10_orig | 1e-2 | pooldp | empty | -35000 (+5.68%) | -35445 (+4.48%) **NEW BEST** | -34520 (+6.97%) | -37107 | 150 | — | 7/44 |
| po_a050_t10_orig | 1e-2 | pt | best | -34520 (+6.97%) | -34520 (+6.97%) **≤prev** | -34520 (+6.97%) | -37107 | 164 | 0.68–0.78–0.84 | 6/42 |
| po_a050_t10_orig | 1e-2 | pt | empty | -1520 (+95.90%) | -1520 (+95.90%) | -34520 (+6.97%) | -37107 | 164 | 0.95–0.96–0.97 | 1/3 |
| po_a050_t10_orig | 1e-2 | pt | mix | -34520 (+6.97%) | -34520 (+6.97%) **≤prev** | -34520 (+6.97%) | -37107 | 152 | 0.00–0.59–1.00 | 6/42 |
| po_a050_t10_orig | 1e-2 | pt λ-temp | overfull | -1520 (+95.90%) | -1520 (+95.90%) | -34520 (+6.97%) | -37107 | 162 | 0.00–0.00–0.03 | 1/3 |
| po_a050_t10_orig | 1e-2 | sa | best | -35445 (+4.48%) | -35445 (+4.48%) **≤prev** | -35445 (+4.48%) | -37107 | 152 | — | 7/44 |
| po_a050_t10_s01 | 1e-2 | pooldp | empty | -1000 (+55.52%) | -1000 (+55.52%) **≤prev** | -1000 (+55.52%) | -2248 | 150 | — | 0/0 |
| po_a050_t10_s01 | 1e-2 | pt | mix | -1000 (+55.52%) | -1000 (+55.52%) **≤prev** | -1000 (+55.52%) | -2248 | 152 | 0.28–0.97–1.00 | 0/0 |
| po_a050_t10_s01 | 1e-2 | pt λ-temp | overfull | -1000 (+55.52%) | -1000 (+55.52%) **≤prev** | -1000 (+55.52%) | -2248 | 152 | 0.00–0.07–0.72 | 0/0 |
| po_a050_t10_s02 | 1e-2 | pooldp | empty | -45069 (-3.45%) | -45069 (-3.45%) **NEW BEST** | -40149 (+7.84%) | -43565 | 150 | — | 6/43 |
| po_a050_t10_s02 | 1e-2 | pt | best | -40149 (+7.84%) | -40149 (+7.84%) **≤prev** | -40149 (+7.84%) | -43565 | 151 | 0.65–0.68–0.88 | 5/40 |
| po_a050_t10_s02 | 1e-2 | pt | mix | -40149 (+7.84%) | -40149 (+7.84%) **≤prev** | -40149 (+7.84%) | -43565 | 152 | 0.00–0.64–1.00 | 5/40 |
| po_a050_t10_s02 | 1e-2 | pt λ-temp | overfull | -17145 (+60.65%) | -17145 (+60.65%) | -40149 (+7.84%) | -43565 | 151 | 0.00–0.00–0.47 | 1/14 |
| po_a050_t10_s02 | 1e-2 | sa | best | -40149 (+7.84%) | -40149 (+7.84%) **≤prev** | -40149 (+7.84%) | -43565 | 151 | — | 5/40 |
| po_a050_t15_s00 | 1e-2 | pooldp | empty | -39797 (+9.12%) | -41504 (+5.22%) **NEW BEST** | -39280 (+10.30%) | -43792 | 150 | — | 9/45 |
| po_a050_t15_s00 | 1e-2 | pt | mix | -39280 (+10.30%) | -39280 (+10.30%) **≤prev** | -39280 (+10.30%) | -43792 | 152 | 0.00–0.56–0.98 | 7/44 |
| po_a050_t15_s00 | 1e-2 | sa | best | -41504 (+5.22%) | -41504 (+5.22%) **≤prev** | -41504 (+5.22%) | -43792 | 151 | — | 9/45 |
| po_a050_t15_s01 | 1e-2 | pooldp | empty | -6580 (+51.99%) | -8301 (+39.44%) | -12983 (+5.28%) | -13706 | 150 | — | 1/16 |
| po_a050_t15_s01 | 1e-2 | pt | mix | -12983 (+5.28%) | -12983 (+5.28%) **≤prev** | -12983 (+5.28%) | -13706 | 152 | 0.00–0.79–1.00 | 4/34 |
| po_a050_t15_s01 | 1e-2 | sa | best | -12983 (+5.28%) | -12983 (+5.28%) **≤prev** | -12983 (+5.28%) | -13706 | 150 | — | 4/34 |

Note on the SA rows. The `best` start is read when the job starts. For t10_orig, t15_s00 and
t15_s01, the SA job started after the pool-DP job had already recorded its new best, so SA started
from that new solution (−35 445 / −41 504; t15_s01 had no new best, so −12 983). For t10_s02 it
started from the old −40 149. In every SA run, SA plus polish returned its start unchanged. So SA
did not beat anything either.

Not run: the po_a050_t10_s00 PT jobs. I cut them when I replaced the first job queue, because that
instance already beats its best known value. I also did not run the optional quantum hybrid
(seeding replicas with PCE-QLNS samples, or using the `move_qubo` large move). The time box ran out,
so there is **no quantum contribution** to report.

## Swap-rate diagnostics

- **Narrow ladder (T = 10…3000), empty start.** The swap rate is 0.95–0.97 on every pair. Every
  replica, even at T = 3000, sits near the empty portfolio (hot E ≈ −235). One unit of a single
  asset costs about 18 k in risk (median diag R_t ≈ 1.5–1.9e4), so the whole ladder is effectively
  frozen in the empty basin, and the high swap rate means nothing.
- **Adaptive ladder.** It degenerates. Replicas that share the same funnel swap trivially, so the
  adaptation packed 11–13 of the 16 levels at T ≈ 10, left one big jump to 948 → 3000 at the top,
  and the rates stayed at 0.68–0.84 throughout. Fixed endpoints plus burn-in did not fix this:
  there is no smooth energy-vs-T curve to equalise.
- **Wide fixed ladder (T = 10…1e5, 24 levels).** Every instance shows a hard **swap wall at
  T ≈ 22–50**: the pairs at levels 2–4 have a swap rate of **0.00**. Below the wall, replicas sit in the
  hedged, factor-neutral basin (−34 520 on t10_orig). Above it, they sit in the empty or random
  basin (≥ −1 000). The two basins differ by about 3e4 in energy, and the energy jumps across the
  wall instead of changing smoothly with temperature. Above the wall the rates fall smoothly
  from 1.0 to about 0.2–0.3 at T ≈ 1e4 and rise again to about 0.45 at the hot end.
  - Example, t10_orig: `[0.92, 0.93, 0.00, 0.00, 0.00, 1.00, 1.00, 0.99, …, 0.42, 0.35, 0.29, 0.29, 0.30, 0.35, 0.40, 0.46]`.
  - Example, t15_s01: `[0.98, 0.98, 0.98, 0.00, 0.00, 1.00, …, 0.21, 0.20, 0.20, 0.24, 0.30, 0.35]`.
  
  Hot replicas therefore never deliver a new hedged configuration to the cold end.
- **λ-tempering.** Swap rates are about 0.00 on almost every pair (median 0.00–0.07). A state that
  is good at s ≈ 0.7 still holds a factor exposure worth about 1e5 at s = 1, and at T ≈ 10 that is
  far too much to swap. Making the swaps work would need far more than 24 levels near s = 1. In move
  acceptance, the cold levels accepted only 0.0–0.2 % of moves.
- **Move acceptance at the cold end** (t10_orig, best start). single 0.3 %, transfer 0.5 %,
  hedge 0.2 %, time-asset 0.1 %, time-day 2.5 %, shift 2.6 %. The cold replicas do move, but
  only inside the local hedged configuration, and the LNS polish confirms that configuration is a
  local optimum.

## What failed, and why

1. **PT cannot leave the empty basin on these instances.** The factor eigenvalues reach about 6e5
   against a median of about 2e3. Any partially built hedge is therefore exposed to a factor and
   costs 1e4–1e5. The good states are narrow holes in a landscape shaped like a golf course, and
   tempering does not lower that barrier.
   - t10_orig from empty reached −1 520 (+95.9 %).
   - s01 stayed at −1 000 in all 4 runs (2 pilots and 2 production runs), against a best known of −2 248.
2. **The adaptive ladder is unreliable with a bimodal energy distribution.** It equalises
   trivially high acceptances inside one basin and cannot see the wall between basins.
3. **λ-tempering is the right idea, but 24 replicas are far too few.** It was the only PT variant
   that left the empty basin from the over-full start, reaching −17 145 on t10_s02. That is still
   +60.7 % and far from −40 149. On t10_orig it reached −1 520, and on s01 −1 000.
4. **From a good start, PT returns the start unchanged.** In all 7 runs with `best` or `mix`
   starts, PT plus LNS returned exactly the previous best. The stronger move set and the exact
   polish found no better configuration within 150 s.

What did work: **seed diversity in the pool-DP + LNS pipeline.** With seed 1 instead of seed 0,
and the same 150 s, it improved 3 of 5 targets. On po_a050_t10_s02 the result now beats the QOBLIB
best known value (−45 069 vs −43 565). It got worse only on t15_s01 (−8 301 vs −12 983), where the
earlier result was kept. The cheapest next step is therefore more pool-DP seeds per instance, not PT.

## Reproduce

```bash
cd portfolio_QIcing/pt
NUMBA_NUM_THREADS=2 ../../.venv/bin/python run_pt.py --instance po_a010_t10_orig --lam 1e-3 \
    --start empty --time 30 --polish 10 --replicas 16 --tmin 5 --tmax 5000 --no-record   # sanity
./run_queue.sh jobs2.txt           # wide-ladder PT + baselines, 3 workers
../../.venv/bin/python summarize.py > table.md
```
