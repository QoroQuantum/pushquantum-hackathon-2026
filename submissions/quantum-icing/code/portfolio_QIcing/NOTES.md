# Portfolio (QOBLIB 06) — working notes, QIcing team

Running log. Newest entries at the bottom. Concise report: `REPORT.md`. All solutions: `results.csv` + `solutions/<instance>/`.

## Setup
- Folder copy of notebook 01: `01_portfolio_optimisation.ipynb` (starting state). Heavy lifting moved to `.py` files (long runs, numba).
- `pf_core.py`: BKV table (`qoblib_bkv_table.md`, copy of QOBLIB `06-portfolio/solutions/README.md`, 2026-11-21), reduced-state helpers, exact DP, `record_solution` (writes `.sol` via referee + row in `results.csv`).
- Referee = `qoblib_tools.PortfolioModel.score` (unchanged).

## Model facts worth knowing (derived from the referee code)
- Interest = −10·(C − net) per period → **linear** in positions; cash slack bits are not needed.
- Holding long and short of the same asset in one period is **dominated** (same net, extra fees; transaction cost |ΔL|+|ΔS| ≥ |Δv|). So reduced state per (period, asset) = net position v ∈ {−3..3}, cost |v|·tc.
- With B = 4 (10 assets), net ∈ [−4, 4] ⊂ [C−15, C] = [−5, 10] → **capital constraint never binds**; only the budget Σ|v| ≤ 4 matters.
- No transaction cost between period T−2 and T−1, liquidation cost at T−1, no return in T−1 → last period is always empty in an optimum.
- λ = 1e-2 on 10 assets: BKV −1000 (−1500 for T = 15) = empty portfolio (pure interest). Risk too large for any position.

## Result 1 — exact DP proves all 64 ten-asset BKVs optimal (classical)
- `run_dp_all.py`: DP over periods; state = v vector with Σ|v| ≤ 4 (8 341 states for n = 10), transition weighted L1 (numba). ~2 s per instance.
- Validated: matches `brute_force` on a003 (B = 2, 3, 4) and the 10 "optimal"-labelled BKVs.
- **All 64 a010 instances: DP optimum = BKV.** 54 of them were labelled "best known" only → now proven optimal (given the dominance argument). Nothing to beat at 10 assets; they become ground truth for quantum runs. Table: `logs/dp_all.csv`.
- Consequence: improving a BKV needs a050 (Gurobi "best known", n = 50, B = 20) or a200/a400 (no published values at all).

## Quantum pipeline (target: easiest case po_a010_t10_orig, B = 4, λ = 1e-4; proven optimum −69 482)
Why this case: smallest leaderboard instance (notebook warm-up), proven optimum known (DP) → we can measure exactly how close quantum gets.

### Compact QUBO (`pf_quantum.CompactQUBO`)
- Unary copies as official (3 long + 3 short bits per asset-period), interest folded linear, **no cash slack**, 3-bit budget slack → **630 vars** (official 710). Verified: objective energy = referee objective on DP optimum and random positions; penalty 0 on feasible.

### Route B (notebook 1.6) on compact QUBO — `run_partitioned.py`
- CommunityDecomposer cluster 12, QAOA n_layers 2, MC 50×10, W = 100 normalized, 62 sub-programs ≤ 12 qubits, 31k circuits, 79 s local.
- Stitched + repair: −3 484 (gap +95%). + referee local search: −54 923 (gap +21%). Raw partitioned output is weak: blocks are cut through the budget constraints, stitching loses feasibility.

### Quantum LNS (`run_qlns.py`) — main quantum method
- Window = 1 period × 2 assets (12 bits + 3 slack = 15 qubits) or, p = 0.3, 1 asset × 2 periods (18 qubits). Exact conditional sub-QUBO (others fixed to incumbent).
- Solved by QAOA on local Maestro (15–18 qubits < 25 → CPU), MC 30×5, 4000 shots, **warm-started** from Characterization Service angles (job `c3a51b2e-…`, 15-var window, 12 s, recommended 5 layers; we use 4).
- Every sampled sub-bitstring spliced into incumbent → referee score (repair if infeasible) → accept best improving. **No classical local search inside the loop.**
- Control (measurement only): exact window optimum by enumeration → logs "window-opt sampled" rate = how often QAOA's samples contained the window optimum.
- Characterization note: on an empty-incumbent window the service recommends λ ≈ 0.1 (budget can't bind there); kept W = 5000 because windows next to held positions do bind.

### QLNS results, po_a010_t10_orig λ = 1e-4 (optimum −69 482)
| run | sub-solver | start | windows | max qubits | result | gap |
|---|---|---|---|---:|---:|---:|
| s1 | QAOA 4 layers, warm start | partitioned-QAOA + LS (−54 923) | 1p×2a / 1a×2p | 18 | −67 739 @125 it (running) | +2.5% |
| s2 | QAOA 4 layers, warm start | empty | same | 18 | −50 250 @54 it (killed: ~10 s/window, 7 cores) | +27.7% |
| s3 | **PCE PolyEncoding**, 3 layers, SoftObjective α=2 | **empty** | 1p×4a (27 vars) / 2a×2p (30 vars) | **8** | **−69 397** (300 it, 9 min, 96k circuits) | **+0.12%** |
- PCE packs 27–30-variable windows into 7–8 qubits and runs ~1 s/window vs ~4–18 s for 15–18-qubit QAOA → many more windows per minute; bigger windows escape local optima better. Clear winner here.
- QAOA control metric: window optimum appeared among QAOA's top-64 samples in roughly half the 15/18-qubit windows (s1 log); PCE windows are > 20 vars so no enumeration control.
- s3 path is fully quantum-driven: start from empty, no classical local search; only referee repair on infeasible splices + accept-if-better.
- Gotcha: `pkill -f` with a pattern also matches the invoking shell → kill by PID.

### PCE-QLNS continuations (start = previous PCE-QLNS .sol, windows up to 6 assets × 1 period or 3 assets × 2 periods, ≤ 9 qubits)
- λ = 1e-4: −69 397 → **−69 482 = proven optimum** (s7).
- λ = 5e-4: −26 882 → **−27 044 = proven optimum** (s10).
- λ = 1e-3: −6 962 → −7 759 (+7.6%) (s8); one more 10-min continuation s15.
- s02 λ = 1e-4 from empty: −53 136 (+0.22%) (s9); continuation s16.

### Scaling PCE-QLNS to 50 assets (B = 20)
- `CompactQUBO` now adds the 4-bit cash register when B > 5 (capital binds): 3 090 vars for a050_t10. `DenseQUBO` extracts conditional sub-QUBOs by matrix slicing (copy+fix_variables on a 450k-coupling BQM was too slow). Validated: sub-QUBO energies equal the dimod path; objective = referee on the LNS solution.
- From empty at λ = 1e-2: stuck at −1000 (empty) with random, correlation-guided (s13) and objective-only multi-period "hedge" windows (s14). Reason (see `NOTES_lns.md`): high-λ optima are ~15–37 assets at ±1 unit with the budget saturated; any small window from empty sees only positive-cost moves.
- From the classical LNS best (−17 675, already beats BKV −12 340): 600 PCE windows found no further improvement (s12) → that solution is window-locally optimal.
- Lesson: at 50 assets the quantum windows are a polishing step, not a constructor. The fork's pool-DP + LNS (`lns.py`) is the constructor; its `solve_sub` hook is where a QAOA/PCE sub-solver plugs in.

## Final state (2026-11-21 ~22:55)
- λ = 1e-3 chain s6 → s8 → s15: **−8 397 = proven optimum** (0.91 M circuits, 31 min). s02 λ = 1e-4: −53 220 (+0.066%) after s9 → s16.
- → PCE-QLNS reached the proven optimum on all three a010_t10_orig λ tried (1e-4, 5e-4, 1e-3), each from an empty portfolio.
- The a050 s00 λ=1e-2 QLNS file (s12) was relabelled `quantum: False` in its header and in results.csv: it is the classical LNS solution, unchanged.
- `verify_solutions.py`: 127+ .sol files, all feasible, objective lines match → `best_solutions.md`.
- QOBLIB Rust checker (`06-portfolio/check`, `cargo build --release`, run `check_portfolio ../qoblib_data/06-portfolio/instances <file.sol>`): all 127+ pass, incl. the 4 a050 BKV beats (2026-11-21).
- PDF summary: `report/portfolio_qicing.pdf` (source `report/portfolio_qicing.tex.in` + `plotdata.json`; the .tex is generated).
- Gotcha: `pgrep -f`/`pkill -f` patterns also match the waiting shell → hung waiter loop; use PIDs.
