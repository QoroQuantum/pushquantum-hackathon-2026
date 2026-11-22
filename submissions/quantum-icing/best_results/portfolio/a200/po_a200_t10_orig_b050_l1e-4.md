# po_a200_t10_orig_b050_l1e-4

- **Instance:** `po_a200_t10_orig` (200 assets × 10 days), budget B = 50, λ = 1e-4
- **Official objective (referee re-score):** **-794168**, feasible: True
- **Breakdown:** risk=96317, return=-963783, transaction=73971, cash_interest=-1230, short_cost=557, liquidation=0
- **QOBLIB best known:** none published
- **Size:** 12110 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**Previous team pipeline** (old repository, `portfolio_QIcing/` and `portfolio_search.py`): exact DP over periods (a010), pool-DP + exact-DP LNS, day-by-day tabu search, ILS, parallel tempering, and PCE-based quantum LNS for some a010 files. See the comment lines of the .sol and the old repository's notes for the exact run.

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `QIcing portfolio attack | method: QLNS-pce sparse (quantum sub-solver, Maestro)`; `settings: block-sparse QUBO (12100 vars, 56 MB), PCE PolyEncoding n_layers=3 mc=40x8 shots=4000 window_assets=5 time_window_assets=2 p_time=0.4 p_corr=0.5 penalty_weight=5000.0 windows=9424 seed=1 start=empty circuits=3025104`; `quantum: True | qubo vars: 12100 | max qubits: 9`

Source file: `/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/portfolio_QIcing/solutions/po_a200_t10_orig_b050_l1e-4/po_a200_t10_orig_b050_l1e-4__qlns-pce-sparse__20261122-015251.sol`

## Reproduce

```bash
cd opus_lab
# produced by the previous pipeline, see the source path above
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
