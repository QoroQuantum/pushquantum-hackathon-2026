# po_a050_t10_orig_b020_l1e-3

- **Instance:** `po_a050_t10_orig` (50 assets × 10 days), budget B = 20, λ = 1e-3
- **Official objective (referee re-score):** **-206239**, feasible: True
- **Breakdown:** risk=63952, return=-295914, transaction=26563, cash_interest=-1060, short_cost=220, liquidation=0
- **QOBLIB best known:** -206239 (best known); gap +0.000%
- **Size:** 3110 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**Previous team pipeline** (old repository, `portfolio_QIcing/` and `portfolio_search.py`): exact DP over periods (a010), pool-DP + exact-DP LNS, day-by-day tabu search, ILS, parallel tempering, and PCE-based quantum LNS for some a010 files. See the comment lines of the .sol and the old repository's notes for the exact run.

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `converted from a050_t10_orig_b020_l0.001.sol (bqp-lp)`

Source file: `/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/qoblib_data/06-portfolio/solutions/po_a050_t10_orig/po_a050_t10_orig_b020_l1e-3.bst.sol`

## Reproduce

```bash
cd opus_lab
# produced by the previous pipeline, see the source path above
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
