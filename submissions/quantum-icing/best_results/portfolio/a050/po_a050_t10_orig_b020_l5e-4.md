# po_a050_t10_orig_b020_l5e-4

- **Instance:** `po_a050_t10_orig` (50 assets × 10 days), budget B = 20, λ = 5e-4
- **Official objective (referee re-score):** **-254972**, feasible: True
- **Breakdown:** risk=73640, return=-355927, transaction=28215, cash_interest=-1130, short_cost=230, liquidation=0
- **QOBLIB best known:** -254972 (best known); gap +0.000%
- **Size:** 3110 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**engine v1 (classical; 0 qubits).** Per-day candidate pools from iterated tabu search (single and pair unit moves, exact deltas) plus multi-day hold blocks, exact chain dynamic program over days with true transaction costs, then a refinement loop that re-solves each day with its neighbours fixed and re-runs the chain DP.

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab engine: pool chain-DP + context ILS-tabu`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a050_t10_orig_b020_l5e-4.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a050_t10_orig 5e-4 --budget 20 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
