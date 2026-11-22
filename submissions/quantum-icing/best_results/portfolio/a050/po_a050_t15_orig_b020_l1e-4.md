# po_a050_t15_orig_b020_l1e-4

- **Instance:** `po_a050_t15_orig` (50 assets × 15 days), budget B = 20, λ = 1e-4
- **Official objective (referee re-score):** **-572490**, feasible: True
- **Breakdown:** risk=105223, return=-723425, transaction=46904, cash_interest=-1520, short_cost=328, liquidation=0
- **QOBLIB best known:** -572490 (best known); gap +0.000%
- **Size:** 4665 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**engine v1 (classical; 0 qubits).** Per-day candidate pools from iterated tabu search (single and pair unit moves, exact deltas) plus multi-day hold blocks, exact chain dynamic program over days with true transaction costs, then a refinement loop that re-solves each day with its neighbours fixed and re-runs the chain DP.

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab engine: pool chain-DP`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a050_t15_orig_b020_l1e-4.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a050_t15_orig 1e-4 --budget 20 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
