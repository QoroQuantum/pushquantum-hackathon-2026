# po_a003_t02_orig_b003_l5e-5

- **Instance:** `po_a003_t02_orig` (3 assets × 2 days), budget B = 3, λ = 5e-5
- **Official objective (referee re-score):** **-2277**, feasible: True
- **Breakdown:** risk=286, return=-2676, transaction=300, cash_interest=-190, short_cost=3, liquidation=0
- **QOBLIB best known:** none published
- **Size:** 58 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**Exact dynamic programming (proven optimal; classical, 0 qubits).** Every feasible day state v ∈ {−3..3}ⁿ with Σ|v| ≤ B and the capital rule is enumerated; a DP over days with the exact transaction costs returns the global optimum (validated against the referee's brute force).

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab exact DP over all 63 feasible day states (proven optimal)`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a003_t02_orig_b003_l5e-5.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a003_t02_orig 5e-5 --budget 3 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
