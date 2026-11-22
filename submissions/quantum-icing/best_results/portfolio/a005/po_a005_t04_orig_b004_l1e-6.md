# po_a005_t04_orig_b004_l1e-6

- **Instance:** `po_a005_t04_orig` (5 assets × 4 days), budget B = 4, λ = 1e-6
- **Official objective (referee re-score):** **-16347**, feasible: True
- **Breakdown:** risk=352, return=-17708, transaction=1411, cash_interest=-420, short_cost=18, liquidation=0
- **QOBLIB best known:** none published
- **Size:** 164 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**Exact dynamic programming (proven optimal; classical, 0 qubits).** Every feasible day state v ∈ {−3..3}ⁿ with Σ|v| ≤ B and the capital rule is enumerated; a DP over days with the exact transaction costs returns the global optimum (validated against the referee's brute force).

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab exact DP over all 671 feasible day states (proven optimal)`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a005_t04_orig_b004_l1e-6.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a005_t04_orig 1e-6 --budget 4 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
