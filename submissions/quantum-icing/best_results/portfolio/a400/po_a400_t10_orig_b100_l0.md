# po_a400_t10_orig_b100_l0

- **Instance:** `po_a400_t10_orig` (400 assets × 10 days), budget B = 100, λ = 0
- **Official objective (referee re-score):** **-2695464**, feasible: True
- **Breakdown:** risk=0, return=-2850040, transaction=154244, cash_interest=-720, short_cost=1052, liquidation=0
- **Status:** **proven optimal** (MILP solved to zero gap)
- **Size:** 24110 binary variables in the official QUBO; 0 qubits (classical)

## Method

**Exact MILP, proven optimal.** With λ = 0 the risk term vanishes, so the official model is a linear integer
program: integer long/short units L, S ∈ {0..3} per asset and day, auxiliary |ΔL| and |ΔS| variables for the
transaction costs, and the per-day budget (Σ(L+S) ≤ B) and capital rules. Solved to optimality with HiGHS via
`scipy.optimize.milp` in about 1 s.

## Reproduce

```bash
cd opus_lab && ~/.venvs/opus_pf/bin/python milp_l0.py po_a400_t10_orig 100
```
