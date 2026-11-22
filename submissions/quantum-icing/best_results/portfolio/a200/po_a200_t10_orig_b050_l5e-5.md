# po_a200_t10_orig_b050_l5e-5

- **Instance:** `po_a200_t10_orig` (200 assets × 10 days), budget B = 50, λ = 5e-5
- **Official objective (referee re-score):** **-1112868**, feasible: True
- **Breakdown:** risk=109388, return=-1299210, transaction=77549, cash_interest=-1160, short_cost=565, liquidation=0
- **QOBLIB best known:** none published
- **Size:** 12110 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**engine2 (classical; 0 qubits).** For each trading day, a pool of candidate net-position vectors v = long − short ∈ {−3..3}ⁿ is built by iterated tabu search (ILS: random restarts + kicks; single-unit and pair-unit moves with exact integer deltas; candidate-list pair moves for n ≥ 150) on the single-day problem and on multi-day *hold* blocks (one vector held over 2–5 consecutive days). An exact dynamic program over days then picks one pool state per day with the true transaction costs. Refinement loop: every day is re-solved with its neighbour days fixed (transaction weight 1 and 0.5), new states go into the pools, and the chain DP is re-run; parallel fork workers. Last day kept empty (provably optimal).

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab missing: pool chain-DP + context ILS-tabu`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a200_t10_orig_b050_l5e-5.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a200_t10_orig 5e-5 --budget 50 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
