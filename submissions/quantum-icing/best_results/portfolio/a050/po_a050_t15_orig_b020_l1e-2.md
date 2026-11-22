# po_a050_t15_orig_b020_l1e-2

- **Instance:** `po_a050_t15_orig` (50 assets × 15 days), budget B = 20, λ = 1e-2
- **Official objective (referee re-score):** **-84973**, feasible: True
- **Breakdown:** risk=129650, return=-257061, transaction=43666, cash_interest=-1540, short_cost=312, liquidation=0
- **QOBLIB best known:** -31504 (best known); gap -169.721%
- **Leaderboard value to beat:** -77193; gap -10.079%
- **Size:** 4665 binary variables in the official QUBO formulation; largest quantum program used for this file: **0 qubits** (classical)

## Method

**engine2 (classical; 0 qubits).** For each trading day, a pool of candidate net-position vectors v = long − short ∈ {−3..3}ⁿ is built by iterated tabu search (ILS: random restarts + kicks; single-unit and pair-unit moves with exact integer deltas; candidate-list pair moves for n ≥ 150) on the single-day problem and on multi-day *hold* blocks (one vector held over 2–5 consecutive days). An exact dynamic program over days then picks one pool state per day with the true transaction costs. Refinement loop: every day is re-solved with its neighbour days fixed (transaction weight 1 and 0.5), new states go into the pools, and the chain DP is re-run; parallel fork workers. Last day kept empty (provably optimal).

Reduced model used by the solver (equivalent to QOBLIB `bqp_u3_c10.zpl`): v = L − S per day and asset (long and short in one asset is dominated), interest = −10·(C − net) is linear, the capital rule is −5 ≤ Σv ≤ 10 and the budget rule Σ|v| ≤ B. The objective is checked to equal the official referee on every file.

Comment lines in the .sol: `opus_lab campaign: pool chain-DP + context ILS-tabu`

Source file: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab/solutions/po_a050_t15_orig_b020_l1e-2.sol`

## Reproduce

```bash
cd opus_lab
~/.venvs/opus_pf/bin/python engine2.py po_a050_t15_orig 1e-2 --budget 20 --time 1800 --workers 6
~/.venvs/opus_pf/bin/python status.py   # re-score with the official referee
```

_Generated 2026-11-22 09:04 by make_submission.py._
