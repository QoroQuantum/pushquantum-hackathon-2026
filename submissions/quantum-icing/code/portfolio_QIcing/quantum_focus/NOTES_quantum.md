# quantum_focus — notes (2026-11-21/22)

Goal: a quantum result on a050 within +10% of the BKV that is not the empty portfolio. Also test exchange moves starting from a precomputed solution (the network 2-exchange idea).

## Results
| instance, λ | method | result | gap vs BKV |
|---|---|---:|---:|
| a050_t10_s00, 1e-4 | PCE-QLNS from empty (`run_qlns.py`, seed 22, 30 min, 0.9 M circuits, ≤9 qubits) | **−340 222** (20 units/day, 42 assets) | **+4.31%** ✅ |
| a050_t10_s00, 1e-3 | PCE-QLNS from empty (seed 21) | −157 229 | +18.1% |
| a050_t10_s00, 1e-2 | PCE-QLNS from the rounded-relaxation start (seed 24) | +37 234 | +402% |
| a050_t10_s00, 1e-2 | PCE-QLNS from our λ=1e-3 portfolio re-scored at 1e-2 (seed 23) | +69 194 | +661% |
| a050_t10_s00, 1e-2 | Q-MOVES PCE (24 / 60 moves), QAOA 18 moves on qoro_cpu | no change | – |
| a050_t10_orig, 1e-2 | Q-EXCHANGE XY-QAOA from the classical −34 520 (smoke test) | no change | – |
| a050_t10_s00, 1e-2 | λ-homotopy chains (`homotopy.sh`, seeds 1 and 2) | running | – |

## Diagnosis at λ = 1e-2
- The risk matrix has a few huge eigenvalues: 674 714, 256 935, 161 992, against a median of 1 539. The best day portfolios have zero exposure to the top 5 factors. In a normalised QUBO these factor terms swamp everything else.
- Hedge pairs on their own are never profitable (best pair costs +3 137 per day). Profit only comes from baskets of about 20 units.

## Files
- `qf_common.py`: loaders, per-day ±1 QUBO (validated exact), repair.
- `q_round.py`: Q-ROUND (relaxation → rounding QUBO → PCE / SA / naive / random → chain DP). The naive control gives −1000. Even the optimal rounding is positive (SA: +8 293 against a best day of −6 900). Dead end.
- `q_exchange.py`: XY-mixer QAOA that starts in the incumbent via `CustomPerQubitState`, with `QAOA.mixer_hamiltonian = xy_mixer(edges)`. The unit count is conserved in 100% of samples (verified). Windows above 16 qubits run on qoro_cpu.
- `q_moves.py`: move-selection QUBO. The energy is exact (0 mismatches in 880 checks) once conflicts are defined on (asset, day) and moves are drawn conflict-free. SA improves on it; PCE and QAOA samples are near-uniform.
- `homotopy.sh`: λ-continuation with PCE-QLNS. Intermediate λ results are saved to `starts/` and not recorded; only the final official λ is recorded.
- `starts/`: classical starting portfolios. These are NOT submissions.

## Gotchas
- WSL crashed (OOM) with 8 `q_round.py` workers. Keep it to ≤4 Python processes, each about 0.8 GB with the a050 dense QUBO.
- Cloud QAOA on qoro_cpu at 18 qubits takes 115–150 s per call. That is too slow for one call per LNS step; batch windows instead.
- The `q_moves.py` SA probe recorded one classical-control .sol (a +256 613 start-descent result). It is labelled classical and harmless.
