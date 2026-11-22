# sparse/ — block-sparse QUBO for a200 and a400 (2026-11-22)

## Why this exists
The dense compact QUBO (`pf_quantum.DenseQUBO`) needs n_vars² floats, which comes to 2.3 GB for a200 (12 100 vars) and 9 GB for a400 (24 110 vars). The machine has about 9 GB, and WSL already crashed once on memory.

## Structure used (`block_qubo.py`)
Every coupling in the compact QUBO is one of two kinds:
- **within one day**: risk between unit bits, plus the budget and capital penalties. This is a dense block of size (6n + slack)² per day;
- **the same unit bit (asset, copy, direction) on adjacent days**: the rebalancing XOR, one weight per bit.

So the QUBO is stored as T dense float32 blocks plus a vector of adjacent-day weights.

| instance | vars | block memory | dense equivalent | build time | window sub-QUBO |
|---|---:|---:|---:|---:|---:|
| a050_t10 | 3 090 | 7 MB (float64) | 76 MB | <0.1 s | — |
| a200_t10 | 12 100 | **56 MB** | 2.3 GB | 0.4 s | 5 ms |
| a400_t10 | 24 110 | **222 MB** | 9 GB | 2 s | 1.4 ms |

The whole test script peaked at 0.97 GB RSS, including loading the a400 instance (18 s).

## Validation (a050, against the dense path)
- Full energy (objective + penalty weight × penalty) equals dimod's energy on random bitstrings, exactly.
- Window sub-QUBO energy differences equal the `DenseQUBO.sub` differences: 0 error over 100 checks.
- With penalty weight 0 on feasible portfolios, the energy equals the official referee objective (0 of 20 mismatches).

## Runner (`run_qlns_sparse.py`)
Same PCE quantum LNS as `../run_qlns.py`:
- windows of 1 day × 5 assets, or 2 assets × 2 days;
- PCE PolyEncoding on at most 9 qubits, with 3 layers and a Monte Carlo optimiser of 40 × 8, on local Maestro;
- every sample is spliced in and scored by the referee; infeasible splices get a cheap per-day repair;
- a window is accepted only if the score improves.

About 0.3–0.5 s per window at a200 and a400. The window size does not grow with the instance, so no cloud is needed for the circuits.

Smoke test: a200 λ=1e-4, from empty, 40 s → −259 885 (126 assets held). Recorded, because QOBLIB publishes nothing for a200.

## Runs (60 min each, from an empty portfolio)
See `logs/a200_l1e-4_s1.txt`, `logs/a400_l1e-4_s2.txt` and `logs/a200_l1e-3_s3.txt`. Results go in the table below when the runs finish.

| instance, λ (60 min, from empty) | result | assets held | units/day | windows | circuits | max qubits |
|---|---:|---:|---|---:|---:|---:|
| a200_t10_orig, 1e-4 | **−794 168** (first value, no BKV) | 154 | 50 of 50 | 9 424 | 3.0 M | 9 |
| a400_t10_orig, 1e-4 | **−1 397 396** (first value, no BKV) | 313 | 100 of 100 | 6 866 | 2.2 M | 10 |
| a200_t10_orig, 1e-3 | −469 837 (our classical result: −818 362, so 43% worse) | 160 | 49–50 of 50 | 9 479 | 3.0 M | 9 |

All three were still improving at the 60-minute cut. They are recorded in `../solutions/` and verified: 148 .sol files, all feasible.
