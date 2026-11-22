# hopqubo_newTesting: constraint-respecting partitioned QAOA / PCE for QOBLIB 08

This folder holds everything from the 21 Nov 2026 evening session. It continues
[ai_context/quantum_hopqubo_handoff.md](../ai_context/quantum_hopqubo_handoff.md) and fixes the two
problems the report had parked: partitioned QAOA breaking the degree rules (0–15% valid samples), and
PCE decoding to invalid topologies (0 of 100 valid).

## Ideas

1. **Bands: partitions that respect the constraints.** A band is a rectangle of the adjacency matrix.
   - A *row band* takes k tails A and, as heads B, the current topology's heads of A's out-links.
   - A *column band* does the same with the roles of tails and heads swapped.
   - All other arcs stay fixed. The current topology fixes each tail's and each head's link count
     inside the band, so a band assignment is globally valid exactly when it meets those counts.
   - The bands of one round use disjoint arcs. Therefore **any combination of locally valid band
     solutions is a valid topology**, and stitching never breaks a constraint.
   - With k=3 a band has at most 18 qubits. network24 splits into 8 bands per round.
   - Row and column bands alternate between rounds. Nodes on the routing LP's bottleneck arcs are
     spread over different bands.
2. **Exact sub-cost.** Each band is scored by the exact H2 hop polynomial (`network_mixer.h2_hubo`),
   with the rest of the topology fixed.
3. **Sub-solvers**, all inside one Divi `ProgramEnsemble` so they run in parallel:
   | name | mixer / encoding | constraints | warm start |
   |---|---|---|---|
   | `xch` | 2-exchange mixer inside the band, started from the current topology's band state | rows and columns exact by construction | angle transfer from the first band, or `--blind` |
   | `xy` | complete XY mixer per row block, started from the current topology's band state | rows exact, columns penalised | `--char`: service with `mixer="xy"` + row `eq_cardinality` |
   | `xpen` | X mixer | all penalised | `--char`: `characterize_and_validate` angles |
   | `pce` | PolyEncoding, soft values decoded by a band b-matching | exact after decoding | none |
   | `enum` / `random` | classical: every / random locally valid band state | exact | n/a |
4. **Stitching.** Beam search over band candidates by global H2, then the exact LP on the top few,
   then the 2-exchange polish at the end.
5. **Full-problem PCE with constraint-aware decoding** (`run_pce.py`).
   - Parity encodings decode samples into a GF(2) subspace of only 2^q assignments. That subspace
     almost never contains a valid topology, which is why Divi's default decode gave 0 of 100 valid.
   - Instead we take the trained soft values (1+tanh(α⟨P⟩))/2 and decode them with a max-weight
     2-in/2-out b-matching. The b-matching LP's constraint matrix is totally unimodular, so its
     optimum is integral. Gumbel noise on the scores gives diverse candidates. All decoded
     topologies are valid.
   - New **node-pair encoding**: one qubit per node, with arc i→j as X_iX_j for i<j and as Y_iY_j
     for i>j. That puts 132 variables on 12 qubits, or 552 on 24, and node v's degree rule sits on
     qubit v.

## Divi extensions (`divi_ext.py`, no edits to the installed package)

- `WarmQAOA` / `WarmPCE` carry their own `warm_params`. Divi's ensembles call `run()` without
  `initial_params`, so without this there would be no warm start inside an ensemble.
- `BlockEnsemble` is a `ProgramEnsemble` over caller-built programs.
- `NodePairEncoding`: the node-pair PCE encoding described above.
- `pce_soft_values` resets the cancellation event that a finished ensemble leaves set; without that,
  evaluations after the run fail with "cancelled".
- `characterize_cached` caches job ids in `runs/characterization_cache.json`, so reruns are free.

## Results

Gap = (score − best known) / best known. network12 best known 276 474, network24 best known 663 688.
All runs used the local Maestro simulator, from a random start (`--start random`, LP 411 250 on
network12), with 8 rounds and about 110–118 LP calls each.

### network12, band LNS, 3 seeds

| method | circuits/run | gap s0 | gap s1 | gap s2 | **mean** |
|---|---:|---:|---:|---:|---:|
| `xch` blind (MC from random angles on every band) | 3 872 | +2.36% | **+1.21%** | +6.70% | **+3.42%** |
| `pce` + band b-matching | 3 904 | +4.53% | +4.21% | +3.75% | +4.16% |
| `enum` (classical, exact per band) | 0 | +4.65% | +1.67% | +8.65% | +4.99% |
| `random` (classical) | 0 | +5.11% | +4.29% | +8.36% | +5.92% |
| `xch` with angle transfer (2 MC iterations after the transfer) | 1 392 | +5.12% | +7.64% | +6.46% | +6.41% |

- **Best: 279 819 (+1.21%), quantum `xch` blind, seed 1.**
  - Written to `runs/network12_bands_xch_s1.sol`; the referee says valid with objective 279 819.
  - It beats the earlier network12 best, SA + polish (+2.3%), and the earlier best quantum pipeline
    (+17.5%).
- Every sample from `xch` and every stitched topology was valid: 100% by construction.
- `xch` puts 15–37× more probability on each band's exact-H2 optimum than uniform valid sampling.
- The angle-transfer warm start with only 2 MC iterations was the worst quantum setting. Too few
  iterations after the transfer.
- Caveat: the final polish carries much of the score, and seeds vary by up to 7 points. The means
  are more reliable than any single run.

### network12, other mixers (seed 0, 2 rounds)

| method | P(valid) | gap |
|---|---:|---:|
| `xy` blind | 0.93–1.00 | +2.15% (before polish 333 800) |
| `xy` + service warm start (`mixer="xy"`) | 0.85–0.99 | +6.01% |
| `xpen` blind | ≈0 | n/a |
| `xpen` + service warm start | 0.07–0.08 | +5.47% |

The service angles for `xy` did not concentrate on band optima. The likely reason: the service
assumes its own XY setup and initial state, while ours starts from a basis state.

### network12, full-problem PCE (`run_pce.py`, 70 LP calls, 3 seeds)

| encoding | qubits | mean gap | best |
|---|---:|---:|---:|
| dense | 8 | **+3.58%** | +1.77% |
| poly | 16 | +5.50% | +3.85% |
| node-pair | 12 | +5.98% | +5.33% |
| random start + polish (same LP budget) | – | +5.95% | +5.30% |

### network24 (552 vars, 8 bands ≤18 qubits, 6 rounds, seed 0)

| method | gap |
|---|---:|
| `enum` | +6.95% |
| `xch` (angle transfer) | +8.22% |
| `pce` | +8.37% |

The pipeline scales to network24. The classical team result (−0.99%) is still far ahead.

## Commands

```bash
.venv/bin/python hopqubo_newTesting/run_bands.py --n 12 --methods enum random xch pce --rounds 8 --seed 0
.venv/bin/python hopqubo_newTesting/run_bands.py --n 12 --methods xch --blind --rounds 8 --seed 1   # best run
.venv/bin/python hopqubo_newTesting/run_bands.py --n 12 --methods xy xpen --char --rounds 2         # service warm start (credits)
.venv/bin/python hopqubo_newTesting/run_pce.py   --n 12 --encodings nodepair poly dense --pop 30 --iters 15
# --cloud qoro_cpu for jobs too slow locally. A reachability test on 21 Nov returned in 24 s for 0.02 credits.
```

Logs go to `runs/` (txt + json per run; gitignored by the repo-wide `runs/` rule).

## Open work

1. **`xch` on `qoro_cpu`** with a full MC budget on every band. Blind `xch` with the full budget was
   the best method, and it is slow locally: 15 min for network24.
2. **More seeds.** Seed variance is about 7 points, so 3 seeds can't separate methods reliably.
3. **Start from good topologies** (`--start path.sol`), for example the 279 819 above, and check
   whether band QAOA improves on it beyond the polish.
4. **`xy` from a per-row Dicke superposition**, which matches what the service assumes, so its
   warm-start angles should transfer.
5. **Bigger bands (k=4)** via PCE: 24–32 variables on 6–8 qubits. QAOA can't fit them.
