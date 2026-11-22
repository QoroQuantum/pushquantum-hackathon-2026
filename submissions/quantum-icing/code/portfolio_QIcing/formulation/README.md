# formulation/: QUBO/HUBO reformulations of QOBLIB 06 built for PCE

Session of 2026-11-22. All energies are checked against the referee's coefficient tables. Every recorded `.sol` passes QOBLIB's Rust checker. The `quantum:` label in a `.sol` header and in `results.csv` is `True` only when PCE produced the candidates; the random and classical controls are labelled `False`.

## Files
| file | what |
|---|---|
| `formulations.py` | Per-period reduced problem and `chain_dp` (exact multi-period optimum over per-period pools). 4 cell encodings: unary, sign-magnitude HUBO, ternary, ternary-SM. 3 constraint modes: slack, unbalanced, none. `validate()`: 0 mismatches vs the referee on a010 and a050 |
| `codes.py` | Divi parity encodings viewed as linear codes. `MaskEncoding` (custom `ParityEncoding`), re-centring (complement trick), codeword enumeration |
| `code_diag.py` | Enumerates every codeword (2^q) to show what a PCE sampler can reach at all |
| `slots.py` | **Slot formulation** (constraint-free, log-encoded, Gray code over correlation-ordered assets, optional fixed base portfolio) and the closed-form mean-field PCE loss `SlotObjective` |
| `pce_tools.py` | `HarvestPCE` (keeps every expectation vector it evaluates), runners, ideal-relaxation control |
| `exp_period_pce.py`, `exp_slots.py` | Period-decoupled pools → chain DP, optionally with L1 (period descent) and L2 (multi-period LNS) applied identically to every candidate source |
| `slot_qlns.py` | **slot-QLNS**: an LNS move removes ≤k units of one period and re-chooses k slots jointly; sub-solver is `pce`, `rand` or `ideal`; optional period descent; exact multi-period acceptance |
| `lns_control.py`, `campaign.sh`, `sweep_*.sh`, `run_a050.sh` | Controls and batch drivers; logs are in `logs/` |

## 1. Formulation findings
1. **The periods decouple, and the decoupling is exact through a DP.** In every a010 case tested (orig and s02 at λ 1e-4, 5e-4 and 1e-3), the optimal state of each period ranks among the 4 best states of that period's *single-period* problem. A pool of the top 5 states per period plus an exact chain DP returns the proven optimum. The quantum task is therefore T single-period problems: 20–33 variables instead of 630 for a010, 140–159 instead of 3 090 for a050.
2. **Sign-magnitude cells** use 3 bits per asset instead of 6, and the budget stays linear. The HUBO terms cost no qubits, because the PCE loss is a classical polynomial of the expectations. Per period, a050 goes from 309 to 159 variables (25 → 18 Poly qubits).
3. **Divi's Poly code is a poor fit for sparse portfolios.** The decoded value is x = 1 ⊕ Mz, so a pair variable equals [z_a = z_b]. Every shot therefore has about N/2 ones, and **0 %** of the sign-magnitude codewords are feasible.
4. **In the earlier pipelines, the repair step was doing the optimisation.** Greedy repair applied to 32 random codewords of a 5-qubit Dense code already reaches the per-period optimum.
5. **Slack and penalty formulations give a bad relaxed landscape.** Even the ideal relaxation (z unconstrained) rounds to states ranked between 100 and 2 000.
6. **Slot formulation.** A period is B slots. Each slot is a choice of (asset, sign) or nothing, encoded in ⌈log₂(n+1)⌉+1 bits. The budget holds for every bitstring and every fractional point, so there is no slack and no penalty. The energy is exactly pairwise between slots. The rank bits are a Gray code over assets ordered by the leading correlation factor, so correlated assets are one bit flip apart. a010 has 20 variables per period (5 qubits with MultiBasis-2); a050 has 140 (11 qubits).

## 2. PCE without any local search: slot pools + exact chain DP
Setup: MultiBasis-2, L-BFGS, α=6, 6 layers, 12 restarts, no repair, no local search.

| instance, λ | reference | slot-PCE | random, same bitstring count |
|---|---:|---:|---:|
| a010_t10_orig, 1e-4 | −69 482 (opt) | **+1.49 %** (45 k circuits, 5 qubits) | +19.97 % |
| a010_t10_s02, 1e-4 | −53 255 (opt) | **+4.45 %** | +13.67 % |
| a010_t10_orig, 5e-4 | −27 044 (opt) | +10.49 % | +15.74 % |
| a010_t10_orig, 1e-3 | −8 397 (opt) | +25.64 % ✗ | +20.08 % |
| a050_t10_s00, 1e-4 | −355 530 (BKV) | +39.6 % (11 qubits) | +71.2 % |
| a050_t10_s00, 1e-3 | −191 879 (BKV) | +88.5 % | +99.5 % |

PCE clearly beats random sampling when no local search follows, except at high λ, where the optima are sparse. The earlier PCE-QLNS needed 0.2–0.9 M circuits *with* repair to reach these levels.

## 3. With local search, applied identically to every source
L1 runs period descent (single and pair unit moves) on every pool state, then the chain DP. L2 adds 120 s of `lns`.

| a050_t10_s00 | PCE pool | random pool | ideal | empty |
|---|---:|---:|---:|---:|
| λ 1e-4, L2 | −355 530 (= BKV) | −355 530 | −355 530 | −355 396 |
| λ 1e-3, L2 | −179 281 | **−186 426** | −177 897 | −165 265 |
| λ 1e-2, L2 | −2 441 | **−3 537** | −1 000 | −1 000 |

Once local search follows, the diversity of the starts matters more than their quality. Random produces about 1 700 distinct starts against about 300 for PCE, and PCE's raw advantage disappears.

## 4. slot-QLNS: the slot neighbourhood as an LNS move. New BKVs (classical credit)
| instance, λ | start (our previous best) | QOBLIB BKV | slot-QLNS | gap vs BKV | sub-solver |
|---|---:|---:|---:|---:|---|
| a050_t10_orig, 1e-2 | −35 445 | −37 107 | **−47 800** | **−28.8 %** | rand + descent |
| a050_t10_s00, 1e-3 | −190 685 | −191 879 | **−191 991** | **−0.058 %** | rand + descent |
| a050_t10_orig, 1e-3 | −205 278 | −206 239 | −206 153 | +0.042 % | rand + descent |
| a050_t10_orig, 5e-4 | −254 640 | −254 972 | −254 958 | +0.005 % | rand + descent |

All four pass the Rust checker. Same start (s00, 1e-3), same wall time:

| sub-solver | result |
|---|---:|
| `lns` alone | −190 685 (stuck, 285 k iterations) |
| PCE MultiBasis | −190 685 (no change) |
| PCE MultiBasis + descent | −191 446 |
| random + descent | −191 991 |

PCE produces only 3–7 distinct candidates per move, against 40–97 for random. What works is the *formulation*: jointly re-choosing k ≤ 6 units of one period, a neighbourhood of size ~(2n+1)^k that the pair moves of `lns` cannot reach. It is not the quantum sampler.

## 5. Campaign (stopped early to free the machine; 8 of 22 cases run)
Each case got 1 500 s of slot-QLNS (random + descent), starting from the best `.sol` held for that case.

| instance, λ | QOBLIB BKV | before | after | gap vs BKV |
|---|---:|---:|---:|---:|
| a050_t10_s01, 1e-2 | −2 248 | −1 000 (empty) | **−4 267** | **−89.8 %** |
| a050_t10_orig, 1e-2 | −37 107 | −47 800 | **−47 967** | **−29.3 %** |
| a050_t10_s01, 1e-3 | −141 376 | −139 048 | **−141 431** | **−0.039 %** |
| a050_t10_orig 1e-3 / 1e-4 / 5e-4 / 5e-5, a050_t10_s00 1e-3 | — | — | no change (≈1 200 moves each) | +0.042 / +0.007 / +0.005 / +0.018 / −0.058 % |

**PCE head-to-head.** Same start (−35 445, a050_t10_orig λ 1e-2) and the same 1 500 s. PCE-Poly on 9 qubits + descent reached **−44 514 (−20.0 % vs BKV)**, with 12 of 408 moves accepted. Random + descent reached −47 800. So it is a BKV beat with PCE producing the candidates (labelled quantum, hybrid with local search), but random at equal wall time does better.

Every new `.sol` passes the QOBLIB Rust checker, and `best_solutions.md` was rebuilt (160 files, 0 invalid).

**Not run yet** (all still above BKV, start = best held `.sol`): a050_t10_s01 1e-4 / 5e-4, a050_t10_s02 1e-3 / 5e-4, a050_t15_orig 1e-3 / 5e-4, a050_t15_s00 1e-2 / 1e-3 / 5e-4, a050_t15_s01 1e-2 / 1e-3 / 5e-4, a050_t15_s02 1e-3 / 5e-4. To resume, use `campaign.sh` with `logs/campaign_targets.txt` trimmed to these lines. Lower `-P 8` to share the machine.

**Known biases and open questions for a fresh look.**
- Random has the advantage at equal *wall time* because PCE costs about 2 s per move. Equal *move counts* have not been compared.
- The PCE settings (α=6, L-BFGS, MultiBasis-2) were tuned on a010 λ=1e-4 only.
- The slot model's L/S cost penalises long+short on the same asset, which net decoding then removes. This is exact, but it biases the relaxation.
- No study has been made of k (free slots per move), and k=6 was never varied.

## Takeaways
- **Constraint-free formulations matter most for PCE.** The slot encoding took PCE without local search from +92–95 % to +1.5 % on a010 λ=1e-4, on 5 qubits per period.
- **Equal-budget random controls are essential.** With local search in the loop, PCE has so far never beaten random sampling.
- The slot neighbourhood is the most useful single output: 2 new a050 BKVs and two cases within 0.05 % of the BKV.
