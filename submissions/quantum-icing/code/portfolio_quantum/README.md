# Portfolio optimisation (QOBLIB 06) with a quantum large-neighbourhood search

This folder holds a quantum pipeline for notebook 01. Every improvement it accepts is a bitstring
sampled by a **Divi QAOA program** on a sub-QUBO of the official model. The classical side builds the
windows, scores each sample with the exact objective and keeps the improvements. It never runs a
classical optimiser on the QUBO: no local search, no repair, no simulated annealing.

## Update 2026-11-22 06:20: paired-transfer QAOA

A second quantum sub-solver, `paired_trajectory_qaoa.py`, is run on any configuration by
`paired_transfer_run.py`. Here one qubit is one transfer of up to 3 units between two assets on the
same side of one day, so every bitstring keeps the budget and the net position. The QAOA has p = 3,
≤ 16 qubits and 448 shots. It is simulated with the module's own statevector (numba), **not Divi**.
Each run started from the best `.sol` already in the repo and ran 35 minutes. The table lists the best
file per configuration, all re-scored by the referee:

| Instance | B | λ | Binary variables | Largest QAOA (qubits) | Start | Quantum LNS | QOBLIB best known | Gap | Earlier classical (repo) | vs classical |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `po_a050_t10_orig` | 20 | 0.001 | 3,110 | 16 | best | -205,699 | -206,239 | +0.26% | -205,672 | -27 |
| `po_a050_t10_s00` | 20 | 0.001 | 3,110 | 18 | best | -191,256 | -191,879 | +0.32% | -190,685 | -571 |
| `po_a050_t10_s01` | 20 | 0.001 | 3,110 | 18 | best | -140,407 | -141,376 | +0.69% | -139,048 | -1,359 |
| `po_a050_t15_orig` | 20 | 0.001 | 4,665 | 16 | best | -321,183 | -321,907 | +0.22% | -321,070 | -113 |
| `po_a050_t15_orig` | 20 | 0.0005 | 4,665 | 16 | best | -397,384 | -397,497 | +0.03% | -397,236 | -148 |
| `po_a050_t15_s00` | 20 | 0.001 | 4,665 | 18 | best | -312,668 | -314,736 | +0.66% | -312,312 | -356 |
| `po_a050_t15_s01` | 20 | 0.0005 | 4,665 | 16 | best | -304,333 | -304,289 | -0.01% | -303,567 | -766 |
| `po_a400_t15_orig` | 100 | 0.001 | 36,165 | 16 | best | -3,164,781 | — | — | -2,986,716 | -178,065 |

`po_a050_t15_s01` at λ = 5e-4 reaches **−304,333**, below QOBLIB's best-known value (−304,289), a gap
of −0.01 %. `po_a400_t15_orig` starts from the −3,163,545 file in `deliverables/portfolio_quantum`
(`*__paired-trajectory-qaoa.sol`). That file was built by the same module from a 15-day relaxation
guide. `po_a050_t10_orig` at λ = 5e-4 did not change. Logs: `ai_context/portfolio_quantum/logs/paired/`.

## Results (22 Nov 2026, every file re-scored by the official referee)

The pipeline improved every configuration it was run on long enough: four `po_a050` configurations, and
the largest instance of the challenge, `po_a400_t15_orig` with 36 165 binary variables.
"Earlier classical" is the best earlier classical solution in this repo, from `deliverables/portfolio`
and `portfolio_QIcing`, frozen in `classical_baseline.json`. "Start: best" means the run was warm-started
from that classical solution, and every gain on top of it comes from QAOA samples.

| Instance | B | λ | Binary variables | Largest QAOA (qubits) | Start | Quantum LNS | QOBLIB best known | Gap | Earlier classical (repo) | vs classical |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `po_a050_t10_s00` | 20 | 0.001 | 3,110 | 18 | best | -191,256 | -191,879 | +0.32% | -190,685 | -571 |
| `po_a050_t10_s01` | 20 | 0.001 | 3,110 | 18 | best | -140,407 | -141,376 | +0.69% | -139,048 | -1,359 |
| `po_a050_t15_s00` | 20 | 0.001 | 4,665 | 18 | best | -312,668 | -314,736 | +0.66% | -312,312 | -356 |
| `po_a050_t15_s01` | 20 | 0.0005 | 4,665 | 18 | qbest | -304,271 | -304,289 | +0.01% | -303,567 | -704 |
| `po_a400_t15_orig` | 100 | 0.001 | 36,165 | 26 | best | -3,015,526 | — | — | -2,986,716 | -28,810 |

All five files are in [`deliverables/portfolio_quantum/`](../deliverables/portfolio_quantum/). None goes
below the QOBLIB best-known value yet. `po_a050_t15_s01` at λ = 5e-4 is 18 units, or 0.01 %, above it.

**Independent quantum result, from the empty portfolio.** On `po_a050_t10_s01` at λ = 1e-3, in 12 minutes on
the local Maestro simulator, the pipeline reaches **−139 634**. That is better than the earlier classical
best, −139 048 (ILS + parallel tempering + DP). The run used 1 229 windows and 3 084 circuits, 626 of
them for angle training, with QAOA programs of at most 18 qubits. With the QAOA replaced by the exact
window optimum (`--solver oracle`) and the same settings, the same run reaches −140 770. That is the
ceiling of these windows.

**How good the QAOA is.** On windows also solved exactly, in a random 5–10 % of the windows, the best
QAOA sample was the exact window optimum in **120 of 244 cases (49 %)**. The circuits use p = 5 and
transferred angles, 2 circuits per window and 4 000 shots, in subspaces of up to C(16, 8) = 12 870 states.

**Scale (cloud).** On `po_a400_t15_orig` (B = 100, λ = 1e-3), `cloud_batch.py` ran 8 rounds of 7
QAOA programs in parallel on Qoro's cloud. Each program had up to **26 qubits**. The run used 112
circuits and 84.7 credits, about 4 minutes per round. It improved the classical solution by 28 810, or
0.96 %. Windows on days of the same parity are independent, and the code asserts that their energy
changes add up exactly.

**What did not work, or was not done.**
- `po_a050_t10_orig` and `po_a050_t15_orig` at λ = 5e-4: no improvement in 1 028 and 209 windows.
  Those classical solutions are already within 0.05 % and 0.07 % of the best-known values.
- At λ = 1e-2, rounding the relaxation breaks the hedges and makes the risk explode. Even the exact
  window optimum finds nothing from an empty portfolio. This regime is still classical (see
  `deliverables/portfolio`).
- The uniform-sampling ablation (`--solver random`) and the remaining λ = 1e-3 and λ = 5e-4
  configurations were queued (`jobs/`) but not run, for lack of time.
- The warm-started variant (`--warm`, a seed state plus XY rings that conserve the + and − groups) is
  implemented. Its trained angles barely mix, with β ≈ 0.03, so the runs above use the Dicke family.


## How it works

**Model.** We work with one integer per asset and day, v = long − short ∈ {−3..3}. Holding long and short
units of the same asset never helps. The objective then equals the referee's
`qoblib_tools.PortfolioModel.score`. The cost function is `portfolio_search.full_obj`, which its `selftest`
checks against the referee. Every saved `.sol` is re-scored by the referee.

**Sub-QUBO (move selection).** A window is a list of unit moves (day t, asset i, ±1), one qubit each.
The energy of the QUBO is *exactly* the change of the official objective (`qlns_core.build_qubo`,
tested on 6 000 random assignments, 0 mismatches). The terms are:

- risk couplings within a day;
- returns, short and buy-in or liquidation costs, and cash interest as linear terms;
- rebalancing between consecutive days, which is exactly quadratic for one move per slot. Two opposite
  moves on one slot cost exactly zero when both are on.

**Windows.**
1. *Relaxation-guided rounding* (most of the gains). The continuous relaxation of one day is convex. It is
   solved with exact coordinate descent and a budget multiplier, with the neighbouring days fixed
   (`dayrelax.py`). It gives fractional positions v\*. Each fractional slot (up to 16 locally, 26 on the
   cloud) gets one qubit, which chooses between trunc(v\*) and one more unit away from zero. The
   relaxation saturates the day budget, so a window must add exactly `room` units. The QUBO therefore
   lives in a fixed-Hamming-weight subspace. Random noise on the returns and randomly excluded assets
   diversify the windows.
2. *Budget-preserving swaps.* These are unit moves on 7 assets of one day, in both directions. Moves that
   free a unit are flipped, y = 1 − x, an exact affine change of variables (`qaoa_solver.flip_qubo`). A
   saturated budget then becomes a Hamming-weight constraint as well.

**Quantum solver (`qaoa_solver.py`, Divi 0.16.2).**
- *Constraint-preserving QAOA.* The circuit starts from a Dicke state |D^M_k⟩ (`DickeState`) and uses a
  ring XY mixer (`divi.hamiltonians.xy_mixer`). Every sample therefore meets the day budget exactly,
  with no penalty term and no slack qubit.
- *Capital rule on a400.* There the rule C − 15 ≤ net ≤ C binds. The circuit then starts from the
  relaxation's preferred rounding (`CustomPerQubitState`) and uses one XY ring per group of +1 and −1
  moves, which conserves both counts.
- *Angles.* We use p = 5 layers. The angles are trained once with COBYLA on a real window, after the
  Hamiltonian is normalised by the standard deviation of the window energy. They are re-trained every
  300 windows and otherwise **transferred**, so a window costs 2 circuits: one loss evaluation and the
  4 000-shot sampling.
- *Transfer vs linear ramp.* On 14-qubit windows the trained angles, transferred unchanged, put about
  8× the uniform probability on the exact window optimum and found it in 4 of 5 unseen windows. A
  linear-ramp schedule found it in 0 of 5.
- *Backends.* Windows run on the local Maestro simulator, or in parallel on Qoro's cloud
  (`cloud_batch.py`). Windows on days of the same parity are independent, so their energy changes add
  up exactly (asserted in the code).

**Classical parts, stated plainly.** These are the day relaxation that chooses the window, the exact
scoring of each sampled bitstring, and accepting a window when it lowers the official objective.
`--solver oracle` replaces QAOA by exact enumeration of the same window, the classical ceiling.
`--solver random` replaces it by uniform samples from the same subspace with the same shots, the
ablation.

## Reproduce

```bash
source .venv/bin/activate
# one run (QAOA on the local Maestro simulator); solutions go to portfolio_quantum/solutions/<stem>/
python portfolio_quantum/qlns.py --inst po_a050_t10_s01 --lam 1e-3 --solver qaoa --start empty \
  --minutes 12 --seed 3 --kround 16 --kmove 7 --p-round 0.7 --p-block 0 \
  --angles portfolio_quantum/angles/a050_l1e-3.json
# cloud batch on the largest instance (26-qubit programs, 7 in parallel per round)
cd portfolio_quantum && python cloud_batch.py --inst po_a400_t15_orig --lam 1e-3 --start best --kround 26 --rounds 8
# verify every file with the referee and print the table
python portfolio_quantum/report.py
```

`--start best` starts from the repo's earlier classical solution. `--start qbest` starts from the best of
those and of this pipeline's own files. `--start empty` starts from the empty portfolio. The job lists of
the campaign are in `jobs/`. The logs, experiments and reading guide are in
[`ai_context/portfolio_quantum/`](../ai_context/portfolio_quantum/READ_ME_FIRST.txt). Each `.sol` has a `.log.json` next to it with every
accepted window (kind, qubits, gain).
