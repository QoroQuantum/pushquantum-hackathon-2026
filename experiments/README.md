# Experiments not used by the hand-in

Scripts kept for the record: dead ends, side tests and teammates' solvers. None of them
produced a file in `handin/`. Run them from the repository root
(`python experiments/<script>.py …`); each one puts the root on `sys.path`, so the shared
modules (`qoblib_tools`, `asset_groups`, `portfolio_compact`, …) still import.

| Script | What it was for | Outcome |
|---|---|---|
| `check_portfolio_compact.py` | Checks that the compact QUBO (`portfolio_compact.py`) is exact | Passes; the compact QUBO is used everywhere |
| `per_day_pce.py` | PCE on one day at a time on the compact (unary) QUBO | PCE fails on the unary encoding |
| `pce_signed_day.py` | PCE on a whole day in the signed 3-bit encoding | Still fails on whole days |
| `benchmark_groups.py` | First grid runner for the asset-group pipeline | Replaced by `grid_runs.py` |
| `lower_bound.py` | LP lower bound on the optimum, for a200/a400 (no BKV) | Not cited in the hand-in |
| `init_comp.py`, `run_init_comp.py` | QAOA/PCE initialisation comparison, through `auto_runner.py` (Hugo) | Results stay in `AutoData/` (not committed) |
| `solver1.py` | Partitioned QAOA as in notebook 01, through `auto_runner.py` (skoder09) | Baseline |
| `solver2.py`, `solver3.py` | PCE solvers for a200/a400 in the signed 3-bit encoding; `solver3.py` builds on `solver2.py` (skoder09) | Not used for hand-in files |
| `solver_partitioned_sa.py` | Asset/day block SA solver for `auto_runner.py` (Tim) | Overlaps with `asset_groups.py` |

Outputs from the early runs are in `results/early/`.
