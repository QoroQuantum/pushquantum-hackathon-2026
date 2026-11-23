#!/usr/bin/env python
"""Fixed-angle TQA: deep QAOA (p = 30) on the TQA ramp, no angle optimisation.

Each block QAOA call evaluates the energy of the TQA ramp at 10 time steps Δt in one job
(one random draw in each 0.4-wide slice of (0, 2.5), fixed per run), keeps the Δt with the
lowest energy and samples there in a second job. That is the colleagues' block solver
with Monte Carlo limited to one iteration and the TQA ramps as its population. Noiseless
simulation makes depth free, so a deep ramp approximates an adiabatic sweep.

Measured on a050 blocks (λ 1e-4): p = 20 and p = 30 found the exact block optimum on both
blocks tested (p = 3 and p = 10 did not); energy vs Δt falls smoothly up to Δt ≈ 1.3-1.6
and jumps up beyond (Trotter breakdown).

    python fixed_tqa.py                 # Qoro cloud, po_a050_t10_orig_b020, all 8 λ, seed 1
    python fixed_tqa.py --local         # same on local CPUs
    python fixed_tqa.py --quick         # cloud calibration (λ 1e-2, 2 sweeps, a few credits)
    python fixed_tqa.py --instance po_a010_t10_orig_b004 --lams 1e-6 1e-4 1e-2

Other options pass through to tqa_vs_random_cloud.py / tqa_vs_random.py (e.g. --hours,
--min-credits, --in-flight, --workers, --dt-sampling geom). Results go to
AutoData/quantum_annealing/<problem>_p30_cloud_fixedtqa/ (or _p30_fixedtqa locally); each
run.json holds dt_grid / dt_curve_mean (mean energy vs Δt). Replot:
python plot_tqa_vs_random.py <folder>
"""

import sys
from pathlib import Path

DEFAULTS = [
    "--instance",
    "po_a050_t10_orig_b020",
    "--lams",
    "0",
    "1e-6",
    "1e-5",
    "5e-5",
    "1e-4",
    "5e-4",
    "1e-3",
    "1e-2",
    "--seeds",
    "1",
    "--inits",
    "tqa",
    "--layers",
    "30",
    "--population",
    "10",
    "--iters",
    "1",
    "--tag",
    "_fixedtqa",
    "--root",
    str(Path(__file__).resolve().parent / "AutoData" / "quantum_annealing"),
]
DT_RANGE = [
    "--dt-range",
    "0",
    "2.5",
    "--dt-sampling",
    "stratified",
]  # one random Δt in each 0.4-wide slice of (0, 2.5)


def main() -> int:
    argv = sys.argv[1:]
    local = "--local" in argv
    argv = [a for a in argv if a != "--local"]
    if local:
        import tqa_vs_random as driver
    else:
        import tqa_vs_random_cloud as driver
    return driver.main(DEFAULTS + DT_RANGE + argv)


if __name__ == "__main__":
    sys.exit(main())
