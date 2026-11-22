#!/bin/bash
# 3 workers, 2 numba threads each; PT 120 s + LNS polish 30 s per job (equal wall for baselines)
cd "$(dirname "$0")"
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 -I{} sh -c '../../.venv/bin/python run_pt.py {} --lam 1e-2 --time 120 --polish 30 > /dev/null 2>&1' < "${1:-jobs.txt}"
