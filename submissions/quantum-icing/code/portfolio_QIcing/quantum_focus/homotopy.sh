#!/usr/bin/env bash
# λ-homotopy with quantum windows (PCE-QLNS): start from a quantum-built portfolio at a small λ and
# follow the optimum through intermediate λ values to the target. Only the final (official) λ is recorded.
# usage: homotopy.sh <instance> <start.sol> <seed> <lams...>   (last λ = official target)
set -e
cd "$(dirname "$0")/.."
INST=$1; START=$2; SEED=$3; shift 3
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2
P="--instance $INST --routine pce --n-layers 3 --iters 100000 --no-char --mc-population 40 --mc-iterations 8 --window-assets 5 --time-window-assets 2 --p-time 0.4 --p-corr 0.6"
CUR=$START
N=$#
I=0
for LAM in "$@"; do
  I=$((I+1))
  LOG=quantum_focus/logs/homotopy_${INST}_s${SEED}_step${I}_l${LAM}.txt
  if [ $I -lt $N ]; then
    OUT=quantum_focus/starts/homotopy_${INST}_s${SEED}_l${LAM}.sol
    ../.venv/bin/python run_qlns.py $P --lam $LAM --start "$CUR" --seed $((SEED*10+I)) --time-limit 540 --save-sol $OUT > $LOG 2>&1
    CUR=$OUT
  else
    ../.venv/bin/python run_qlns.py $P --lam $LAM --start "$CUR" --seed $((SEED*10+I)) --time-limit 1200 > $LOG 2>&1
  fi
  echo "step $I λ=$LAM: $(grep FINAL $LOG | cut -c1-60)"
done
