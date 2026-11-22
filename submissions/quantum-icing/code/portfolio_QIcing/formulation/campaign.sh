#!/bin/bash
# slot-QLNS (rand sub-solver + period descent) on every a050 case above BKV; 8 workers, 1500 s each
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 NUMBA_NUM_THREADS=1
run() { set -- $1; ../../.venv/bin/python slot_qlns.py --instance $1 --lam $2 --start best --solver rand --inner-descent --time-limit 1500 --record --seed 7 --tag _camp > logs/camp_${1}_l$2.txt 2>&1; }
export -f run
cat logs/campaign_targets.txt | xargs -P 8 -I{} bash -c 'run "{}"'
