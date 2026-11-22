#!/bin/bash
# slot-PCE + L1/L2 polish on po_a050_t10_s00, 4 λ; 3 workers; logs unbuffered
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
run() { ../../.venv/bin/python exp_slots.py --instance po_a050_t10_s00 --optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis2 --polish --lns-time 120 --record "$@" > logs/a050_slots_$(echo "$*" | tr ' ' '_' | tr -d '-').txt 2>&1; }
export -f run
printf '%s\n' "--lam 1e-2 --repeats 8 --tag _r8ls" "--lam 1e-3 --repeats 8 --tag _r8ls" "--lam 1e-4 --repeats 8 --tag _r8ls" "--lam 5e-4 --repeats 8 --tag _r8ls" \
 | xargs -P 3 -I{} bash -c 'run {}'
