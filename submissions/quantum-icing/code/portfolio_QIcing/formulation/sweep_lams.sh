#!/bin/bash
# best slot-PCE config across λ and instances (a010); 2 workers
cd "$(dirname "$0")"
run() { ../../.venv/bin/python exp_slots.py --no-ideal --optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis2 "$@" 2>&1 | grep -v -i warn | grep "==\|reference" | sed "s|^|$* :: |" | cut -c1-400; }
export -f run
printf '%s\n' \
 "--lam 1e-4 --repeats 12 --tag _best_r12_std" \
 "--lam 5e-4 --repeats 12 --tag _best_r12_std" \
 "--lam 1e-3 --repeats 12 --tag _best_r12_std" \
 "--instance po_a010_t10_s02 --lam 1e-4 --repeats 12 --tag _best_r12_std" \
 | xargs -P 2 -I{} bash -c 'run {}'
