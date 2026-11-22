#!/bin/bash
# restarts + multibasis sweep, slot formulation, a010_t10_orig λ=1e-4; 2 workers
cd "$(dirname "$0")"
run() { ../../.venv/bin/python exp_slots.py --lam ${LAM:-1e-4} --no-ideal "$@" 2>&1 | grep -v -i warn | grep "==" | sed "s|^|$* :: |"; }
export -f run
printf '%s\n' \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis2 --repeats 4 --tag _mb2a6r4" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 15 --code multibasis2 --repeats 4 --tag _mb2a15r4" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis3 --repeats 4 --tag _mb3a6r4" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --qubits 10 --repeats 4 --tag _a6q10r4" \
 "--optimizer lbfgs --iters 200 --n-layers 10 --alpha 6 --code multibasis2 --repeats 4 --tag _mb2a6L10r4" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis2 --qubits 7 --repeats 4 --tag _mb2a6q7r4" \
 | xargs -P 2 -I{} bash -c 'run {}'
