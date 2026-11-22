#!/bin/bash
# alpha x qubits sweep for the slot formulation (a010_t10_orig, λ=1e-4); 2 workers
cd "$(dirname "$0")"
run() { ../../.venv/bin/python exp_slots.py --lam 1e-4 "$@" 2>&1 | grep -v -i warn | grep "==" | sed "s|^|$* :: |"; }
export -f run
printf '%s\n' \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --tag _a6" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 15 --tag _a15" \
 "--optimizer mc --iters 40 --alpha 6 --tag _mca6" \
 "--optimizer mc --iters 40 --alpha 15 --tag _mca15" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --qubits 10 --tag _a6q10" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 2 --qubits 10 --tag _a2q10" \
 "--optimizer mc --iters 40 --alpha 6 --qubits 10 --tag _mca6q10" \
 "--optimizer lbfgs --iters 200 --n-layers 6 --alpha 6 --code multibasis2 --tag _a6mb2" \
 | xargs -P 2 -I{} bash -c 'run {}'
