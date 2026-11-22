# network18

- **Nodes:** 18 (306 binary arc variables)
- **Official objective (ported QOBLIB checker):** **469000** — valid: True (valid)
- **QOBLIB best known:** 473375 (best known); gap -0.924%
- **Largest quantum program used:** 0 qubits (classical)

## Method

**Previous team pipeline** (old repository `network_design/`: annealing, ILS, parallel tempering, block search, Divi proposals). See the comment lines of the source .sol.

Comment lines in the source .sol: none

Source: `/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/deliverables/network18/network18.sol`

## Reproduce

```bash
cd opus_lab_network
~/.venvs/opus_pf/bin/python search.py 18 --time 1800 --workers 4
```

_Generated 2026-11-22 09:35._
