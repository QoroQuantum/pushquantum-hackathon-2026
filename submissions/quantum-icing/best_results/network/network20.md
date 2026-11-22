# network20

- **Nodes:** 20 (380 binary arc variables)
- **Official objective (ported QOBLIB checker):** **518305** — valid: True (valid)
- **QOBLIB best known:** 528528 (best known); gap -1.934%
- **Largest quantum program used:** 0 qubits (classical)

## Method

**Previous team pipeline** (old repository `network_design/`: annealing, ILS, parallel tempering, block search, Divi proposals). See the comment lines of the source .sol.

Comment lines in the source .sol: none

Source: `/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/deliverables/network20/network20.sol`

## Reproduce

```bash
cd opus_lab_network
~/.venvs/opus_pf/bin/python search.py 20 --time 1800 --workers 4
```

_Generated 2026-11-22 09:35._
