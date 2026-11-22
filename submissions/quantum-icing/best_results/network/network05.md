# network05

- **Nodes:** 5 (20 binary arc variables)
- **Official objective (ported QOBLIB checker):** **65500** — valid: True (valid)
- **QOBLIB best known:** 65500 (optimal); gap +0.000%
- **Largest quantum program used:** 0 qubits (classical)

## Method

**Exact enumeration** of every valid topology (each node picks 2 out-arcs; filtered to in-degree 2), each scored by the routing LP. Proven optimal.

Comment lines in the source .sol: `opus_lab_network exact: enumeration of all 216 valid topologies (proven optimal)`

Source: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab_network/solutions/network05.sol`

## Reproduce

```bash
cd opus_lab_network
~/.venvs/opus_pf/bin/python exact.py 5
```

_Generated 2026-11-22 09:35._
