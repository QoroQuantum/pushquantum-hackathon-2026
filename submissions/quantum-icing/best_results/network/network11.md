# network11

- **Nodes:** 11 (110 binary arc variables)
- **Official objective (ported QOBLIB checker):** **241000** — valid: True (valid)
- **QOBLIB best known:** 238334 (best known); gap +1.119%
- **Largest quantum program used:** 0 qubits (classical)

## Method

**opus_lab_network search (classical; 0 qubits).** Parallel ILS-tabu over degree-preserving moves (2-exchange (a→b),(c→d)⇒(a→d),(c→b) and 3-rotations). Candidate moves must touch a congestion bottleneck, i.e. an arc with positive LP dual in the min-congestion routing LP (the network analogue of the portfolio engine's neighbour-context re-solves). Fractional LP optimum as search objective, kicks on stall, an elite pool with path relinking between elites (pool recombination), and disk merge between parallel workers. Every saved file is integer-routed and checked by the port of QOBLIB's checker.

Comment lines in the source .sol: `opus_lab_network sprint: dual-guided ILS-tabu + path relinking`

Source: `/mnt/d/Repos/Quantum_Munich_Reset_3-30AM/opus_lab_network/solutions/network11.sol`

## Reproduce

```bash
cd opus_lab_network
~/.venvs/opus_pf/bin/python search.py 11 --time 1800 --workers 4
```

_Generated 2026-11-22 09:35._
