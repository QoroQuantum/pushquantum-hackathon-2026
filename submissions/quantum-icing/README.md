# Quantum Icing: QOBLIB 06 portfolio + QOBLIB 08 network design

Team Quantum Icing. Both problems, every instance. Each `.sol` is re-scored by the referee in
`qoblib_tools.py`; every best file comes with a `.md` card (objective, QOBLIB best-known value, gap,
qubits, method, reproduction command).

| Folder | Contents |
|---|---|
| [`best_results/`](best_results/README.md) | **Hand-in:** best `.sol` per instance + cards. [network](best_results/network/INDEX.md), [portfolio](best_results/portfolio/INDEX.md), LaTeX [reports](best_results/reports/) |
| [`code/`](code/README.md) | Solvers, quantum pipelines, notebooks, run logs |

## Network design (QOBLIB 08): network05–network24

- 20/20 instances valid. network05, 06, 07, 09 match the best-known value.
- **network18–network24 all beat QOBLIB's best-known value** (gap −0.82 % to −2.98 %), e.g.
  network24 (552 binary variables): 657 121 vs 663 688, gap −0.99 %.
- These files come from a classical ILS-tabu search over degree-preserving moves, guided by the
  routing LP's duals.
- Quantum track ([`code/hopqubo_newTesting/`](code/hopqubo_newTesting/README.md)): constraint-respecting
  *band* partitions (≤ 18 qubits each, 100 % valid samples by construction) run as one Divi
  `ProgramEnsemble` with a 2-exchange-mixer QAOA, plus full-problem PCE with b-matching decoding.
  Best quantum network12: +1.21 % gap; the pipeline scales to network24 (8 bands/round).

## Portfolio optimisation (QOBLIB 06): 352 configurations, 3 to 400 assets

- 109 match the best-known value, **19 beat it** (all `po_a050`), 224 have no published value.
- Every λ = 0 file (up to 400 assets × 15 days, 36 165 variables) is proven optimal by an exact
  MILP; the 3–5-asset files at every λ, and all 10-asset files, are proven optimal by exact DP.
- Quantum tracks:
  - [`code/portfolio_QIcing/`](code/portfolio_QIcing/REPORT.md): PCE quantum large-neighbourhood
    search from an empty portfolio, ≤ 9 qubits, no classical local search, reaches the proven optimum
    on `po_a010_t10_orig` at λ = 1e-4, 5e-4, 1e-3.
  - [`code/portfolio_quantum/`](code/portfolio_quantum/README.md): QAOA large-neighbourhood search
    where every accepted move is a QAOA sample; improves on our classical starts at `po_a050` and on
    `po_a400_t15_orig` (36 165 variables, QAOA up to 26 qubits).

**Scale:** largest instance solved 36 165 binary variables (`po_a400_t15_*`); largest quantum
program 26 qubits.

## Not included

`code/qoblib_data/` (downloaded by the notebooks), the static website, and our AI-session
transcripts/logs folder were left out to keep the PR small. Some scripts still expect our original
repo layout (`deliverables/`, `solutions/`); see `code/README.md`.
