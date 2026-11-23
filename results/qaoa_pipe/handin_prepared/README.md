# Portfolio (QOBLIB 06) solutions to hand in

Best verified solution per instance so far, re-scored with the official referee in
`qoblib_tools.py` (all feasible). Binary variables = the official QUBO size,
`(6n + cash slack + count slack) * T`. λ = 1e-2 `s01` is left out: the pipeline
found nothing better than the empty portfolio (-1000, best known -2248).

| file | binary vars | score | best known | gap | method |
|---|---:|---:|---:|---:|---|
| `po_a010_t10_orig_b004_l1e-3.sol` | 710 | -8397 | -8397 | +0.00% | SA + **QAOA blocks** (9 qubits) + iterated search, 600 s |
| `po_a050_t10_orig_b020_l1e-4.sol` | 3,110 | -370869 | -370869 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s00_b020_l1e-4.sol` | 3,110 | -355367 | -355530 | +0.05% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s01_b020_l1e-4.sol` | 3,110 | -301400 | -301400 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s02_b020_l1e-4.sol` | 3,110 | -443266 | -443357 | +0.02% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_orig_b020_l1e-3.sol` | 3,110 | -205926 | -206239 | +0.15% | SA + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a050_t10_s00_b020_l1e-3.sol` | 3,110 | -189555 | -191879 | +1.21% | SA + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a050_t10_s01_b020_l1e-3.sol` | 3,110 | -139281 | -141376 | +1.48% | SA + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a050_t10_s02_b020_l1e-3.sol` | 3,110 | -233672 | -233766 | +0.04% | SA + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a050_t10_orig_b020_l1e-2.sol` | 3,110 | -26186 | -37107 | +29.43% | empty start + blocks + whole-day blocks + iterated search, 480 s, seed 1 |
| `po_a050_t10_s00_b020_l1e-2.sol` | 3,110 | -9631 | -12340 | +21.95% | empty start + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a050_t10_s02_b020_l1e-2.sol` | 3,110 | -32367 | -43565 | +25.70% | empty start + blocks + whole-day blocks + iterated search, 1200 s |
| `po_a200_t10_orig_b050_l1e-3.sol` | 12,110 | -798331 | — | — | SA + blocks + iterated search, 1200 s |
| `po_a400_t15_orig_b100_l1e-3.sol` | 36,165 | -2815258 | — | — | empty start + blocks + iterated search, 1500 s |

Method: SA on the compact QUBO (`portfolio_compact.py`) + fast repair, then exact
day × asset-group blocks (groups of 3, net-position space) and iterated local search
(`asset_groups.py pipeline`). Rows marked "whole-day blocks" also re-solve every day
with all assets at once (integer SA on the block's exact form, `big_blocks.py`;
`--day-blocks`), alternating with the small blocks and with kicks; seed 0 unless the
row says otherwise. Quantum programs: only the a010 λ1e-3 file comes from one (below); for the others,
QAOA is still an experiment (`qaoa_signed_blocks.py`: up to 24 qubits on Qoro's cloud).

**QAOA row (a010 λ1e-3).** Same pipeline, but every block is solved by QAOA instead of
enumeration (`asset_groups.py pipeline --solver qaoa`, `QAOABlockSolver`). Each block is 3
assets × 1 day on the signed 3-bit encoding = **9 qubits** (largest quantum program),
QAOA p = 2, TQA start (dt = 1), COBYLA 30 iterations, 10,000 shots, local Maestro
simulator. The 64 most probable samples are decoded and the best feasible one by true
block cost is proposed. The budget is a Lagrange price μ, raised (0, 0.25, 1, 4 × a
per-block unit cost) only when the lowest-energy sample breaks a rule. No block is
enumerated. The start (SA + repair, −7983) and the kicks of the iterated search are
classical. 7,982 QAOA runs in 600 s, seed 0. A diagnostic enumeration (not used by the
solver) found that QAOA's proposal was the block optimum on 7,299 of 7,302 blocks.

Re-check a file:

```python
import qoblib_tools as qt
h, pos = qt.read_portfolio_solution("handin/po_a050_t10_orig_b020_l1e-4.sol")
m = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]), lam=h["lambda"])
print(m.score(pos))
```
