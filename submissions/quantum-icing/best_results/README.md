# Best results

Final hand-in: one best `.sol` per instance, each with a `.md` card (objective, QOBLIB best-known value, gap, qubits, method, reproduction command).

| Folder | Contents |
|---|---|
| [network/](network/INDEX.md) | QOBLIB 08 network design, `network05`–`network24` (20 instances) |
| [portfolio/](portfolio/INDEX.md) | QOBLIB 06 portfolio optimisation, 352 configurations grouped by asset count (`a003`–`a400`) |
| [reports/](reports/) | LaTeX reports: [network24_report.tex](reports/network24_report.tex), [portfolio_report.tex](reports/portfolio_report.tex) |

## Network design

| Instance | Objective | QOBLIB BKV | Gap |
|---|---:|---:|---:|
| [network05](network/network05.md) | 65500 | 65500 | +0.000% |
| [network06](network/network06.md) | 101000 | 101000 | +0.000% |
| [network07](network/network07.md) | 142400 | 142400 | +0.000% |
| [network08](network/network08.md) | 170400 | 170231 | +0.099% |
| [network09](network/network09.md) | 196750 | 196750 | +0.000% |
| [network10](network/network10.md) | 216429 | 210800 | +2.670% |
| [network11](network/network11.md) | 241000 | 238334 | +1.119% |
| [network12](network/network12.md) | 280834 | 276474 | +1.577% |
| [network13](network/network13.md) | 308500 | 300620 | +2.621% |
| [network14](network/network14.md) | 352650 | 343091 | +2.786% |
| [network15](network/network15.md) | 385091 | 370870 | +3.834% |
| [network16](network/network16.md) | 411546 | 397908 | +3.427% |
| [network17](network/network17.md) | 455690 | 437000 | +4.277% |
| [network18](network/network18.md) | 469000 | 473375 | -0.924% |
| [network19](network/network19.md) | 488895 | 492924 | -0.817% |
| [network20](network/network20.md) | 518305 | 528528 | -1.934% |
| [network21](network/network21.md) | 562362 | 569273 | -1.214% |
| [network22](network/network22.md) | 604540 | 618135 | -2.199% |
| [network23](network/network23.md) | 639739 | 659369 | -2.977% |
| [network24](network/network24.md) | 657121 | 663688 | -0.989% |

Negative gap = better than QOBLIB's best-known value (minimisation). `network18`–`network24` all beat it.

## Portfolio optimisation

352 configurations: 109 match QOBLIB's best-known value, 19 go below it (all `po_a050`), and 224 have no published value. Every λ = 0 file is proven optimal with an exact MILP; the 3–5-asset files at every λ are proven optimal with exact DP. Full table: [portfolio/INDEX.md](portfolio/INDEX.md).
