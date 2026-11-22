# TheGOAT_Testing: what one night of measurements says (network24)

Snapshot 2026-11-22. All numbers are measured; scripts and commands are at the end.

## Verdict

**network24 stays at 657,121 (−0.99% vs 663,688).** Nothing tonight beat it: not block moves
(exhaustive, MILP, quantum or classical), not regrowing it from the team's network23, not
recombining the two elite topologies, not a global search on the hop count from random starts. The quantum proposers are valid by construction and run on
the full instance at 14–25 qubits, but on these block subproblems classical SA on the same QUBO
is the better optimizer. That matches what the team found independently with its own QAOA
compound-move search on network24 (SA matched or beat QAOA in 21/21 rounds,
`ai_context/quantum_networks/READ_ME_FIRST.txt`): two groups, three encodings, same answer.

1. **The surrogate decides, not the sampler.** On network24 blocks the existing
   `network_qlns` sub-QUBO ranks rewirings with Spearman 0.33–0.67 against the exact LP; the
   demand-weighted hop count reaches 0.85–0.89 and puts the LP-best rewiring first in 6/6
   blocks. Every quantum run below samples a QUBO learned from the hop count.
2. **657,121 is certified optimal within three arcs of itself.** All 35,720 moves at radius 2
   and 3 enumerated and refuted in 6 min 21 s (section 6): a proof, not a search result. Beyond
   that radius it is evidence only — six blocks enumerated exhaustively at r = 6–7 and 182
   time-limited MILP block steps at r = 8–16 found zero improvements, but those six blocks are
   six subsets out of the C(48,6) ≈ 12.3 million there are.
3. **Quantum proposers run in the loop on the full network24 instance, always valid, at
   14–25 qubits.** Hungarian decoding makes every PCE sample a valid topology; the QAOA
   warm-started at the incumbent with a 2-exchange mixer keeps 100% of raw samples on valid
   permutations, also at 25 qubits on Qoro's GPU cluster.
4. **As optimizers of the block QUBO they lose to classical SA on the same QUBO.** PCE
   (14 qubits) beats random rewirings in 60/60 blocks and the strongest sampling control in
   51/60, but loses to SA in 55/60 (ties 5). Seeding a local search with quantum candidates
   gives the same result as seeding it with random ones (6 wins, 31 ties, 7 losses in 44 blocks).
5. **Changing the classical optimizer does not close the gap.** Five optimizers on the same five
   blocks reach 0.17–0.23 of the way from SA's optimum to a random permutation; quantum natural
   gradient costs 6,782 circuits per block for no gain; QN-SPSA matches Monte Carlo with a third
   of the circuits.
7. **The certificate machinery moved three other hand-ins, and bounds how much is left.**
   What could not beat network24 beat its smaller siblings. Exhaustive radius-3 descent took
   `network18` to 472,112 and `network22` to 604,540 (section 7); kicking a *proven* optimum by
   five arcs and exhausting radius 3 again took `network18` to **469,000**, −0.27% → −0.92%
   against best known, and `network20` to 519,207 (section 10). The winning topology on
   network18 is **ten arcs** from the certified one, which is the clearest thing measured all
   night about what a local certificate buys. And from the other side, a bound that holds for
   every 2-in/2-out topology puts every hand-in within 28.5–32.3% of the true optimum
   (section 9) — the one number here that no better search can improve.

8. **Angle warm start halves the optimizer's distance to the QUBO optimum, and the LP does not
   follow.** Starting each block from the previous block's best angles: gap 0.100 vs 0.202 cold
   (4 blocks, same 412 circuits), but the LP value of the resulting candidates was better in 1
   block and worse in 3. A better-optimized surrogate is not a better topology: the bottleneck is
   back to the model.

## 1. Which surrogate ranks like the referee

Every valid rewiring of each block scored with the exact LP; Spearman of each surrogate against
it (`block_fidelity.py`). `fit_*` are fitted on half the block and scored on the other half.

| block | r | valid rewirings | improving | team sub-QUBO | dual bound | hop count | fit linear | fit quadratic |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 362 | 0 | +0.44 | +0.40 | +0.88 | +0.83 | +0.94 |
| 1 | 6 | 72 | 0 | +0.67 | +0.19 | +0.89 | +0.82 | +0.78 |
| 2 | 6 | 102 | 0 | +0.43 | +0.55 | +0.85 | +0.78 | +0.86 |
| 3 | 6 | 93 | 0 | +0.33 | +0.57 | +0.88 | +0.72 | +0.78 |
| 0 | 7 | 2,428 | 0 | +0.39 | +0.41 | +0.87 | +0.83 | +0.96 |
| 1 | 7 | 249 | 0 | +0.60 | +0.19 | +0.87 | +0.76 | +0.85 |
| **range** | | | **0** | **0.33–0.67** | **0.19–0.57** | **0.85–0.89** | **0.72–0.83** | **0.78–0.96** |

At the incumbent 23 of 48 links carry the maximum load: a wide plateau, which is why no small
block can lower the maximum.

## 2. Exact checks on the 657,121 hand-in

| check | what was searched | better than 657,121 |
|---|---|---|
| exhaustive blocks (`block_fidelity.py`) | 6 blocks, r = 6–7, every valid rewiring (72–2,428 each) | 0 |
| exact LNS (`network_lns.py`, MILP with joint routing) | 182 block steps, r = 8–16, 20 s each, 2 workers × 30 min | 0 (most MILPs stopped at the time limit) |
| regrow from network23 (`node_insertion.py`) | all 989 degree-preserving insertions of node 24 into the 639,739 network23 topology; best 670,942; then team ILS 12 min + PT 10 min | 0 (stuck at 666,036, +0.35%) |
| recombination | the 6 arcs where 666,036 and 657,121 differ, tails kept, heads permuted: 330 valid recombinations | 0 (the best is the hand-in itself) |
| block LNS from the stall point (`goat_lns.py --start-file`, r = 8, 30 steps) | from 666,036, where the team's ILS and PT stalled for 22 min: a quantum-led and an SA-led walk | 0 (both reach 665,351 on the same block, step 28; the quantum proposer found that move in the quantum-led run, not in the SA-led one) |
| global hop-count search (`hop_search.py`) | 87 SA restarts from random topologies on Σ d·hops, 2 workers × 6 min, then the exact LP on the best 150 | 0 (best hop count 29,834 is 0.96% *above* the hand-in's 29,549; best new basin 686,875, 32–37 arcs away) |

The regrown topology differs from the hand-in in only 6 of 48 arcs: the team derived network23
from this network24 topology, so regrowing returns to the same basin, slightly worse.

The hand-in is also excellent on the surrogate: no random restart of the hop-count search got
below its hop count. Its LP value is 6.74% above its own hop bound (615,604 = Σ d·hops / 48),
and among topologies that already have a low hop count the hop count no longer predicts the LP
(Spearman 0.17 inside the search pool): what is left is load balance, which the hop count does
not see. A surrogate that models balance is the next step, not a better sampler.

## 3. Classical optimizer of the circuit parameters (`optimizer_bench.py`)

PCE, 14 qubits, block QUBO with r = 10 learned from the hop count, five network24 blocks, 60
iterations each. `gap` = (best QUBO energy among decoded candidates − SA's best) / (median random
permutation − SA's best): 0 = SA's optimum, 1 = a typical random permutation.

| optimizer | mean gap | median gap | circuits per block | seconds per block |
|---|---|---|---|---|
| Monte Carlo (divi default) | 0.172 | 0.175 | 1,202 | 10.6 |
| QN-SPSA (natural-gradient SPSA) | 0.182 | 0.168 | 412 | 4.0 |
| quantum natural gradient | 0.204 | 0.194 | 6,782 | 50.9 |
| COBYLA | 0.219 | 0.253 | 169 | 4.9 |
| L-BFGS-B | 0.227 | 0.254 | 136 | 5.7 |
| QN-SPSA + angle warm start (blocks 1–4) | **0.100** | | 412 | 4–23 |
| QN-SPSA cold, same blocks 1–4 | 0.202 | | 412 | 3–7 |

## 4. Quantum LNS walks on network24

Each walk starts from the greedy topology (768,375, +15.77%) so that there is room to descend,
and follows the quantum proposer only. On every block the controls get the same number of exact
LP solves (K = 10 each). W / T / L = blocks where the quantum proposer's best LP value was lower /
equal / higher.

| run | proposer | qubits | blocks | reached | vs SA on same QUBO | vs sampled (best of 1,500 by hops) | vs random | quantum+LS vs random+LS |
|---|---|---|---|---|---|---|---|---|
| `pce_r10_greedy` | PCE, cell variables, Monte Carlo | 14 | 60 | 701,615 (+5.71%) | 0 / 5 / 55 | 51 / 1 / 8 | 60 / 0 / 0 | – |
| `pce_r10_greedy_ls` | PCE, cells, COBYLA + soft read-out | 14 | 44 | 700,934 (+5.61%) | 1 / 1 / 42 | 33 / 0 / 11 | 43 / 0 / 1 | 6 / 31 / 7 |
| `pce_r10_greedy_qnspsa_warm` | PCE, cells, QN-SPSA + angle warm start | 14 | 20 | 705,228 (+6.26%) | 0 / 1 / 19 | 13 / 1 / 6 | 20 / 0 / 0 | 2 / 16 / 2 |
| `sw_r16_greedy` | PCE, swap variables, trust region | 14 | 30 | 723,657 (+9.04%) | 0 / 0 / 30 | 1 / 0 / 29 | 1 / 0 / 29 | – |
| `swap_r4_greedy_local` | QAOA, incumbent start + 2-exchange mixer | 16 | 6 | 725,134 (+9.26%) | 0 / 6 / 0 | 0 / 6 / 0 | 0 / 6 / 0 | 2 / 4 / 0 |
| `swap_r5_greedy_gpu` | same, Qoro `qoro_gpu` | 25 | 10 | 731,000 (+10.14%) | 0 / 5 / 5 | 0 / 5 / 5 | 0 / 5 / 5 | – |

Raw samples that already are valid permutations: 0% for PCE (repaired by Hungarian decoding),
**100% for the 2-exchange-mixer QAOA** at 16 and 25 qubits. The 25-qubit cloud run is still
going (about 6–16 min and 30–50 credits per step, 380 credits for the first 10 steps); `summarize.py`
regenerates this table from `runs/goat/`.

Reading: the quantum proposer carries real information (it beats random rewirings almost always,
and the best of 1,500 hop-ranked samples in most blocks with cell variables), but SA on the same
learned QUBO finds better candidates, and a local search started from quantum candidates ends
where one started from random candidates ends. On blocks of 4–5 arcs every proposer finds the
block optimum, so the 16- and 25-qubit runs show validity and scale, not an edge.

## 5. A surrogate that models the max instead of bounding it — measured, and it fails

Every surrogate above (`hops`, `bound`, the team sub-QUBO) is the weak-duality bound
`sum_ij d_ij dist_w(i,j) / sum_a w_a`: a **sum** over paths, while the referee minimises a
**max** over links. That gap is where the remaining margin lives — the hand-in is 6.74% above
its own hop bound (657,121 against 615,604) and its routing leaves many links at the maximum —
and it explains why the hop count ranks block rewirings at rho 0.85–0.89 but collapses to
rho 0.17 among already-good topologies (section 4).

So `load_surrogate.py` routes the demand instead of bounding it: all-pairs shortest paths with
the demand split evenly over equal-length paths, loads accumulated per arc, the arc weights
then raised on congested arcs and the routing repeated. Its maximum is a feasible routing's
maximum, hence an *upper* bound on the LP where the hop count is a lower one.

| | value on the hand-in | vs the LP | time |
|---|---|---|---|
| exact LP (referee) | 657,121 | — | **60 ms** |
| ECMP, 1 iteration | 813,000 | +23.7% | 17 ms |
| congestion-aware, 6 iterations | 777,000 | +18.2% | 68 ms |

**The idea is dead, and the reason is worth keeping: the exact LP costs 60 ms.** A surrogate
has to be much cheaper than that to be worth any approximation error, and a routing good enough
to track the LP is not. The premise that the referee is expensive — which is what justifies a
surrogate for *ranking* candidates at all — does not hold at n = 24. Surrogates are still
needed for the one thing an LP cannot be: the closed-form quadratic a sampler optimises.

## 6. Certificate: 657,121 is optimal within three arcs of itself (`certify_radius.py`)

Degrees pin the neighbourhood down exactly. If two 2-in/2-out topologies differ in k arcs, each
node loses as many out-arcs as it gains, and the same for in-arcs, so the added arcs carry
exactly the removed arcs' tails and heads: the move is a **permutation of the heads of those k
arcs**, and a permutation fixing an arc is a smaller move. The radius-k neighbourhood is
therefore every k-subset times every **derangement** of k — finite and enumerable:

| radius | subsets | derangements | moves | note |
|---|---:|---:|---:|---|
| 2 | 1,128 | 1 | **1,128** | the 2-exchange scan the team's ILS already does |
| 3 | 17,296 | 2 | **34,592** | the two 3-rotations per triple |
| 4 | 194,580 | 9 | **1,751,220** | |
| 6 | 12,271,512 | 265 | 3.3 billion | the regime the six enumerated blocks sampled |

This claim is what the certificate rests on, so it is checked by brute force rather than
asserted: `tests/test_certify_radius.py` enumerates all 216 valid topologies of network05 and
confirms that every one within distance 5 of a reference is reached by a derangement of exactly
the differing arcs (32 tests pass).

`certify_radius.py` enumerates those moves against a warm-started dual simplex with
`objective_bound` at the incumbent, so HiGHS abandons a candidate as soon as it can prove it is
no better instead of solving it to optimality. Run on 2 workers:

| radius | moves | invalid | LPs | time | per LP | result |
|---|---:|---:|---:|---:|---:|---|
| 2 | 1,128 | 192 | 936 | 10 s | 10.8 ms | **no better topology** |
| 3 | 34,592 | 6,251 | 28,341 | 371 s | 13.1 ms | **no better topology** |

**Certified radius 3, in 6 min 21 s total: every topology differing from the hand-in in at most
three arcs is no better than 657,121.** That is a proof over 35,720 moves, not a search result,
and it upgrades verdict 2 accordingly.

Two details worth reading. The objective bound cut **100% of candidates** (936/936 and
28,341/28,341): not one of them needed solving to optimality, so the certificate costs 13 ms a
move instead of the 60 ms a cold LP costs. The flip side is that the certificate says *none is
better* and cannot say *by how much*, because a cut candidate never returns a value.

It also draws the line where a sampler earns its place. Radii 2 and 3 are settled by
enumeration; radius 4 is 1,751,220 moves, about **2.5 hours** on 2 workers at the measured
rate, and still enumerable; radius 6 — the regime the quantum LNS blocks work in — is 3.3
billion and never will be. Exhaustive methods and proposers divide the problem at radius 4–5,
and that boundary is now measured rather than assumed.

```bash
venv/bin/python TheGOAT_Testing/certify_radius.py --n 24 --r 3 --workers 2   # 6 min, done
venv/bin/python TheGOAT_Testing/certify_radius.py --n 24 --r 4 --workers 2   # ~2.5 h
```

## 7. The certificate turned into a search: network18 improved (`certify_descent.py`)

A radius scan that answers "is anything within r arcs better?" becomes a search when you take
the best move it finds and run it again. Each pass enumerates every radius-2 and radius-3 move,
takes the single best improving one, banks it through the official checker and restarts. When a
pass finds nothing the run stops **with a proof**: a verified radius-3 local optimum, not a
point a heuristic got stuck in.

On `network18`, the weakest hand-in at −0.079%:

| pass | radius | LPs | time | result |
|---|---|---:|---:|---|
| 1 | 2 | 490 | 4 s | none better |
| 1 | 3 | 10,887 | 126 s | **473,000 → 472,112** |
| 2 | 2 | 490 | 4 s | none better |
| 2 | 3 | 10,885 | 112 s | none better → certified |

**473,000 → 472,112 in 247 s, gap to best known from −0.08% to −0.27%**, checked by QOBLIB's
own checker and installed as the hand-in.

Then the same descent over the rest of the chain, one worker each:

| instance | arcs | start | final | gap | r = 3 scan | certified |
|---|---:|---:|---:|---:|---:|---|
| network18 | 36 | 473,000 | **472,112** | −0.27% | 10,887 LPs, 126 s | radius 3 |
| network19 | 38 | 488,895 | 488,895 | −0.82% | 13,062 LPs, 154 s | radius 3 |
| network20 | 40 | 519,343 | 519,343 | −1.74% | 15,509 LPs, 211 s | radius 3 |
| network21 | 42 | 562,362 | 562,362 | −1.21% | 18,247 LPs, 265 s | radius 3 |
| network22 | 44 | 606,424 | **604,540** | −2.20% | 21,286 LPs, 456 s | radius 3 |
| network23 | 46 | 639,739 | 639,739 | −2.98% | 24,647 LPs, 610 s | radius 3 |

**Two hand-ins improved, all six certified, in 40 minutes on one core.** Both improving moves
are 3-rotations — a move type the chain's ILS already samples — and that is the finding:
sampling a neighbourhood is not exhausting it. On instances several hours of heuristic search
had polished, enumerating fourteen to twenty-one thousand moves found something in minutes.

The starting topologies came from the team's hybrid chain, so the 18-qubit QAOA step stays in
these solutions' ancestry; only the final refinement is classical and exhaustive.

### Radius 4 on network18 and network19

| instance | radius | moves | distinct valid | LPs cut by the bound | improving | time |
|---|---:|---:|---:|---:|---:|---:|
| network18 | 2 | 630 | 490 | 490 | 0 | 5 s |
| network18 | 3 | 14,280 | 10,885 | 10,885 | 0 | 115 s |
| network18 | 4 | 530,145 | 370,545 | 370,545 | 0 | 4,527 s |
| network19 | 2 | 703 | 551 | 551 | 0 | 9 s |
| network19 | 3 | 16,872 | 13,062 | 13,062 | 0 | 226 s |
| network19 | 4 | 664,335 | 472,184 | 472,184 | 0 | 9,651 s |

**`network19`'s 488,895 is a certified radius-4 local optimum**, 485,797 LPs over 2 h 41 on one
core, every one abandoned by the dual simplex at the incumbent bound.

**472,112 is a certified radius-4 local optimum**: 381,920 LPs, 77 minutes on one core, every
one abandoned by the dual simplex as soon as the incumbent bound was reached. 30% of the
enumerated moves are not distinct valid topologies — a duplicated arc, a self-loop, or a
permutation that rebuilds the same topology — and are rejected before the LP.

That certificate was true and the hand-in is no longer that topology. §10 found **469,000** on
network18, and it differs from 472,112 in **ten** arcs — outside every radius any certificate
here can reach. The two results do not conflict: nothing within four arcs of 472,112 is better,
and the thing that is better was never within four arcs. It is the cleanest statement in this
report about what a local certificate does and does not buy.

The nine derangements of a 4-subset split into six 4-cycles and **three double transpositions**,
and a double transposition is exactly two disjoint 2-exchanges applied together — the compound
move the team's QAOA step was built to find. Radius 4 therefore contains all C(36,4) × 3 =
176,715 pairs of disjoint 2-exchanges on network18, and all C(48,4) × 3 = 583,740 of them on
network24, enumerated rather than sampled.

## 8. How much further the certificate can reach

Each radius costs a factor of thirty: radius 3 on network18 is 10,885 LPs, radius 4 is 370,545,
radius 5 would be about 11 million and radius 6 about 400 million. At the measured 12 ms a move
that is 37 hours and 55 days. Two ways out were measured, and neither works.

### A cheap lower bound in place of the LP (`bound_prune.py`)

The certificate only needs to know whether a candidate *can* beat the incumbent, so a valid
lower bound that already reaches the incumbent would let it skip the LP and stay a proof. LP
duality hands one over for free: for any nonnegative arc weights `w`,

    z(arcs)  >=  1000 * sum_ij d_ij dist_w(i,j) / sum_a w_a

Unit weights give the demand-weighted hop bound; the incumbent's own duals give the weights that
are optimal *there*, and the script checks that it reproduces the LP to six digits as a test of
the dual extraction. On a candidate the k new arcs have no dual, so they are filled with 0, the
mean or the max. 800 sampled candidates of network18 at radii 3 and 4:

| weights | ms | valid bound | bound / LP | pruned at the incumbent |
|---|---:|---|---:|---:|
| incumbent duals, new arcs 0 | 1.46 | yes | 0.849 | 33.5% |
| incumbent duals, new arcs mean | 1.22 | yes | 0.853 | 3.9% |
| incumbent duals, new arcs max | 1.15 | yes | 0.810 | 1.4% |
| unit (hop bound) | 1.10 | yes | 0.897 | 20.0% |

Every bound held on every sample, and every one is far too loose. Candidates around a polished
incumbent sit a fraction of a percent above it, so pruning needs a bound within a fraction of a
percent; the best of these is 15% low. The ceiling on the idea is **1.27x** — a third of the
candidates skipped at 1.5 ms against 12 ms for the exact answer. Radius 5 needs 20x.

The reason is the same one that sinks the hop count as a surrogate, and it is worth stating
because it generalises: the objective is a *max* over arcs, so the dual concentrates on the few
congested ones (13 of 36 arcs carry the whole dual on network18), and a radius-k move is chosen
precisely to disturb those arcs. The weights that certify the incumbent are the weights a
promising candidate invalidates. Re-tightening them per candidate *is* the dual simplex, which
is what the warm LP already does in 12 ms.

### Re-deriving the chain, and where the hand-ins' quality actually comes from (`node_deletion.py`)

The hand-ins network18–23 were built downwards, each from the next larger topology with a node
deleted. `node_deletion.py` goes back down and enumerates the whole step: delete a node, repair
the four broken degrees (two ways), relabel, score with the exact LP.

The demand matrices are exactly nested — `load_network_demand(n)` is `load_network_demand(n+1)`
cut to its leading block, asserted in the tests — so deleting the **last** node is the only
deletion that keeps every survivor's demand attached to its own label. The measurement says that
is also the best one, at every level and by a wide margin: on all 44 deletions of network22, the
chain's node gives 568,429 and the next best is 587,516, +4.5%.

What the full sweep shows is more interesting than the node choice. Delete one node from the
hand-in above, repair the degrees, and stop — no search of any kind:

| n | delete a node and stop | vs best known | our polished hand-in | vs best known | what polishing added |
|---:|---:|---:|---:|---:|---:|
| 18 | 481,400 | +1.70% | 472,112 | −0.27% | 1.97% |
| 19 | 502,600 | +1.96% | 488,895 | −0.82% | 2.80% |
| 20 | 529,967 | +0.27% | 519,343 | −1.74% | 2.05% |
| 21 | 568,429 | −0.15% | 562,362 | −1.21% | 1.08% |
| 22 | 610,880 | −1.17% | 604,540 | −2.20% | 1.05% |
| 23 | 642,084 | −2.62% | 639,739 | −2.98% | 0.37% |

**At n = 21, 22 and 23, deleting a single node from the solution one level up already beats
QOBLIB's published best-known value for that instance** — with no search at that level at all.
The level above is of course not free, but everything the hybrid chain then does at each level,
over hours of ILS, parallel tempering and QAOA sampling, is worth between 0.37% and 2.80%.

That is a statement about the instance family, not about the search: these demand matrices are
nested, so a good solution at n = 24 contains a good solution at every smaller n, and the whole
descending chain is carried by the quality of the one topology at the top. It also says where
the remaining effort belongs. Improving network24 by one point improves the floor of all six
instances below it; improving network18 improves network18.

The improved `network22` does buy its child a better start — 568,429 against the old 573,625,
two topologies three arcs apart — and a radius-3 descent from it is the one branch of the chain
that was never run with this ancestor.

## 9. A bound from below: how far from optimal can the hand-ins be? (`instance_bound.py`)

Everything above bounds the optimum from *above* -- a solution is an upper bound, and a
certificate says no nearby solution is better. Nothing so far says the optimum is not far
below. This does, for every 2-in/2-out topology at once, in microseconds.

1. Each unit of demand from i to j crosses at least `dist(i,j)` arcs, so the total load over all
   arcs is at least `sum_ij d_ij dist(i,j)`.
2. Out-degree 2 caps proximity: at most 2 nodes at distance 1 from i, 4 more at distance 2, 8 at
   distance 3 (the directed Moore bound). For a fixed multiset of distances the sum in (1) is
   smallest when the largest demands take the shortest distances, so sorting row i downwards and
   assigning it to 1, 1, 2, 2, 2, 2, 3, ... lower-bounds `sum_j d_ij dist(i,j)` for *every*
   topology. In-degree 2 gives the same argument on the columns; both hold, so take the larger.
3. The objective is the maximum arc load, which is at least the average, total / 2n.

Independently, all demand leaving node i must go out on its 2 arcs, so the busiest arc carries
at least `sum_j d_ij / 2`. On these instances that is weaker, but the maximum of valid bounds is
valid.

| n | QOBLIB reference | proven? | bound | bound / reference | our hand-in | provable gap |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 65,500 | yes | 63,000 | 0.962 | 65,500 | 4.0% |
| 6 | 101,000 | yes | 99,500 | 0.985 | — | — |
| 7 | 142,400 | yes | 133,000 | 0.934 | — | — |
| 8 | 170,231 | yes | 153,000 | 0.899 | — | — |
| 9 | 196,750 | yes | 173,000 | 0.879 | — | — |
| 10 | 210,800 | yes | 173,000 | 0.821 | — | — |
| 18 | 473,375 | — | 362,111 | 0.765 | 469,000 | 29.5% |
| 19 | 492,924 | — | 372,184 | 0.755 | 488,895 | 31.4% |
| 20 | 528,528 | — | 392,650 | 0.743 | 518,305 | 32.0% |
| 21 | 569,273 | — | 428,238 | 0.752 | 562,362 | 31.3% |
| 22 | 618,135 | — | 459,704 | 0.744 | 604,540 | 31.5% |
| 23 | 659,369 | — | 490,760 | 0.744 | 639,739 | 30.4% |
| 24 | 663,688 | — | 511,229 | 0.770 | 657,121 | 28.5% |

The bound holds on all twenty instances with a published value, which the tests assert, and
QOBLIB proves the optimum for n = 5..10, so its looseness is measurable where the truth is
known: **1.5% below the optimum at n = 6, 3.8% at n = 5, then 6.6%, 10.1%, 12.1%, 17.9%** as n
grows and both the Moore step and the max-versus-average step weaken. From n = 11 it flattens
at 0.74-0.77 of the best-known value and stays flat to n = 24.

Two readings, and only one of them is a proof.

The proof: **every network hand-in is within 28.5% to 32.3% of the true optimum**, and that is
the first statement in this repo that cannot be improved by a better search -- only by a better
bound. `network05` shows the scale: it sits 4.0% above the bound and is nevertheless optimal.

The inference, which is not a proof: the ratio is flat across fourteen instances where the
optimum is unknown, and decaying where it is known. A family of best-known values that drifted
away from optimal as n grew would show the ratio falling, not flattening. So the large instances
are not a different regime from the small ones -- but a uniformly good family and a uniformly
bad one look the same from here, and this bound cannot separate them.

## 10. Certified local optima are plural (`certified_ils.py`)

The re-derived chain start for network21 gave a negative result and a better idea. Descending
from it -- 568,429, the deletion of the improved network22 -- lands on **565,343, a certified
radius-3 local optimum that is worse than the 562,362 already handed in**. The experiment fails:
the improved ancestor does not buy a better child.

What it shows is that network21 has at least two proven radius-3 local optima, 3,086 apart, and
that the one we hold was reached by luck of trajectory rather than by any property of the
landscape. network05 makes the same point at a scale where everything is checkable: 10 arcs, 216
valid topologies, the optimum proven at 65,500 -- and a radius-3 descent from a kicked start
still stalls at 70,000, because radius 3 on 10 arcs cannot see the whole space either.

So the certificate is worth turning into an iterated local search. The team's ILS samples
2-exchanges, 3-rotations and cycle reversals and stops when it stops improving, which is a
statement about the sampler. Here the local search *exhausts* radius 2..3, so each stop is a
proven local optimum, and the certified radius tells the kick exactly how far it must reach to
escape: kicks of 5, 6 and 7 arcs on an incumbent certified to 4.

One detail the tests caught. A radius-k move does not always land k arcs away: every node has
in-degree 2, so two arcs in the chosen subset can share a head, and swapping those two heads
rebuilds the same pair. Inside the certificate that is harmless -- such a move was already
enumerated at a smaller radius -- but a kick that lands inside the exhausted radius is undone by
the first descent step, so `kick()` measures the distance and draws again.

Validation on network05, where the answer is known: 61 restarts, every one returning to 65,500,
one distinct optimum, spread 0.

### network18: 472,112 -> 469,000

| restart | kick | after the kick | descended to | LPs | time |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 548,858 | 474,038 | 13,818 | 196 s |
| 2 | 6 | 552,334 | 476,218 | 16,257 | 258 s |
| 3 | 5 | 516,500 | 472,112 | 14,307 | 182 s |
| 4 | 5 | 541,385 | 472,112 | 12,844 | 193 s |
| 5 | 5 | 567,000 | **469,000** | 37,538 | 565 s |

**469,000, gap −0.92% against QOBLIB's 473,375** — from −0.27%, in five restarts and
twenty-four minutes on one core. Checked by QOBLIB's own checker and installed as the hand-in.

Three of the five restarts landed on a *different* certified radius-3 optimum than the one they
left, which is the measurement the design was for: 472,112, 474,038, 476,218 and 469,000 are all
proven radius-3 local optima of the same instance, spread over 1.5%.

### network20: 519,343 -> 519,207 -> 518,305

The same search on network20, seed 20, kicks of 4 to 6 arcs. The first improvement came after
18 minutes, 519,207, seven arcs from the topology it left. The second came 95 minutes later:
**518,305, −1.738% → −1.934% against best known**, eight arcs from the first.

Both are outside radius 3, and outside the radius-4 sweep that was running on this very instance
when the first kick found it. Two improvements on one instance also answer the obvious worry
about a search with a 24-minute local step: it does keep moving, it is not a one-off.

### And the headline

**469,000 differs from 472,112 in ten of thirty-six arcs.** The radius-4
certificate on 472,112 enumerated 530,145 moves and was right about every one of them. The
better solution was simply never inside it. A certificate bounds a neighbourhood; it says
nothing about the instance, and the only way past it is to leave.

## What did not work, and why

| attempt | measured | reason |
|---|---|---|
| PCE on cell variables (arc f takes slot s) | loses to SA 55/60 | the poly encoding makes the r² cells rigid parities of ~√2·r qubits that ignore rows and columns: 2^14 reachable assignments for 3.6 M permutations |
| soft read-out of PCE (per-cell expectations → Hungarian) | no better than bitstrings | the expectations are far from the QUBO optimum: the optimizer, not the read-out, limits |
| swap variables (y = 1 if two arcs swap heads), exact second-order QUBO | without a trust region SA and PCE score worse than random | the QUBO optimum stacks many swaps, where third-order terms dominate; a cost per swap fixes SA, not PCE |
| quantum natural gradient | gap 0.204 at 6,782 circuits per block | no gain over Monte Carlo (0.172) or QN-SPSA (0.182 at 412 circuits) |
| weak-duality bound in place of the LP in the certificate | valid on 800/800 samples, 15% too loose, 1.27x ceiling | the objective is a max over arcs; the dual concentrates on the congested few and a radius-k move disturbs exactly those |
| deleting a node other than the last to re-derive the chain | best alternative +4.5%, the chain's choice +1.1% | the demand matrices are nested, so only the last node keeps every survivor's demand on its own label |
| QAOA at 25 qubits on qoro_gpu | 16 min and ~34 credits per step | each batch of 20 circuits is 84 s of GPU time with the 800-string exchange mixer; blocks of 5 arcs are small enough that every proposer finds the block optimum |

## Reproduce

From the repo root, with `venv/` from the README:

```bash
venv/bin/python TheGOAT_Testing/block_fidelity.py --n 24 --r 6 --blocks 4 --workers 2
venv/bin/python TheGOAT_Testing/block_fidelity.py --n 24 --r 7 --blocks 2 --workers 2
venv/bin/python TheGOAT_Testing/goat_lns.py --tag pce_r10_greedy --start greedy --solver pce --r 10 --steps 60
venv/bin/python TheGOAT_Testing/goat_lns.py --tag pce_r10_greedy_ls --start greedy --solver pce --r 10 --steps 60 --optimizer cobyla --iters 100
venv/bin/python TheGOAT_Testing/goat_swaps.py --tag sw_r16_greedy --start greedy --r 16 --iters 80 --trust 1.0 --steps 30
venv/bin/python TheGOAT_Testing/goat_lns.py --tag swap_r5_greedy_gpu --start greedy --solver qaoa_swap --r 5 --steps 30 --cloud qoro_gpu
venv/bin/python TheGOAT_Testing/optimizer_bench.py --blocks 5 --optimizers mc cobyla qng qnspsa lbfgs
venv/bin/python TheGOAT_Testing/optimizer_bench.py --blocks 5 --optimizers qnspsa_warm qnspsa
venv/bin/python TheGOAT_Testing/goat_lns.py --tag pce_r10_greedy_qnspsa_warm --start greedy --solver pce --r 10 --steps 20 --optimizer qnspsa --iters 60 --warm-angles 1
venv/bin/python TheGOAT_Testing/summarize.py

# regrow network24 from the team's network23, then descend with the team's ILS and PT
venv/bin/python TheGOAT_Testing/node_insertion.py --n 24 --from deliverables/network23/network23.sol
# bank the best insertion into a scratch folder (npl.bank), then, with NETWORK_SOL_DIR set to it:
python network_ils.py run --n 24 --minutes 12 --chains 2 --starts banked,banked --sync
python network_pt.py 24 --minutes 10 --workers 2 --replicas 10 --banked 10 --tag grow24

# block LNS from the stall point (copy runs/goat/sol_ins/network24.sol first), and the global hop search
venv/bin/python TheGOAT_Testing/goat_lns.py --tag A_quantum_r8_from666 --start-file runs/goat/sol_ins/network24.sol --solver pce --r 8 --steps 30 --optimizer qnspsa --iters 60 --warm-angles 1 --follow quantum
venv/bin/python TheGOAT_Testing/goat_lns.py --tag A_sa_r8_from666 --start-file runs/goat/sol_ins/network24.sol --solver pce --r 8 --steps 30 --optimizer qnspsa --iters 60 --warm-angles 1 --follow sa
venv/bin/python TheGOAT_Testing/hop_search.py --n 24 --minutes 6 --workers 2 --lp 150
```

Runs that start from the greedy topology bank into a scratch folder
(`NETWORK_SOL_DIR=runs/goat/sol_greedy`) so they never touch `solutions/`.
