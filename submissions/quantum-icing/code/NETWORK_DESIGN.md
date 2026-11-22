# QOBLIB 08 network design

Build a directed network with exactly two links in and two out per node. Route all demand and minimize the load on the busiest link.

See [the hand-in](deliverables/) for the final checked solutions and reports, and [the methodology guide](ai_context/README.md) for a walkthrough and run evidence. The reported solver implementations are in `network_design/`. Their original root commands remain available.

1. **Verify the pipeline.** Reroute the published `network05` topology and check that the exported `.sol` scores **65,500**.
2. **Bank a baseline.** Score one cycle-based and 16 seeded random `network24` topologies with the routing LP. Compute integer flows for the best candidate and check the `.sol`. The published target is **663,688**. [QOBLIB results](https://github.com/ZIB-AOPT/QOBLIB/blob/main/08-network/solutions/README.md)
3. **Improve the topology.** Try degree-preserving link swaps. Use Divi to propose batches of compatible swaps and compare with classical proposals under the same routing evaluation budget.
4. **Submit the best checked result.** Keep the `.sol`, score, gap, instance size, largest circuit width, and settings for reproduction.

The `network_design/` package contains the baseline generator, annealing search, and checked solution I/O. The commands below run from the repo root.

```bash
source venv/bin/activate
python network_search.py --nodes 11 --seconds 180 --seed 42
```

`network11` needs a score below **238,334** to set a new best. The search builds a baseline in `runs/network-baseline/` on first use and saves checked improvements in `runs/network-search/`. Repeat runs start from the best saved solution for that size. `python network_baseline.py` still verifies `network05` and builds the `network24` baseline, and `python network_search.py` defaults to `network24`.

Try larger moves from the saved `network11` solution with `python -m network_design.block_search --seconds 180 --block-size 6 --seed 42`. Each block keeps all other links fixed and searches degree-preserving replacements together. Checked improvements go to `runs/network-block-search/network11.sol`.

## Qoro pipeline

```bash
source venv/bin/activate
python -m network_design.qoro_search --nodes 11
python -m network_design.qoro_search --nodes 11 --cloud
```

The first command runs Divi on local Maestro. The second uses Qoro's cloud Maestro simulator and takes `QORO_API_KEY` from `.env` or the environment. Add `--characterize` to run Qoro's Characterization Service before QAOA. Both cloud features use credits.

Divi partitions the topology QUBO, runs a QAOA ensemble, and stitches samples by beam search. Degree repair and routing turn those samples into checked candidates in `runs/network-qoro/`. By default eight incumbent links are freed. Use `--free-links 0` for the full topology QUBO or `--nodes 24` for the largest instance. The output reports partition width, circuit count, and the best new candidate against the classical incumbent.

## Swap reachability and a better QUBO

The reported search rebuilt its shortlist around the current topology at each step. Within one circuit, QAOA picked three swaps from a fixed list of 16, with conflicting selections repaired before routing. [Experiment loop](network_design/divi_proposer.py#L337)

Greenhill's Lemma 2.2 guarantees reachability across simple regular directed graphs when each move may use links created by earlier swaps. A single fixed shortlist lacks that guarantee, even if it includes every swap available at the start. [Greenhill, Lemma 2.2](https://arxiv.org/pdf/1105.0457#page=9)

I would extend `qoro_search.py` by freeing a small group of links and assigning one bit per possible replacement link, with penalties for the remaining degree requirements. This can express coordinated rewirings that change an odd number of links. To guide the QUBO, set a congestion cap just below the incumbent and add cuts from the routing LP that rule out networks unable to meet that cap. Verify each proposed topology with exact routing and rebuild the QUBO after an improvement. [Focused adjacency QUBO](network_design/qoro_search.py#L25)

## Template labeling as a QUBO

Fix a directed graph template and optimize which traffic node occupies each position. Relabeling can change many labeled links in one move while preserving the template's degrees, diameter, and path structure. Our ILS already uses a two-node version of this move. [Existing relabel move](network_design/iterated_local_search.py#L237)

Let $x_{ip}=1$ mean traffic node $i$ occupies template position $p$. Add squared penalties so each node occupies exactly one position and each position hosts exactly one node. With demand $t_{ij}$ and template shortest-path distance $d_{pq}$, the quadratic assignment objective is

$$
\sum_{i,j,p,q} t_{ij}\,d_{pq}\,x_{ip}x_{jq}.
$$

This objective gives heavy traffic an incentive to travel fewer hops. The routing LP still scores the resulting topology by its busiest link. [Quadratic assignment formulation](https://www.jstor.org/stable/1907742)

In a `network24` pilot, three methods started from the same labeling of the 24-node Kautz template, at **821,937**. Each used ten blocks of six positions and scored six proposed permutations per block with the routing LP. The distance variants ranked permutations by classical enumeration. The LP-priced variant weighted template links using capacity duals from the current labeling. [Kautz construction](network_design/iterated_local_search.py#L44) · [Routing LP and duals](network_design/pipeline.py#L61)

| Proposal ranking | Best routing score after 60 evaluations |
| --- | ---: |
| Random permutations | 806,182 |
| Shortest-hop distance | **776,077** |
| Distance with routing-LP link prices | 783,895 |

The Kautz template has diameter four. Our checked **657,121** topology has diameter six, so diameter alone gives an incomplete picture of demand-specific congestion. I would use template labeling alongside link rewiring so the search can improve both placement and graph structure. A block of six labels has **36 assignment bits** and **720 feasible permutations**, making exact enumeration a practical control for Divi under the same routing evaluation budget. [Saved solution](deliverables/network24/network24.sol)
