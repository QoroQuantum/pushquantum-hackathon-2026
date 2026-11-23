# Pitch brief: portfolio optimisation (QOBLIB 06), 5 minutes

This is the brief for building the slides. Every number comes from `QUANTUM_STORY.md`, `README.md` or
`RESULTS.md` in this folder. Don't add numbers that aren't here; check them against those files.

## Context for the slide builder

- **How the challenge is judged** (challenge PDF, "How solutions are scored"):
  - **Scale:** the binary variables in the instance you solve, and the qubits in your largest quantum program.
  - **Quality:** the gap to the best-known value (BKV), `gap = (score − best) / |best|`. Lower is better; 0% matches the BKV and below 0% beats it.
  - A solution that breaks a rule doesn't count. The referee recomputes the score from the `.sol` file.
- **The problem:** pick holdings in n assets over T trading days, 0–3 units per asset and direction
  (long/short). The score is risk minus return, plus transaction, short-selling and liquidation costs,
  minus interest on cash. A budget limits the total units held on each day. λ weights the risk term.
- **Instances:** a010 / a050 / a200 / a400 = 10 / 50 / 200 / 400 assets; t10 / t15 = 10 / 15 days.
- **Story arc:** where to cut the problem → classical results → quantum at scale, tested against a
  fair control → an honest takeaway.
- **Tone:** factual, with no hype. Every quantum claim is measured against a control. Always say
  "simulated" for the MPS and `qoro_cpu` runs; nothing ran on quantum hardware.
- **Time budget:** about 5 slides plus a title slide; slide 4 (quantum) gets the most time.

---

## Slide 0: Title (no speaking time)

- Title idea: "Whole-day blocks: classical search for the scores, warm-start QAOA up to 1,200 qubits"
- Team name, challenge: PushQuantum 2026, Problem A, Portfolio optimisation (QOBLIB 06).

## Slide 1: What we hand in (≈ 30 s)

**Message:** a feasible, re-scored file for every instance, including the largest.

Content (three or four big-number tiles):
- **130** solution files, all feasible under the official referee (`qoblib_tools.py`).
- **Largest instance:** `po_a400_t15_orig`, 400 assets × 15 days = **36,165 binary variables**, score
  **−3,162,386**. QOBLIB has no BKV for a200 / a400.
- Also a200_t10: 12,110 binary variables, score −840,155.
- **Largest quantum program:** **1,200 qubits** (simulated). Slide 4 explains it.

Speaker note: "The challenge scores scale and quality, so here's both."

## Slide 2: Where to cut the problem (≈ 60 s)

**Message:** the problem is too big to solve in one piece, and cutting it by day loses almost nothing,
while cutting it by asset group loses almost everything. This one finding shapes the whole pipeline.

Visual: a simple two-column table or bar chart (share of |J| coupling weight lost by each cut;
`python asset_groups.py cuts`):

| Instance | λ | Lost when cutting by asset group | Lost when cutting by period (day) |
|---|---|---|---|
| a050_t10, B=20 | 1e-4 | 76–83% | **1.0%** |
| a050_t10, B=20 | 1e-3 | 77–84% | **0.1%** |
| a400_t15, B=100 | 1e-4 | 96.5–98% | **0.2%** |
| a400_t15, B=100 | 1e-3 / 1e-2 | 97–98% | **~0.0%** |

Points to say:
- Risk couples **every pair of assets on the same day**: the links are dense. Days are linked only through
  the trading cost between consecutive days, one small link per asset: the links are sparse.
- Smarter groupings (Louvain, spectral) only bring the loss on a050 down from about 84% to 77%.
- The solvers agree. Asset-group blocks of 6–12 assets barely improved anything; whole-day blocks did
  (e.g. a050 s02 went from a 3.07% to a 1.52% gap).
- **So our block is one whole trading day, with every asset in it.**

## Slide 3: Quality: classical search sets the scores (≈ 60 s)

**Message:** we match or beat the best-known value almost everywhere.

Content:
- Of the **128** instances with a BKV: **19 beat it, 108 match it, 1 is above it** (+0.045%,
  a050_t10_orig λ5e-4). A stacked bar or three tiles works well.
- **All 64 a010 instances are proven optimal** with an exact dynamic program over days (`exact_dp.py`).
  This certifies 50 values that QOBLIB lists only as "best known".
- How the scores were found, one line each:
  - **Whole-day pair-move search** (`day_swap.py`): the budget binds, so an improving move takes a unit
    from one asset and gives it to another. Steepest descent plus iterated local search, on every instance.
  - **Exact recombination of days across runs** (`recombine_runs.py`).
- Optional small callout: at λ = 1e-2 we beat the BKV by a wide margin (e.g. a050_t10_s00: −50,445 vs
  −12,340). Those BKVs are weak hedged long/short books. Every file is re-scored by the referee.

## Slide 4: Quantum at scale: warm-start QAOA on whole days (≈ 1 min 45 s, the core slide)

**Message:** we run QAOA on the natural block (one whole day) at up to 1,200 qubits, and test it
against a control that isolates the quantum cost layer.

### 4a. The program (left half of the slide, or a small diagram)

- One whole day as a QUBO, with the other days fixed. Each asset's net position takes **3 qubits** (a
  signed 3-bit encoding: every bitstring is a valid holding, and the block QUBO is exact).
  - a050 day = **150 qubits**, a200 day = **600 qubits**, a400 day = **1,200 qubits**
    (718,842 couplings: every pair of qubits is coupled).
- **Why warm-start:** standard QAOA from the uniform superposition gave **0 feasible samples in 500** at
  150 qubits, because the feasible region is a tiny corner of the space. Warm-start QAOA (Egger et al.,
  *Quantum* 2021) starts each qubit tilted toward the current solution's bit.
- Settings: p = 1, γ = 0.25, β = −0.4, tuned on 30-qubit a010 days.
- **Where it ran:** 150 qubits on **Qoro's cloud** (`qoro_cpu`; the cloud accepts at most 156 qubits per
  circuit), 10–15 programs per merged job. 600 and 1,200 qubits with **Maestro's MPS simulator on a
  laptop** (χ = 16; the 1,200-qubit circuit took 25–33 min and 2.2 GB).
- Mention in **one short line only**: "We lay out the all-to-all cost layer as a swap network, which made
  simulation 20× faster and 1,200 qubits feasible." No detail on the slide.

### 4b. The control: the "γ = 0 twin" (must be clear; it's the key idea)

- γ is the strength of the QAOA cost layer, the only part of the circuit that knows the portfolio problem.
- With **γ = 0** the circuit just reproduces its warm start. Each bit comes out as the current
  solution's bit, flipped independently with a small probability ε. That's a **classical random-flip
  sampler**.
- The **twin** is that γ = 0 circuit with the same block, warm start, ε, shots and scoring. Any
  difference is the work of the cost layer. It's a much fairer control than uniform random bitstrings
  (which give none feasible at this size).
- Suggested one-liner: *"Our control is the same circuit with the cost layer switched off. Every gain we
  report comes from the QAOA layer itself."*

### 4c. Results (right half: a small table plus the chart)

Sample quality by width (QAOA vs γ = 0 twin, same shots; Δ = cost above the current day, lower is better):

| qubits | block | feasible share, QAOA / twin | mean Δ of feasible samples, QAOA / twin |
|---:|---|---|---|
| 150 | a050_t10_s00 day 3 (cloud-size, from the empty portfolio) | **0.873** / 0.711 | **+76,479** / +101,035 |
| 600 | a200_t10_orig day 4 | **0.185** / 0.061 | **+3,576** / +9,219 |
| **1,200** | **a400_t15_orig day 7** | **0.179** / 0.085 | **+2,024** / +11,439 |

- Overall: at 150, 600 and 1,200 qubits QAOA's samples are **1.2–3× as often feasible** and their mean
  cost is **20–82% lower** than the twin's.
- **Hybrid loop on Qoro's cloud** (`warm_day_qaoa.py loop`): start from the empty portfolio. Each round
  samples every day in one merged cloud job and keeps the best improving sample per day. The twin loop
  does the same with γ = 0 samples. **QAOA is ahead of its twin after every round, on all 9 loops**
  (a050, λ 1e-4 to 1e-3). Example after the last round: −334,346 vs −233,287 on a050_t15_orig λ5e-4.
- **Visual: `handin/qaoa_loops.png`** (objective per round, QAOA vs twin, one panel per loop). This is the
  slide's main image.
- Cost: 9 loops, 112 rounds, 1,360 circuits of 150 qubits = **721 credits** (≈ 0.53 credits per circuit).

## Slide 5: Honest takeaway (≈ 45 s)

**Message:** the quantum layer helps measurably, but classical search still sets the scores.

Points:
- **What quantum gives:** at every width we tried, up to a whole day of our largest instance, the QAOA
  cost layer moves probability toward feasible, cheaper portfolios compared with its exact classical
  twin. In the cloud loop that means a much faster descent.
- **What it doesn't give:** a better final score than classical search. It is a local sampler around the
  current solution, and classical whole-day search finds the same improvements, and more, in
  milliseconds. No QAOA sample improved an already polished hand-in day.
- One hand-in file comes from the hybrid pipeline: **a050_t10_s00 λ1e-3, −192,036** (BKV −191,879):
  8 cloud QAOA rounds, then classical polish. Classical search alone reaches −191,991, so the polish
  does most of the work. Say this plainly.
- Closing line idea: "We found the cut that makes quantum blocks meaningful, whole days, ran QAOA on
  them at up to 1,200 qubits, and measured it honestly. Classical whole-day search sets the scores."

---

## Backup slides (Q&A only, not in the talk)

1. **Swap network / simulability:** Divi emits the cost layer's RZZ gates in an order that spans the
   whole MPS chain (440 s locally for one 150-qubit circuit). The gates commute, so we reorder them as a
   swap network (n layers of nearest-neighbour RZZ+SWAP): 22 s, 20× faster. It produces the same
   samples: checked against Divi's own QAOA on 30-qubit days, feasible share 0.757 vs 0.761 and
   0.775 vs 0.776. χ = 16 vs χ = 64 at 150 qubits: feasible share 0.873 vs 0.857, same direction against
   the twin.
2. **Full hybrid loop table** (9 loops, with the polish and classical columns): copy from
   `QUANTUM_STORY.md`, "Hybrid loop on Qoro's cloud".
3. **Earlier experiments and dead ends:**
   - 9-qubit QAOA block solver (TQA start, local): reaches the BKV on a010 λ1e-3 (−8,397), but is
     throughput-limited on a050.
   - 30-qubit uniform-start QAOA on whole a010 days (`qoro_gpu`, 20 programs, 65 credits): 8× more
     feasible shots than random, best sample 5× closer to the day optimum, but 0/10 day optima.
   - Batched cloud probe on a400 (100 × 9-qubit programs in one ensemble): 7 submissions whatever N
     is, 98–99% block optimum, but about 3.6 s per block on the cloud vs 0.09 s locally.
   - PCE failed (unary encoding, and whole 50-asset days); uniform-start QAOA at 150 qubits gave no
     feasible sample.

## Likely questions and prepared answers

- **"Is 1,200 qubits real hardware?"** No. It's simulated with Maestro MPS (χ = 16) on a laptop, and
  `qoro_cpu` is a simulator too. The scale criterion counts the qubits in the quantum program; we say
  "simulated" every time.
- **"Isn't the twin a weak baseline?"** Yes, on purpose. It isolates the effect of the cost layer. Against
  strong classical search QAOA doesn't win, and slide 5 says so.
- **"Your λ = 1e-2 gaps (−100% to −300%) look too good."** Those BKVs are weak hedged long/short books.
  Every file is re-scored by the official referee, and the reference solutions re-score exactly to their
  published values, so the gaps are like for like.
- **"Does hybrid beat classical at the same budget?"** Only the polish step is budget-matched (better on
  4 of 9, equal on 5). The QAOA rounds themselves took extra cloud time and credits. With more classical
  seeds, classical search catches up everywhere except the one hybrid hand-in file.
- **"Why not uniform-start QAOA?"** At 150 qubits it gave 0 feasible samples out of 500, no better than
  random bitstrings.

## Assets available

- `handin/qaoa_loops.png`: QAOA vs γ = 0 twin, objective per round, 9 loops (slide 4).
- Tables in `QUANTUM_STORY.md` (cut table, sample-quality table, loop table) and `README.md` (headline
  rows) for charts.
- No figure exists yet for the cut table (slide 2) or the 19 / 108 / 1 split (slide 3); the slide
  builder should make simple bar charts from the numbers above.
