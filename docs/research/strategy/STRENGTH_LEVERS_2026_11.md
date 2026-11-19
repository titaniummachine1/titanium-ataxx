# Strength levers, ranked by measured evidence (2026-11-14)

Everything below is ranked from what this repository actually measured, not from
what is fashionable. Each item states the evidence, the next step, and the cost.

## Where Titanium stands after the 2026-11-13/14 session

| area | verdict | evidence |
|---|---|---|
| search selectivity (E1/E2/NMP/per-phase) | **exhausted** | 10 timed gates, all neutral or negative; none reached PASS |
| SPSA on lmr/fut/irr | **no gradient** | 38 iterations, mean R −0.025 ± 0.026 |
| NNUE on the shipped dataset | **saturated** | val 0.018171 (lr 3e-4) and 0.018000 (lr 1e-4), both early-stop |
| label depth | **5k is the knee** | teacher moves 31 cp RMS from 1k→5k but only 21 cp 5k→20k; static-to-label 60/61/63 cp |
| lambda (temporal) targets | **negative** | lambda=0.9, champion weights: −12.2 ± 26.6, LLR −2.41 |
| SEE (ordering / eval / guard) | **rejected 3 ways** | −31.5, +0.6 (+9% slower), −127 (real-game NPS −24%) |
| policy head | **improved (merged)** | TPOL5 SEE-off: 36 138 → 28 146 nodes, +6.4 @100 ms, +23.2 @1000 ms |
| book bias | **not the limiter** | split rate ~50% uniformly across lines; pruning 2.9% of lines moved decisive share 49.9%→52.1% |
| max-NPS kernels | **prototypes work** | 67–180 Mnps; D4-canonicalised variants reach depth 8.0–8.1 @100 ms vs 6.9 |

The single most important reading: **the eval is saturated and the search knobs
are flat.** More of either does nothing. Strength has to come from *different
data* or *different information*, not from more tuning.

## Ranked levers

### 1. Fresh positions from the current engine (highest expected value)
The dataset's *positions* come from games played by an engine ~1000 Elo weaker
than today; only the labels are current. That biases the state distribution.
`ti data games-tdat` (added 2026-11-14) is the missing link: games → relabelled
5k rows + chains + manifest, stamped `relabel_src` so generations stay filterable.
- **Next step:** generate ~30k games at 100 ms with the current engine from book
  lines that reach the weak buckets (e25–34), convert, fine-tune, gate at 100 ms
  and 1000 ms.
- **Cost:** ~2 h of 8-core self-play + cents of GPU. **Judge:** gate only.

### 2. D4-canonicalised search (best measured search idea left)
In the max-NPS prototypes the D4-canonicalising variants (`brute-c3/c4/c5`)
reach **depth 8.0–8.1 in 100 ms** versus 6.9 for the plain kernel, because the
canonicalised tree is ~2.4× smaller (8.7M vs 21.2M nodes at depth 8) — the win is
transposition merging, not raw speed.
- **Next step:** port TT canonicalisation (min over the 8 symmetries of the
  position key) into the real search behind a flag; defaults bit-identical; gate.
- **Risk:** hashing cost per node; measure NPS first. **Upside:** >1 ply at 100 ms.

### 3. Exact endgame proofs under Botzone rules (`exp/endgame-solver`)
The dataset's 2.44M proven rows came from Zataxx under **classic** rules (pass when
stuck, 100-ply draw rule), while Botzone ends the game the moment either side is
stuck and the mobile side absorbs the empties (D197). A proof derived under the
wrong terminal can disagree with what Botzone actually scores.
- **Next step:** run the exact solver over proven rows with ≤4 empties, count
  mismatches, and re-prove the wrong ones. Exact solver already committed with
  2-ply/terminal semantics and two tests.
- **Cost:** minutes for 4 empties. **Upside:** removes wrong hard labels in the
  phase where the eval is most used.

### 4. A trained SEE/volatility table (only SEE form left)
SEE as a leaf eval costs 1800× (measured); as a policy input it lost 31 Elo. The
remaining honest form is a **table** indexed by (phase bucket, my conversions,
opponent's best reply), fitted offline like TPOL5 — i.e. pay SEE once per row, not
per node.
- **Next step:** build the table from the dump's `backed − stat` labels and gate a
  policy head that uses it *only* where it disagrees strongly with the pattern
  ordering.

### 5. Capacity, only after data (last)
W=512 lost 344 Elo historically; Net2Net recovered. Capacity is the right lever
only once the dataset stops being the limit. Do not start here.

## What not to do (measured dead ends)

- More epochs on `train_v9` (saturated at two learning rates).
- Deeper labels (20k buys 21 cp of teacher movement for 4× compute; 1k adds noise).
- More search-knob tuning (E1/E2/NMP/SPSA all flat).
- SEE as a per-leaf eval or as a policy input.
- Rebalancing the opening book by decisive rate alone (uniform ~50% split; no
  concentration to remove).
