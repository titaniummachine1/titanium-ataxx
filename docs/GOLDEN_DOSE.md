# GOLDEN DOSE — what actually produced champions elsewhere, and how to port it

Sources read 2026-10-20:
- `C:\Users\Terminatort8000\Downloads\ACE_APEX_ARCHITECTURE.md` (Quoridor "ace apex", 26 Sep 2026)
- `C:\Users\Terminatort8000\Downloads\apex_rows\README.txt` + `nodes.jsonl` + `screens.jsonl`
  (search-labelled export, 30 Sep 2026)
- `C:\Users\Terminatort8000\Downloads\apex_golden_1300k_values.csv`, `apex_p1_sweep_values.csv`
- Moonbird README (tsoj, Nim Ataxx engine)
- Stockfish NNUE wiki "Training datasets" + fairy-stockfish `generate_training_data.md` + TalkChess

## 1. ACE APEX (Quoridor) — the recipe that won

Architecture: AlphaZero-style PUCT MCTS + 8x512x8 transformer (25.6M params), 145 tokens,
bidirectional attention, value head tanh, policy over 209 actions. Rust batched MCTS; exact race solver.

Training loop:
- Continues from the champion, NEVER from scratch. MuonAdamW.
- LR = 0.05x nanochat base on a warm start (full LR was tried and FAILED — NANO-LR-1):
  matrices 1e-3, embeddings 1.5e-2, head 4e-4, scalars 2.5e-2; warmup 40 steps, flat,
  linear warmdown over the last 65% to 5%.
- Window = newest self-play + previous loop rows (~27% replay) + golden rows.
- **Value target = 0.5 x game result + 0.5 x search value.** (Outcome anchoring: the
  anti-echo-chamber device. Pure search targets drift; pure results are noisy.)
- Policy target = search VISIT COUNTS, not first-guess x Q (that target keeps blind-spot
  moves starved forever).
- Rows deduplicated by position (mirror-folded): visits summed, values averaged.

Golden rows (the method that produced the last two champions):
- Deep searches at **1.3M sims/move** along the main opening line and at the face-off.
- A position becomes golden only if the search put **>= 10,000 visits within 6 plies**
  (1,547 positions qualified).
- Each golden row written **20x** in the dataset (so it is not outvoted) and oversampled
  to **3-10% of training draws** ("golden dose"; 10% was the best of the sweeps).
- Effect: the face-off move went from 0.03% to 99.9% of the net's first guess; its value
  went +0.20 -> +0.56.
- **Lesson (binding): lightly searched deep positions (500+ visits) HURT as training
  targets, dosed or not. Only well-searched positions help.**
- Whole-tree data experiments (golden3 / set1 / mix1) scored 47.5-48.4% vs champion and
  were NOT promoted. Curated+deep beats bulk.

Sweeps (blind-spot finder): every legal move at a position gets its own 100k search, then
the top moves (within 0.1) get 1.3M. Prevents the net from hiding a move from itself.

Evaluation/promotion: 640 games = 320 openings x both seats, 400 sims/move, paired
bootstrap CI. Promote if score > 50% vs champion. Early stop only if promotion is
practically impossible (<1%) or mathematically impossible. Fixed anchor (older champion b3)
as yardstick; diagnostics (held-out loss, face-off probe) never promote alone.

Data schema actually exported (worth copying):
- `nodes.jsonl`: state, side_to_move, visits, value, children[{move, visits, value}].
- `screens.jsonl`: state, side_to_move, moves[{move, value, sims}], best_move, best_value, net.
- Values are side-to-move, [-1, +1]. Duplicates merged. **Only trust children with many
  visits; a child with 1-10 visits is a near-raw net guess, not a search result.**

### Pack 2 (exported 2 Oct 2026) — the scalable shape

- `selfplay.jsonl.gz`: 1,677,689 positions from apex's own games, each with `visits` per move
  (policy target), search `value`, and `game_winner` — "use this as the bulk of a fresh
  training run instead of noisy self-play".
- `nodes.jsonl.gz`: 150,284 deep-search positions (only heavily-visited children trusted).
- `screens.jsonl.gz`: 1,468 positions with every legal move searched (most reliable labels).
- `quick.jsonl.gz`: 98,341 cheap 50-sim labels — "good for breadth at a LOW weight; values
  from short searches are squashed towards 0".
- Suggested use: value from selfplay value (or mix with game_winner) + nodes/screens
  best_value; policy bulk from selfplay visits, screens/nodes at higher weight; mirror for
  2x; hold out ~5%; **judge by games, not loss**.

## 2. Stockfish / Moonbird (context)

Stockfish NNUE data generation: self-play from an opening book with randomized multi-PV moves
(`random_multi_pv 4`, diff 100), fixed depth ~9 or fixed nodes ~5000/sample; `PruneAtShallowDepth
false`; eval_limit to cut mate-adjacent rows. Best nets mix Stockfish self-play data with
Lc0-derived rescored data; "better evaluations do not always give better training" (quality vs
learnability). TalkChess: search-only targets destabilize after a few iterations; fixed-depth
self-play + mixing in the game result allows many iterations. "Wrong binpacks" (static eval
disagrees with search eval) are valuable rows.

Moonbird (Ataxx, superhuman): `genData --run Moonbird 6_000 50_000_000 false` = self-play data
generation with **6000 search nodes per sample**, **50M samples**, 4x2 and 2x4 tuple tables tuned
with SGD, plus SPRT and dedicated eval/search tuning loops.

## 3. Port to Titanium Ataxx (concrete, in priority order)

1. **Golden rows, not bulk deep labels.** Pick a small curated set (opening/book lines and
   face-off positions, plus the highest-wsum / highest-residual rows). Deep-search each at
   100k-1M nodes with the current champion. Keep only rows where the search is CONVERGED
   (best-move share high, e.g. >= 0.9, and the root score stable across depths/nodes).
   Write each 10-20x and dose at 3-10% of each training mix. Shallow 5k labels remain bulk
   data only — never dressed up as golden.
2. **Outcome anchoring.** Value target = a x real outcome + (1-a) x search score, with a in
   [0.3, 0.5] justified by Apex. (Run B tonight starts at 0.1; the Apex evidence says go higher.)
3. **Dedup by position** (sum wsum, average teacher, mirror-fold). cache12M surely repeats
   transpositions; dedup shrinks the effective dataset and sharpens targets.
4. **Record provenance in every row** (the user's requirement): occ0/occ1/blk/stm, teacher
   score, nodes_used, depth_reached, best_move + share, engine binary hash, net hash, source
   (selfplay / golden / arena), real outcome, visits/wsum. Turso (libSQL) fits this schema and
   keeps it queryable; JSON only for Botzone wire format, never in the hot path.
5. **Self-play tree collection**: while generating self-play games, store the root tree
   (children + visits + values) at fixed nodes, like apex `nodes.jsonl`. Use only strong
   children as targets, anchor values with outcomes, and keep the golden dose small to avoid
   the echo chamber. Deep-tree vs shallow-tree alignment is exactly the Apex value-target
   idea: shallow search = cheap bulk, deep search = trusted labels, results = anchor.
6. **Gate like Apex**: >= 640 games over many openings x both seats with paired bootstrap.
   Our fixed-node gates are cheap (~17 s per 104 games), so large-N is affordable.
7. **Do not chase megasearches on CPU before the dose experiment.** Apex shows the CURATION
   and DOSE matter more than raw depth everywhere: 1,547 golden rows moved the champion.
   Budget: curated golden set (thousands of rows at 100k-1M nodes) is CPU-feasible; full-corpus
   deep relabel is not needed ("we don't need so much deep labels").

## 3b. Quoridor_best_AI teacher doctrine (TEACHER_DERIVATION + V7 plan), distilled

- **Source routing by reach: exact verdict > outcome > own-search.** Exact solutions override
  everything; outcomes are cheapest and least biased where continuations are SHORT (backward
  starts); own-search is a dense residual initializer, last in the registered hypothesis.
- **Registered blend: 0.75 x search + 0.25 x outcome** (V7 `L*=16B`, 800 ms/mode-L) — a
  conservative pre-registered baseline, not a proven universal optimum. Per-cell weights
  are the derived answer; a single global weight is not automatically optimal.
- **Outcome weight and outcome count move together**: at k=3 independent continuations the
  outcome SD bound is 1/(2*sqrt(k)) = 0.289; keeping equal noise when raising the weight `a`
  requires k >= 3*(a/0.25)^2 (k>=27 at 75%, k>=48 at pure outcome). Never raise outcome
  weight while keeping k=3 and claim equal precision.
- **N=3 unanimity** applies to every atomic own-search label (same move+score three times) —
  search reproducibility is checked separately from outcome sampling.
- **Far from terminal, the gauntlet (games) is the only unbiased estimator.** Label-fit
  metrics (teacher correlation, MSE, agreement) are diagnostics and can move while strength
  does not (exactly what our C0/C1/C2 gates just showed: better teacher agreement, no Elo).
- **Backward starts**: a label 2x closer to terminal is 2x cheaper and (under the model)
  4x better in variance-per-dollar. Economics: labels are the bottleneck, not training.

## 3c. Replay-until-learn / regret mining (owner idea, 2026-10-20)

Idea: play one self-play game, then mine it deeply instead of discarding it.
Pipeline (CPU-feasible with the current CLI, design ready):

1. Play a game at N nodes; log per-ply best move + eval (`bestmove` and `label`).
2. Find suspect positions: eval swings above a threshold, or where a deeper re-search picks a
   different move. These are the moments that decided the game.
3. Deep-mine each suspect: re-search at 10-50x nodes (optionally give the LOSING side the
   deeper search; "until it actually wins" = adversarial deepening). Record:
   position, shallow move/value, deep best move/value, regret = deep value(chosen) - deep
   best value.
4. Train on the mined rows: value target = deep value (or 0.75 search + 0.25 outcome, or
   outcome-anchored 0.5/0.5), oversample the high-regret rows as a golden dose (3-10%),
   dedup positions, and gate by games only.
5. Regret weighting: sample rows proportional to regret (clipped), so the net learns exactly
   the moves it was blind to — the Ataxx analogue of Apex sweeps and Stockfish "wrong binpacks".
   Antidote to echo chamber: outcome anchor + small dose + gauntlet promotion rule.

Why it fits Ataxx: games are volatile (flip avalanches), so critical moments are few and
high-signal; deep-mining those positions is cheap on this box (label --nodes 20000 measured
34.7 pos/s/worker; ~0.14 s/pos at 100k, ~1.4 s/pos at 1M single-thread). A few thousand
mined rows at 500k-1M nodes is ~10-40 min with 6 workers — the definition of a golden dose.

Caveat from the record: lightly searched labels hurt; only CONVERGED deep labels help
(best-move share high / stable across depths). Mine with verification, not one shot.

## 3d. Tree tightening / PV distillation (owner idea, 2026-10-20)

Idea: when a deep search produces a PV line, train the evaluation on that line so the eval
"tightens" toward the PV; then a shallower search is more likely to reproduce the same PV.

Research grounding:
- Supervised distillation of strong engines into AlphaZero-shaped nets works and is cheap
  ("Distilling AlphaZero on one GPU", 2026): 5M labeled positions -> 1810, 30M -> 2009,
  deeper search at play time + deeper training -> 2301. No self-play loop required.
- AlphaZero/Lc0 train the policy toward SEARCH VISIT distributions, not the single PV move;
  the PV is the sharpest part but not the whole signal.
- WARNING (arXiv 2609.37447, efficient self-play under limited compute): "closeness to a deep
  policy does not measure how much a target will improve the next network. Easy-to-predict
  targets can agree with deeper search while teaching little." Their controller improved
  policy agreement but produced WORSE learning. This is exactly the val-lies/agreement-lies
  pattern we measured tonight (C0/C1 agree better with the deep teacher, do not win games).
- Therefore: PV tightening must be (a) restricted to CONVERGED deep searches, (b) dosed small
  (3-10%), (c) outcome-anchored, and (d) promoted only by games, never by agreement.

Concrete Ataxx form:
1. Probe set: positions where shallow (5k) and deep (200k-1M) best moves DIFFER. Measure
   shallow->deep agreement before/after training as a diagnostic only.
2. Targets on PV rows: value = deep search value (optionally 0.75 search + 0.25 outcome).
   Add a margin term: make the eval of the PV child exceed its siblings by a calibrated
   margin (ranking/pairwise loss on (PV move, alternative moves) after each move).
3. Oversample PV/disagreement rows (golden dose) and dedup by position.
4. Track: (i) agreement on held-out probes (diagnostic), (ii) gate vs previous champion at
   5k and time controls (the only promotion evidence).
5. The point is not to imitate the deep move per se: it is to move the eval's decision
   boundary so shallow search stops needing the deep tree to find the same move — more
   strength per node, which is what time-control gates reward.

### Owner ratio (2026-10-20)

Effort split: **>=50% on outcome-anchored value**, the remainder on deep/PV precision
("golden dose" as Apex found). Rationale: while tightening the eval to fit a deeper PV, the
net must not lose much on predicting the actual game outcome — the outcome anchor is what
stops the echo chamber.

Experiment grid queued tonight:
- Run C: outcome-weight 0.50, golden dose 3% (3% tested first, Apex best was 10%).
- Run D (queued): outcome-weight 0.25 (Quoridor 0.75 search / 0.25 outcome doctrine),
  golden dose 6% — tests the dose axis at the doctrine blend.
Both warm-start from champ, blend 1.0 (pure teacher column, deep labels on golden rows),
promotion by 5k gates on fresh paired-book lines only.

## 4. Where tonight's evidence fits

- Full-corpus 5k-node retrain (tonightA_fixed, all families): 5k gate vs main = +6 Elo at
  2008 games (straddle). The 5k labels are "lightly searched" in Apex terms.
- The extra heads (q2 active -30, capconv+pa -71) hurt: consistent with training on
  unconverged/uninformative targets (Wfeataudit2: pa/capconv carry ~zero real-outcome signal).
- Conclusion: stop feeding raw 5k residuals into more heads. Spend the next data budget on
  CURATED, CONVERGED, DEEP-labeled rows + outcome anchoring, dosed 3-10%.
