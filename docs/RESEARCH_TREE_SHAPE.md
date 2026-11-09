# Tree shaping per game phase (user D494 / D533 / D547 / D549, 2026-11-07)

Goal (user): per game phase, find which nodes the search **over-prunes** (cut or reduced, but a deeper search shows
the line mattered) and which it **under-prunes** (searched in full, but the result was never close to the window),
then fit every pruning knob per phase so the tree is spent where it changes the move. Objective: lower value lost
per move (CPL) in every phase at the played clock; screened with `ti regret`, decided by `ti gate` (5k, then 100 ms).

## 1. Phases
16 phases by empty squares (the board fills monotonically except for jumps, so empties is the natural clock):
`phase = clamp((45 - empties) / 3, 0, 15)`; phase 0 = 43-45 empties (opening), phase 15 = 0-2 empties.
Each knob becomes a 16-entry table whose default entries equal today's constant, so the default build stays
bit-identical (node counts are checked before any gate).

## 2. Knobs on main (crates/titanium/src/search.rs, origin/main)
| knob | where | today |
|---|---|---|
| RFP (reverse futility) | node entry, depth <= 4, ply > 0 | margins 257/347/478/774 + dis x 32/128 |
| NMP (pass null move) | depth > 2, >= 11 clone targets, fill < 0.54 | R = 3 |
| ProbCut | depth >= 6, null window | R = 3, margin 140 + dis x 32/128 |
| IIR | no TT entry, depth >= 4 | -1 ply |
| LMR (Moonbird) | from move index 6 | depth x 47 / (47 + n reduced) |
| futility reduction | from move index 1 | -1 ply per 57 cp short of alpha (minus dis credit 24/128), skip when <= 0 |
| SmallNet prefilter | before the child NNUE eval | lower bound - 0 cp |
| dis extension | child dis > 300 | +1 ply back |

## 3. Tool (plan)
1. **Trace build** (`cargo feature trace`, compiled out of every normal build): every negamax node and every skipped
   child is a record: hash, parent, move, ply, depth in / searched depth, alpha, beta, static eval, dis, score, which
   knob fired and its slack (how far inside its margin it was), iteration number. A 500k-node tree = ~20 MB.
2. **`titanium-cli trace`**: dump the tree of a position at a small budget (50k-100k nodes) and a big one (200k-500k).
3. **Diff along the deep PV** (the user's idea): where the deep and shallow searches pick different moves, walk the
   deep PV through the shallow tree by hash; the first PV move that the shallow tree skipped or reduced names the
   knob that over-pruned (and the phase, depth and slack where it happened).
4. **Per-knob verification**: a sample of fired prunes is re-searched with that knob off at that node (same depth, and
   depth + 2 as the deeper truth): false-prune rate and cp cost per knob x phase x slack bucket. Under-pruning = nodes
   searched in full whose result missed the window by more than the knob's margin: nodes that could have been saved.
5. **Fit** per phase: move each margin until the marginal false-prune cost equals the nodes saved; pad toward today's
   value (generalisation); screen with `ti regret` per phase; gate.

## 3b. Tool as built (exp/tree-shape, worktree C:/gitProjects/tree-ataxx, from origin/main 5d2eead)
- `crates/titanium/src/trace.rs` + `trc!` hooks in search.rs, Cargo feature `trace` (the default build compiles the
  hooks out; `negamax` is an inline(always) wrapper of `negamax_body`). One record per negamax call (parent, move,
  via first/scout/research/null/probcut, move index, depth in, depth after IIR, full depth wanted, window, static
  eval, dis, score, how it ended, knob slack) + one record per child skipped without a search (small net, futility,
  repetition). The trace keeps one ID iteration (cleared at the iteration start).
- `Searcher::pv_from_tt` (PV walk through the TT).
- `titanium-cli treediff --games G.jsonl [--max-games 40] [--every 4] [--per-phase 12] [--snodes 50000] [--dnodes 500000] [--out PREFIX] [--dump N]`
  (build: `cargo build --release -p titanium-cli --features trace`): deep reference search, shallow search rerun at
  the deepest depth that completes within snodes with the trace on, deep PV walked through the shallow tree.
  Files: PREFIX.pos.tsv (per position: phase, depths, nodes, both moves, agree, first PV event), PREFIX.pv.tsv (per
  deep-PV ply: full / reducedR / skip_sn / skip_fut / cut_by_sibling / node_rfp / node_nmp / node_probcut /
  node_tt_cut / horizon), PREFIX.nodes.tsv (phase x kind: count, subtree nodes; null/probcut searches that cut vs
  failed; RFP near misses by gap bucket and depth), PREFIX.treeK.tsv (full shallow tree dumps).

## 4. Results
### r1 (D555): 260 positions, deep 500k vs shallow 50k-depth, logs/tree/r1.*
See D555: null move cuts only 3.5-3.9% of its tries in phases 0-7 and its failed searches are 19-26% of all nodes; in the dumped trees no cut below static >= beta+100. ProbCut 49% success in phases 12-15, 11% in 0-3. Disagreements come with the opponent's deep reply not reached (sibling cut) 14.7% vs 2.9%; deep-PV skips are rare.

## 5. Not analysed yet
Everything above; no tree has been dumped yet (2026-11-07).

### r2 (D559) and r3 (D560): knob buckets; r3 by NNUE bucket (logs/tree/r3.buckets.txt)
- Search knobs are now ALL per NNUE bucket (exp/tree-shape 5078c12, user D558): 19 knobs x 16 buckets, bucket =
  empties*16/50 (as nnue.rs; 14 = start, 0 = full board), file `--knobs data/knobs/knobs_<sha8>.txt`
  (`name = v` or 16 values, bucket 0 first). Default = bit-identical to main.
- Null move (only buckets 7-14 run it): tries with static < beta+200 cut 0-3% and cost 5-18% of the tree
  (b7 5.0, b8 10.4, b9 12.3, b10 12.2, b11 17.5); +200..300 cuts 9-27%; >= +300 45-52%.
- ProbCut: still 24-46% cuts below pc_beta in b0-b6: no clear waste.
- Move index (user D556): raise-alpha at index 6-8 = 5-6.5% in b1-b11 (flat), 1.3% in b0 (0-3 empties), 4.7% in b12.
- Estimate (D559): +15..40 Elo total at the played clock, mostly null move (+5..15).
- Candidate A = knobs_89EE6ED4.txt (nmp_margin 200 in buckets 7-15): 5k gate logs/gates/d560_nmpA_5k.out (running).
- r4 (running): all 118 gate-game files (stride 13) + 300 opening playouts, 80 positions per bucket, logs/tree/r4.*.

### r5 (D567): 600 positions, all buckets (gate games + opening playouts), logs/tree/r5.summary.txt
- Legal moves per node: b0 9.6, b2 30.8, b4 46.5, b8 61.3 (peak), b12 42.3, b14 23.1.
- Raise-alpha by relative index falls much faster in the endgame (b0 at 6/10 of the list: 2.3%; b8 2.8%).
- Clone vs jump: ahead -> clones raise alpha more (index 1-3: 40-44% vs 30-36%); behind -> about equal in b0-b11,
  jumps better in the opening b12-15 (22.0% vs 17.4%). Matches the user's experience.
- Candidates queued at 5k vs main 285a127: B (lmr_min 3 in b0) +14.3 +-17.4 at 438 g (LOS 94.6%, running),
  C (tighter RFP b4+), D (user LMR schedule 3..13), E (clone killers by destination), F (ordering bias ahead/behind).

## 5. Caches (TT + eval cache), D570-D592
- Census over a real game at the Botzone budget (D575): the 4M-entry TT is never full (23% empty at the end), ~50% of
  occupied slots dead late (fewer stones than the root, user rule D571); the 64k eval cache is 100% full and thrashing.
- Botzone memory (D573): Titanium ~450 MB, other bots report exactly 512 -> limit >= 512 MB; budget kept equal.
- Eval cache size (D579, old layout): 2^16 18.6% hits / 30.69M NNUE evals; 2^20 35.7% / 25.02M; 2^22 39.5% / 23.68M.
- Ranked eval cache (D585/D586/D588, --ecache-mode 1): 8 x 8-byte entries per 64-byte line (32-bit check = 25 hash
  bits + importance bit + 6 bits empties), victim = empty > dead > unimportant (beyond the top-12 moves) > LRU,
  prefetch of the child's line: 2^22 in 32 MB = 40.9% hits / 23.20M evals (-24% vs today, better than old 64 MB).
- Ranked TT (D590, --tt-mode 1): 16-byte entries 4 per line, importance 3/2/1/0 (first / top-12 / later / null probe),
  victim = empty > dead > lowest worth (depth + 4 x importance + 16 proven - 3 x age).
- Same memory as today: ranked TT 2^22 (64 MB) + ranked eval cache 2^22 (32 MB) = 96 MB = today's TT alone.
- Speed A/B (idle machine) running: logs/tree/cache_speed_ab.txt. Then a gate near the Botzone budget.
- Tools: serve 'ttstats' (census), --tt-bits/--ecache-bits/--ecache-mode/--tt-mode, docs/analysis/cache_speed_ab.ps1.

## 6. Knob gates so far (5k vs main 285a127, 600-800 g, D560-D592)
A nmp200 b7-15 +5.8 | B lmr_min 3 in b0 +11.0 +-15.9 | C RFP tighter -9.8 | D user LMR schedule -11.6 | E clone killers
+5.2 | F ahead/behind ordering -3.5 | H (A on b9-14 + B) -1.7. All within ~+-10 Elo, below what 600-game gates resolve.
Per-bucket reading tool: ti insight --by-bucket (D578). Viewers: docs/analysis/tree_viz.py, tree_graph.py (D568, D587).
