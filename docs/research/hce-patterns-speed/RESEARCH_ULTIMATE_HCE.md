# RESEARCH_ULTIMATE_HCE: how fast and how good can a hand-crafted Titanium be? (D963, 2026-11-16)

User: research and document what is best; how fast can the ultimate HCE be, any cheats allowed, SPSA it like Moonbird; maybe speed is most important and we never had a good enough static eval. Remember: SmallNet, policy head and disagreement head are extra costs (user).

## Measured (PC, single thread)
| item | number | source |
|---|---|---|
| champion NNUE full forward | 3,559 ns/eval | `lab hce-fast` on 90,786 positions |
| best fast integer HCE (material, frontier, holes, no-move flags, square x weakness table own/enemy, soft weaknesses; ~20 bitboard ops + one table lookup per stone) | 127 ns/eval (28x faster) | same run |
| its quality vs the champion | R2 0.737, sibling best-move agreement 40.4%, regret 78.6 cp/move (the full slow linear HCE: 0.80 / 48.7% / 50 cp) | logs/hce/d963_hcefast.txt |
| main 0120fd12 bench depth 12 (start position), NNUE + SmallNet + policy | 261k nps | `bench --depth 12` |
| main, NNUE without SmallNet and policy | 181k nps (SmallNet + policy make the NNUE engine FASTER: they prune evals) | `--no-fut-sn --policy none` |
| main with its old hand eval (`--network none`; SmallNet, policy and dis head are then inactive: node counts identical with or without them) | 352-362k nps | `--network none` (+ `--no-fut-sn --policy none`) |
| Moonbird 1.1.0 (hand-crafted, tuned) | 390-480k nps | scripts/moonbird, `go movetime 2000` |
| Zataxx (2x2 tuple net) | 1.4-1.8 Mnps claimed (D167, not re-measured) | KEY_FACTS |
- With the hand eval, main's node costs ~2.8 us, almost all of it search machinery (TT, ordering, history, movegen), not the eval. A 127 ns eval inside today's search would give maybe 400-500k nps; an ultimate HCE engine needs a LEAN search core too (Zataxx-class 1-2 Mnps, ~0.5-1 us/node).

## What the strength data says about speed vs eval quality
- Equal search: an eval at ~50% sibling agreement (TSTG 51.6%, regret 41 cp) scored 0 points vs the NNUE (D413/D436); pattern net k3s at 65% agreement and 1.7x nps lost -147 Elo at 100 ms (D481/D482). Our best fast HCE is at 40% / 79 cp, below both.
- Fast engines win short time controls and lose long ones: Moonbird beats Titanium at 100 ms, Titanium wins at 1 s (D105); Zataxx +354 at 100 ms but -81 at 5k nodes (D166). Botzone plays ~3.3 s per move (B13), where eval quality dominates.
- In the middlegame a DEPTH-1 search already gives 80-90% of what 5,000 nodes correct (D962): extra depth from speed buys little there; the endgame (0-12 empties) is where search pays, and there exact solving / counting beats any eval.
- Kata: one evaluation ~ 1,000 Titanium nodes in strength (D755): quality per eval is worth huge node factors.
Reading: speed cannot close the gap between a 40-50% HCE and the NNUE at the Botzone clock; an ultimate HCE engine would likely beat main only at bullet time controls. This is a prediction from existing gates, NOT a measured gate.

## The best possible HCE (if built)
Terms in order of value (D950-D962): material; per-stone square x weakness table (the biggest single term, with soft weaknesses); 3x3 neighbour counts (dynamic PST); frontier; holes (private / shared); no-move flags (Botzone pass rules); the best 2-ply exchange of the side to move (costs a movegen, ~1 us: only at shallow nodes). Area of effect 5x5. Integer tables per phase (4 buckets; finer buckets cheap). Cost: ~130-250 ns/eval depending on terms.
Tuning: closed-form ridge to the champion eval (done, saturated by construction = Texel-style distillation), then game-level SPSA of the table scale + search knobs with the hub's `ti spsa` (Rust, LAWS A7) at the target clock.
Engine: a lean search core (bitboard movegen, small TT, killer/history ordering, no policy head / SmallNet / dis head), on an exp/* branch from origin/main; gates vs main at 100 ms and 1,000 ms + the Botzone clock (B9).

## Recommendation
Do NOT spend the Botzone effort on an HCE engine. Use the findings where they are cheap: (1) the weakness / soft-weakness / pressure terms as INPUT FEATURES or a residual for the NNUE or the SmallNet (the net already uses them; check whether an explicit per-stone weakness input lowers the training loss); (2) the search core speed (2.8 us/node of machinery even with a trivial eval) is a lever for main itself, independent of the eval.
NOT measured: an actual HCE engine gate; Zataxx nps today; a mid-game bench (bench uses only the start position).

## Max-speed search kernels re-measured (D965, 2026-11-16)
Branch exp/max-nps b847b5c (worktree C:/gitProjects/maxnps-ataxx on exp/max-nps-check 9f727bb, binary of 2026-11-13): crates/titanium/src/brute.rs = material-only alpha-beta kernels (register-passed state, staged bitboard movegen, 16 B TT; `bare` = no TT, no hash). Results were never written to DECISIONS before this row. `titanium-cli sperft --eval X`, 8 positions, single thread:
| kernel | Mnps | nodes to depth 9 (8 positions) | time |
|---|---|---|---|
| main search, NNUE | 0.08 | 206,793 | 2.53 s |
| main search, hand eval | 0.11 | 234,905 | 2.08 s |
| brute (material, TT) | 74.4 | 755,656,854 | 10.16 s |
| brute-q1 | 158.3 | 755,656,580 | 4.78 s |
| brute-zob | 76.2 | 770,367,279 | 10.11 s |
| bare (no TT), depth 8 | 96.2 | 700,927,920 | 7.29 s |
| bare-q1, depth 8 | 212.6 | 700,927,920 | 3.30 s |
Maximum = ~213 Mnps single thread (not 500M; perft move generation alone is 1.8-3.1 billion pairs/s, D674/D676). The kernels have no pruning: ~3,600x more nodes than main to the same depth, so main reaches depth 9 faster at 0.1 Mnps. Raw nps is not strength.
External engines for the eval-swap test: Zataxx = Rust source C:/gitProjects/Zataxx (+ Botzone fork zataxx-bz). Fairy-Stockfish (C:/gitProjects/engines/fairy-stockfish.exe, fairy13.exe): neither local .nnue (ataxx-e631fe1b1b6d, weights-8b5d1db5176c) loads ("network evaluation parameters compatible with the engine must be available"); it falls back to its classical eval and prints no `info depth` lines in Ataxx mode, so its nps / NNUE-vs-classical delta could not be measured with these files.
