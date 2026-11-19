# HCE + SEARCH SPEED: the one index (D966, 2026-11-16)

Every hand-crafted-eval and search-speed result we have, in one place, with its number and source.
Read this BEFORE any new HCE or speed work. All speeds are SINGLE THREAD (user rule). `sperft` = the fixed 8-position
search suite (`titanium-cli sperft --depth N`), `bench` = start position only.

## 1. Timeline (oldest first)
| date | what | speed (1 thread) | strength | source |
|---|---|---|---|---|
| 2026-09-16 | first engine c7910ce: bitboard movegen, material eval, TT/killers/history/RFP/LMR/NMP | perft 3.25 B pairs/s; search ~2.9 Mnps sperft | - | HANDOFFS_GIT_UNTIL_2026_10_28.md (movegen table, speed log) |
| 2026-09 | move-ordering A/B (sperft d8) | lazy selection 4.11 Mnps; insertion sort 1.77; LSD radix 2.00 | - | same |
| 2026-09 | E2 virus contact eval (-15 per endangered stone, user) | - | +190 Elo, merged | same |
| 2026-09 | E3b partial-iteration adoption | - | +190 Elo, merged | same |
| 2026-09 | E4 multi-capture exposure (-20 per extra stone in one landing, user) | - | +330 Elo, merged; cumulative ~+700 self-play; Moonbird still 0-32 | same |
| 2026-09-20 | S1 slim-eval-speed 7ab844e: eval = material + PST + tempo, dead paths deleted | sperft 4.11 vs 2.40 (+71%); main sperft 4.29 Mnps; bench 4.6-8.0 M nps | 100 g @100 ms 73-15-12 (+230), merged | same; branch origin/exp/slim-eval-speed |
| 2026-09-21 | S2 sancta NNUE, lazy incremental accumulator | 5.0-5.2 M nps | 5k nodes 10-20 (-140); 100 ms 14-16 (about parity: speed compensates) | same |
| 2026-09-21 | S4v3 own net vs classical | net bench d8 2.76 M nps vs classical 6.5 M nps | 0-30 at 5k nodes | same |
| 2026-09-21 | S13 contact eval MERGED (C:/gitProjects/wt-classical 1a27b92): material + PST + tempo + E2 contact 15 + E4 multicap 20 | sperft ~4.3 M; NPS -34% vs slim | +163 @5k, +150 @100 ms, +270 @200 ms | same; wt-classical search.rs |
| 2026-09-21 | S21 tuple eval (pairs) SHELVED | cost floor -44% | +125 @5k but time gates -55..-66 | same |
| 2026-10-19 | winter-clean (tuple champion S55) | sperft 0.95 -> 1.20 Mnps | - | same |
| 2026-11-01 | NNUE main profile | 110-148 knps in real positions; eval 74% of the node, 1.40-1.48 evals/node | - | PROFILE_2026_11_01.md, D241, D305 |
| 2026-11-06 | TSTG (tuple distill) as sole eval | - | 0 points vs NNUE; ~-500 at equal search (sibling top-1 51.6%) | D413, D436 |
| 2026-11-06 | k3s pattern net | eval 1.45x faster, played nps +3.8% | -147 @100 ms | D481, D482 |
| 2026-11-09 | node profile, champion 285a127 | 7.9 us/node: NNUE 65%, SmallNet bound 11.7%, policy 7.3%, movegen+ordering 3.4%, TT 0.5% | - | D627 |
| 2026-11-09 | fast 25-window net in the engine (1.8x nps) | 217k vs 120k knps | -342 (mode 1), -330 (PV hybrid) @100 ms | D628 |
| 2026-11-10 | perft single thread | raw 1.8-3.1 B pairs/s; `perft --engine` 55 M positions/s (18 ns each) | - | D674, D676 |
| 2026-11-11 | exp/max-nps brute.rs material-only kernels (no pruning) | brute 74 Mnps, brute-q1 158, bare-q1 213 Mnps (re-measured 2026-11-16) | not strength: ~3,600x the nodes of main to depth 9 | D965, RESEARCH_ULTIMATE_HCE.md |
| 2026-11-13 | SEE as eval in the brute kernels | material 20.2 Mnps vs exact 2-ply SEE 11.2 knps (~1,800x) | agreement with a d8 judge 36.5% vs 37.0% (noise) | SEE_EVAL_MEASUREMENT.md (was only on exp/max-nps-check, restored D966) |
| 2026-11-16 | what the champion eval looks at (ti/lab hce-fit) | - | linear HCE R2 0.80, 48.7% sibling agreement | RESEARCH_HCE_FIT.md, D948-D962 |
| 2026-11-16 | fast integer HCE (lab hce-fast) | 127 ns/eval vs NNUE 3,559 ns (28x) | R2 0.74, 40.4% agreement (not gated) | RESEARCH_ULTIMATE_HCE.md, D963 |
| 2026-11-16 | today's main with its hand eval (`--network none`) | bench d12 352-362k nps (SmallNet / policy / dis head inactive then) | not gated | D963 |
| 2026-11-16 | Moonbird 1.1.0 (tuned HCE) | 390-480k nps | beats Titanium at 100 ms, loses at 1 s (D105) | D963 |

## 2. What is NOT on record
- A "~500 million nodes per second" search result: not in any doc, any git commit message of any branch, or the Claude Code transcripts on this PC (searched 2026-11-16). The highest recorded single-thread search number is bare-q1 213 Mnps; perft (move generation, not search) is 1.8-3.1 B pairs/s. OpenCode / Gemini transcripts were not searched (search stopped by the user).
- A gate of the HCE-era champion (S13 contact eval, wt-classical 1a27b92, ~4.3 Mnps) against today's NNUE main at 100 ms / 1 s / Botzone clock: never run. This is the cheapest real answer to "HCE at full speed vs NNUE".

## 3. Rules for this topic
- Speed is always single thread (user). A kernel without pruning is not a strength result; compare time to depth or a gate.
- Every new HCE / speed number goes into THIS file and DECISIONS the same turn (global docs, never only on a branch: D966).

## 4. Correction (D968, 2026-11-16): main on the 8 real sperft positions, depth 9, 1 thread: NNUE 0.18 Mnps (5.5 us/node), old hand eval 0.25 Mnps (4.0 us/node), NNUE without SmallNet/policy 0.12 Mnps. Replacing the NNUE with the OLD hand eval = only 1.4x per node; the hand path (different tree, more child evals, loop-based E2/E4 eval) has never been profiled. The 127 ns HCE was never inside the engine. Use sperft (real positions), never bench (start position only), for speed claims.

## 5. External engines (local sources, single thread)
| engine | language / size | eval | speed (1 thread) | depth | strength notes | source |
|---|---|---|---|---|---|---|
| Sanctaphraxx 0.1.11 (Ciekce) | Rust, 3,642 lines, C:/gitProjects/sanctaphraxx | NNUE (src/nnue, net004.nnue 19 KB) | **14.7 Mnps** (36.8M nodes in 2.5 s, midgame position, measured 2026-11-16) | 6 in 2.5 s: very little pruning | lost 2-0 to Titanium at 5k nodes (HANDOFFS_ARCHIVE external ladder) | D969 |
| Zataxx | Rust, 2,101 lines, C:/gitProjects/Zataxx (+ Botzone fork zataxx-bz) | 2x2 tuple net, 36 rows -> 256 SCReLU -> 1, ~0.3 us/eval, ~0.6 us/node | 1.52 Mnps (same position, 2026-11-16); 1.42-1.78 Mnps on 3 midgame FENs (D167) | 17 in 2 s | +81 for Titanium per node at 5k, but Zataxx +354 at 100 ms (D166/D167, old Titanium build) | D167, AUDIT_2026_10_30 |
| Moonbird 1.1.0 | Nim, 2,653 lines, C:/gitProjects/moonbird-src (no Nim compiler here) | tuned HCE | 390-480k nps | 12-16 in 2 s | beats Titanium at 100 ms, loses at 1 s (D105) | D963 |
| sotax (Botzone, binary only) | - | ab-nnue | 403-435k nps (Botzone logs) | 6-23 | beat Titanium 2-0 (D278) | D278, D547 |
| tiktaxx (kz04px) | C++ libataxx, 19,869 lines, C:/gitProjects/tiktaxx | HCE (eval.cpp) | not measured | - | - | NNUE_EVAL_ANALYSIS |
| autaxx (Tuxel) | C++, 38,651 lines | NNUE | not measured | - | stronger than Titanium at equal nodes in Oct (10-20 at 5k) | HANDOFFS_ARCHIVE |
Reading: raw nps is not strength (Sanctaphraxx 10x Zataxx's nps reaches depth 6 vs 17). Best target for "replace the engine's net with our HCE": Zataxx (small Rust, efficient search, one eval function in nn.rs); Sanctaphraxx's search is too weak for the swap to say much.

- **2026-11-17 01:38 D986-D1012 (eval study on our data):** HCE trained like the NNUE (16/32/45 buckets), + gataxx pair tables (0.010904), gataxx / Zataxx shipped evals scored on our val, Zataxx net retrained with D4 (0.009928, best non-NNUE). Table: RESEARCH_HCE_FIT.md TL;DR; raw logs logs/hce/d985*..d1005*.
