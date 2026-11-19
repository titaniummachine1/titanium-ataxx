# Engine code review, 2026-11-18 (D1226, user: "review titanium code if we could improve titanium")

Target: `origin/main` 0120fd12 (`C:/gitProjects/mainpromo-ataxx`, clean). Read in full: `search.rs` 1-2125 (TT, eval cache,
move scoring, negamax, root / iterative deepening), `nnue.rs` 330-960 (forward pass, FT sums, head). Cross-checked against
the earlier reviews (RESEARCH_FUNDAMENTALS_REVIEW.md F1-F13, ENGINE_REVIEW.md A/B, ENGINE_PROFILE_2026_11_17.md, the D241
speed rounds in DECISIONS, BACKLOG "TRIED AND REJECTED") so that nothing tried is re-proposed as new.

NOT done (say it): no measurement and no build (the PC runs the 47-stone proof hunt, then pc_run16 generation, D1225);
`board.rs`, `nnfeat.rs`, `smallnet.rs`, `policy.rs`, the FFI driver and the hub were not re-read; nothing gated.
Every Elo figure below is a hypothesis until gated.

## 1. Correctness: no bug found
TT (packed 10-byte entries, empties check, mate-distance adjust), eval cache (25+2 check bits + empties, about 0.07
expected false hits per 500k-node move), TT-move fast path, Botzone ban/taint bookkeeping, mate-distance pruning,
RFP / ProbCut / futility guards against mate scores, partial-iteration adoption: all consistent with what they claim.

## 2. Where the time goes (unchanged picture, ENGINE_PROFILE_2026_11_17)
Big-net evals 0.82 per node = 66% of node time (FT row sums alone 36%). Titanium 203k nps vs Zataxx 1,514k at 1 s,
median depth 13 vs 19 (D1208): with equal tree shape the depth gap is all nps. The bit-identical speed work is done
(D241 rounds 1-3: psqt chains interleaved, scans, L1, L2); incremental accumulator (D188, D1106) and software prefetch of
FT rows / L1 columns were measured NEGATIVE. So there is no cheap exact speedup left in nnue.rs; the eval lever is the
architecture (fewer rows, narrower accumulator, tighter SmallNet bound: arch hunt D1110, RESEARCH_SMALLNET_ARCH).

## 3. Levers, ranked (NEW = not found in any earlier doc; OWED = undecided earlier gate the gate rule says to extend)

| # | Lever | Code | Status | Why | Test |
|---|---|---|---|---|---|
| R1 | Search margins in the NET's own scale | RFP_MARGINS 257/347/478/774, MB_FUT_DIV 57, PC_T 140, PC_K 32, DIS_RFP_NUM 32, DIS_FUT_NUM 24, `child_dis > 300` (search.rs:192-210, 1293, 1328, 1638-1641) | NEW | All are fixed cp constants tuned on older nets. Newer nets are 10-15% more confident and their dis head +33% in the endgame (D1115), so the same constants prune differently. This fits the pattern that val gains do not convert: nn_v11bsat -28 (D1065), bundle BB9CECE8 5k +25 / 100 ms -39 (D1112), nn_v13sat equal per node (D1207). | Export-time scale match: fit k per bucket so the new net's cp matches the champion's on a fixed position set; store k in the header, or scale the margins. Then gate scaled vs unscaled nn_v13anneal (5k + 100 ms). |
| R2 | Duplicate non-PV re-search | search.rs:1695 re-searches at full depth after a null-window fail-high even when the move was NOT reduced (same window, same depth) | OWED (D721/D723) | Fix `score > alpha && (is_pv \|\| new_depth < depth)`: bench -12.2% nodes, 800 g +7.8 +-23.4, LOS 74%, parked and never extended. Still on main. | Rebase exp/nonpv-research 16739d22 onto 0120fd12, extend the 100 ms gate, then 1000 ms. |
| R3 | Singular extensions | none in search.rs | OWED | exp/se-ext: about 1000 g, +3 +-21.7 (2026-10-31, BACKLOG line 682), on an engine with no policy head, a classic search with null move, and an older net. Every strong chess and Ataxx engine has SE (cheetax log). The engine changed too much for that number to stand (B10, D1188). | Rebase or re-implement on 0120fd12 (exclude TT move, depth/2 search at tt_score - margin), gate 100 ms + 1000 ms. |
| R4 | Root ordering ignores the policy | search.rs:1918 `gen_and_score_into::<false, false>` (no policy); later iterations only swap the best move to the front (2079-2082), the old best lands in the new best's slot | NEW (root sort by previous scores was tested D667: z -0.91, but those scores are upper bounds under the open root window) | Every Botzone search ends in a partial iteration, in 59% of them >= 2 root moves are searched and the played move changes in 12-17% (D667): the order of moves 2..k decides which alternatives get searched at all. | Order root moves 2..n by the policy once, then by last iteration's subtree node count. Agreement harness (docs/analysis/move_agreement.ps1) first, then a 1000 ms gate. |
| R5 | PV re-search after a reduced fail-high | search.rs:1695-1698 goes straight to full window + full depth | NEW to gating (F6 noted 2026-11-09, never gated) | Standard: null-window full-depth first, full window only if that also fails high. Saves work in PV nodes; small. | Stack with R2 on one branch, measure nodes at fixed depth, then gate. |
| R6 | ProbCut overwrites the node's own TT entry | store_e (search.rs:658-666) replaces a same-position entry whatever its depth; ProbCut (1330) searches the SAME position at depth-3 and stores first | NEW detail of F9 | The deeper entry from the previous iteration is lost whenever ProbCut fails and the node is then stopped / tainted / repetition-skipped (otherwise the node's own end store repairs it). The general depth guard F9 gated -20.9 +-34.8 at 400 g (D707, undecided). | Narrow version: keep a same-position entry that is deeper and from the current age when the new store comes from a ProbCut inner search; bench parity on depth + gate. Low priority. |
| R7 | History weight is ~0 | ORD_HIST_FULL 2000 = 0.02 policy logit (search.rs:262, 385) | KNOWN (B9) | H3 x10: 100 ms +16.7 LOS 97% (1600 g) but 1000 ms -47.2 (200 g, D730/D731): clock-dependent, undecided at 1000 ms. "History off" (save the update cost) never tested. | Low priority; ablation "history off" at fixed nodes. |

## 3b. R8 (user idea D1228): semi-terminal nodes without the NNUE
Main has no small-empties solver: a Botzone-search leaf with 1-2 empties returns the big-net eval (search.rs:1355), even
when a clone into the last square decides the game. zataxx-bz already has an O(1) 1/2-empty solver (board.rs:356, used in
the proof hunt, D1224). Port it plus a leaf check for Botzone stalemate-mates (a clone that leaves the opponent no move and
wins by absorbed count) for <= ~6 empties. Exact by construction, so it changes play only where the NNUE was wrong or a
mate was beyond the horizon: the late game, where our losses are decided (D784, D1065). Full design, rules and test:
BACKLOG top, R8.

## 4. Things checked and NOT levers (so nobody re-proposes them)
- Child TT probe before the child's big-net eval: a child that has a TT entry was searched before, so its eval is almost
  always already in the 32 MB eval cache (checked first, search.rs:1590); the 1.80 cache misses per node are mostly new
  frontier nodes. Expected gain small.
- Lazy node static eval (after the TT move): the eval is needed for RFP at the many depth <= 4 non-PV nodes before any
  move, so only depth-5 and PV nodes could defer it. Small.
- check_stop: one clock read per 64 nodes, negligible. Hand eval in the SmallNet input: 1.4% (known, ENGINE_REVIEW B2).
- Sparse L1 sums are serial add chains (4-cycle latency x about 72 inputs); two partial sums would cut that but change
  the float order (not bit-identical) for about 1% of eval. Not worth a gate alone.

## 5. Recommended order
R1 first (it decides whether the net research can show up in play at all, and the nets are where most effort goes),
then R2 (code exists, one rebase), R3, R4 + R5 together. R6/R7 only if a cheap slot is free.
All on the PC after the generation run, or on a Docker box under the budget rule.
