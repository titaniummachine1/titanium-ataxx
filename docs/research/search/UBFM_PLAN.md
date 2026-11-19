# UBFM (unbounded best-first minimax) preparation plan - 2026-11-01 (D264/D266, user: "worth checking at the very least")

Status: PLAN ONLY. No code written. PyJataxx's source was not found online (web search 2026-11-01), so its algorithm is UNVERIFIED; this plan follows the published
algorithm of Cohen-Solal (arXiv 2008.01188, JMLR 27) and the 22-game study arXiv 2505.04525 (which lists Ataxx and finds the best variant = transposition table +
completion + exact terminal evaluation + full backpropagation to the root; it contains NO comparison with alpha-beta or MCTS under a fixed evaluation function).

## Algorithm (what we would build)
- Tree of nodes keyed by Zobrist hash (a DAG via a transposition table). Node value = minimax of its children's values (negamax), a leaf value = NNUE eval (+ disagreement).
- One iteration: from the root repeatedly step to the child with the best current value (ties: prefer the deeper/solved-win one) until an unexpanded leaf, expand it
  (generate children, give each a value), then propagate to the root ("full" backprop: always back to the root, restart from the root every iteration).
- Completion: a solved win is preferred, a solved loss is avoided, solved nodes are never selected again. Exact terminal scores (stone count / wipe-out, Botzone rules).
- Anytime: after the time is up play the root child with the best value (or most visited); no iteration depth, so "depth" is the length of the principal line (maxLevel).
## Why it could help us
- D210: PyJataxx's tree reaches maxLevel 38-62 in the middlegame while our alpha-beta reaches 11-21, and its root value was decisive 15-20 plies before our score moved.
  Best-first search narrows the tree on the promising line instead of widening it ply by ply (the "keep the width bounded as depth grows" idea).
- Our late-move pruning is already extreme (docs/research/search/RESEARCH_LMR2.md: 96-99.7% of late moves pruned), so the alpha-beta tree is already narrow; UBFM changes WHERE the nodes go (one deep line), not how many.
## Risks (why it may not work)
- Cost per expansion: evaluating every child of a leaf with the big net is ~50 x 6.5 us = ~0.3 ms, i.e. ~13k expansions in 4 s. Mitigation: lazy children (value of an unexpanded
  child = the futility SmallNet bound / 940 ns, big-net eval only when the child is selected), or an alpha-beta search of depth 2-3 as the leaf evaluator (hybrid).
- Memory: Botzone gives 256 MB; node arena must be bounded (reuse the TT budget, evict by age).
- Value noise: best-first search is greedy on the NNUE value, a single overestimated leaf attracts all the effort (the known weakness); our dis head could widen/narrow selection.
- Non-transitive strength (Mistake 36): beating PyJataxx is not implied by beating Titanium-ab.
## Steps (nothing started)
1. Branch `exp/ubfm` from main in a new worktree; module `crates/titanium/src/ubfm.rs` with arena nodes {hash, first_child, n_children, value, dis, solved}; reuse movegen, NNUE accumulator update, `terminal_score`.
2. Offline probe (no gate): for ~1000 positions from the std5k book plus the 3 Botzone loss games, compare UBFM's chosen move at a fixed expansion budget with the 5k-node alpha-beta best move
   (agreement %) and the value at the known late-lost positions (does UBFM see the loss earlier, like PyJataxx's h? D210 positions: 6ae38cd7.., 6ae22cc6.., 6ae23097..).
3. Gate vs main with `ti sprt` at 1000 ms and 4000 ms (the played clock, LAWS B9) using the B8 rule (promote at P >= 95%, else extend). A 100 ms leg only as diagnosis.
4. If promising: late-game-only UBFM (stones >= N) or the alpha-beta/UBFM hybrid; else leave it alone (low-probability work is not continued).
Sources: https://arxiv.org/html/2008.01188v5 , https://arxiv.org/html/2505.04525v1 .

## Status 2026-11-01 18:20: prototype built (exp/ubfm 23bfada, worktree C:/gitProjects/ubfm-ataxx, exe logs/ubfm_a.exe a62230a0)
- `serve/bestmove --ubfm K`: unbounded best-first minimax, children valued at expansion by a depth-K alpha-beta (K=0 static NNUE eval). Tree not DAG yet, no TT merge, arena cap 1.5M nodes, best-valued unsolved child is followed (completion), full backprop.
- Sanity (start position, 1000 ms, net sat16a_cal): normal ID depth 11, 120,617 nodes; K=0: 12.5k expansions, principal line 9; K=2: 150k nodes, line 8; K=4: 167k nodes, line 7. All pick a7b6.
- User D270/D272 refinement (2026-11-01): look for the best WIDTH-TO-DEPTH balance. K is the depth side (leaf search depth); the width side still to add is a widening rule (only the best W children by value are selectable, or a value-gap window) and a tie/exploration rule. Planned sweep: K in {0,2,4,6} x width in {all, 8, 4} at 300 ms, then the best at 1000/4000 ms.
- Screen chain logs/gates/chain5_ubfm.sh (candidate --ubfm K vs the same exe with UBFM off, 100 g, 300 ms, k=2,4,0,6); results as 'UBFM ...' lines in data/nnue/proxy/progress.txt.

## Result 2026-11-01 18:30: EAGER variants are hopeless at 300 ms
`--ubfm 2` (children valued by a depth-2 alpha-beta at expansion) vs the same exe with UBFM off, 300 ms, 7 parallel games: 0-28 after 28 games (14 pairs, all pairs lost 0-2), Elo about -2400, chain stopped (not a verdict for good-cost variants).
Cause (measured on the games): ~14.5k nodes per move = only ~10 expansions per move, because each expansion values ~50 children (k=2: ~1500 alpha-beta nodes per expansion; k=0: ~80 us per expansion with 50 big-net evals).
The cost per expansion decides everything: PyJataxx reports 70-300k expansions per ~4 s (D210), i.e. 13-55 us per expansion.
=> LAZY mode (K=99, written, not yet built): children get the futility SmallNet LOWER BOUND (940 ns each, + running bias correction learned from the exact evals), the big net is paid only for the one node that is expanded (1 eval instead of ~50). Expected ~15-25 us per expansion.

## Idea 2026-11-01 (user, 4th message): PV TUNNEL instead of best-first
"Maybe the variant isn't best-first but more like beam or tunneling: it finds the expected PV a couple of moves ahead at depth and tries to proceed the search from the PV."
Implemented (source only, not built or gated) as knob `pvx` in exp/lmr2: the FIRST move of a PV node (window wider than 1) near the horizon (remaining depth <= pvx_d, default 2) is searched one ply deeper while ply < iteration depth + pvx. The PV therefore tunnels pvx plies past the nominal depth, everything else keeps the normal alpha-beta depth and accuracy (a width-narrow, depth-extended principal line, verified by alpha-beta, which is the accuracy-preserving cousin of UBFM). Gate plan: pvx 2/4/8 at 100/300 ms, then 1000 ms. Related earlier result: exp/reply-ext (recapture-style extension) was flat/negative at 5k (D2xx, see HANDOFFS 14:30 entry), so the expectation is modest.

## Result 2026-11-01 18:55: LAZY variant also far behind
`--ubfm 99` (exp/ubfm 40e1764, exe ubfm_b.exe) vs the same exe with UBFM off, 300 ms, 100 games (50 pairs): 1-99-0, score 0.010, Elo -797 (CI [-2400,-562]), P(better) 0.0%. The 100 ms leg was not run (chain stopped by the agent).
Speed was as designed (~38k expansions/s, principal line 29 plies at 1 s) so the loss is not a cost problem any more: greedy best-first on the SmallNet-bound estimates plays far worse than alpha-beta. Not analysed: how much is the estimator bias / optimism (an unexpanded child carries a LOWER bound of the net value), the missing TT merge, no exploration term, or the final-move rule (best value vs most visited, Cohen-Solal 'safe decision'). Those are untested fixes, not evidence for the idea.
Verdict by the gate rule: not undecided - 100 games at about -800 Elo is conclusive below, but the agent does not reject (law 4/B8): reported as a number, the user decides. Parked; exp/ubfm keeps the code.

## PV tunnel first screen was an implementation error (2026-11-01 18:35)
pvx=2 (lmr2_b.exe 3cb38e5c) scored 0.3300, Elo -123 [-172,-79], 200 g, 100 ms. Cause found afterwards: the tunnel condition was 'window wider than 1', but the root searches EVERY root move with the window (-INF,-alpha), so every root move's leftmost chain was extended (about 40x the intended work). Fixed in exp/lmr2 (explicit `pvon` flags: only the root's first move chain and fail-high re-searches of PV nodes count as the principal line), exe logs/lmr2_c.exe; bench identical with the knob off (68961), pvx=2: 58,827 nodes, pvx=4: 63,242 nodes at depth 10. The -123 is NOT a measurement of the PV tunnel idea. Re-screen queued (chain7 with lmr2_c.exe).

## PV tunnel re-screen (fixed, lmr2_c.exe, 100 ms, 200 g): pvx=2 -> score 0.4725, Elo -19.1 CI [-62.0,+23.2], P(better) 18.8% - flat/undecided (the first -123 was the bug, D277). Further LMR2C lines (pvx4, r_npv +-8, r_idx -4, r_hist +-8, r_impr 8): docs/research/search/RESEARCH_LMR2.md and progress.txt 'LMR2C'.

## 2026-11-02: best-first with policy vs PUCT (user question, D318)
Answer given: in a tactical game like Ataxx (800 cp swings per ply) best-first MINIMAX (UBFM / Athenan-style, Cohen-Solal) fits better than PUCT: PUCT backs up AVERAGES, so one refutation deep in a line is diluted by the other visits and needs many visits to surface; minimax backup lets a single refutation flip the value at once. PUCT works 'algorithmically with enough training' only because the policy+value nets are trained by huge self-play (AlphaZero, Kata) and run on a GPU; on our 1 CPU core at 4 s a net with a policy head costs ~1000 Titanium nodes per eval (Kata, D156). Our UBFM prototype lost 1-99 (D276) WITHOUT a policy and with a value net trained for alpha-beta leaves. Both PUCT and UBFM need a policy first, and so does better alpha-beta move ordering. Order: (1) policy head trained on Titanium's own search best moves (hub `ti train`), used for move ordering / LMR in alpha-beta (cheap to gate); (2) if the policy is good, a UBFM prototype with policy-ordered expansion and minimax backup, gated vs main at 1 s and 4 s; PUCT only as a comparison.
