# Search literature check: what the best-known implementations do (2026-11-01, user: "find the most optimal implementation online")

Method: web search + fetch of primary sources on 2026-11-01. Quotes limited; numbers are from the fetched pages. NOT done: no paper PDF was read end to end, no Zataxx/Kataxx source found (not indexed), PyJataxx source not found.

## 1. Stockfish (current search.cpp, fetched from official-stockfish/Stockfish master)
- No unconditional PV extension. The "narrow but deep" effect comes from SINGULAR extensions: TT move, depth >= 6 (+1 if ttPv), ttValue lower bound with depth >= depth-3,
  verification search EXCLUDING the TT move at depth/2 with window ttValue - (59 + 66*(ttPv&&!Pv))*depth/63; extend +1 (+2 / +3 when far below, margins larger in PV nodes);
  multi-cut returns when the reduced search fails high; NEGATIVE extension (-3) when the TT move is not singular and (ttValue >= beta or cut node).
- LMR is node-type aware: PV node r -= 512 (/1024 ply) and +1 ply on the reduced depth; cut node r += 4026; not improving r += S*197/512; ttPv adjustments; history term r -= statScore*439/4096;
  allNode and cutoff-count terms. Titanium's LMR has NONE of these (index-only Moonbird scheme + static-eval shortfall).
## 2. Moonbird (strongest known Ataxx engine, tsoj/Moonbird src/search.nim + searchParams.nim)
- Plain PVS alpha-beta: TT, IIR (depth >= 4), null move (depth sub 4 + depth/2, >= 11 target squares), LMR from move 7 depth*47/(47+n), futility reduction (div 35) and skip at depth <= 0,
  aspiration windows, history ordering. NO extensions of any kind, NO PV/cut-node/improving/history term in LMR, no ProbCut, no razoring, no quiescence. Titanium already has all of that plus ProbCut and the disagreement-head terms.
  So the Ataxx state of the art is "fast eval + tuned parameters"; the chess-style extension zoo is not part of it.
## 3. Our own results on the chess-style techniques (docs, repo)
- Singular extension: exp/se-ext, 100 ms vs ProbCut main, ~1000 g, +3 +-21.7 (BACKLOG 424) - no gain, parked. Double extension: ~300 g, ~49%, parked. Reply-ply extension: v1..v5 at 5k, flat to -64 (D244/D249).
- LMR2 100 ms screens (docs/RESEARCH_LMR2.md): extra reduction by move index (r_idx=+8) = -137 Elo; no-capture reduction flat. PV-aware term (r_npv) and PV tunnel (pvx) not yet run.
## 4. Best-first minimax literature
- Cohen-Solal, "Learning to Play Two-Player Perfect-Information Games without Knowledge" (arXiv 2008.01188) and "Minimax Strikes Back" (arXiv 2012.10700): Unbounded Minimax = repeatedly extend the leaves of the principal
  variation; Descent = follow the best action to a terminal state each iteration; "Safe decision" plays the most-explored move. In THEIR learning setting (learned eval, self-play) Descent > Unbounded Minimax > alpha-beta > MCTS; the paper states
  "the worse the evaluation function is, the wider the exploration is". Not a head-to-head with a tuned alpha-beta engine at play time.
- Cohen-Solal & Cazenave, "On some improvements to Unbounded Minimax" (arXiv 2505.04525): 22 games incl. Ataxx, variants compared only with each other (best = TT + completion + exact terminal eval + full backprop); no alpha-beta comparison.
- Baier & Winands, MCTS-minimax hybrids (JAIR 2018): shallow minimax inside MCTS wins 66.9% vs a strong alpha-beta in Breakthrough but ONLY 17.4% (Catch the Lion) and 26.9% (Othello) - alpha-beta stays superior in the capture/flip games closest to Ataxx.
## 5. Conclusions (agent, for the user to overrule)
- Literature plus our data say a best-first tree is unlikely to beat a tuned alpha-beta in an Othello-like game with a good fast eval: eager UBFM measured 0-30 at 300 ms; lazy UBFM screens run (docs/UBFM_PLAN.md).
- The cheap, known-good pieces that Titanium lacks are in the LMR node-type terms (PV less reduction, cut-node more, history, improving). They are already in exp/lmr2 as knobs (r_npv, r_hist, r_impr). Because r_idx=+8 was -137 Elo the response curve is steep; both signs must be probed.
- Sources: https://www.chessprogramming.org/Singular_Extensions , https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/search.cpp , https://github.com/tsoj/Moonbird ,
  https://arxiv.org/html/2008.01188v5 , https://arxiv.org/html/2012.10700 , https://arxiv.org/html/2505.04525v1 , https://dke.maastrichtuniversity.nl/m.winands/documents/mcts-minimax_hybrids_final.pdf

## Botzone loss to [sotax]sotax, 2026-11-01 (match 6ae774774b9c28024545d6c3, user: "Destroyed")
Raw JSON: data/botzone/6ae774774b9c28024545d6c3.json (via curl -H 'X-Requested-With: XMLHttpRequest' '...?lite=true'). sotax = player 0 (moves first), Titanium (titanium_ataxx_PY6, driver debug 'bundled-main/22') = player 1. Result 2-0 for sotax; the end was a forced win (sotax reports mate score 29951 from its turn 126, depth 63).
- Even game for ~90 turns: Titanium scores -77..+28 until turn ~97 (e.g. +21 at turn 97); sotax scores +42..+104 early, +27 at turn 54, 0 at 60, 6 at 66, 76-77 at 72-78.
- Titanium's evaluation LAGS: sotax already sees +130 at turn 84, +213 at turn 90, +154 at turn 96, +163 at 102, +453 at 114; Titanium says -54 at 85, -145 at 91, +21 at 97, -260 at 103, -349 at 115, -945 at 121, -2719 at 127. Titanium only accepts that it is lost about 6-10 turns (12-20 plies) after sotax, the same pattern as D210 (PyJataxx).
- NOT a depth problem: Titanium reached depth 15-27 (sotax 14-23). It is a speed difference only in nodes/s: median Titanium 165k nps (3.0-4.0 s per move, 15-20 depth) vs sotax 435k nps (up to 3.3M nps in the endgame).
- Conclusion for the strategy: both this and PyJataxx losses show the late-middlegame VALUE error (eval blind to who wins the final distribution), not search width/depth. That supports BACKLOG 375 (final-stone-distribution head), the lambda-by-phase retrain and the stronger-eval work over more search tuning. sotax is not in the opponent notes yet (rank not looked up).
