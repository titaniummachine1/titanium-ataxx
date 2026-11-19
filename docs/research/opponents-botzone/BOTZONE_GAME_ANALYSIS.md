# Botzone game analyses (hub: `ti botzone <match> --out DIR --games-jsonl FILE`, then `ti analyze FILE`)

## 2026-11-07 match 6aef432692eeeb215c899ecc: Titanium (x, player 0) vs [jacek]PyJataxx: LOST 16-33 (D546)
Raw: data/botzone/6aef432692eeeb215c899ecc.json; gate-format game logs/botzone/6aef4326.games.jsonl; report logs/botzone/6aef4326_report.txt;
deep re-search logs/botzone/6aef4326_deep.txt. Titanium version: driver debug says `bundled-main/22` (no version tag in the log; which .so was live is NOT known from the log).
- Engines: Titanium median depth 16, 558k nodes, 161k nps, 3.6 s/move (first move 10.2 s). PyJataxx (UBFM-like best-first): median tree depth 38
  (up to 308), 84k expansions, 3.9 s/move; its debug `h` is a value in [-1, 1], converted to cp = 400 ln((1+h)/(1-h)).
- Eval trajectories (Titanium's view): both near +30..+80 for the first 30 plies; from ply 32 PyJataxx's value swings +-100..350 between its
  own moves (noisy, odd/even-depth effect) while Titanium stays within +-80.
- Decisive phase plies 92-106: after Titanium's ply-92 move PyJataxx says -251, after ply 94 -515, ply 96 -721, ply 104 -1124, proves the
  win (h = 8540) at ply 108. Titanium says -43, -82, -45, -140, -534, then -873 at 108 and mate only at ply 120 (ti analyze turning
  point 120: -1546 -> -99974; 'loser saw it (<= -200)' at 106). PyJataxx saw the loss ~12 plies (6 own moves) earlier.
- Deep check (30 s per move, ~8x the game time, main engine, same positions): Titanium keeps its game move at plies 86, 90, 92, 96
  (scores -49, -94, -31, -77); at ply 88 prefers d4f4 over the played d3f4 (same destination), at ply 94 prefers d4b3 (+36) over the
  played g3g5 (-82). Even with 8x time Titanium does not see what PyJataxx saw at plies 92-96 => an EVALUATION/horizon blind spot in the
  late middlegame, not a time/depth problem (same pattern as D210 PyJataxx and the sotax loss in RESEARCH_SEARCH_LITERATURE).
- NOT checked: which positions/features PyJataxx's value reacts to; a 5-minute or 50M-node Titanium search; Kata's view of plies 88-96.

## 2026-11-11 match 6af3f64401a9a145b459dfef: [Terminator_PL]titanium_ataxx_PY6 (x, player 0) vs [jacek]PyJataxx: LOST 0-2 (20-29)
Raw: fetched via `ti botzone 6af3f64401a9a145b459dfef`; game log logs/botzone/6af3f644.games.jsonl; report logs/botzone/6af3f644_report.txt; positions logs/botzone/6af3f644_pos/positions.bin (128 positions -> 128 unique under D4); deep re-search logs/botzone/6af3f644_deep.txt.
- Timing & live build confirmation: first turn 11,989 ms engine / 11,961 ms Botzone (d15, 2,603,456 nodes); all subsequent 64 turns used 3,989 ms engine / 3,988 ms Botzone (until forced mate <=0 ms at plies 115+). Confirms the `ti_botzone_serve` deadline-search driver is live on Botzone (up from 3.6 s/move in match `6aef4326`). All 130 plies verdict `OK` (0 timeouts/forfeits/bans).
- Engines: Titanium (65 moves) median depth 16, 648,576 nodes, 166,506 nps, 3,989 ms engine / 3,988 ms bz, eval[cp] median +12, range [-99998, +115]. PyJataxx (65 moves) median depth 28, 78,747 nodes, 3,899 ms bz (max 4,148 ms), eval[h] median +0.047, range [-0.3406, 8700].
- Repetition / Botzone revisit check: 0 board revisits and 0 repeated position pairs across all 130 plies; Botzone repetition rules had zero interaction in this game.
- Game trajectory & collapse (`ti analyze`: turning point 0-idx 112 [1-idx p113]: `-1020 -> -99982`; loser saw it `<= -200` at 0-idx 98 [p99, `-356`]; biggest disagreement 0-idx 110: `-1020 vs -19990`):
  - Balanced opening through p83 (`+1..+90 cp`). At p86 (`30->11`) PyJataxx spiked to `h=+0.640` (~`+607 cp`), then dropped back to `-0.126..+0.192` through p94 (`h=+0.045`) while Titanium sat flat at `+115 cp` (p87/p89/p91, d24..28) then `+5 cp` (p93, `e5c6`).
  - Collapse trigger: after Titanium played p95 (`f7e5`, `-127 cp`, 775k nodes, d24), PyJataxx jumped to `h=+0.731` at p96 (`+745 cp`), `+0.774` at p98, `+0.993` at p110, and solved a forced win (`h=8680`, `19990 cp`) in `78 ms` at p112 (`61->42`). Titanium reached `-356 cp` at p99, `-1020 cp` at p111, and proved mate `-99982` at p113 (`d29`, 8.09M nodes).
- Deep re-search check (`exp-max-nps` classic search, `2,500,000` nodes = ~2.2-3.5x game nodes on 1-idx plies 85, 87, 89, 91, 93, 95, 97, 99):
  - Same move kept at 6 of 8 plies (`b3, a5, d3c1, f2d1, d6c4, d4f3`), though deeper eval dropped immediately on p87 (`+115 -> -58 cp`, d29) and p89 (`+115 -> -13 cp`, d28).
  - Move flips at the two collapse turns: p93 (`0-idx 92`) prefers clone `c6` (`+96 cp`, d24) over jump `e5c6` (`+5 cp`); p95 (`0-idx 94`) prefers `a2` (`-65 cp`, d26) over `f7e5` (`-127 cp`, d24). Mix of slight depth/search miss around p93/p95 (`c6`/`a2`) plus late-middlegame eval blind spot (`p87/p89` scored `+115 cp` at game nodes vs `-58/-13 cp` at 2.5M).

## 2026-11-14 14:15 six matches vs [jacek]PyJataxx (user sent 6 links, D784): Titanium 3 wins, 3 losses
HOW TO REPORT A BOTZONE MATCH (the hub does it, no scripts, no deep re-search unless the user asks for one): (1) 	i botzone <full URLs https://www.botzone.org.cn/match/ID ...> --out logs/botzone/<tag>_pos --games-jsonl logs/botzone/<tag>.games.jsonl > logs/botzone/<tag>_report.txt (a bare ID does NOT work, it needs the URL; one --out dir per call, it refuses to overwrite positions.bin); (2) 	i analyze logs/botzone/<tag>.games.jsonl > logs/botzone/<tag>_analyze.txt; (3) 	i insight only with >= 50 games (it refuses fewer); (4) report: players, result, stones, plies, timing/depth/nodes medians (report file), analyze table (turning point, ply the loser saw <= -200), and PyJataxx's h converted to cp = 400 ln((1+h)/(1-h)) next to Titanium's cp at the same plies. Never pause the generator or start side jobs for this.
Raw: logs/botzone/d784.games.jsonl (6 games), d784_report.txt, d784_analyze.txt, positions logs/botzone/d784_pos/positions.bin (661 positions, 622 unique under D4; usable for `ti relabel`). Matches (Titanium = titanium_ataxx_PY6, all verdicts OK, no timeouts):
| match | Ti seat | result | stones x:o | plies | turning point (loser's eval) | loser saw <= -200 at |
|---|---|---|---|---|---|---|
| 6af742d9...aec2 | x | WIN | 31:18 | 113 | 103: PyJ -3085 -> -19990 | ply 65 |
| 6af742d9...aec5 | o | LOSS | 35:14 | 113 | 103: Ti -1558 -> -99990 | ply 77 |
| 6af73fff...ac3c | o | WIN | 20:16 | 65 | 52: PyJ +230 -> -24 | never (short game; not examined further) |
| 6af70b4b...7169 | x | LOSS | 18:31 | 158 | 130: Ti -1528 -> -99974 | ply 90 |
| 6af70b4b...716c | o | WIN | 20:29 | 113 | 102: PyJ -2065 -> -19990 | ply 72 |
| 6af76327...c6c8 | o | LOSS | 29:20 | 111 | 91: Ti -1467 -> -99988 | ply 65 |
- Engines (report medians): Titanium depth 15-19, 615k-673k nodes, 154k-182k nps, 3,989 ms engine / 3,988 ms Botzone per move, first move 11.9-12.0 s (intended doubled first turn); PyJataxx depth 23-34 (up to 55), 73k-97k nodes, 3,899 ms (max 4,191).
- The losses repeat the known pattern (D210, D679): PyJataxx's value turns clearly positive 8-10 plies before Titanium's own score reaches -200. 6af742d9...aec5: PyJ h 0.419 (about +357 cp) at ply 74 while Titanium said -103; Ti reached -243 at ply 78 and -1558 at 103. 6af70b4b...7169: PyJ h 0.728 (about +740 cp) at ply 81 while Titanium sat at -116, -25, -26, -2 (plies 81-87, depth 23-35); Ti first <= -200 at ply 90. 6af76327...c6c8: PyJ h 0.617 (about +576 cp) at ply 58 while Titanium said -95; Ti <= -200 at ply 66, -1060 at 86, mate at 92. In the wins the roles reverse (PyJ <= -200 at plies 65 and 72, turning points 102-103).
- Reading: 3-3 over six games, as x 1 win 1 loss, as o 2 wins 2 losses. All six games are Botzone-legal, no timing issue, so the losses are evaluation/horizon in the plies 55-90 range, not time. NOT checked: whether the move at the turning ply was avoidable (no deep re-search was run, on purpose); which Titanium build was live (log has no version tag); the 65-ply game (replayed stones 20:16 in Titanium's favour) beyond the table.
- 2026-11-14 14:20 follow-up (D785, from d784_report.txt, each engine's own view in cp, PyJataxx h converted with 400 ln((1+h)/(1-h))): at ply 60 PyJataxx is optimistic in ALL six games (+117 .. +607 cp) while Titanium sits at -115 .. +/-0, so ply 60 does not tell who is right. The split comes at plies 64-80: in the three losses PyJataxx keeps rising (+344 / +740 / +818 cp at ply 70-80) and Titanium follows only 8-10 plies later (-171 .. -363 at 70-90); in the three wins PyJataxx falls (m1 +133 at ply 60 -> -364 at 70 -> -585 at 80; m5 +181 at 70 -> -103 at 80 -> -671 at 90; m3 +320 at ply 50, game over at 65) while Titanium turns positive (+141 at 80, +329 at 90). So the games are decided by who judges the plies 55-85 middlegame correctly; Titanium's search is deeper in the endgame (d29-51 at 3-4M nodes when few empties) but its eval is the later one in the losses. Sample is 6 games (3-3): no strength conclusion beyond 'about equal', consistent with Botzone ratings 2171 (Titanium, rank 6) vs about 2157 (PyJataxx, rank 7, D210). NOT verified: that PyJataxx's early positive value in the losses was right and not luck; no deep check was run.

## 2026-11-16 match 6afb1fcc36b19c237a3f701b: [Terminator_PL]titanium_ataxx_PY6 (x, player 0) vs [fab]cheetax: WON 2-0 (final discs 25-24) (D936)
Raw: data/botzone/6afb1fcc36b19c237a3f701b.json; game logs/botzone/6afb1fcc.games.jsonl; report logs/botzone/6afb1fcc_report.txt; positions logs/botzone/6afb1fcc_pos/positions.bin (97 unique).
Which .so played is NOT known from the log (driver label "bundled-main/rust" only; the v13 .so storage upload was still in progress, D935).
- Engines: Titanium 50 moves, median depth 17, 737k nodes, 194k nps, 3,989 ms/move (first 11,962 ms), all verdicts OK. cheetax median depth 25, 3,800 ms/move.
- Evals: both sides near +0..+70 (each from its own view) through ply 80, i.e. both thought they were slightly better.
- Decisive phase plies 83-93: Titanium's own eval climbs 705 (83), 1051 (85), 1489 (87), 1555 (89), 2107 (91) and proves the win at ply 93 (99989); cheetax still says +455..+521 at plies 84-94 and drops to +98 only at ply 95 (ti analyze: loser's turning point 95, biggest disagreement ply 90: +2107 vs -519). Titanium saw the forced win ~10 plies before cheetax noticed anything.
- Margin: the proven win ends 25-24 (one disc).
- NOT checked: deep re-search of plies 60-83 (where the win was set up), cheetax's engine type.
- **CORRECTION (D938):** the cheetax numbers above are WRONG: the hub read cheetax's `eval score` (another number) instead of its search score `evaluation` (fixed in titaniumtest a6c8aa2). Real cheetax trail: -28..-70 (plies 40-54), ~0 (56-66), -185 (68), -245 (74), -325 (78), -524 (86), proven loss at 92. cheetax is an ALPHA-BETA searcher (depth 24-26, seldepth up to 99, median 8.4M nodes/move = ~12x Titanium), NOT PUCT.
- Main deep check (10M nodes, depth 19-30, logs/botzone/6afb1fcc_deep.txt): ply 61 +2, 65 -75, 71 +348 (game said -34), 77 +240, 81 +179, 85 +1356. Titanium's in-game value was ~380 cp too pessimistic at ply 71; cheetax was right that it stood worse.

## 2026-11-16 match 6afb232636b19c237a3f725d: [fab]cheetax (x) vs [Terminator_PL]titanium_ataxx_PY6 (o): LOST 0-2 (discs 18-31) (D938/D939)
Raw: data/botzone/6afb232636b19c237a3f725d.json; report logs/botzone/6afb2326_report.txt; deep check logs/botzone/6afb2326_deep.txt (10M) + logs/botzone/6afb2326_deep100m_raw.txt (100M, plies 66/68).
- cheetax search score (own view): +17..+117 (plies 43-65), -9 (67), +221 (69), +429 (71), +524 (75), proven win at ply 83.
- Titanium in game (own view): about -20..-70 through ply 68, +66 at ply 70, -538 at 72: it saw the loss ~3 plies after cheetax.
- Main deep check, 10M nodes (~13x game nodes, depth 22-30): Titanium's game move = deep best move at ALL 6 positions.

| ply | deep 10M | game | move |
|---|---|---|---|
| 62 | -67 | +35 | d6 |
| 64 | -46 | -70 | b2d4 |
| 66 | +63 | -56 | c2 |
| 68 | -92 | -26 | d4b6 |
| 70 | -211 | +66 | b3d1 |
| 72 | -718 | -538 | b5d5 |

- Reading: no single Titanium move is a blunder at 10M nodes; the value fell +63 -> -718 over cheetax's replies 67-71, which 10M nodes do not foresee at ply 66 = an evaluation / horizon blind spot (same pattern as the PyJataxx losses), not a time-per-move problem. 100M nodes (D943, ~11 min each): ply 66 depth 34 = -185 (best c2, same as played), ply 68 depth 38 = -228 (best d4b6, same as played). More depth lowers the value (the game search was ~150-200 cp too optimistic) but finds no better move: the position was already lost at ply 66; the first real error is earlier (NOT located yet; next: 30M re-search of plies 50-64).

## 2026-11-18 matches 6afc50b3 / 6afc50ac / 6afc52be vs [noexcept]Ataxx: LOST 0-3 (D1151)
Raw: data/botzone/6afc50b3.json, 6afc50ac.json, 6afc52be.json (full ?lite=true logs, cached). No `ti botzone --games-jsonl` run: the hub build failed (wrong package name, then shell error), so turning points below are the hub's exact formula (`explain()`: biggest drop between the loser's own non-mate evals; saw-it = first own eval <= -200) computed in scratch Python over the recorded per-move debug, no engine. Opponent scores mapped q -> cp = 400*atanh(q), the PyJataxx precedent (D546); |q| = 1.0 observed -> treated as solved (+-19990), same boundary rule.
| match | seats (Ti) | result | final stones x:o | turns | turning point (Ti eval) | Ti saw <= -200 at |
|---|---|---|---|---|---|---|
| 6afc50b3 | o (P1) | LOSS 0-2 | 27:22 | 134 | ply 105: -2701 -> -99984 | ply 89 |
| 6afc50ac | x (P0) | LOSS 0-2 | 19:30 | 138 | ply 126: -2101 -> -99988 | ply 90 |
| 6afc52be | x (P0) | LOSS 0-2 | 20:29 | 108 | ply 88: -1649 -> -99982 | ply 72 |
- Engines: Titanium 54-69 moves, median depth 17-18, 718-770k nodes, 188k nps, 3,989 ms engine / 3,988 ms bz (first move ~11.6-12.0 s, the intended doubled turn). noexcept (py36): full 3.94 s/move clock like us (first ~11.6 s), memory 146 -> 512 MB over a game (persistent growing tree/table), debug `e/s/v/q/t` with q = expected score in [-1, 1].
- Trajectories: noexcept q starts ∓0.14 (correct side: +0.14 as P0, -0.14 as P1), crosses |q| = 0.5 at plies 40 / 81 / 67 and climbs to exactly 1.000 in all three; max single-move |Δq| over 190 assessed moves is 0.29 (mean 0.03): no re-search cliffs anywhere. Titanium s goes from +50s/-60s to mate (-9999x) in all three. Material (|x-o| >= 10) turns decisive only at plies 88 / 109 / 85: their value leads material by 18-48 plies.
- Biggest mover-vs-next-mover gaps (hub `disagreement` formula, their view negated): m1 ply 102 (+1661 vs +2701, both say noexcept winning, magnitudes differ; q-resolution collapses near 1.0, so this gap is partly the mapping); m2 ply 88 (-113 vs +775); m3 ply 76 (-361 vs +1122). M2/M3 fit the known pattern (opponent sees it ~8-10 plies first, cf. D210/D784); m1's q crossing at ply 40 with the win 90+ plies later carries the D785 optimism caveat (their q runs warm early in all three games).
- Architecture read (no source, from behaviour only): MCTS-family value search (PUCT-like). Against: classic alpha-beta (no discontinuous jumps in 190 moves; value leads material by 20-50 plies; 11.6 s init cost; monotonic 366 MB growth = tree reuse across moves, not a fixed TT). MCGS-vs-PUCT is NOT resolvable from outside traces. e-field semantics (15k -> 0 in decided endgames) not decoded; left as raw.
- What it means: a veteran multi-game programmer's fresh v1 (ranked #7, climbing 46 -> 7 in 4 days) just took 3 straight off us from both seats in long endgames. The 36-Elo rank gap is already gone if not inverted.
- NOT checked: deep re-search of the collapse plies (would need engine time; the user forbade engine runs for this analysis); whether q's early warmth (ply-40 crossing in m1) is bias or genuine depth. UPDATE (D1158): hub q-parser BUILT (target_main ti.exe BA7FD881) and verified — ti botzone --games-jsonl + ti analyze reproduce the Python turning points exactly and fill the disagreement column (102/88/76); artifacts logs/botzone/noexcept_q.games.jsonl + positions.bin.

## 2026-11-18 match 6afcf9fa vs [fab]cheetax: Titanium (o, player 1) WON 2-0, final stones x:o 24:25 (D1154)
Raw: data/botzone/6afcf9fa.json (full ?lite=true log); gate-format game log logs/botzone/6afcf9fa.games.jsonl (hub `ti botzone --games-jsonl` on the local file, cheetax `evaluation`/`eval score` parsed natively since D938). No deep re-search: `ti analyze` output only (replay, no engine).
- Engines: cheetax 50 moves, median depth 26 (seldepth 42, up to 99 in the endgame), ~8.4M nodes/move, 3.8 s/move; search eval ("evaluation", mover's view) range [-19952, +70], static "eval score" stayed +100..+500 deep into the lost endgame (blind). Titanium 49 moves, median depth 17, 712k nodes, 187k nps, 3,989 ms/move (first 11.94 s), eval median +53, range [-63, +99997]. All verdicts OK, no timeouts.
- Game (`ti analyze`): turning point ply 84 (-513 -> -19990, cheetax's search eval collapses into proven loss); cheetax saw it (own eval <= -200) at ply 76; biggest disagreement ply 84 (-19990 vs -1803: cheetax proves its own loss ~10 plies before Titanium's eval proves the win at +99981+). 99 plies total (97 positions at min-ply 3, all unique under D4).
- Reading: won by ONE stone (25:24) from the second seat. Cheetax out-searches 12x on nodes (8.4M vs 0.7M) and proves terminals earlier, but its static eval never sees the defeat (positive to the end) while Titanium's eval climbs steadily -63 -> +99997. Same lesson as the anneal/HCE work: the eval, not the node count, decides these games.


## 2026-11-18 02:41 all 7 matches vs [noexcept]Ataxx (2026-11-17 21:03-21:12): LOST 0-7 (D1155, extends the 3-game section above)
Raw: data/botzone/{6afc50ac,6afc50b3,6afc52be,6afc52b8,6afc52c3,6afc52c8,6afc52cd}.json (?lite=true, the last 4 fetched 2026-11-18 02:41); gate-format logs/botzone/d1155_noexcept.games.jsonl; ti analyze logs/botzone/d1155_noexcept_analyze.txt. The hub binary used does NOT contain the q-token fix (the 'biggest disagreement' column of ti analyze shows noexcept's s COUNTER, e.g. -16372 at ply 17 = s16372, not a score: INVALID); the trajectory table below reads q straight from the files. Titanium's own columns (turning point, saw it) are valid.
| match | Ti seat | plies | noexcept q at ply ~20/40/60/80 | first ply q >= 0.5 | Ti eval at ~20/40/60/80 | Ti first <= -200 | noexcept ahead by |
|---|---|---|---|---|---|---|---|
| 6afc50ac | P0 | 138 | -0.09/0.03/0.04/0.48 | 82 | +39/+27/+14/-62 | 91 | 9 plies |
| 6afc50b3 | P1 | 134 | 0.21/0.52/0.51/0.62 | 41 | -44/-94/-104/-188 | 90 | 49 |
| 6afc52be | P0 | 108 | 0.00/0.10/0.28/0.91 | 68 | +40/+37/-59/-1074 | 73 | 5 |
| 6afc52b8 | P1 | 115 | 0.30/0.68/0.75/0.95 | 31 | -87/-142/-222/-940 | 60 | 29 |
| 6afc52c3 | P1 | 103 | 0.12/0.36/0.71/0.96 | 53 | -24/-21/-150/-1216 | 62 | 9 |
| 6afc52c8 | P0 | 108 | -0.12/0.00/0.33/0.91 | 68 | +47/+64/-78/-1112 | 73 | 5 |
| 6afc52cd | P1 | 117 | 0.17/0.21/0.30/0.95 | 69 | -48/-15/-48/-378 | 70 | 1 |
- Search sizes per move (medians): noexcept e ~15.0k / s ~15.7k / v 22-48k counters at 3.94 s, memory grows to 512 MB; Titanium 710-784k nodes, depth 16-19, same 3.99 s. If s = simulations, ~4,000 per second = ~250 us per simulation = a large neural evaluation per node (AlphaZero/PUCT-style), ~45x fewer nodes than Titanium. Inference only: e/s/v semantics are not documented.
- 0-7 head-to-head: at equal strength P(7-0) = 1/128 (0.8%), so it is genuinely stronger; its 2078 rating is a fresh v1 still climbing.
- Reading: noexcept's value turns decisive a median 9 plies (1-49) before Titanium's eval drops below -200; as P0 Titanium still says +37..+64 at ply 40 while noexcept is level, then both collapse 60-80. Same signature as the cheetax and PyJataxx losses (D943, D784): Titanium's eval is too optimistic in long late middlegames; time per move is not the cause (both use the full clock).
- cheetax 3 games (logs/botzone/d1155_cheetax_analyze.txt, parser valid for cheetax): W (6afcf9fa), L 6afcfc2c (turning point ply 93, Ti saw <= -200 at 73, cheetax proven win at 87 while Ti -1478), L 6afcff26 (turning point 105, saw at 79, cheetax proven at 99 while Ti -1310).
- NOT checked: deep re-search of the decisive plies; whether noexcept is trained on millions of self-play games (not knowable from the logs); a rebuilt hub with the q fix.

## 2026-11-18 cheetax morning set: full artifacts (D1157)
Raw: data/botzone/6afcf9fa.json (win); gate-format logs/botzone/6afcf9fa.games.jsonl (win) and logs/botzone/cheetax_losses.games.jsonl (both losses, 109 + 117 plies); positions logs/botzone/cheetax_win/positions.bin (97 unique) + logs/botzone/cheetax_losses/positions.bin (220 unique). Analyze tables: D1154 section (win: turning point 84, saw-it 76) + D1155 line 110 (losses: turning points 93/105, saw-it 73/79, cheetax proves win ~6 plies first). Morning set: L, L, W, all Titanium o, player 1.

## 2026-11-19 12:17 match 6afed87d36b19c237a430f3b (user link, D1285): [noexcept]Ataxx (x, player 0) vs [EmptyBlue]Ataxx_CPP: noexcept WON (Botzone score 2-0). Titanium did NOT play.
Raw: logs/botzone/6afed87d_report.txt; positions logs/botzone/6afed87d_pos/positions.bin (115 positions, 115 unique under D4; usable for ti relabel); games.jsonl empty (no Titanium seat, so ti analyze has nothing to check).
- 117 plies, all verdicts OK. noexcept: 59 moves at 3,940 ms each (first 11,600 ms = full Botzone budget). Ataxx_CPP: 58 moves at ~970 ms each (first 1,940 ms) = it uses only ~1/4 of the 4 s budget.
- noexcept value q (cp = 400 ln((1+q)/(1-q))): +0.14 (~+113 cp) at the start, drifts to ~0 at plies 45-55 (min -0.018 = -14 cp), rises from ply 57 (+0.096 = +77 cp), +0.311 (+257 cp) at ply 63, back to +0.05..+0.21 through ply 91, then decisive: ply 93 +0.318 (+264), ply 95 +0.629 (+593), ply 97 +0.786 (+848), ply 101 +0.864 (+1047), ply 109 +0.987 (+2011), solved ~0.999 from ply 111 (moves in 1-10 ms).
- Reading: the same pattern as noexcept vs Titanium (D1155): long balanced middlegame, then a short window around plies 91-97 where noexcept's value jumps ~+600 cp in 2 own moves; even Botzone #3 (Ataxx_CPP, 2354) cannot hold it. Ataxx_CPP spends ~1 s/move, so this match says nothing about Ataxx_CPP at full time.
- NOT done: Titanium's own eval of these 115 positions (ti relabel --bin ... --nodes 5000, or a deep search of plies 89-97) to see whether Titanium would also miss the ply 91-97 turn.

## 2026-11-19 12:18 match 6afed83536b19c237a430f13 (user link, D1286): [noexcept]Ataxx (x, player 0) vs [EmptyBlue]Ataxx_CPP: noexcept LOST (Botzone 0-2). Titanium did NOT play.
Raw: logs/botzone/6afed835_report.txt; positions logs/botzone/6afed835_pos/positions.bin (115 positions); games.jsonl empty (no Titanium seat). Same pairing and seat as 6afed87d (above, noexcept WON 2-0), same timing (noexcept 3.94 s/move, Ataxx_CPP ~0.97 s/move), 117 plies, all OK.
- noexcept value q (cp = 400 ln((1+q)/(1-q))): +0.144 (+116 cp) opening, ~0 at plies 27-35, slides negative from ply 37 (-0.022), -0.14 (-113 cp) at ply 51, -0.246 (-201 cp) at ply 55; brief recovery to -0.033 at ply 65; then collapses: ply 67 -0.141, 69 -0.316 (-262 cp), 71 -0.364 (-305 cp), 73 -0.549 (-494 cp), 75 -0.681 (-665 cp), 85 -0.809 (-899 cp), 95 -0.930, 107 -0.990, lost (-0.999) from ply 109.
- Reading: noexcept is NOT unbeatable. Against the same opponent and seat it won one and lost one (opposite outcomes), so its q is noisy/odd-depth-like: it saw -0.2 (about -200 cp) already at ply 55 = ~15 plies before the collapse (ply 69-75, -262..-665 cp). Ataxx_CPP (private v4+, searches ~1 s) beats a 3.94 s noexcept here, so per-second strength is not the story; the opening/middlegame line decides.
- NOT done: Titanium eval of the 115+115 positions of these two matches (ti relabel --bin ... --nodes 5000): would show whether Titanium sees the ply 55-75 swing earlier than noexcept (-200 at ply 55).
