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
