# Titanium vs KataAtaxx arena: how many Titanium nodes equal N Kata visits

Measured 2026-10-31 / 2026-11-01 on Modal GPUs (L4 / A10G), ONE Kata search thread, Kata b12c384 (KataGomo release
Ataxx_20260501, ONNX). Titanium = main `3f5816e` (ProbCut), net `nnue_main.tnn`, classic rules, std5k book, every
book line from both seats, Botzone end rules. 1837 games, 26 pairings, joint maximum-likelihood fit with Titanium
1000 nodes = 0 Elo. This is the file to read; HANDOFFS only has the story.

## THE GRAPH
- `data/kata_arena/live.html` ("Titanium vs KataAtaxx: fitted Elo"). Blue = Titanium by nodes, orange = Kata by visits,
  one shared Elo axis, table underneath, auto-refreshing fit from the per-leg logs next to it.
- View: `cd data/kata_arena && python -m http.server 8765`, open `http://localhost:8765/live.html` (it fetches the
  `*.log` files beside it; opened as a file:// page it shows nothing). A tracked copy of the page is
  `docs/kata_arena_live.html` (copy it next to the logs to use it). `data/` is gitignored, so the logs/jsonl are local only.
- Data: `data/kata_arena/*.log` (per-game lines `Ti W-L-D`), `*.jsonl` (per move: Titanium depth/score/nodes, Kata raw
  value + policy, `kata-last-stats`). First ~250 ladder games and several partial legs have scores but no moves (Mistake 49/50).

## NUMBERS (fit, Titanium 1000 nodes = 0)
| Titanium nodes | Elo | | Kata visits | Elo |
|---|---|---|---|---|
| 1000 | 0 | | 1 | -30 |
| 2500 | +137 | | 2 | +233 |
| 5000 | +315 | | 4 | +283 |
| 10000 | +422 | | 5 | +302 |
| 20000 | ~+470 (23 g) | | 8 | +320 |
| | | | 10 | +295 |
| | | | 16 | ~+546 |
| | | | 20 | ~+489 |
Kata points at 4-10 visits are a noisy plateau (+-45..65); 16 and 20 visits are noisy too.

## HOW MANY TITANIUM NODES BEAT KATA (interpolated in log nodes from the table above, so approximate)
| Kata | Titanium nodes needed to match it |
|---|---|
| 1 visit (-30) | about 850-1000 (1000 nodes drew 47-53; 5000 nodes won 41-9; 100 nodes lost 2-48) |
| 2 visits (+233) | about 3,600 |
| 4 visits (+283) | about 4,400 |
| 5 visits (+302) | about 4,800 |
| 8-10 visits (+295..+320) | about 4,600-5,000 |
| 16 visits (~+546) | above 20,000 (beyond the measured range) |
So one Kata evaluation is worth roughly 1000 Titanium nodes.

## PER TIME
- GPU (L4): Kata 1 visit 50-55 ms, Titanium 1000 nodes 8 ms, 5000 nodes 40-46 ms. Kata 2 visits 17 ms is about +233,
  Titanium 5000 nodes 40 ms is about +315. At equal GPU time Kata is somewhat ahead at small sizes.
- CPU (PC, 1 thread, measured while another gate loaded 6 threads, so upper bounds): Kata 300-600 ms per evaluation
  (2 visits 617 ms, 4 visits 1287 ms, 8 visits 2532 ms). Titanium about 12 us per node. At equal CPU time Titanium is far
  ahead; Kata only catches up on a GPU. Re-measure on an idle box before quoting it.
- Earlier arena numbers (D156/D105): at 1 s Titanium beat KataAtaxx by +191 Elo (75-25-0) while using ~1.8x LESS time; at 100 ms KataAtaxx scored -108 Elo
  against Titanium even with ~9x the wall time. Titanium is far stronger per second; Kata only competes on a GPU.

## WHAT THIS IS AND IS NOT
- Strength per evaluation: Kata's policy+value network is far better per evaluation than ours; we win by evaluations per second.
- It does NOT say anything about PyJataxx (an MCTS bot, rank 7 on Botzone, beats Titanium; see D210, Mistake 36). Matchups are
  not transitive: Kata loses to Titanium, Titanium loses to PyJataxx, PyJataxx loses to Kata according to the Kata developer.
- Tools: `tools/modal_kata_arena.py` + `tools/kata_arena_run.py` are STOPGAP scripts (Mistake 49) to be ported to a hub
  subcommand `ti kata-arena`. Modal spend limit was hit on 2026-11-01 ~23:50; do not relaunch without credit from the user.
- Owed: 1 s vs 1 s leg, idle-box Kata CPU timing, 20000-node legs, port to the hub.
