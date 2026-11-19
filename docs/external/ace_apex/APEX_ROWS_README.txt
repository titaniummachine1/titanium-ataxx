ace apex - search-labelled Quoridor positions (exported 30 Sep 2026)
=====================================================================

Two files, JSON Lines (one JSON object per line). Every value comes from ace apex's own MCTS search
(8x512 transformer + PUCT), NOT from game results.

VALUES: always from the point of view of the SIDE TO MOVE in that row's "state", in [-1, +1]
(+1 = side to move wins, -1 = it loses; roughly a win-probability-like score, 0 = even).

1) screens.jsonl  (678 positions)  -- best for move ordering / evaluation training
   Every legal move at the position was searched as its OWN tree ("sweep"): first every move at 10,000
   simulations, then the top candidates again at 100,000. So each move has its own value.
   {"state": ..., "side_to_move": 0|1,
    "moves": [{"move": "a3h", "value": 0.6487, "sims": 100000}, ...],   sorted best first
    "best_move": ..., "best_value": ..., "net": the apex net that did the search}
   A move's "value" = how good that move is for the side to move (the position after it, flipped back).
   "sims" 100000 = deep search; 10000 = screen only (a bit noisier). Moves within ~0.03 of the best are
   effectively tied.

2) nodes.jsonl  (75,384 positions)  -- best for evaluation (value) training
   Positions that received >= 10,000 visits inside a 100,000-simulation search.
   {"state": ..., "side_to_move": 0|1, "visits": N, "value": search value of the position,
    "children": [{"move": ..., "visits": n, "value": q}, ...]}
   Child "value" = that move's value for the side to move. ONLY trust children with many visits: a child
   with 1-10 visits has a near-raw network guess, not a search result.

STATE FORMAT   p0=e4;p1=e6;w0=9;w1=10;h=a3,f6;v=e5;t=1
   p0 / p1  pawn squares. Files a..i (left to right), ranks 1..9.
            P0 starts on e1 and must reach rank 9; P1 starts on e9 and must reach rank 1.
   w0 / w1  walls left in hand for P0 / P1 (10 each at the start).
   h / v    horizontal / vertical walls on the board, listed by anchor ("-" = none).
   t        side to move (0 = P0, 1 = P1).

MOVE TOKENS
   pawn:  the destination square, e.g. "e5" (jumps included).
   "a3h"  horizontal wall along the TOP edge of squares a3 and b3 (between ranks 3 and 4, covering files a-b).
   "e5v"  vertical wall along the RIGHT edge of squares e5 and e6 (between files e and f, covering ranks 5-6).
   Anchors run a1..h8. This is the same notation the claustrophobia bot arena uses.

NOTES
   - Duplicate positions were merged (screens: latest search kept; nodes: the most-visited kept).
   - Positions come from ace apex's own lines, arena openings, and positions from lost games; the labels
     are all ace apex's search. Different nets labelled different files (see "net" in screens).
   - Left-right mirror positions are NOT added; mirroring a position mirrors its moves (file x -> the
     opposite file) and keeps the values.
