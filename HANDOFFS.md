# TITANIUM — engineering handoff notes

Everything measurable lives here so we never re-derive it. Update after every
optimization round. Local game logs accumulate in `logs/` (gitignored but
NEVER deleted — played games are training data for a future learned eval).

## Machine

4 cores / 8 threads, Windows, x86-64-v3 build (POPCNT/BMI2 enabled via
`.cargo/config.toml` — this alone was +40% on bitboard ops).

## Rules & geometry (do not re-litigate)

- 7x7, one piece per corner (black a1 bottom-left, white g7), black first.
- Clone = compass distance 1 (8 dirs). Jump = full 5x5 ring (16 cells,
  knight offsets included). After ANY landing, adjacent enemies convert.
- Jump vacates the origin (this was a real bug once — caught by the
  `perft_bb_differential` test; keep that test green).
- No legal move = forced pass; double pass / full board / wipe = game over.

## Movement generator (perft, all `(from,to)` pairs, mirror-checked)

| depth | nodes | knps |
|---|---|---|
| 5 | 498,060 | ~2.0M |
| 6 | 11,823,096 | ~2.3M |
| 7 | 371,874,782 | **3.25B pairs/s** |

Path to 3.25B: adaptive leaf side (count from whichever of pieces/targets is
smaller) → hardware POPCNT (x86-64-v3) → depth-2 bulk inlining (no child
recursion, no targets recompute). Clone folding: all clones to the same
destination give the identical child → one recursion × multiplicity.

Movegen micro (genbench, 2M iters): bulk targets 66M/s, full move-list fill
5.0M/s (~76 moves avg), make 90M/s, make_via_lut 50M/s, extract8 portable
175M/s vs PEXT 600M/s.

## LUT architecture (your Quoridor-style design, as implemented)

- `NEIGH_POS[sq]`/`NEIGH_CNT[sq]`: per-square neighbor slots (corner 3, edge
  5, center 8) — off-board is a permanently-empty padding bit; no special
  casing anywhere.
- `CONVERT8[256]`: ONE universal table, square-independent (identity under
  standard rules — key IS the answer). Rule variants (support-required
  infection, shielding) change this table, not make().
- PEXT (`_pext_u64(occ, RING1[to])`) gives the same key as the portable
  loop because NEIGH_POS enumerates ring squares in ascending square order.
- Direct `opp & RING1[to]` beats the LUT for standard rules (measured):
  the LUT is the rule-variant hook, not the default path.

## Search (native binary; wasm is only the browser compat layer)

Current: negamax alpha-beta + iterative deepening + 1M-entry TT (zobrist,
incremental hash on Board) + killer moves + history heuristic + eval cache
(TT depth-0 EXACT entries) + clone-dedup movegen + lazy move selection.
Eval v0.2: pure material (piece diff × 100); mobility switchable via
`MOBILITY` const (currently 0). Node/time/depth limits: whichever trips
first stops the search (partial ID iterations discarded).

Ordering A/B (`sperft --depth 8`, 8-position suite, identical scoring):

| strategy | nodes | time | speed |
|---|---|---|---|
| **lazy selection** (default) | 72.6M | 17.7s | **4.11 Mnps** |
| insertion sort (full) | 71.3M | 40.2s | 1.77 Mnps |
| LSD radix (3×8-bit passes) | 71.3M | 35.7s | 2.00 Mnps |

Radix got a fair trial (user suggestion) and lost 2.3×: cut-nodes only
examine the first few picks, so pay-as-you-go selection beats full sorts.
Insertion == radix node counts exactly (both stable, same keys).
Search nps ~4-8M vs movegen 3.25B pairs/s — the gap is eval/TT/overhead per
node; that's the next optimization surface (sperft is the metric).

Search speed log:
- pre-TT/killers/history (2026-09-16): ~2.9 Mnps suite total
- post: 4.11 Mnps lazy (suite depth 8)

## Benchmarks vs competitors

Gates (user rule: NEVER cap depth for opponents — node or time limits only;
`--opp-nodes N` sends Moonbird `go nodes N` so both sides get equal compute):

- **Gate A — equal compute**: both engines 5000 nodes/move.
- **Gate B — equal strength**: both engines 100 ms/move.
- 8 parallel match shards × 4 games = 32 games per gate; a full gate finishes
  in <1 minute. Games log incrementally (interrupt-safe).

Results (2026-09-16, engine v0.2: TT+killers+history+eval-cache, material eval):

| gate | result |
|---|---|
| 5000 nodes vs 5000 nodes | **titanium 0 - 32 Moonbird** |
| 100 ms vs 100 ms | **titanium 0 - 32 Moonbird** |
| 1000 ms vs 1000 ms (pre-TT baseline, 50 games) | titanium 1 - 49 Moonbird |

Reading: Moonbird wins at EQUAL node counts too — the gap is eval + pruning
quality (its tuned eval, LMR, null-move, aspiration), not speed. Our nps is
comparable; our node *efficiency* is not. Strength work = better per-node
decisions, in the order listed under "Next levers".

Opponent: **Moonbird 1.1.0** (tsoj, prebuilt exe, UAI, "superhuman" classical
AB), `scripts/moonbird/`. Backup: Ciekce/sanctaphraxx (Rust/cargo/UAI/NNUE).
Ceiling: KataAtaxx (hzyhhzy/KataGomo releases, AlphaZero MCTS) — upper bound,
not a peer.

## Fast iteration protocol (4c/8t)

Short gates only: **5000 nodes/move** (equal-compute) and **100 ms/move**
(equal-strength). 8 parallel shards, 4 games each, both engines on the same
gate. Example shard:

```powershell
Start-Process target\release\titanium-cli.exe -ArgumentList "match","--games","4",
  "--nodes","5000","--opp","scripts\moonbird\Moonbird-1.1.0-windows-amd64.exe",
  "--opp-nodes","5000","--start-game","$sg",
  "--out","logs\gateA_shard$i.txt" -WindowStyle Hidden
```

`--start-game` balances colors across shards (game index parity decides
color). 20000-node budgets for self-play regression checks (self-play
@ 20k nodes: 8 - 8 over 16 games — no color bias).

## Data retention (training)

`logs/*.txt` keeps every played game: header, per-game result + full move
list (space-separated, `e2e3`/`pass`). Never delete. When we train a learned
eval, parse logs for positions+results; `data/` is the designated export
target (both gitignored).

## Next levers (in the order we'd take them)

1. Quiescence search (captures-only at depth 0) — biggest strength gap vs
   Moonbird-class engines.
2. Null-move pruning + LMR (needs care with Ataxx zugzwang — test via sperft
   node counts + selfplay).
3. Aspiration windows at root.
4. Mobility re-enabled + piece-square/edge-correction eval terms.
5. SEE-style exchange eval for infection chains (probably overkill: Ataxx
   conversions are not sequential like chess captures).
