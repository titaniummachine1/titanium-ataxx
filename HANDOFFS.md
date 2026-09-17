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

**Fishtest protocol (established 2026-09-16):** every experiment = branch +
binary snapshot; A/B vs `main` at BOTH gates (5k nodes = equal compute,
100ms/10ms = equal strength); merge only on a win; **Moonbird runs only after
a proven self-play win**. Full history in `LEDGER.md`.

Gate results timeline (all 32 games, 8 shards, <2 min per gate):

| engine state | vs Moonbird 5k nodes | vs Moonbird 100ms |
|---|---|---|
| E0 baseline | 0–32 | 0–32 |
| E0 baseline | — | 0–32 (1s gate: 1–49) |
| after E2+E3b+E4 merged (~+700 self-play Elo) | 0–32 | 0–32 |

Opponent: **Moonbird 1.1.0** (tsoj, prebuilt exe, UAI, "superhuman" classical
AB), `scripts/moonbird/`. Backup: Ciekce/sanctaphraxx (Rust/cargo/UAI/NNUE).
Ceiling: KataAtaxx (hzyhhzy/KataGomo releases, AlphaZero MCTS) — upper bound,
not a peer.

## Fast iteration protocol (4c/8t)

Short gates only: **5000 nodes/move** (equal-compute) and **10–100 ms/move**
(equal-strength). 8 parallel shards, 4 games each. `--opp-nodes` for
Moonbird's `go nodes N`. `--start-game` balances colors. Games append to the
log immediately (interrupt-safe). A/B between own binaries:
`--opp "scripts\bench\titanium_main.exe serve"` (UAI). NOTE:
`Start-Process -ArgumentList` does NOT quote elements with spaces on PS 5.1 —
embed quotes manually: `$oppQ = "`"$m serve`""`.

## Experiment log (see LEDGER.md for full detail)

- E1 autaxx-style overhaul — retained (architecture), no match gain.
- E2 **virus-strategy eval** (user): endangered-stone contact −15/stone,
  clone/jump ordering — **+190 Elo. MERGED.**
- E3 tuning bundle — REJECTED (node-budget prediction under-consumes budget).
- E3b adoption-only — **+190 Elo. MERGED.** (partial-iteration adoption =
  real Elo at node gates)
- E4 **multi-capture exposure** (user): −20 per extra stone convertible in
  one enemy landing — **+330 Elo. MERGED.**
- Cumulative ~+700 self-play Elo; Moonbird still 0–32 at both gates.
- E6 weakness (user: `conv × attackers` per enemy-reachable landing) —
  REJECTED as classical term (4–28 vs no-holes; 14–18 at WEAK_PEN=1), kept as
  NNUE dense feature (`weakness_features`, 8 scalars/record). Prod eval is
  no-holes now (MASK_HOLES out of MASK_ALL).
- ASP1 aspiration 8cp×1.9 (Moonbird) — REJECTED 3–29 @5k, 15–17 @100ms.
  Thrashes on quantized eval.
- ASP1b ladder 50/200/800 (autaxx bounds + Quoridor 4x widening) — PARKED:
  13–19 @5k / 20–12 @100ms, pooled 33–31 ≈ +7. Window buys nothing with
  quantized eval; smoothness must come from NNUE. BUG LESSON: mate scores
  saturate fixed window ceilings → infinite research; saturated bounds MUST
  accept (guard in `search.rs` root loop).
- DAG1 fold — COMMITTED (dagfold cmd + Board PartialEq, fcff86b): shard0
  339k lines → 195k nodes / 42% dedup / edges, 23.8MB db. DB feeds Net0.
- NET0-plan: NO GPU on box → torch-cpu trainer for 28k params. Eval-first
  CONFIRMED (ASP proved search can't win quantized; Moonbird 13x slower/node
  still 0–32 on eval quality). Net0 before more search work.
- A2 LazySMP phase 1 (private TTs, `search_smp`, --threads) — REJECTED:
  11–21 @5k TOTAL / 17–15 @100ms. Helpers tax shared budget. Phase 2
  (shared atomic TT) or drop — USER DECISION (recorded 2026-09-17).

## Session state 2026-09-17 (READ FIRST if session starts cold)

Branches (`titanium-ataxx` repo): `main` (E2+E3b+E4), `ablation/eval-inputs`
@f089f37 (harness + no-holes + nnue.rs + datagen), `exp/ataxx-aspiration`
(ASP1b PARKED), `exp/ataxx-lazysmp` (A2 REJECTED — do not merge).
`titanium_smp4_cand.exe` = A2 candidate (REJECTED); `titanium_abl_base.exe` =
f089f37 baseline for all A-gates. A2 logs: `logs/a2_smp_5k.txt` (11–21),
`logs/a2_smp_t100.txt` (17–15). Gate scripts `scripts/bench/a{1,1b,2}_*.ps1`.
Binaries in `scripts/bench/` (gitignored, rebuild if missing):
`titanium_abl_base.exe` = f089f37 baseline, `titanium_asp_cand.exe` = 8cp
REJECTED, `titanium_asp100_cand.exe` = ladder PARKED.
Game logs `logs/a1_asp_5k.txt` (3–29), `logs/a1_asp_t100.txt` (15–17),
`logs/a1b_asp100_5k.txt` (13–19), `logs/a1b_asp100_t100.txt` (20–12).
Datagen: `datagen --games 2000 --nodes 5000 --seed 2000+N --opening-plies 8
--out data/nnue/datagen_v2_shardN.txt` (4-field records with weakness
scalars) + old 3-field `datagen_shardN.txt` (~190k pos, Net0-usable).
Quoridor side: engine-submodule branches `exp-ataxx-iir` (gating),
`exp-ataxx-nmp-gate`, `exp-ataxx-lazy-order` (chained); see
Quoridor_best_AI/LEDGER.md 2026-09-17 entry.

## OPERATING RULES (learned 2026-09-17, do not re-derive)

1. Tool-call timeouts TREE-KILL the shell including `Start-Process` children.
   Long jobs MUST spawn via WMI `Win32_Process.Create` (fully detached).
   Reboot kills everything: after any restart relaunch datagen + gate chains
   (builds skip if binary present, datagen appends — all resumable).
2. Ataxx node gates REQUIRE `--opp-nodes N`: without it the serve side plays
   `movetime 1000ms` default while the driver is node-capped (silent unfair
   gate; burned one 32-game run before catching).
3. 32-game gates ≈ ±45 Elo CIs (50-game ≈ ±40): only large effects decidable;
   20–30 range needs expansion, never merge on it.
4. Gate scripts live in `scripts/bench/a1*.ps1` (gitignored, on disk only).

## Next levers (in the order we'd take them)

1. **Eval weight tuning loop** — automate: parameter sweep over CONTACT_PEN,
   MULTICAP_PEN, TEMPO, HOLE_PEN curve with the gate harness (their Elo came
   from self-tuning, ours are hand-guessed).
2. **2×2 structure pattern eval** (Moonbird's biggest feature): per-square
   2×2-neighborhood index, needs the tuning loop first.
3. Correction history keyed by occupancy-hash bucket (titanium pattern).
4. ~~Aspiration windows~~ DONE 2026-09-17: 8cp REJECTED, ladder 50/200/800 PARKED (pooled +7). Dead until eval is smooth (NNUE).
5. ~~LazySMP phase 1 (private TTs)~~ DONE 2026-09-17: REJECTED 11–21 @5k TOTAL / 17–15 @100ms. Phase 2 = shared atomic TT (real Quoridor pattern) or drop — USER DECISION PENDING. Next otherwise: Net0 torch-cpu trainer on DAG db.
