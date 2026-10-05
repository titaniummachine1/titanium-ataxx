## STATE FOR NEW SESSION (2026-10-05 S31 PROMOTION PROTOCOL DONE — MAIN STANDS)

- Champ tupS25ow20 vs MAIN 1a27b92 @5k 100g = 25-73-2 (CI [0.174,0.346], -182). Conclusive REJECT, no merge. Self-check pooled to 100g: 45-51-4, identity-consistent. Law 10: no gate stops below 100g.
- Logs: `logs/s31_selfcheck_5k.txt` + `logs/s31_selfcheck_5k_ext.txt`, `logs/s31_champ_5k.txt`. Binaries: main = mainwt rebuild, champ = target/release + data/nnue/tupS25ow20.tup.
- S32 LADDER DONE: moon@5k beats main-classical@200k (0-99-1) and champ@200k (0-100). Margins flat -20 → -19 across 4x-40x. Nodes cannot buy this; knowledge gap confirmed. Next levers: eval knowledge or SMP, not nodes.
- S33 SPOT DONE: parity = titan@50k-100k vs moon@100 (~500-1000x handicap). 20k conclusively loses (0.375); 100k pooled 0.545 [0.476,0.614] coin flip. Fair-sparring recipe: moon@100 vs titan@50k+. Logs logs/s33_*.txt.
- Nothing else qualifies: corrhist/countermove/singular/aspiration all conclusive rejects on record; LazySMP unbuilt. main = strongest. Tuple line stays on exp/tuple-eval.
- Logs: `logs/s31_selfcheck_5k.txt`, `logs/s31_champ_5k.txt`. Binaries: main = mainwt rebuild, champ = target/release + data/nnue/tupS25ow20.tup.

## STATE FOR NEW SESSION (2026-10-05 S30 OPENING-HANDOVER DONE)

- Champ tupS25ow20 vs Moonbird 1.1.0 @5k (+/-20% jitter): takeover-after-moon-opening 0-100 AND direct 0-100. Margins -20 vs -21 stones. OPENING EXONERATED — gap is mid/endgame, not opening. First-ever champ-vs-Moonbird gate.
- Log: `logs/opening_gate_5k.txt`. Harness: `C:\Users\TERMIN~1\AppData\Local\Temp\opencode\opening_gate.py` (copy into scripts/ if reused; needs --jitter since both engines are deterministic at fixed nodes).
- Still FROZEN otherwise: branch exp/tuple-eval @576699d, resume via 09-22 pointers below.

## STATE FOR NEW SESSION (2026-09-22 FROZEN — user switched projects)

- STOPPED clean: all trainers + Moonbird workers killed, NOTHING running (verified empty process list).
- Frozen commit `1e88d3a` on `exp/tuple-eval`. main untouched throughout.
- Resume: `python training/moon_distill.py --top 200000 --nodes 5000 --ows 0.0,0.10,0.20`
  (resumable: moon_*.txt/.out skip-OK, then join -> train -> gate).
  Resweep resume: `python training/resweep.py --ows 0.05,0.10,0.15,0.20,0.25,0.30` (ow0 done: 32-63-5 -111).
- Data on disk: cache12M.npz (12.4M), hard200k.npy, fly12M_ep*.tup (14 ckpts), lab_moon/probe200 (2000/2000 moon scores OK).
- Champion: tupS25ow20. Morning tables pending: RESWEEP.txt, MOONDISTILL.txt.

## STATE FOR NEW SESSION (2026-09-22 S26-S30 OVERNIGHT)

- Champion: `data/nnue/tupS25ow20.tup` (ow 0.20, 73-26-1 +177 @5k 100g vs tupS21hvW). `CHAMPION.txt` points here. main untouched, all on `exp/tuple-eval`.
- cache12M.npz DONE (12,398,075 rows, 178.8MB, ow20-5k labels, 6wk ~100min). relab12m/ chunks retained.
- Flywheel S26: 25 epochs plateau-stopped, PROMOTION=REGRESSION-vs-base (fly 8-90-2 -402 vs S13 base; 60-134-6 -135 vs champ). ECHO CHAMBER at 12M: val 43k improves, play collapses. ckpts fly12M_ep*.tup retained, best ep14.
- Nightly: every ckpt STALLED vs champ (ep20 22-71 -186 worst). Nothing promotes without LB>0.
- Resweep S29 RUNNING (ow 0.0-0.30 on fresh 12M, table -> RESWEEP.txt): ow0-12M gated 32-63-5 -111 — old 0.20 blend did NOT survive teacher change. ow5+ training.
- Moondistill QUEUED (moon_distill.py, UAI-verified labeler): hard200k (rank done, wsum_mass 0.034) x Moonbird @5k, 8 workers -> cache12M_moon200k -> ow 0/10/20 train+gate -> MOONDISTILL.txt. Probe showed Moonbird hard-row speed ~1-3/s (slow!) + probe200.out came back EMPTY (0 rows, needs debug before full launch).
- Trainer is S28 streaming (int8 feats, per-batch index math, 20k-probe parity 13-17). OOM fixed (was 11.5GB swap death).
- NEXT: (1) fix moon probe (empty output), measure true Moonbird rows/s; (2) read RESWEEP.txt when done; (3) if moon wins -> expand; else capacity wall -> mobility/per-anchor tables, then SMP.

## STATE FOR NEW SESSION (2026-09-21 S22R2M VERDICT)

- 2M@20k self-teacher student: +0 (16-16). Champion stays tupS21hvW (+125, 2M dag/wsum). ECHO CHAMBER confirmed at scale: self-distillation fossilizes; diversity wins. 20k verdict: TOO MUCH. Label cheap (dag-5k), spend on rows/capacity/SMP.
- All label data retained (lab2m/lab400k/lab10k .out, cache2mR/400kR npz, gitignored) for future capacity experiments.
- NEXT: LazySMP build (exp/lazysmp SharedTt checkpointed). Then mobility tables, Moonbird re-gate, SPSA.

## STATE FOR NEW SESSION (2026-09-21 S22 CAPACITY-WALL)

- Ladder: +340/+220/+140 per 3x (log decay, user model confirmed). Relabel: 400k x 20k tuple-teacher -> R1 +56; teacher-parity 17-15 -> CAPACITY wall (29k), not teacher wall. Bootstrap needs capacity/architecture per round, escalating depth to stay ahead.
- Branches: tuple-eval (ahead, label cmd + dump_fens + tcol + R1/dag4k weights local), lazysmp (SharedTt checkpoint, paused).
- NEXT: LazySMP (run search, the unbottleable part) -> capacity bumps (mobility tables! per-anchor?) -> rebootstrap with escalating teachers -> Moonbird re-gate.

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

**MAIN IS UNTOUCHABLE (user law, Quoridor/Stockfish rule).** All work on
`exp/*` branches off `main`. Merge only when a branch decisively beats main
with no doubt it is stronger. Session corrupts ~every 5 min — LEDGER.md,
HANDOFFS.md, BEHAVIOR.md updated in real time so no state lives in chat only.

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
- S1 **slim-eval-speed** (2026-09-20): eval = material+PST+tempo;
  Ordering/radix/insertion/LMP/LUT paths deleted (−307/+41). Sperft 4.11
  vs old-lazy 2.40 (+71%). 100g@100ms 73–15–12 (+230). **MERGED to main
  `7ab844e`, pushed.** Main now = slim, sperft 4.29 Mnps.
- S2 sancta lazy-incr (branch `exp/sancta-incr`, off new main): minimal
  sancta.rs (load+refresh+SIMD forward, no S4Undo/PROF), root refresh once,
  child acc = parent acc + diffs at make(). OFF zero-cost (sperft 4.22 vs
  4.29). ON 5.0–5.2M NPS (swap ~free). 5k-node gate 10–20 (−140); 100ms
  gate 14–16 (≈parity — speed compensates). Sign double-negation bug
  found+fixed. NOT merged. Next: retune RFP/LMR for net scale OR saturate
  own net — user decision.
- S4v3 own-net (2026-09-21): v3 147x64+128→16→1 exact-distill (r4 25ep flat
  .294/.296, v3exact stale-math ep12 flat .2602/.2606 KILLED). Gate
  own_v3.s1 @ep9 vs classical: **0–30 @5k nodes** (identical 0-piece wipe
  pattern as v1/v2; bench d8 34043 nodes/2.76M vs classical 9401/6.5M —
  net multiplies nodes 3.6x, buys zero). Epoch check S4v3e: e1 0–20,
  e5 0–20 — never strong, NOT overfit, init/scale broken from ep0.
  Harness S5tool: `--net/--opp-net` presets on bench/serve/match
  (env vars DEAD), `scripts/gate.ps1 <games> <net> [5k|100ms|both]`,
  `training/export_epoch.py` re-exports any ckpt. Committed 4271d6b.
- S7quality zero-scaffold (2026-09-21, COMMITTED 2a0a0ba on
  `exp/sancta-incr` BEFORE net capacity experiments — user ordered
  Stockfish-grade quality first): Arc/unsafe OUT (borrowed `&'a S1Net`),
  MoveStack owned by Searcher (no per-search alloc), score_into free fn,
  integer NMP fill, fallthrough-terminal, aspiration ladder ±50→200→800
  from depth 5, dead perft()/imports/from_v1_parts pruned. 26t green,
  sperft 4.22–4.34M x3 (28.2M nodes — aspiration deeper, expected change
  from 535k), bench d8 classical 17034/6.9M, self-check 10–10.
  S6score ep25 gated: 20–0 classical @5k (real games, not wipes).

## STATE FOR NEW SESSION (2026-09-21 S21 SHELVED)

- Branch exp/tuple-eval: full campaign done, SHELVED unmerged. Shipped infra (geometry/loader/forward/flags/native-mode, all tested+committed) + trainer (base ports, wsum, TUP4-i8). Weights local only (data/nnue/tupS21*.tup, gitignored): best = tupS21hvW.tup (H+V wsum, +125@5k). Proved: pairs carry +125-family knowledge (x5 runs); cost floor -44pct (splat+i8+L1+s kips, lookups dominate); time gates unwinnable (-55..-66); singles flat; outcome-columns poison.
- main = 1a27b92. Shelved branches: search-retune, corrhist, countermove, singular, search-tweaks, tuple-eval.
- NEXT (recommended): LazySMP - 8 idle threads, discontinuous throughput (~6-7x => +2+ plies => +200+), multiplies ALL banked knowledge; re-gate tuples at depth after (knowledge compounds with depth per S13 pattern). Then Moonbird re-gate, then SPSA.

## STATE FOR NEW SESSION (2026-09-21 S13 MERGED)

- main = S13contact MERGED (merge commit, pushed): eval = material+PST+tempo+E2contact+E4multicap. tune_contact.exe == new main baseline (sperft ~4.3M, bench d8 a1b1/100 13252n). Logs logs/s13_* (gitignored).
- exp/contact-eval merged, exp/search-retune closed (S11+S12 docs live on that branch, no code delta). mainwt worktree untouched (dirty Cargo.toml hack left alone).
- USER INSIGHT (2026-09-21): per-node information is capped in Ataxx (800k transformer +25 Elo same-nodes; NNUE wins at 4-5x nodes) -> intelligence must come from RELATIONSHIPS BETWEEN NODES, not richer nodes. Agenda: correction history (learns eval-error = search-vs-static = pure node-relationship signal), singular extensions (sibling comparison), counter-move history, SPSA-from-game-outcomes (inter-node learning). E2/E4 reframed: already proto-relational (stone-vs-enemy-reachability), which is why they beat the pixel-net.
- NEXT: pick first relational lever (recommend correction history: online ML, ~zero cost, Stockfish-proven) + Moonbird re-gate at new strength.

## STATE FOR NEW SESSION (2026-09-21 close)

- Branch: `exp/sancta-incr` = 3 commits past main `7ab844e`:
  `f8b98b5` (v3 loss contract), `4271d6b` (harness presets + S6),
  `2a0a0ba` (S7quality). main UNTOUCHED = slim `7ab844e`.
- S7quality = REGRESSION both gates (QB 16–84 @100ms rerun + 45–53–2 @5k).
  Aspiration + borrowed-net rewrite lost quality for speed. Branch holds it.
- Working tree dirty (uncommitted, measured no-gain micro): `pattern.rs`
  LUT draft + `is_clone`/`FULL` shift-spill fixes. Safe to discard or keep.
- Movegen SATURATED: LUT thread closed 3 ways (key sizes 2^48/2^32 dead,
  identity argument, measured make 74M vs via_lut 33M). perft_bb bulk
  3.25B pairs/s @d7. Remaining ±10% micro only.
- NEXT TARGET (user-ordered): eval-saturate. Incremental PST in Board
  (+pst:i32, fold delta into make() capture loop = free), evaluate() →
  3 ops, DELETE eval_cached (double TT probe + pollution per node).
  O(1) eval, bit-identical values → 5k gate must be 15-15 by construction,
  100ms gains from speed. Then score_into popcounts + NMP dist_union.
- Gates (S8gatedef): 5k nodes = quality-per-node, 100ms = strength. Run
  BOTH for every experiment. `scripts/gate.ps1 <games> <net> [5k|100ms|both]`.
- Nets: own_v3.s1 learned (pred_cp 108/112 vs tgt 120) but gates 0 both.
  sancta_w.s1 = owner's brain. data/ logs/ scripts/ target/ gitignored.


## Next levers (in the order we'd take them)

1. **Eval weight tuning loop** — automate: parameter sweep over CONTACT_PEN,
   MULTICAP_PEN, TEMPO, HOLE_PEN curve with the gate harness (their Elo came
   from self-tuning, ours are hand-guessed).
2. **2×2 structure pattern eval** (Moonbird's biggest feature): per-square
   2×2-neighborhood index, needs the tuning loop first.
3. Correction history keyed by occupancy-hash bucket (titanium pattern).
4. Aspiration windows (ladder ±50/±200/±800 from depth 3, autaxx).
5. LazySMP with TOTAL node budget across workers (Arc<AtomicU64> counter —
   titanium commit 8a0399d pattern) — we have 4c/8t idle during gates.
