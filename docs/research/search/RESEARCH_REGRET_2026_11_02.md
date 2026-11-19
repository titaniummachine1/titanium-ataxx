# REGRET TUNING — deep-reference regret instead of game noise (2026-11-02)

Implements the user idea logged as D373(A): score a search configuration by how much value it
throws away against a DEEP search on the same positions, instead of by W/L over ~100 games.
Built and first measured the same day; nothing promoted by this document.

## Why

A paired game gate costs ~200 games to resolve a few Elo (score noise ~±5 points per 100 games).
Regret against a deep reference is measured per POSITION, is deterministic at fixed nodes, and
prices exactly what pruning buys (depth) and costs (missed value) — the quantity the knobs act on.
The final proof of strength stays the paired game gate (LAWS B1/B2).

## Tool: `ti regret` (hub, Rust — no Python)

Code: `titaniumtest/src/main.rs`, `run_regret_cli` (dispatch arm `regret`). Build:
`$env:CARGO_TARGET_DIR="C:/gitProjects/titaniumtest/target_phase"; cargo build --release --bin ti`.

Method, per position:

1. Reference search at `--ref-nodes` (default 500,000) with the reference engine (default = baseline
   command) → `(m_ref, v_ref)`, value from the mover's point of view.
2. Shallow engine at `--nodes` (default 10,000) → `(m_shallow, v_shallow)`.
3. If `m_shallow == m_ref` → move regret 0.
   Otherwise re-search the child position after `m_shallow` with the reference engine at
   `--ref-nodes` → child value `v_child` (opponent's view), so the value of the played move is
   `-v_child` and `regret = v_ref + v_child`, clipped to ±2,000 cp (clip count reported).
4. Also reported: move agreement %, value gap `v_ref - v_shallow`, search depth, per-empties phase.

Determinism: fixed node budgets, `uainewgame` before every search (LAWS B4), one engine process
per thread, no clock. Mate scores map to `±(30000 - 100·|mate|)` so ordering is preserved.

```powershell
ti.exe regret --book data/nnue/book50k_std5k.txt --book0 33000 --limit 200 `
  --ref-nodes 500000 --nodes 10000 --threads 8 `
  --candidate "tunepol.exe serve --tune <theta> --network nnue_main.tnn" `
  --baseline  "tunepol.exe serve --network nnue_main.tnn" `
  --out logs/gates/regret_C_200.csv
```

## Result — theta variant C vs defaults (200 positions, 2026-11-02)

Same exe `logs/tunepol_1cf931f.exe` (3060E318), net 46FCB5A4, book `book50k_std5k.txt` from 33000,
ref 500k nodes, probe 10k nodes, 8 threads, 487.1 s. Variant C = theta@k=277 with all LMR knobs at
defaults (D382).

| | agreement with ref move | mean regret (cp/move) | mean value gap (cp) | mean depth |
|---|---|---|---|---|
| variant C | 55.0% (110/200) | **7.0** | +2.2 | 8.22 |
| defaults | 57.5% (115/200) | **8.6** | +3.3 | 8.14 |

- Paired regret difference (C − defaults): **−1.58 cp/move, sd 12.1, SE 0.85, t = −1.84**
  → normal-approx P(C better on this objective) ≈ 96.7% (one-sided). Not a promotion: it is a
  regret objective, and the game gate still decides (LAWS B1/B2).
- Agreement is slightly WORSE for C (−2.5 pp) while its misses are cheaper: only 13 of 200
  positions differ in agreement (106 both agree with the reference, 81 neither does). Agreement is
  a coarse binary read; the regret magnitude is the informative statistic.
- Value gap difference −1.1 cp, SE 1.25 → not distinguishable.
- No clips (max regret 260 cp, one position >100 cp).
- Power: per-position regret sd ≈ 12 cp, so SE = 12/√n; a 1.6 cp/move effect needs ~200 positions
  (as here), a 1 cp effect ~600, a 3 cp effect ~70. A 200-position run costs ~8 min on 8 threads.

Small-sample warning (recorded because it happened): the first 8 positions of the same book range
gave C +12.5 pp agreement and −7.8 cp/move; the 200-position run reversed the agreement sign.
Never read this tool below ~100 positions.

## Limitations / not checked

- **Phase coverage**: all 200 positions sit in the 30-49 empties bucket, because the paired book
  only contains opening lines. This run says nothing about the endgame — exactly where the 100 ms
  game legs showed variant C's damage (ply 60+, D379/D380). A mid/endgame position set is required
  before this metric can arbitrate the TC-dependence question.
- The reference is the same engine at 500k nodes, not ground truth: regret is measured relative to
  the reference's own blindness (it inherits the reference's errors). A 1M-node or different-engine
  reference is a separate arm.
- The reference re-search of the child uses a cleared TT (`uainewgame`), so no TT information leaks
  from the parent search; it also means the child value is a fresh 500k search, not a PV extension.
- Mate handling is a heuristic ordering map, not exact solve; unexercised here (no mates in the set).
- Not yet used as an SPSA objective, and not yet used to screen single knobs.

## Next steps (ranked)

1. Build a position set with midgame and endgame coverage (real game positions, e.g. the analysed
   Botzone games) and re-run the same comparison; this is the run that can judge variant C's
   late-game damage without paying for 1,000 game legs.
2. Use regret as the SPSA objective at the depth that is played (ref 500k-1M, probe 20k-50k), then
   confirm the theta with the normal paired time gate.
3. Single-knob regret screens (one knob moved vs defaults) as a cheap pre-filter before game gates.

## Raw data

- `logs/gates/regret_C_200.csv` — per-position rows (book_idx, empties, v_ref, both moves, both
  regrets, agreement, depth).
- `logs/gates/regret_smoke.csv` — the 8-position smoke run.
- Command lines are in this document; the tool lives in the hub repo (`src/main.rs`), no Python.

## Result 2 - real game positions (endgame coverage), 2026-11-02 22:05

`ti regret --positions FILE.bin` was added: it reads the hub's `positions.bin` (25 B per row, LE:
occ0 u64, occ1 u64, blockers u64, turn u8) written by `ti botzone --out DIR`, so real game positions can
be probed. Source: `logs/botzone_6ae8ba1d4b/positions.bin` = 100 unique positions from match
`6ae8ba1d4b9c28024547d962` (V18 won 2-0 vs PyJataxx), ref 500k, probe 10k, 8 threads, 234.8 s.

| | agreement with ref move | mean regret (cp/move) | mean depth |
|---|---|---|---|
| variant C | 62.0% | **3.0** | 9.0 |
| defaults | 57.0% | **8.6** | 9.0 |

Paired regret difference (C - defaults): **-5.51 cp/move, sd 33.9, SE 3.39, t = -1.63**
=> P(C better) ~94.8% (one-sided normal approx). Agreement differs on only 19 of 100 positions
(50 both agree, 31 neither).

Per empties bucket (paired diff, cand - base):

| empties | n | paired diff cp | SE |
|---|---|---|---|
| 0-9 | 31 | **-14.5** | 8.9 |
| 10-19 | 29 | -0.3 | 5.1 |
| 20-29 | 24 | -0.2 | 1.7 |
| 30-49 | 16 | -5.6 | 7.3 |

Reading: on real positions the opening-only result holds (C ahead) and the ENDGAME does **not** show the
damage the 100 ms game legs suggested (D379/D380); if anything C is better at 0-9 empties, but n=31,
SE 8.9, and the sample is one match of Titanium's own (winning) games -> biased and wide.

Sampling cost: per-position sd is 33.9 cp on real positions vs 12.1 cp on book positions, so the same
SE needs ~8x more positions here (sd^2 ratio). Budget 300-600 real positions for ~1-2 cp resolution.

Caveats: positions come from Titanium's own moves in one match (selection bias toward lines our engine
chose); the reference is still the same engine at 500k nodes; the value gap on these positions is
-279 cp on both sides, i.e. the shallow search is systematically ~280 cp more optimistic than the deep
reference on this sample (expected: deep search finds the opponent's counterplay).

Raw: `logs/gates/regret_bz_C_100.csv` (smoke: `logs/gates/regret_bz_smoke.csv`).
