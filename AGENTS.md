# AGENTS.md — Titanium Ataxx (auto-loaded every session, read this first)

> This file is the whole project briefing. A fresh session must know the rules
> WITHOUT opening anything else. Details live in `BEHAVIOR.md` (laws),
> `HANDOFFS.md` (current state, newest entry on top), `LEDGER.md` (experiment
> history). Read those three before touching code.

## 1. What this is

Rust alpha-beta Ataxx engine (7x7), workspace `C:\gitProjects\titanium-ataxx`:
`crates/titanium` (core: bitboard rules, movegen, search, tuple net),
`crates/titanium-cli` (native binary: perft/bench/sperft/selfplay/serve/match),
`crates/titanium-wasm` (browser compat only), `web/` (Vite GUI),
`training/` (harness + Botzone entry), `data/`/`logs/` (gitignored scratch).

## 2. Where the strongest engine lives (do not guess)

- `main @ 73a859d` + default net `data/nnue/tupS55tree.tup` (726440 B,
  sha256 `F66A00FE…899D7C`) is the STRONGEST. `main` default eval IS that net.
- THIS folder is usually on an `exp/*` branch (e.g. `exp/search-buckets`),
  which defaults to CLASSICAL hand-eval (`net: none`) ≈ **-500 Elo** vs
  net-main. Classical-vs-classical gates NEVER prove strength.
- Clean copies: `C:\gitProjects\main-ataxx` (`[main]` worktree),
  `C:\gitProjects\titanium-main` (detached `73a859d`).
  `C:\gitProjects\baseline-ataxx` = pre-batch baseline `b62a78e`.
- First question on any strength task: "which binary + which net hash on BOTH
  sides?" Log exe sha + tup sha in every gate (harness does this).

## 3. Iron laws (verbatim from BEHAVIOR.md — never re-litigate)

1. **MAIN IS UNTOUCHABLE.** Work on `exp/*` only. No local main edits, ever.
2. **TIME CROWNS — but NODES CAN MERGE.** Intelligence-only change (speed
   provably untouched): 5k-node gate CAN merge alone (fairest, zero clock
   noise). Speed affected: chosen time controls (50/100/200ms per gate,
   both chosen legs must win); 5k is diagnosis.
   **STOCKFISH STACKING:** main absorbs EVERY proven gain, however small.
3. **MAIN IS ALWAYS STRONGEST.** Nothing merges (code or weights) without a
   sufficient-games gate proving the new build stronger. Straddling CI =
   EXTEND, never merge, never reject on the bound alone.
4. **NEVER REJECT ON A DIP — EXTEND BOTH WAYS.** Ahead-but-insufficient →
   EXTEND, never merge early. Behind → EXTEND too (FOMO). Elo/score tells
   ahead from behind (most important metric) — the gate proves DIRECTION
   (better/worse), never magnitude. Reject only on conclusive CI below 0.5
   AND user confirmation (user calls every merge AND every rejection).
5. **NO GATE STOPS BELOW 100 GAMES.** Every leg ≥100g, then extend on straddle.
6. **GATES ARE BY SEAT + PAIRED BOOK.** Exactly 50/50 per side from log;
   every book line (`data/nnue/book50k.txt`) from BOTH seats. Win-both-seats
   = real edge (even 1pt is signal, esp. fixed-nodes). Same line+seat must
   replay bit-identically or the run is VOID (infra bug).
7. **CLEAR ENGINE MEMORY EVERY GAME.** `uainewgame` before every gate game,
   both sides — TT/killers/history reset. No reset = VOID run.
8. **GATES RUN BETWEEN GIT REFS ONLY (fishtest rule).** Both sides = branch
   or commit, never dirty. Record exe sha + commit hash + net hash both sides.
8. **ONE variant per binary.** Experiments live on branches, not flags in hot path.
9. **TRASH UNTIL PROVEN.** Microbenchmarks don't crown — alternating `sperft`
   pairs + real SPRT do. Single/min-of-N probes are jitter.
10. **GATE LAW v4 / SPRT:** `training/gate_check.py --cand <tag>` recomputes
    LLR (H0 elo≤0 vs H1 elo≥20); PASS needs LLR ≥ +2.94. Merge needs tripwire +
    explicit user approval naming branch/net + LEDGER promotion row with DB
    entries. `training/results.py backfill` parses `logs/*.txt` into
    `data/nnue/results.db`. (Both scripts currently missing on some branches
    — hand-compute LLR + seat counts and note it until restored.)
11. **NO REJECTION WHILE FOMO LIVES.** Undertrained net that loses stays
    PARKED, never rejected. **ENGINE ADAPTS TO NET** — never strip/convert a
    net blob to fit a loader. Load-time allzero skips (W4 has_q2) are NOT
    damage: path stays intact, nonzero weights take the full path.

## 4. Rules: two of them (do not mix)

- **Classic** (UAI serve default): pass on no-move, double-pass/full/wipe ends.
- **Botzone** (judge): NO passes — either side stuck ends immediately, empties
  go to mobile side (`49 - stuck`); `-1 -1 -1 -1` reply = INVALIDMOVE loss;
  2-position repetition = loss. Engine has BOTH via build-time const generic:
  `search`/`parallel_search` = classic, `search_bz`/`parallel_search_bz` =
  Botzone (NMP off, `bz_over`/`bz_counts`). Zero hot-path runtime flag.
- Judge wire format is JSON-in/out; the shipped bundle
  (`C:\gitProjects\botzoo\titanium-ataxx-v1.rs`, built by
  `training/bundle_botzone.py`, checked by `training/verify_bundle.py`) still
  speaks the LINE simple-protocol — JSON front-end NOT built (needs
  player-side schema confirmation).

## 5. Tournament entry (current shape)

- `serve --tournament [--threads N] [--bz-rules]`: 128 MB TT default
  (tt_bits 23, ~192 MiB actual), startup autotune 1,2,3,4,6,8… while total
  grows >5%, `Threads` + `BotzoneRules` setoptions, silent auto-ponder after
  `bestmove`, next `position`+`go` aborts and converts with warm TT.
- `training/botzone_main.rs`: stateless per-turn, 800 ms / 1500 ms first move,
  TT bits 23, threads = `available_parallelism` clamped 1..=4, hardcoded
  `_bz`, repetition guard stricter than judge. Turn peak ≈196 MB vs 256 MB
  cap (bits 23 kept; bits 22 fallback if MLE signals).
- NOT built: Botzone JSON I/O; per-move thread auto-scale; position-without-go
  premature search.

## 6. Commands (PowerShell 5.1 — no `head`/`sed`/`&&`)

```powershell
cargo test --workspace
cargo run -p titanium-cli --release -- bench --depth 8
cargo run -p titanium-cli --release -- sperft --depth 8   # speed truth, alternating pairs
python training/results.py backfill; python training/results.py gates
python training/gate_check.py --cand <tag>
python training/bundle_botzone.py; python training/verify_bundle.py
```

## 7. Box traps (do not relearn)
- Edit preserves mtime → `os.utime` after EVERY edit, verify `git diff`.
- Stale `training/__pycache__` after mtime-preserving edits → delete it.
- `place_bench` orphans accumulate → `taskkill /F /IM titanium-cli.exe /T`;
  they steal CPU but can't skew fixed-node results.
- `schtasks /tr` 261-char limit; STOP file = `C:\gitProjects\logs\STOP`.
- `LEDGER.md` is CRLF — Edit tool fails on it; append via python script.
- `uai_probe.py` hangs here — use direct `serve` smoke via `uai_smoke.in`.
- Determinism proven (identical engines replay bit-identical 200g). Any wild
  swing = INFRA BUG until proven otherwise; check beta/INF arithmetic first.
- Held decisions (need user letter-choice): (a) cherry-pick guard+saturation
  onto main, (b) rebase batch onto main + re-gate, (c) park branch as Botzone
  line. Do NOT merge anything until told.

## 8. Memory protocol (knowledge accumulates, sessions never start lobotomized)

- Memory lives ON DISK, not in chat: this file (briefing) + `BEHAVIOR.md`
  (laws) + `HANDOFFS.md` top entry (current state) + `LEDGER.md` tail
  (verdicts) auto-load every session via `opencode.json` instructions and
  survive compaction (system prompt, not chat history).
- Rule change the user states → update `BEHAVIOR.md` immediately + mirror the
  summary into section 3 above. It takes effect for all future sessions after
  an opencode restart (config loads once at startup, no hot-reload).
- End of every work block → run `/handoff` (prepends HANDOFFS entry, LEDGER
  row via python, BEHAVIOR if laws changed, diff verify).
- After compaction or fresh start → run `/rules` (rebuilds working memory
  from disk in 5 lines).
- Gate verification → run `/gate <tag>` (backfill + 100g/seat/net checks +
  SPRT tripwire, verbatim verdict, never upgrades).
- A compaction plugin (`.opencode/plugins/session-guard.ts`) forces the
  summarizer to preserve branch, dirty files, strongest-build pointer,
  HANDOFFS head, law changes, and what's owed.
