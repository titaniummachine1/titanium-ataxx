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

- **UPDATE 2026-10-24 late: STRONGEST = main 653d9a1 (pushed) + S57 tupS57v2.tup (53F2B29C).** 653d9a1 = the engine now evaluates TUP7 nets exactly as trained (NET_ANALYSIS section 27; +77 @5k, +45 @100ms, +26 @50ms vs 298b7ae). Any tuple-net work must keep training/tuple_parity.py at 0 cp.
- **UPDATE 2026-10-24 (read with HANDOFFS top entry): strongest is still main 298b7ae + S57. A real NNUE is in progress on branch `exp/nnue` (worktree `C:\gitProjects\nnue-ataxx`, committed locally 02c3f29). Its design must follow `docs/NET_ANALYSIS.md` (what Kata/Ataxx_CPP/Moonbird/autaxx/Stockfish look at, anchor test, policies). Titanium 5k labels = BASE stage data, KataGo = TEACHER.**
- **UPDATE 2026-10-23 (supersedes the bullet below): strongest = `main-ataxx` 298b7ae (S57) + net `data/nnue/final_v2_wdl_ep8.tup` (sha 53F2B29C, main default name tupS57v2.tup). Dataset v3 plan + the Ataxx_CPP/Kata findings: `docs/AXNET_PROBE.md` (read it for any teacher/eval/dataset work).**
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
10. **GATE LAW v5 / OBJECTIVELY BETTER (user order 2026-10-22, replaces the
    old +20 Elo / +2.94 LLR threshold):** the gate asks only "is it better
    than main (elo > 0)?" — NO fixed magnitude. PASS when the 95% CI lower
    bound on score clears 0.5. `training/gate_summary.py` computes W/L/D, seat
    balance, score, Elo, CI from match logs. Merge needs tripwire + explicit
    user approval naming branch/net + LEDGER promotion row. (gate_check.py /
    results.py may be absent on some branches — use gate_summary.py and note
    it.)
11. **NO REJECTION WHILE FOMO LIVES.** Undertrained net that loses stays
    PARKED, never rejected. **ENGINE ADAPTS TO NET** — never strip/convert a
    net blob to fit a loader. Load-time allzero skips (W4 has_q2) are NOT
    damage: path stays intact, nonzero weights take the full path.

12. **(2026-10-24, BEHAVIOR 19-22)** Titanium 5k labels = BASE data, KataGo = TEACHER;
    no spiralling (one plan, explicit go); protect the 16 GB box (RAM checks, no 13 GB memmap);
    research before building (values AND policies of the strong engines, docs/NET_ANALYSIS.md).
13. **(2026-10-24, BEHAVIOR 23) STANDING SIGN-OFF:** the user signs every agent proposal "as long as it wins":
    build / train / ablate without a per-step go (log a DECISIONS row); one job at a time, RAM + audit rules stay;
    merging into main still needs explicit user approval naming branch + net after a passed gate.
    Ranked "what matters" summary for any eval/NNUE work: docs/WHAT_MATTERS.md.

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
- RAM (16 GB): never hold a whole multi-GB npz + temporaries (np.savez(**cols)) -> swap freeze/crash (2026-10-24). Never np.memmap a 13 GB file in a long run (pinned working set grew to 10.7 GB); read chunks with file reads. Check `GlobalMemoryStatusEx` available RAM before and during big jobs.
- `npz[key]` re-reads the whole array from the zip on EVERY access: load columns once into variables.
- Moonbird exe = tag 1.1.0 (2.58 MB eval); `data/engines/moonbird_src` is a newer unreleased checkout.
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

Decisions live in docs/DECISIONS.md: nothing counts as the user-signed unless signed there; agent rows default NEEDS-REVERIFICATION (BEHAVIOR law 18).
