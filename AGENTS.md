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

- **DOCKER: docs/DOCKER_SANDBOXES.md = how to run, watch, STOP and pay for `sbx --cloud` sandboxes (money rules, ledger, current sandbox tit-sn2 and how to stop it). Read it before touching any sandbox.**
- **MODAL (GPU training, cloud gates): docs/MODAL.md = commands, $30/month hard cap, how to list/STOP apps, no secrets. Docker preferred for CPU work.**

- **UPDATE 2026-10-29 ~17:00: STRONGEST = origin/main `55a62ab` (PUSHED), default net `data/nnue/nnue_main.tnn` sha `7AE0122F` (W=256 Net2Net downscale of the W=512 ep7 teacher, TNN3 Disagreement Head).** Promoted for BOTH target tiers after gating: 5k +34 Elo over 832 g (scoreCI [0.5015,0.5971] PASS, tripwire BIT-IDENTICAL), 100 ms on AVX2 +47 (208 g, scoreCI [0.5000,0.6346]), 100 ms on AVX-512 +73 (208 g, scoreCI [0.5369,0.6698] PASS, Modal A10G). One net is strongest on both tiers (D154). Previous champion `31212A7B` (+72 vs `e11827d1`) is history. Builtin SmallNet `fut_small.tsn` (52,844 B) is FROZEN (D148). `C:\gitProjects\main-ataxx` worktree sits on `exp/search-spsa-disagree` `5eea2d3`; `C:\gitProjects\mainpromo-ataxx` is a worktree on `main` `55a62ab` (holds the promoted net + the V12 Botzone upload folder).
- **Candidates:** W=512 D4 v7 ep7 disagree `8AB5B82E` = +52 Elo per node (D151) but at 100 ms on AVX-512 only +8 (100-95-13, scoreCI [0.4441,0.5800] INCONCLUSIVE) with NPS 149k vs 230k for W=256 -> **PARKED, not deployed**. W=256 sat ep6 `976C515F` = 0-413-3 (-977) representation collapse (D153 / Mistake 39) — never fine-tune a downscale at lr 2e-4.
- **Data (D149):** `data/nnue/modal_up/final_{train,val}_v7.npz` = 32,622,551 train rows (740.8 MB) + 677,670 val (16.9 MB) = v6 + 2,025,000 fresh 5k disagreement relabels + 88,295 Botzone >=2200 WDL anchors.
- **READ FIRST for training/relabel/Botzone work: docs/TRAINING_RUNBOOK.md, docs/RELABEL_RUNBOOK.md, docs/BACKLOG_AUDIT.md, HANDOFFS top entry. No external teacher beats Titanium (D105); never fine-tune on the Kata columns (D98/D101).**
- THIS folder is usually on an `exp/*` branch; check the exe sha AND net sha on both sides before any gate (a branch can default to no net or classical eval).
- Clean copies: `C:\gitProjects\main-ataxx` (worktree of `main`), `C:\gitProjects\staging-ataxx` (`staging`, stale at `0f27043`).
- First question on any strength task: "which binary + which net hash on BOTH
  sides?" Log exe sha + net sha in every gate (harness does this).

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
14. **(2026-10-24, BEHAVIOR 24) STAGING:** branch `staging` (worktree C:\gitProjects\staging-ataxx) = main + proven gains.
    The agent merges a candidate into staging itself once it passed the full gate vs the STAGING TIP (laws 3-6/8/10; speed-affected
    = 100 ms + 50 ms legs); LEDGER row STAGE-<tag>. New candidates gate vs the staging tip. staging -> main = user approval naming
    commit + net, after a staging-vs-main confirmation gate. Staging is not pushed unless the user says so.

15. **(2026-10-25, BEHAVIOR 25) DOCS IN REAL TIME:** every important change (LR algorithm, data, labels, recipe, what a net was really trained with, run start/stop) goes into DECISIONS + HANDOFFS the moment it happens, with a TRUE status; read DECISIONS + run logs + the nnue-pytorch recipe before choosing any training setting.
16. **(2026-10-29, BEHAVIOR 27-31) NNUE + SAFETY LAWS:** every candidate is gated pre- and post-promotion vs `main` >=100 g/leg (D124); capacity changes only via Net2Net with 0 cp step-0 parity, never from scratch (D130); cold ID starts at depth 1, warm ID only from a valid non-mate root entry, depth u8 255, solved mates stop (D138); SmallNet tau 0.02 and the LMR baseline are frozen (D147/D148); a fresh downscale must keep val-MSE parity and is never fine-tuned at lr 2e-4 (D153/Mistake 39).
17. **(2026-10-29, BEHAVIOR 32) CREDENTIALS: USE FREELY, NEVER STORE.** Creds the user gives in conversation may be used for their intended action, but are NEVER written into any repo file, committed, pushed, or bundled into artifacts/upload folders. Pass them via env vars or function args at the moment of use; never echo them in logs or commit messages.

## 4. Rules: two of them (do not mix)

- **Classic** (UAI serve default): pass on no-move, double-pass/full/wipe ends.
- **Botzone** (judge): NO passes — either side stuck ends immediately, empties
  go to mobile side (`49 - stuck`); `-1 -1 -1 -1` reply = INVALIDMOVE loss;
  2-position repetition = loss. Engine has BOTH via build-time const generic:
  `search`/`parallel_search` = classic, `search_bz`/`parallel_search_bz` =
  Botzone (NMP off, `bz_over`/`bz_counts`). Zero hot-path runtime flag.
- **Botzone deployment = V12 LIVE (uploaded 2026-10-29):** bot `titanium_ataxx_PY6`
  `6ad7e0c3bdad901f323f6516`, latest version `6ae375db9eb7df09d05cede4`, driver
  `botzone/upload/titanium_botzone.py` sha `9550316F` (14,257 B; hardened: guarded
  stderr, `os.write` + `os._exit(0)`, engine resuscitation `ev ...`). Storage
  `/titanium_v2.so` sha `3F9E10BA` (28,492,848 B, glibc 2.14, **embedded W256 ep7
  `7AE0122F`**, AVX-512 kernel runtime-dispatched), verified by download hash;
  long-running ON (4 s/turn, 12 s first). V11 (`a9e63aca`, `31212A7B`) retired.
  Older crashes were exit-120 stderr flush (Mistake 30/31) and glibc 2.34 (v5).
  Build with `python tools/build_botzone.py` (Docker test step needs Docker Desktop
  running). **Upload trap: `requests` times out the 28.5 MB POST after ~60 s from
  both Modal and the local PC; `curl.exe` works (~640 s)** — see the docstring of
  `tools/upload_botzone.py`. Verify with `GET /userfiles/titanium_v2.so` + sha256.
  Next: autonomous Elo climber (BACKLOG_AUDIT Phase C).

## 5. Tournament entry (current shape)

- **Botzone = Python driver + `titanium-ffi` C ABI** (`ti_new`,
  `ti_new_game`, `ti_apply`, `ti_search_with_limits(time,nodes,depth)`,
  `ti_stop`, `ti_stats`). The engine knows nothing about the protocol; the
  driver replays history after a restart, keeps the clock/watchdog and sends
  `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`. `TI_MODE=long` (default) = persistent
  TT + warm ID from root TT depth; `TI_MODE=oneshot` = cold 6 s reset.
- Botzone judge: 12 s first turn / 4 s later, 256 MB, one core, Ubuntu 16.04
  (glibc 2.23 -> build the `.so` with zig glibc 2.14, never plain cargo on a
  modern Linux). Warm ID starts at `(root_tt.depth-1).max(1)`, depth u8 255,
  solved mates stop.
- `serve --tournament [--threads N] [--bz-rules]`: 128 MB TT default
  (tt_bits 23), startup autotune, silent auto-ponder after `bestmove`, next
  `position`+`go` aborts and converts with warm TT.
- NOT built: per-move thread auto-scale; position-without-go premature search.
  The Botzone `.so` DOES carry the AVX-512 kernel (runtime CPUID dispatch, no
  separate library); it was compiled into V12 (`3F9E10BA`).

## 6. Commands (PowerShell 5.1 — no `head`/`sed`/`&&`)

```powershell
cargo test --workspace
cargo run -p titanium-cli --release -- bench --depth 8
cargo run -p titanium-cli --release -- sperft --depth 8   # speed truth, alternating pairs
python training/gate_summary.py <pooled.txt> [more...]    # positional paths (no --log): W/L/D, seats, score, Elo, CI
python training/gate_net_vs_main.py --tnn <net> --tag <t> --book0 N --shards 16 --games 26 --nodes 5000 --par 10
                                                          # resumes: shards with .done are skipped; tripwire on shard 0
python training/net2net_downscale.py / upscale.py         # capacity change, 0 cp parity at step 0
python training/assemble_v7_dataset.py                    # relabel chunks + Botzone anchors -> v7
python training/overnight_pc_relabel.py --resume          # skips existing chunks, heartbeat
python tools/build_botzone.py --no-pull                   # V12 package: zig glibc-2.14 .so + upload/
python -m modal run tools/upload_botzone.py               # cloud upload (creds via env); if it stalls, curl (docstring)
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
- Downscale fine-tune collapse (Mistake 39, D153): never fine-tune a fresh
  downscale at lr 2e-4 (0-413-3). Downscale must keep val MSE parity first.
- Botzone exit 120 = CPython stream flush at shutdown (Mistake 30/31). The
  driver must never write to a raw stream after the move is sent; hardened
  driver uses `os.write` + `os._exit(0)`. A match log with no stdout and
  `status 120` means the OLD driver ran.
- Relabel resume: `training/overnight_pc_relabel.py` skips existing chunks;
  never delete/rebuild the output dir mid-run, and never rebuild `main-ataxx`
  while a relabel job reads its exe/net.
- Cloud: `sbx --cloud ls` (Docker) and `python -m modal app list` are the
  only truth for "what is running"; check both before claiming idle.
- **Botzone upload (2026-10-29):** `requests` dies on the 28.5 MB POST with
  `TimeoutError('The write operation timed out')` after ~60 s from BOTH Modal
  and the local PC (5+3 attempts). `curl.exe` works (~640 s): login with
  `-c cookies`, then `-F "file=@...so;type=application/octet-stream" -F "path="
  -F "filename=titanium_v2.so"` to `/userfile/uploadfile`. Storage page is
  `/userfiles` (not `/userfile`); verify by downloading `/userfiles/titanium_v2.so`
  and comparing sha256.
- **Botzone upload folder driver must be LF.** Git blob = sha `9550316F`,
  14,257 B; a Windows checkout can hand the script a CRLF copy (14,594 B).
  Rebuild it from `git cat-file blob HEAD:botzone/titanium_botzone.py` and
  re-hash before uploading.
- **Modal hosts:** GPU hosts (A10G/L4) DO have AVX-512 (`avx512f/bw/vl/vnni/vbmi/
  vpopcntdq`); CPU-only containers do NOT. A binary built on rust:1.97 needs
  glibc 2.39 and dies on Modal's image -> build
  `--target x86_64-unknown-linux-musl` (static-pie) and upload that. Set
  `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"` for the CLI.
- **`modal volume get <vol> <dir> <local>` with `--force` can create a FILE**
  when `<local>` does not exist; fetch shard files one by one instead. `modal
  volume put -f` is needed to overwrite.
- **Verifying a stripped .so:** `strings` shows no `avx512`/symbol names. Count
  `%zmm` lines from `llvm-objdump -d` (AVX-512 kernel present) and probe a
  64-byte slice from the middle of the net file to prove the embedded net.
- **sbx:** inline `bash -c '...$var...'` is mangled by PowerShell -> upload an
  LF `.sh` and run it; `sbx cp` prints to stderr (noise, but works). Log actual
  spend in `logs/docker_spend.csv` and delete the sandbox (`sbx --cloud rm
  --force <id>`) when done - `ls` is the truth.
- Held decisions (need user letter-choice): W=512 `8AB5B82E` deploy (AVX-512),
  and every merge/rejection. Do NOT merge anything until told.

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
