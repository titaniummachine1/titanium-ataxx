# BEHAVIOR — standing orders (updated real time, survives session corruption)

Session corrupts ~every 5 min. This file + LEDGER.md + HANDOFFS.md are the
real memory. Update them the moment something happens, not at the end.

## Iron laws (user orders, never re-litigate)

1. **MAIN IS UNTOUCHABLE.** Work on `exp/*` branches only (Quoridor/Stockfish
   rule). Merge only when a branch decisively beats main with zero doubt it
   is stronger. No local main edits, ever.
2. **SPEED FIRST (current order).** User only cares about speed right now.
   Strength gates come after speed is proven.
3. **ONE variant per binary.** If the binary holds >1 variant (ordering
   switches, eval modes, feature flags in the hot path) it gets PURGED.
   Experiments live on branches, not behind flags.
4. **TRASH UNTIL PROVEN.** Every prior claim is trash until re-measured on the
   current branch. Microbenchmarks don't crown — tree NPS (bench/sperft,
   back-to-back alternating) does.
5. **Merge policy:** gains Elo → MERGE; neutral/dry → STILL MERGE; worse by
   ANY → ASK user, no unilateral rejects.

## Mistakes ledger (written when scolded, read before acting)

- 2026-09-20: built S1–S5 sancta scaffolding (PROF atomics, Box/Arc in hot
  path, per-ply acc stack, do/unmake retrofit) that halved tree NPS (1.6M vs
  champ 3.3M same tree) while micro evalbench looked fine. Lesson: measure
  TREE, not micro. Fix: `exp/slim-eval-speed` purge (−307/+41), sperft truth.
- 2026-09-20: worked on `sancta-speed` branched from a dirty commit + edited
  `main.rs` on a branch based off wrong HEAD (main had diverged; build broke
  on `set_nnue`). Lesson: branch `exp/*` off `main` HEAD, verify `git log
  main`, build clean BEFORE experimenting. Fix: fresh branch, baseline first.
- 2026-09-20: used `head`/`sed` on PowerShell 5.1 (not recognized). Lesson:
  this shell is win32 PowerShell — `Select-Object`, no `head/sed/&&`.
- 2026-09-20: presented bench-d8 single numbers as wins (4.6–8.0M jitter).
  Lesson: bench d8 is noisy; sperft 8-pos suite is the speed truth.
- General: if basic git/bench ops fail, the code is hyper-unmaintainable and
  I pay for it — purge, don't patch. Simplicity is the metric.

## Checkpoints (save tokens: doc at end of each TA block, then answer short)

- 2026-09-20 S1: purge done (−307/+41), sperft +71%, 100g@100ms 73–15–12
  (+230). MERGED to main 7ab844e, pushed.
- 2026-09-20 S2: sign double-negation bug (forward_ready already
  stm-relative; extra negate = 0-50). Fixed → 5k-node 10–20 (−140),
  100ms 14–16 (≈parity). Eval swap ~free (5.0–5.2M vs 5.6–8.4M).
  NOT merged. Awaiting user: retune search vs saturate own net.

## Protocol

- Branch: `git checkout -b exp/<name> main`, build clean, record baseline
  (bench d8 x4 + sperft), THEN experiment.
- Baseline law: experiment-OFF must equal baseline or instant revert.
- Commit + push `exp/*` to GitHub (`titaniummachine1/titanium-ataxx`) with
  numbers in the message. `/data /logs /scripts /target` are gitignored —
  code pushes, blobs stay local.
- Update LEDGER (experiment row), HANDOFFS (state), BEHAVIOR (this file)
  in the same session as the work, not later.
