# BEHAVIOR — standing orders (updated real time, survives session corruption)

Session corrupts ~every 5 min. This file + LEDGER.md + HANDOFFS.md are the
real memory. Update them the moment something happens, not at the end.

## Iron laws (user orders, never re-litigate)

1. **MAIN IS UNTOUCHABLE.** Work on `exp/*` branches only (Quoridor/Stockfish
   rule). Merge only when a branch decisively beats main with zero doubt it
   is stronger. No local main edits, ever.
2. **SPEED FIRST (current order).** User only cares about speed right now.
   Strength gates come after speed is proven.
3. **TIME CROWNS (user law 2026-09-21).** Ataxx intelligence is
   LOG-diminishing: 800k-param net @val .143 = +25 Elo same-nodes;
   classical wins only at 4–5x nodes. Strength = speed × log(intel):
   NPS buys depth linearly, eval buys log-wise. Balance the two —
   never trade 2x NPS for +25 eval-Elo. Val loss NEVER crowns — time
   gates do. Speed buys what intelligence can't.
4. **ONE variant per binary.** If the binary holds >1 variant (ordering
   switches, eval modes, feature flags in the hot path) it gets PURGED.
   Experiments live on branches, not behind flags.
5. **TRASH UNTIL PROVEN.** Every prior claim is trash until re-measured on the
   current branch. Microbenchmarks don't crown — tree NPS (bench/sperft,
   back-to-back alternating) does.
6. **Merge policy:** gains Elo → MERGE; neutral/dry → STILL MERGE; worse by
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
- 2026-09-21: user caught UNLOGGED Moonbird parity/wins (moon3_5k 19–13,
  moon5_100g 52–48, moon5_t100 17–15) sitting in logs/ from an external
  duel harness while LEDGER claimed 0–32 everywhere. Lesson: EVERY game
  log in logs/ gets a LEDGER row, regardless of harness. I don't get to
  only record titanium-cli match results. Added S3moonX row.
- General: if basic git/bench ops fail, the code is hyper-unmaintainable and
  I pay for it — purge, don't patch. Simplicity is the metric.

## Checkpoints (save tokens: doc at end of each TA block, then answer short)

- 2026-09-20 S1: purge done (−307/+41), sperft +71%, 100g@100ms 73–15–12
  (+230). MERGED to main 7ab844e, pushed.
- 2026-09-20 S2: sign double-negation bug (forward_ready already
  stm-relative; extra negate = 0-50). Fixed → 5k-node 10–20 (−140),
  100ms 14–16 (≈parity). Eval swap ~free (5.0–5.2M vs 5.6–8.4M).
  NOT merged.
- 2026-09-20 S3: slim vs owner 19-0-1 (his tree is weak), vs Moonbird
  0-20 (wall stands). Data: dag.db 8.7M nodes/167 folds, torch CPU-only
  (no GPU). Decision: saturate own 147x64 net on CPU, then gate.
- 2026-09-21 S4v3: exact-distill run v3exact ep12 flat (tr .2602 va .2606,
  out ~477/sc ~109 = STALE loss math vs HEAD f8b98b5, pred mean ~4700 vs
  tgt O(1)) — KILLED PID 10228. own_v3.s1 @ep9 gated vs classical 0-30
  @5k nodes (same 0-piece wipe pattern as v1/v2). Net adds ~3.6x nodes
  (34k vs 9.4k d8) but zero knowledge. Capacity+scale bug, not data.
- 2026-09-21 S4v3e: NOT overfit — e1 0-20, e5 0-20, same wipe from ep0.
  Verdict: init/scale broken from the start, keep working the net.
- 2026-09-21 S5tool: harness presets done (--net/--opp-net on
  bench/serve/match, scripts/gate.ps1, training/export_epoch.py).
  Committed 4271d6b. No more env vars: `scripts/gate.ps1 30 <net>`.
- 2026-09-21 S6score: score-only CP loss RUNNING (400k-cache 25ep, ep1
  tr .0547 va .0549, pred same scale as tgt — actually learning).
  Threat inputs OPEN: 147 = stones only, net must learn contact/multicap
  via FT; user asked if we add explicit weakness/heat/pressure inputs.
- 2026-09-21 S7quality: zero-scaffold pass on exp/sancta-incr BEFORE any
  net experiments (user: Stockfish-grade engine, no legacy). Borrowed net
  (&'a S1Net, NO Arc/unsafe in tree), MoveStack owned by Searcher (no
  per-search ~180KB alloc, no stack param threading), score_into free fn
  (no split-borrow fight), integer NMP fill check (no per-node float div),
  fallthrough-terminal (empty list = pass-through score, no panic), TT-hit
  flag removed, aspiration ladder ±50/±200/±800 from depth 5, dead
  perft()/StopReason import/from_v1_parts/stack_state pruned.
  COMMITTED 2a0a0ba. 26 tests green. Sperft 4.22–4.34M (x3, 28.2M nodes —
  aspiration CHANGED suite count from 535k: deeper same-time, expected).
  Bench d8 classical 17034/6.9M. Self-check 10–10.
  S6score FINISHED ep25 (tr .0537 va .0549, pred 108/112 vs tgt 120/2000):
  gate vs classical **20–0 @5k** — real games (39-10) not wipes. Progress,
  not parity.

- 2026-09-21 S9night: NIGHT 5/6 DONE (tq=S7quality vs BASE/Moon/own_v3):
  QB5k 45–53–2 BASE (S7 REGRESSES clean-main on quality gate), QBT100 died
  at 12g (both tq PIDs are the QNETT100 tail — no QBt100 runner; 12g
  file = overwritten partial, RERUN NEEDED), QMOON5k 0–100 + QMOONt100
  0–100 (wall stands), QNET5k 100–0 QUAL (own_net still zero), QNETt100
  36/100 running (gm5836). S7-aspiration cut node count 535k→28.2M-same-
  time reading was DEPTH, not Elo: quality per S8gatedef = LOSING.
  Verdict: revert S7 OR re-tune RFP/LMR around aspiration before any net
  capacity work.

- 2026-09-21 SESSION CLOSE (handoff for new chat): branch
  exp/sancta-incr = 3 commits past main 7ab844e (f8b98b5 v3-loss,
  4271d6b harness+S6, 2a0a0ba S7quality). S7quality = REGRESSION both
  gates (QBt100 rerun 16-84). Working tree dirty: pattern.rs LUT draft +
  is_clone/FULL fixes (uncommitted micro, no-gain). Movegen SATURATED
  (LUT closed 3 ways: sizes/identity/measured 74M vs 33M). Next
  user-ordered target: eval-saturate = incremental PST in Board (fold
  delta into make() capture loop, free), evaluate() -> 3 ops, DELETE
  eval_cached (double TT probe + pollution). O(1) eval, bit-identical
  values, 5k gate must be 15-15, 100ms gains from speed. Then
  score_into popcounts + NMP dist_union. Run BOTH gates (S8gatedef:
  5k=quality, 100ms=strength). main UNTOUCHED.

- 2026-09-21 S8incrpst DONE (558ae64): incremental PST shipped.
  pst:i32 on Board, delta folded into make() capture loop, evaluate()
  O(1), eval_cached deleted, aspiration revert (left dirty by prev
  session) finalized. Mystery solved: tq.exe 28.2M sperft nodes was the
  ASPIRATION build exploding on re-searches = the S7quality 16-84
  regression. New build: 317k nodes @5.3-5.9M. Gates vs main:
  **20-12 @5k AND 27-5 @100ms — WIN BOTH, merge candidate.**
  NEXT: (a) merge decision (branch carries zero-cost-OFF sancta
  machinery); (b) score_into capture popcounts (~57 count_ones/node);
  (c) NMP dist_union per node; (d) then incremental CONTACT/MULTICAP
  (E2/E4 knowledge back, per-move delta not per-node loop).

## Protocol

- Branch: `git checkout -b exp/<name> main`, build clean, record baseline
  (bench d8 x4 + sperft), THEN experiment.
- Baseline law: experiment-OFF must equal baseline or instant revert.
- Commit + push `exp/*` to GitHub (`titaniummachine1/titanium-ataxx`) with
  numbers in the message. `/data /logs /scripts /target` are gitignored —
  code pushes, blobs stay local.
- Update LEDGER (experiment row), HANDOFFS (state), BEHAVIOR (this file)
  in the same session as the work, not later.
