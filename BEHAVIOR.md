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
6. **Merge policy (user law 2026-09-21 v2 + 2026-10-19 node-merge rule):**
    merge ONLY what is proven STRONGER beyond reasonable doubt. WHICH gate
    proves it depends on what changed:
    (a) intelligence-only (eval knowledge, net weights — speed provably
    untouched: identical sperft nodes or bench/sperft parity): the 5k-node
    gate CAN merge alone. Fixed nodes = both sides think identical trees =
    zero clock noise, the fairest most precise gate. No time gate owed.
    (b) speed affected (any NPS change): the time crown must be won at the
    CHOSEN controls — 50ms and/or 100ms legs (200ms for confirmation); the
    exact legs are chosen per gate and BOTH chosen legs must win with a
    discernible pattern — speed must convert. The 5k leg is then the quality
    DIAGNOSIS (did decisions change?).
    (c) STOCKFISH STACKING (user law 2026-10-19): main absorbs EVERYTHING
    proven better, however small. Ten stacked +5 gains beat one +50 swing
    that never lands. Iterative improvement is the whole strategy: prove a
    gain, merge it, climb. No gain is too small to merge; no merge without
    proof.
    Neutral/dry does NOT merge — no proof of stronger, no merge. Any
    regression at the required gates = NO merge. Even the smallest gain
    counts, but it must be REAL and repeatable: when in doubt, run more
    games, never merge on vibes. Direction of travel is only forward —
    we prove each kept change didn't make the engine worse.

7. **MAIN IS ALWAYS STRONGEST (Stockfish rule, adopted 2026-10-05 from
    QuoridorBestAI).** Nothing merges to main — code or weights — without a
    sufficient-games gate proving the new build STRONGER (pair-mean +
    paired-bootstrap CI clears 0.5). Record the estimated Elo gain every
    leg — it is data — but promotion proves AHEAD (objectively better,
    stronger; CI clears 0.5 / SPRT pass), never the exact number. No one
    proves "+23"; we prove "ahead". Extend until ahead is proven.
    "Sufficient" is sized until the CI is decisive; a straddling CI EXTENDS
    the gate, it never merges and never rejects on the bound alone. Identical-decision proofs (perft, fixed-node
    parity) excuse nothing: strength is proven by games, and only games.
    Extends law 6: time gates (100ms AND 200ms) are the crown, 5k nodes is
    the quality diagnosis.
8. **NEVER REJECT ON A DIP — EXTEND BOTH WAYS (SPRT gate law, user order
    2026-10-19; threshold amended 2026-10-22 by user order "i said better
    objectively not some specific amount of advantage").** The gate asks ONE
    question: is the build objectively better than main (elo > 0)? There is NO
    fixed Elo magnitude. PASS when the 95% CI lower bound on score clears 0.5
    (one-sided test against 0.5). LLR/Elo are reported as data, never as a
    fixed pass threshold. Elo/score IS the metric — it tells ahead from behind and it
    is the most important number on the board — but the gate proves
    DIRECTION (better or worse than main), never magnitude (no one proves
    "how much Elo it gained"). AHEAD but proof insufficient → EXTEND (more
    games, fresh book), never merge early. BEHIND → EXTEND too (FOMO: fear
    of missing out on Elo — a buried gem rejects at our peril); reject ONLY
    on a conclusive CI below 0.5 AND user confirmation (same as merges: user
    calls every merge AND every rejection). REJECTION IS FOR THE
    UNDOUBTEDLY WORSE: proven worse than main beyond doubt. Anything with
    potential — faster, stronger, or better in any way, even tiny — PARKS
    for more gate, never rejects. An inconclusive gate is an
    order to SPEND MORE GATE, not a verdict.
9. **ENGINE ADAPTS TO NET.** Never damage an experimental net to make it fit
    main: no stripping sections, no converting formats, no rewriting blobs
    to satisfy a loader. A net main cannot load yet waits for its wiring +
    parity proof; it is never cut down. Reads (hash, load-for-eval, carve a
    copy for verification) are free. Load-time allzero skips (W4 has_q2
    pattern) are NOT damage: the code path stays intact and nonzero
    weights automatically take the full path — no silent lobotomy, ever.
    Training may fill any section later and the engine follows.
10. **NO GATE STOPS BELOW 100 GAMES (user law 2026-10-05).** Identity is
    not a reason to stop small: expecting ~50% never justifies a sub-100
    gate. Every gate leg runs to >=100 games, then extends on a straddle.
11. **GATES ARE BY SEAT + PAIRED BOOK (user law 2026-10-05, book rule
    2026-10-19).** Sides swap colors every game (odd/even) — every leg ends
    exactly 50/50 black/white per side, verified from the log, not assumed
    from the game count. Every gate game runs on the prepared book
    (`data/nnue/book50k.txt`): each book line is played from BOTH seats by
    BOTH engines. Win-both-seats of a line = real edge in that situation,
    not noise — even a 1-point margin is measurable signal. Split seats =
    parity. Fixed nodes are deterministic: same line + same seats MUST
    replay bit-identically; any divergence is an INFRA BUG (stop, fix,
    re-gate — never explain away).
12. **CLEAR ENGINE MEMORY EVERY GAME (user order 2026-10-19).** `uainewgame`
    before every gate game, both sides, no exceptions — TT, killers,
    history reset. Cross-game state leaks ordering hints (killers/history
    flip LMR/PVS tie-breaks) and breaks determinism: caught 2026-10-19
    (same seat, same startpos diverged at move 4). In-process sides get a
    fresh searcher per game; UAI children get `uainewgame`. A gate run
    without per-game resets is VOID.
13. **GATES RUN BETWEEN GIT REFS ONLY (user order 2026-10-19, fishtest
    rule).** Both sides run code that exists as a git object — a branch, or
    a commit — never a dirty worktree, never an untracked binary. Otherwise
    the exe hash points at nothing and the gate cannot be linked to code,
    reproduced, or merged. Every gate row records: exe sha + commit hash +
    net hash on BOTH sides. Rebuild from a clean checkout of the recorded
    ref and re-hash before the first game; if the hash differs, stop.

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
- 2026-10-19: match harness never sent `uainewgame` — opp child carried
  killers/history across gate games, flipping LMR/PVS tie-breaks so same
  seat + same startpos diverged at move 4 (games 1/3 identical, game 5
  split). Lesson: VOID any gate without per-game resets (law 12); the
  determinism self-check (same line + seat replays bit-identically) is the
  tripwire. Fixed same session.

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

- 2026-09-21 S8incrpst MERGED to main `293d94b` (pushed): gates vs old
  main 20-12 @5k + 27-5 @100ms, decisive both. main now = slim +
  incremental PST (O(1) eval) + zero-cost-OFF sancta machinery +
  --net/--opp-net harness + training/ scripts. exp/sancta-incr continues
  for next experiments (score_into popcounts, NMP dist_union,
  incremental CONTACT/MULTICAP).

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

- 2026-09-21 S13contact MERGED: E2+E4 back, +163/+150/+270 across 5k/100ms/200ms (96 games). -34pct NPS did not matter. Lesson: KNOWLEDGE-BEATS-SPEED-AT-34PCT. User law upgrade: per-node information is capped (S6big +25 Elo wall) -> future intelligence from NODE RELATIONSHIPS (correction history, singular ext, countermove, SPSA-from-outcomes), not bigger nodes.

- 2026-09-21 S20/S21 TUPLES shelved: pairs +125-family @5k (x5 runs) but -44pct cost floor kills time (-55..-66). Lessons: (1) LOOKUPS-DOMINATE - branchless splat only +14pct, cache class (L1 via i8) is the real lever; (2) OUT-COLUMN-POISON - never train on unproven-semantics columns (-400 in one run); (3) VAL-NEVER-CROWNS redux - 2M data better val (42.8k) but worse play than wsum-400k; (4) SCALE-MATCHES-KNOWLEDGE - 2x2 flat (S13 already owns that scale), 8-cell pairs carry it, diagonals add noise (H+V > full).

- 2026-09-21 VOLATILITY-UNIFIES-ALL (user insight): Ataxx = tactical exchanges until sudden fill/wipe. Positions are VOLATILE (800cp swings per ply, no quiet maneuvering). This one fact explains EVERY failed mechanism: countermove (-137, replies do not persist across exchanges), correction history (flat, shallow verdicts are noise), singular (-112, no lone standout among 57 tactical moves), aspiration (exploded, scores jump window edges), outcome-blend (-400, results are 90-ply credit-assignment lottery). Prescription: dense graded teacher-SCORES as signal (proven +125), depth-first engine, distrust any mechanism assuming positional stability.

- 2026-09-21 FLIP-AVALANCHE (user): one landing converts up to 8; +500 becomes -500 in a ply; ALMOST-WINNING IS UNDEFINED. Corollary: S13 contact/multicap are FLIP INSURANCE (endangered = flip candidates, multicap = multi-flip exposure) - the engine already prices treachery, SPSA tunes the paranoia knobs. Volatility-native proposals: (1) conversion extensions (extend big-capture lines, chase avalanches in search); (2) sharpness-weighted training (pre-avalanche decisions = highest-value rows); (3) SMP depth (see the flip coming). Smoothness-assuming mechanisms stay banned.

- 2026-09-21 S22: teacher depth does NOT distill past student capacity (parity 17-15); rows dominate (+125 needs 2M). Lessons: (1) CAPACITY-WALL - 29k shared pairs saturate; distill gaps need architecture bumps per round; (2) RUN-IT-BEATS-BOTTLE-IT - search strength is runtime process, SMP > distillation for the unbottleable fraction; (3) ladder discipline - measure fitness-vs-search before buying label-compute.

- 2026-09-21 ORPHAN-WRITES (S22 2M relabel corrupted): timed-out cmd leaves CHILD exe alive; my liveness check used the WRONG process name (titanium-cli vs tune_tuple) so 8 orphans + 4 resume workers double-wrote outputs (332k vs 250k rows). Rules: (1) check the EXACT binary name; (2) kill orphans BEFORE resume; (3) NEVER append-resume into shared files - fresh outputs + exact python pre-splits + count-verify before join; (4) corruption signal: output rows != input rows. Redoing clean.

## Protocol

- Branch: `git checkout -b exp/<name> main`, build clean, record baseline
  (bench d8 x4 + sperft), THEN experiment.
- Baseline law: experiment-OFF must equal baseline or instant revert.
- Commit + push `exp/*` to GitHub (`titaniummachine1/titanium-ataxx`) with
  numbers in the message. `/data /logs /scripts /target` are gitignored —
  code pushes, blobs stay local.
- Update LEDGER (experiment row), HANDOFFS (state), BEHAVIOR (this file)
  in the same session as the work, not later.

## Standing orders added 2026-10-22 (user orders, same weight as iron laws)

14. **NO TRAINING WITHOUT A READY DATASET + EXPLICIT ORDER.** A training run
    starts only when (a) the dataset audit script prints ok=true on the exact
    files about to be trained, and (b) the user explicitly says "train".
    Finishing dataset prep never authorizes training. Dataset work and
    training are different jobs.
15. **ONE JOB AT A TIME.** Exactly one long-running task on the box. A new
    task starts only after the previous one is verified dead (empty process
    list), never on assumption. No competing relabel/train/label jobs.
16. **NO RESTARTS, ONLY RESUMES.** Every long job must be resumable
    (chunked outputs, skip-if-exists, tmp+rename atomic writes, resume
    manifests) BEFORE its first long run. A kill may cost at most one
    in-flight chunk. Proof = a kill test, not a claim.
17. **BYTES ARE THE RECEIPT.** Every claim points at a file/log line: audit
    JSON, gate log, checkpoint with timestamp, manifest with hashes. No
    artifact, no claim. Validation loss never crowns anything; only
    paired-book games do.
18. **EVERY DECISION IS SIGNED.** No decision counts as the user's unless the
    user explicitly signed it (docs/DECISIONS.md). Agent-origin rows default to
    NEEDS-REVERIFICATION regardless of age. Lifecycle: PROPOSED -> TESTED ->
    EXPLAINED (user confirms understanding) -> SIGNED or SENT BACK.
    Presenting an old agent decision as the user's is a violation. No status
    claim without the tool call behind it in the same turn.

## Standing orders added 2026-10-24 (user orders, same weight as iron laws)

19. **TITANIUM LABELS ARE BASE DATA, KATAGO IS THE TEACHER.** Titanium 5k search
    labels are the Stockfish base stage (stage 1, whole dataset); they are never
    called or weighted as a teacher. KataGo's own 200-visit search values are the
    teacher (stage 2, Kata rows). Use Kata data wherever it exists.
20. **NO SPIRALLING.** One agreed plan, explicit go, one job, leave it running.
    Never start relabels/trainings/mock runs the user did not order; never flip
    between runs on my own reading of the latest sentence. Talk plainly.
21. **PROTECT THE BOX (16 GB).** Check available RAM before and during big jobs;
    stream big files; never memmap a multi-GB file in a long loop; RAM guards.
    (2026-10-24: a dataset build crashed the PC.)
22. **RESEARCH BEFORE BUILDING.** Before designing eval/NNUE inputs, take the
    strong engines apart (source, exact eval reproduction, internals, VALUES AND
    POLICIES) and test each design choice against them. State what was not
    analysed; never call research done with gaps (docs/NET_ANALYSIS.md).
