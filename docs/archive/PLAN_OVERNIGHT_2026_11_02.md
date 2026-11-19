# Overnight plan 2026-11-02 (D321). Written so any model can continue without chat memory.

User message that created this (2026-11-02, verbatim intent): "go hard, update docs, ledger and backlog so you don't get confused overnight.
Improvements to Titanium: assume you potentially have to run Titanium on the same rubbish as my computer on Botzone (no usable GPU).
For TRAINING you may use Modal freely as long as you don't rack up excessive costs that gain little Elo for the cost. The PC is assumed a
free resource (not entirely, but cheaper than those websites). Do what's best."
Earlier the user asked "you can remember the overnight plan for improvement tests?": there was no single overnight file. The plan was spread over
docs/archive/PLAN_SESSION_2026_11_01.md (work order), D320 (training-method sweep) and D316 (SPSA 1b). THIS file is now the one consolidated list.

## 0. Deployment hardware assumption (NEW LAW-LEVEL FACT, LAWS B13)
Botzone runs Titanium on ONE CPU core (Ubuntu 16.04, 4 s/turn, 256 MB), no GPU, speed roughly like the user's PC (Haswell AVX2, 4C/8T). So:
- Every inference-time change must be CPU-cheap: NNUE-sized layers only. A policy head must be measured in knps on the PC, single thread. No ResNet,
  no GPU net, no PUCT with a big net at play time (Kata = ~1000 Titanium nodes per eval on CPU, KATA_ARENA).
- Time-leg gates (100 ms / 1000 ms) are the played-clock truth (B9). The Docker 16-vCPU boxes are a DIFFERENT machine (D241 bimodal nps): they
  are fine for 5k-node (clock-free) gates and for direction at time legs, but the FINAL time-leg confirmation of any promotion candidate runs on
  the PC with the SPSA job paused (single-thread engines, 1 core each, like Botzone).
- Training and label generation may use any cloud/GPU because they are offline.

## 1. Money rules for the night (LAWS A9 + this order)
- PC = free first. Docker = accelerates short parallel gates beside the PC, never replaces it.
- Modal = training only (GPU). Free to use if (expected Elo gain) justifies the cost: a recipe run is ~$0.2-0.4 + a 5k gate ~$0.3, fine. Do NOT
  spend on big jobs with an unclear gain (arenas, smoke tests, 1M-node relabel before the policy head exists).
- Ledger (checked 2026-11-02 03:10 local): Docker spent ~$5.6 of the $200 cap (tit-cal1 CREATED 01:45 UTC ttl extended, worst case $2.8, expires
  04:05 UTC = 06:05 local); Modal November: ~$0.56 logged of $29.69 (rwphase1 $0.16 + rwphase5/6 $0.40 estimate), ~$29 left.
- Every launch: ttl + delete, ledger row in logs/docker_spend.csv or logs/modal_spend.csv at once, copy results out before deleting.

## 2. State at 03:10 (2026-11-02)
- Main unchanged: c86733b, net F5FA8557. NOTHING merged, pushed or uploaded.
- Candidate net rwphase3_nocal 54A6B02B (result weight 0.55 everywhere, NOT calibrated, dis head refit): 5k vs F5FA8557 = +22.2 (800 g, LOS 96.7%),
  extension batch (book0 78400) +24.8 +-23.7 (800 g, LOS 98.0%); pooled ~1600 g about +23.5 (P(better) ~99%). 1000 ms leg RUNNING in tit-cal1 at
  98/200 g: +64.5 +-57.9 (LOS 98.8%), not final. sat16a_nocal 7048DA8B (calibration off only) pooled +14.8; rwphase3_cal 9373BD03 pooled +13.3.
- Modal: rwphase5 (0.7) finished (early stop sb 44, own-target val 0.0784); rwphase6 (0.85) training. Not yet fetched/exported/gated.
- PC: SPSA stage 1b running (log C:\gitProjects\logs\spsa_stage1b.log, 3 of 600 iterations at 03:08; ~12 min/iteration; checkpoint cmd
  C:\gitProjects\logs\spsa_stage1b.cmd.txt; resume = cmd + --resume with a higher --iters).
- Promotion of rwphase3_nocal (or a better weight) = USER decision after the 1000 ms leg; agent never promotes.

## 3. Ranked work order (each item: command source, done-when)
A. Finish the weight sweep (running). When rwphase6 is done: `ti train modal fetch --tag rwphaseN --out runs/rwphaseN` for 5 and 6, export
   WITHOUT --calibrate but WITH `--result-weight-phase w,w,w` (same flag as training) and the dis-head refit, then 5k gate each (800 g, vs F5FA8557,
   book0 82000 / 84000 so books differ from earlier legs) in a Docker sandbox (tit-cal1 if its ttl allows, else a new 16-vCPU one under A9) with
   `ti insight` + `ti analyze` in the report (B12). Done when 0.7 and 0.85 each have >= 800 paired games and a direction. Monotonic so far (0 mid -32,
   0.4 +4, 0.55 +22): if 0.7 and 0.85 keep rising, next try per-phase asymmetric (more result at >= 13 empties) and 1.0.
B. 1000 ms leg of the best candidate to 200 g (running); then a PC confirmation with SPSA paused at 1 thread/engine (hardware rule 0). Report to the user.
C. Training-method sweep (D320), one variable at a time against the best recipe, each = Modal run + 5k gate: lr / lr-min / warmup, steps + early
   stop (runs stop at sb ~29-44 now: try longer without --plateau), proof-blend, wsum clip, scratch vs fine-tune, data scaling 25/50/100% of v9, EMA.
D. SPSA 1b (PC, running). At ~100 iterations: Docker checkpoint gate of theta vs untouched main at 100 ms + 1000 ms. If the net changes (item A
   promoted), rerun SPSA on the new net. Dis-dynamics knobs are tuned against F5FA8557; a different result weight shifts the dis head, so the
   SPSA result must be re-gated with the promoted net.
E. Hub builds (Rust, no Python; no spend): (1) `ti mine` hard-position dump from ti analyze turning points (PLAN_SESSION sec. 2); (2) deep root-move dump
   in `ti relabel` (docs/research/policy/POLICY_PLAN.md data A); (3) Kata policy reader (streams policyTargetsNCMove, data B); (4) policy loss + head in `ti train`.
   Policy head cost must be measured in knps on the PC (hardware rule 0) before any gate.
F. Policy -> move ordering (user 2026-11-02): policy logits order moves before any child is made or evaluated (search.rs:1007-1033 EVORD today does
   b.make + cached child eval per move, ~4% instructions). Test nodes-to-depth, then 5k / 100 ms / 1000 ms gates. Then UBFM prototype (UBFM_PLAN).
G0 (D322, do before G and before the policy head is used for speed). Eval-source counters on an exp branch: static / futility-child-then-pruned / futility-child-searched / ecache collision; clock-free, PC. Then decide: bigger ecache (exact) and/or a per-move value (Q) head next to the policy head (docs/research/policy/POLICY_PLAN.md last section).
G. Speed: 1.40-1.48 evals/node (D241): reuse/cache to ~1.1 = up to 20-25% faster (this is also strength on a weak Botzone core). Bench node counts identical, then time gates on the PC.

## 4. Rules for the unattended night
- Do not promote, merge, push, or upload. Do not change main. Do not run two long jobs on the PC (SPSA is the PC job).
- Do not rebuild titaniumtest target_dev while SPSA runs (it is the running ti.exe); build hub changes with a different CARGO_TARGET_DIR.
- If the PC reboots again: files can be zero-filled (D314). Re-fetch nets from the Modal volume; check sha before use.
- Gate reports always carry `ti insight` + `ti analyze` (B12); undecided = extend, never "rejected" (B8); at 3,000 pooled games ask the user.
- Time margin of the Botzone driver is tuned (D309): never flag it.
- Write DECISIONS/HANDOFFS/KEY_FACTS BEFORE replying with a result (A1).
