# PLAN 2026-10-30 — full review, bullet-proven training methods, tests, Zataxx proof

User order: "everything needs review, planning, improvements and tests for better training methods from
bullet that are known already; Zataxx (trained with bullet) beats Titanium decidedly." Companion to
`docs/archive/AUDIT_2026_10_30.md`. Rows marked FACT are read from source; PROPOSAL rows are hypotheses, games decide.

## 0. Status of the claim "Zataxx beats Titanium decidedly"

- Direction is credible: Zataxx 74-0-1 (log `logs/zataxx_vs_titanium_100ms.txt`), and Zataxx beat Moonbird
  29-2-1 at 1 s while Moonbird already beats us at 100 ms (D105 -241, D155 -74).
- NOT proof yet: that run had no book (wrong one-piece start), unpaired seats, unknown Titanium exe/net, 75 games,
  6-thread relabel on the box. Step P1 below makes it proof (or refutes it). Until then: "probable".

## 1. Proof gates (first, cheap, decide what matters)

| # | gate | why |
|---|---|---|
| P1 | Titanium (`c62eed4`, exe `16C1B38A`, net `7AE0122F`) vs Zataxx `b24e59f` (built here, hash logged), book `book50k_std.txt` (`621B7830`), paired seats, idle box, **5k nodes** both sides, >=100 g | per-node quality: is Zataxx's tiny net + search better per node? |
| P2 | same, **100 ms** | the real-time verdict |
| P3 | same two legs vs Moonbird 1.1.0 | replace D105/D155 (old book / missing log) |
| P4 | Titanium **with a Zataxx-style net** (sec. 3, M1) vs Titanium main, 5k and 100 ms | isolates eval cost vs eval quality |

Reading: if Zataxx wins P2 but not P1 -> speed (eval cost); if it wins P1 too -> its net/data/search is better per node.
Needs an external-gate wrapper with the same laws as `gate_net_vs_main.py` (T5).

## 2. Our trainer vs bullet's known recipe (FACT columns read from `training/train_nnue.py` and bullet source)

CORRECTION (2026-10-30): the first version of this table said our loss was "plain MSE, no WDL". WRONG — the
trainer has had the WDL/game blend since D11/D38 (user instruction). Read from the loss code, not the runbook.

| aspect | Titanium today (`train_nnue.py`) | bullet reference (`progression/4_multi_layer.rs`) | Zataxx |
|---|---|---|---|
| target | **0.75 * sigmoid(clip(cp,+-1500)/400) + 0.25 * outcome_p** (base stage), optional linear lambda schedule `--lam-start/--lam-end` | WDL 0.75 score + 0.25 result, constant | score + result (config unpublished) |
| loss | weighted mean \|sigmoid(y) - target\|^2.5 (Stockfish power), weights `wsum` | sigmoid(out) vs target, squared error | ? |
| eval scale | **400 cp, never fitted** | 400 (chess cp; scale must match the engine's units) | SCALE 400 |
| arch | 26,196 feats (G1-G6) -> W, dual perspective, CReLU, pairwise (`--arch v2`), **8 phase buckets by empties** + PSQT-style skip, L1/L2 | dual perspective, input buckets + factoriser, CReLU + pairwise, 8 output buckets by material, 2 hidden | 2,916 one-hot -> 256 SCReLU -> 1 |
| optimiser | Ranger (RAdam + Lookahead k 10 a 0.32), hypergrad LR, EMA 0.9995, wd 1.4e-6; adam/adamw available | AdamW | bullet |
| weight clipping | **none found** (only the cp target is clipped) | **L0 +-0.99** (keeps weights inside the i16 x255 quant range) | bullet default |
| LR schedule | val-driven auto-halving + restarts | cosine 1e-3 -> 2.4e-6 over the whole run | ? |
| volume | 32.6M rows, <= ~60 epochs ~ 2e9 samples | ~8e10 samples | ? |
| data | Kata-game rows (kind 1) + own old self-play rows (kind 2) + relabels + Botzone anchors | whole-game binpacks, noisy positions filtered, per-file shuffle + interleave | own 5k/10k-node self-play, 14-17 random plies, |score|<=300 openings |

So the recipe is already close to the modern default (WDL blend, power loss, pairwise, phase buckets, PSQT).
The real gaps are in the DATA feeding the WDL half and in two quantisation/schedule details.

## 2b. Data problems found 2026-10-30 (measured on `final_val_v7.npz`, 677,670 rows)

1. **Own-game outcomes (kind 2, 247k rows = 36%) are nearly flat in the opening**: 0-20 stones (135k rows,
   20% of ALL data): positions our 5k search calls +1200 cp are won 63%, +300 cp 51%, -600 cp 42%. corr(score,
   outcome) 0.15 there. The 25% outcome half of the target teaches noise on a fifth of the data. v3 had
   `--src0-weight` to downweight exactly these rows (memory note: "near-flat calibration"); the v7 base stage
   applies the blend to every row at full weight -> the fix was lost.
2. **Eval scale 400 is mis-calibrated vs real results**: on the clean Kata-game rows the best-fit logistic scale
   is 300 (0-20 stones), 200 (20-35), 150 (35-50): our cp is under-confident by 1.3-2.7x, and the error grows with
   game phase. With a fixed 400 the score half and the result half of the target pull in different directions.
3. outcome_p: 94% hard 0/0.5/1, 6% fractional (averaged duplicates); draws 6%. Perspective is consistent
   (|cp| > 1400 rows agree with the result 99.6%) -> no stm bug.
4. 9% of early val rows carry the one-piece-start signature (sec. 2 of the audit).

## 2c. Kata-teacher verdict REOPENED (user 2026-10-30: "KataAtaxx is 300+ Elo above us; those results were weird")

What D98/D101/D105 actually tested, and why none of it proves "Kata teacher is bad":
- D98 (2026-10-26): stage-2 fine-tune on **Kata rows only**, target = Kata TD values **unscaled** -> -137 vs its own
  stage-1 net. Confounds: (1) Kata-only rows = forgetting of everything else (base val 0.0216 -> 0.0257);
  (2) Kata's value scale is 1.72x ours (D99) -> the search's fixed cp margins (RFP/futility) misfire;
  (3) TD targets carry hindsight (Brier 0.022 at 0-10 empties).
- D99/D102: the ONLY scale-corrected re-test (`s2scaled_vs_s1chg`) was VOID (harness capped searches at 1 s on an
  oversubscribed box, tripwire MISMATCH). No valid re-run exists in the docs.
- D105 arena: KataAtaxx ran on 1 CPU thread at ~2 visits/s (GTP ignores maxVisits, DirectML crashed) -> it measured
  a crippled Kata, not Kata. Botzone: KataAtaxx_mini 2381 vs our bot ~1516 (ratings, not a controlled gate).
- D4: augmentation was missing only from D128 (2026-10-28) to D140; D98 predates it, so D4 is not D98's cause,
  but every net trained 10-28..10-29 is affected and every comparison from that window is suspect.
Status: "never fine-tune on the Kata columns" is downgraded from LAW to UNPROVEN (AGENTS/BEHAVIOR updated).

## 2d. Policy / decision side of every teacher (user 2026-10-30: "policy and eval both should be reviewed")

FACTS: (1) `label`/`labelbin` write only `score depth nodes` (main.rs:1034/1136): the best move of every 5k
relabel search is thrown away. (2) v7 has no move/policy column; Kata's 2-stage policy targets (mean 197 visits)
were never extracted. (3) Titanium @5k picks its own 1M-node move only 46.7%; a policy net's single pass 44.3%,
complementary (17.3% of positions only the policy has it) (NET_ANALYSIS sec. 22). So every teacher today
reaches the net ONLY through its value; the decision (which move) is lost, although that is what wins games.

Review = measure BOTH halves for every teacher (Titanium 5k, Titanium deep, Kata value, Kata policy, game result):
- value: calibration vs results per phase (T10);
- decision: agreement of the teacher's move with a deep reference (1M-node Titanium) and with the game winner's move.

| # | arm | what | cost |
|---|---|---|---|
| PV1 | relabels also store best move + 2nd-best score (Multi-PV 2) | the decision of every search we already pay for; gap = how forced the position is | small code change; MPV-2 ~2x label cost, best move alone is free |
| PV2 | **D1 decision probe (T11)**: on val, our net's depth-1 choice (argmax over children of -eval) vs Kata policy top-1, Titanium 5k move, Titanium 1M move, by phase | what our eval "decides" without search = what it actually learned | probe, minutes-hours |
| PV3 | **decision loss (depth-1 consistency)**: for sampled positions, eval the children with the net, softmax(-v_child / T) vs the teacher's move distribution (Kata 200-visit policy, or Titanium MPV best move), small weight next to the value loss | the teacher's policy flows into the eval itself; D1 search then ranks moves like the teacher | needs child feature lists in the trainer (engine dump of children); days |
| PV4 | separate cheap policy net/head for move ordering (= K3) | ordering instead of eval; 100 ms gate | days |

Order: PV1 goes into every relabel from now on (it is free); PV2 before any teacher fine-tune (it decides
whether K0/PV3 are worth it and is the honest answer to "what did our net learn"); PV3 after K0.

## 3. Training experiments (each = one arm, 5k-node gate on the new 50k book, SPRT; 100 ms leg if speed changes)

| # | arm | change vs current recipe | cost |
|---|---|---|---|
| D1 | **fix the WDL data**: outcome weight 0 on kind-2 rows with < 30 stones (or per-row lambda by source x phase) | removes the measured noise (2b.1) | hours (retrain from current net) |
| D2 | **fit the eval scale**: per phase bucket, scale fitted on held-out results (or one global ~250), applied to the score half | aligns score and result halves (2b.2); engine cp scale unchanged at inference | hours |
| D0 | DONE in code (D162): proven mates = exact 1/0 targets, no clip, no outcome blend; mate distance kept in the raw label | proof is truth, not an approximation | trained with step 5 |
| D2b | replace the +-1500 cp clip by a probability cap chosen with the fitted scale (D163) | the clip was never tuned | with D2 |
| D3 | drop one-piece-start rows (signature filter) | correct game only | minutes |
| R1 | weight clipping +-0.99 on L0 (+ check quant saturation counts at export) | bullet default; quant safety | hours |
| R2 | cosine LR to 2.4e-6 instead of auto-halving (same epochs) | bullet default schedule | hours |
| R3 | AdamW vs Ranger at equal epochs | is the exotic stack earning its keep? | hours |
| M0 | **bullet pipeline**: bullet crate, Zataxx-style 2x2 inputs (+ blocker bit), v7 data after D1-D3 | independent trainer + cheap-eval family | ~1 day + GPU minutes |
| M4 | fresh self-play from the FIXED start (Zataxx recipe), 5k/10k nodes, WDL from those games, 10x volume | data volume + correct game + honest outcomes | days CPU / Modal |
| M5 | best recipe on M4 data | | |

| K0 | **Kata teacher done right**: blend, not replace: target = 0.75 * [a * sigmoid(cp_ti/s_ti) + (1-a) * Kata_value_rescaled] + 0.25 * outcome, Kata value mapped to our scale by the fitted 1.72 slope (or per-phase fit), a in {0.75, 0.5}, ALL rows (no Kata-only stage), D4 aug ON, start-signature rows dropped, warm start from 7AE0122F, low LR | the test D98 never ran; SPRT on the new book decides | hours GPU |
| K-proof | KataAtaxx at proper speed (Modal GPU, ONNX/CUDA) vs Titanium, 100 ms + 1 s, needs a GTP->UAI adapter (D156) | is Kata really +300? if yes it is the best teacher we own | 1 day |
| K1 | **use the KataAtaxx self-play results as the WDL source**: they are the clean half (kind 1, best-fit scale 150-300, corr 0.48 vs 0.22 for own games) -> higher outcome weight on kind-1 rows, ~0 on early kind-2 rows (combines with D1) | Kata outcomes are real strong-play results; D98/D101 forbid Kata VALUE columns as teacher, not the game results | hours |
| K2 | future-stone aux head (Kata channels 2/3 of `valueTargetsNCHW`, NET_ANALYSIS sec. 12): training-only head, free at inference | Kata's only spatial aux task; needs an extraction job from the raw tdata (3.4 GB engines dir + raw files) | 1 day + extraction |
| K3 | policy targets (2-stage, mean 197 visits) -> small policy net for root/ALL-node move ordering (raw top-3 = search best 94%, NET_ANALYSIS) | ordering, not eval; gate at 100 ms | days |

Order: D3 + D1 + D2 together first (data fixes are the biggest measured defects, one retrain), then R1-R3 one
at a time, then M0, then M4/M5. Every arm keeps the current architecture unless named (Net2Net laws, D130).

Rules kept: Net2Net / parity laws for capacity changes of the CURRENT net (D130) do not block a new
architecture family; a new family is gated against main like any candidate. One job at a time.

## 4. Tests (build before more training; each one would have caught a bug of this week)

| # | test | catches |
|---|---|---|
| T1 | `cargo test`: full libataxx/Zataxx perft FEN suite incl. blockers and edge FENs (dedup counting) | movegen/start bugs (start bug: done in `8a114f6`) |
| T2 | UAI conformance test: `position startpos [moves]`, `fen ... moves`, short/long clone, pass `0000`, illegal -> error, `go nodes/movetime/wtime` | serve `0000` bug (done in `8a114f6`, wtime still missing) |
| T3 | gate scripts refuse a book whose lines fail the start-signature test or contain duplicates | the poisoned book |
| T4 | net registry `data/nnue/NETS.json` {sha -> origin, recipe, data}; gate scripts refuse a net whose filename-sha disagrees | `champ_31212A7B.tnn` = f8c4ba1c |
| T5 | one external-gate wrapper with the same laws (book, seats, uainewgame, tripwire, hashes, >=100 g, idle-CPU check) | void Zataxx/Moonbird gates |
| T6 | dataset audit script: start signature %, duplicate rows, stm balance, label range, label vs re-search spot check | contaminated data |
| T7 | pytest for exporter/loader: Rust `nneval` == torch on fixed boards (exists at export), plus feature-index golden tests per G-group | silent feature drift |
| T8 | evidence check: every DECISIONS/LEDGER row with a `logs/...` path must point to an existing file | D155 missing log |
| T10 | calibration report per data source x phase (best-fit scale, outcome-vs-score curve) printed by every dataset build | 2b.1 / 2b.2 |
| T11 | D1 decision probe (PV2) printed for every candidate net next to val loss | value loss alone hides decision quality |
| T9 | CPU-busy guard in every timed gate (refuse if another titanium/python job uses >1 core) | busy-box timed gates |

## 5. Review list (everything, in this order)

1. Engine core: start/UAI (done), serve time controls, perft CLI dedup, Botzone FFI parity with serve.
2. Gate harness: `gate_net_vs_main.py` (patched `--main/--book`), `gate_summary.py`, external gates (T5).
3. Nets: hash every `.tnn`, build NETS.json, delete or rename mislabeled copies (user approves deletions).
4. Data: v7 audit (T6), relabel pool provenance, Botzone anchors.
5. Trainer: `train_nnue.py`/`modal_train.py` vs bullet recipe (sec. 2), loss/target code read line by line.
6. Agent scripts 2026-10-27..30 (harvest/kata/moonbird relabel/modal gate, `kata_bridge.py` diff, LEDGER diff).
7. Docs: fix the 9 holes in the audit; snapshot to `C:\gitProjects\docs-history` after each block.

## 6. Order of execution (final)

1. DONE: start fix, UAI fix, treebook fix, SPRT/pentanomial gate, book/net guards, D154 re-gate (+49).
2. RUNNING: 50k book balanced at 5k nodes (`book50k_std5k.txt`).
3. Gate infra: T5 external-gate wrapper + T9 busy-CPU guard + T8 evidence check (small).
4. Proof gates on the new book: Zataxx and Moonbird (5k + 100 ms); extend the D154 re-gate to an SPRT verdict.
4b. PV1 (best move in relabels) + PV2 D1 decision probe on 7AE0122F / 31212A7B.
5. ONE retrain = data fixes D1 + D2 + D3 (+ K1 outcome weighting), warm start 7AE0122F, D4 on -> SPRT.
6. K0 Kata blend (on top of 5 if it passes) -> SPRT; then PV3 decision loss with Kata policy -> SPRT.  K-proof in parallel only on Modal (no local CPU).
7. R1 clipping, R2 cosine, R3 AdamW (one at a time).
8. M0 bullet pipeline (2x2 cheap net) -> P4 gate per node and at 100 ms.
9. M4 fresh standard-start self-play (Zataxx recipe, WDL from those games) -> M5.
Every step: one job at a time, DECISIONS row + HANDOFFS line when it starts and ends, docs-history snapshot.

## 7. ONE TOOL instead of 234 scripts (user 2026-10-30: "one consolidated thing we constantly improve")

FACT: `training/` holds 204 .py + 30 .bat/.ps1/.sh: 14 relabel*, 11 build*, 9 probe*, 8 export*, 7 train*, 7 smallnet*,
7 kata*, 6 launch*, 5 gate*, ... Each does something slightly different; a weaker model picks the wrong one
(the poisoned book came from a tool nobody re-read; the D154 promotion ran through three different gate paths).

Target layout (one package, one entry point, every subcommand tested):

    python -m ti book build|check            (treebook pool + 5k rescoring; start/dup/balance checks)
    python -m ti gate net|search|external    (one harness: book check, net hash check, seats, uainewgame, tripwire, busy-CPU guard)
    python -m ti gate summary                (pentanomial + SPRT, the only verdict code)
    python -m ti data audit|assemble|calibrate   (start signature, duplicates, per-source x phase calibration)
    python -m ti relabel                     (one resumable relabeler, local or Modal backend, writes score + best move)
    python -m ti train / export / parity     (train_nnue.py stays the one trainer; bullet = a backend later)
    python -m ti botzone build|upload
    python -m ti test                        (all tool tests; run before every commit)

Steps:
1. Inventory: for each script, last use (logs/DECISIONS references), what it imports, which canonical command
   replaces it. Output `training/INVENTORY.md`.
2. Build `training/ti/` from the scripts that were actually used for the current main (gate_net_vs_main,
   gate_summary, build_book_std, overnight_pc_relabel, assemble_v7_dataset, train_nnue, modal_train, export_nnue,
   tools/build_botzone, tools/upload_botzone). Keep their behaviour; tests first (test_gate_tools.py is the seed).
3. `git mv` everything else to `training/attic/` (nothing deleted; history kept). Old commands in docs get a
   pointer to the new subcommand.
4. Enforce: `crates/titanium/tests/repo_hygiene.rs` (exists) gains a check that `training/*.py` contains only the
   allowlisted entry points; a new script fails the test. Agents extend a subcommand instead.
5. Docs: TRAINING_RUNBOOK / RELABEL_RUNBOOK rewritten to `ti` commands only.

