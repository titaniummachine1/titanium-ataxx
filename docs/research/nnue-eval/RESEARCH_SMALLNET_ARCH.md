# SmallNet architecture research (started 2026-11-15; D826-D853)

Last update: 2026-11-15 21:48. Status: leader w2+pol+h1 145.81 cp (-28% vs today); squeeze runs S1 RUNNING; big-net input research B1 being built.
Live numbers: `powershell -File docs/analysis/snarch_status.ps1` (reads the PC log and the Docker box logs).

## TL;DR (valid results only: held-out half B, saturated, ~2% violations)
| design | inputs | B slack (cp) | vs today | row updates per child | state |
|---|---|---|---|---|---|
| **w2+pol+h1+rel** | + move-relative input (user H12) | **142.69** | **-29.8%** | ~17.6 + rel rows | ep 36, 1st decay, STILL IMPROVING (D881); rerun to saturation on PC + with 16 buckets on tit-snd |
| w2+pol+h1 16 buckets | 16 empties buckets (user H2) | 144.29 | -29.0% | 17.6 | last decay, ~saturated (D881) |
| w2+pol+h1 | windows + policy + parent big-net hidden layers | 145.81 (seed 2: 145.58) | -28.2% | 17.6 (+h1 per node) | SATURATED (D869/D881) |
| w2+h1 (ablation: no policy) | | 150.67 | -25.8% | 15.6 | ep 57, 1st decay (D881) |
| w2+pol | 2x2 windows + policy rank/log-prob | 148.42 | -27% | 17.6 | SATURATED |
| dw2sq+pol (all-in) | windows + stones + delta + policy | 150.83 | -25.8% | ~40 | SATURATED, no gain over w2+pol |
| next: +h1 | parent big-net hidden layers (per node) | screen -5.9% on sq+pol | - | +0 | RUNNING 1.2M: sq+pol+h1 (PC), w2+pol+h1 (box C), w2+pol+all4 (box A) |
| w2 | Zataxx-style 2x2 windows | 155.4 | -23.5% | 15.6 | lr 1e-3 DONE |
| dw2sq+pol (all-in) | windows + stones + delta + policy | 156.1 | -23% | ~40 | lr 1e-3 epoch 9 (PC) |
| dsq+pol | stones + delta + policy | 170.8 | -16% | 22.1 | lr 1e-3 DONE |
| **sq+pol** | today's stones + policy | 179.8 | **-11.5%** | **6.5** | lr 1e-3 DONE |
| sq (today, control) | per-stone rows | 203.2 | - | 4.5 | lr 1e-3 DONE; lr 2e-3 205.9 |

Reading so far:
1. The 2x2 windows carry the biggest single gain (-23.5%); the policy input adds on top of them (-27%).
2. The policy input is the cheapest gain: -11.5% for +2 row updates per child (user D841 minimax idea).
3. Delta (parent+move) alone is worth nothing (old-protocol log: 207.4 vs 206.4); combined it helps less than the windows do.
4. Speed is NOT known yet: w2 needs ~3.4x the row updates of sq and a 373 KB table vs 19 KB. An old tighter-bound net did not make the engine faster (D115/D116). Only the P6 ns benchmark and timed gates decide.
5. LR: 1e-3 beat 2e-3 on sq (203.2 vs 205.9); earlier lr-2e-3-only numbers understate every design.
6. The loosest bound by far is the endgame (empties bucket 0: 477-731 cp slack vs 110-180 mid-game) and violations are uneven per bucket (dsq+pol 4.2% in bucket 0 with a global shift): see register N1/N2.

## 1. Question
The futility SmallNet is a LOWER BOUND of the big net's eval of a search child. A tighter bound (less slack at the same violation
rate) prunes more children without a big-net eval, so the search gets faster (user: "the more precise it is the faster our engine
gets"). Which inputs and design give the tightest bound per unit of cost, and does that turn into Elo at the played clock?

How the engine uses it today (search.rs:2254-2298): only late moves (move_idx >= 1), new_depth > 0, alpha not a mate score, and only
when the child eval is not already in the eval cache. The SmallNet outputs ONE number lo (child's side to move); FUT_SN_M = 0.
Prune (skip the move: no big-net eval, no search) when alpha + lo >= 57 * new_depth; otherwise the big net evaluates the child and
search continues; the SmallNet never replaces the eval. Safety comes only from training: pinball loss at tau 0.02 = the net learns the
2% conditional quantile (close where sure, far below where unsure). The parent accumulator is built once per node and every child
adds only the changed stones (smallnet.rs:135-238, child_acc_into); the head (64x16 FMAs) costs the same for every design.

Starting point: bundled SmallNet 8F479ECB; refit on 812C2118 = D70D46DD (exp/smallnet-812 6baba9cc, P(bound>truth) 3.87% -> 2.09%,
slack 232.8 -> 207.7 cp, gate 100 ms +5.8 +-28.1 at 600 g = undecided, extension owed, D831). Zataxx (C:/gitProjects/Zataxx/src/nn.rs):
36 overlapping 2x2 windows x 81 states = 2,916 features, always 36 active rows, 256 hidden, SCReLU.

## 2. Method (protocol D845; tool `ti train snarch`, titaniumtest/src/snarch.rs, hub commit 380f53f + later uncommitted P2 inputs)
- Data: big net 812C2118 (C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn). Train = 1,199,104 rows of train_v10 (1,171
  strided blocks of 1024; cloud copy = exact subset docs/analysis/snarch_subset.ps1, sha256 C05E1F8C6252C885) -> every 3rd child
  by hash, D4 mode per row from (seed, global row index) = 18,236,047 children. Val = val_v9_clean 60k rows split BY POSITION into
  alternating 1024-row blocks: half A (selection) and half B (report only). Targets = big-net eval of the child (cp).
- Samples are byte-identical on every machine and thread count (FNV checksum printed; verified 3 vs 7 threads; P2 control sq+pol
  233.55 vs D849 233.52 at 4 vs 7 threads).
- Model (all designs): input rows -> 2 perspectives x W32 accumulator (CReLU) -> 8 empties buckets x (L1 64->16 CReLU -> 1) + per-bucket
  skips of the parent eval and the child's hand eval (= the engine head). Extra input groups (suffixes) start at 0, so a +X model is
  bit-identical to its control at step 0 (unit test) and any difference is the input's own.
- Training: 4 MSE warm-start epochs, then pinball tau 0.02 to SATURATION (user definition): constant LR; < 0.02% relative gain of A's
  slack@tau over 4 epochs -> LR x0.3, up to 3 decays, then stop after 4 more epochs without gain; best A epoch restored and saved
  (binary .snar). LR sweep 1e-3 / 2e-3 / 4e-3 per design.
- Metrics: B slack = mean (truth - bound) on half B after the shift fitted on A (so B's violation rate is out of sample, ~2%); B quantiles,
  per-empties-bucket slack/violation/n; RMSE; COST = row updates per child given the parent accumulator (sq: changed stones; windows:
  2 x changed windows; delta: its active rows; +pol 2, +dis 1, +mv 1, +ent 2, +h1 0 per child because it is per node).
- Verified before use (unit tests 5/5 + end-to-end): touched counts on a hand-worked move; 2x2 window layout; tau shift gives exactly
  tau violations; best-epoch restore (forced early-best case: epoch 198 of 201 restored exactly); +X = control at init; P2 inputs: big net
  l1 16 + l2 32 = 48 hidden values stored, h in [0,1], parent dis mean 78 max 1054 cp; dis computed exactly as the engine (export.rs:271-276).

## 3. Results
### 3a. P1 held-out (1.2M rows; D853 + 20:10 update)
| model | B slack | B viol | A slack@tau | B q10/50/90 | B RMSE | touched | params | time |
|---|---|---|---|---|---|---|---|---|
| sq lr1e-3 | 203.16 | 2.01% | 206.06 | 52/165/370 | 278.1 | 4.5 | 13,216 | 10 min |
| sq lr2e-3 | 205.86 | 2.05% | 209.66 | 51/167/376 | 283.0 | 4.5 | 13,216 | 7 min |
| sq lr4e-3 | 204.88 | 2.03% | 207.73 | 51/163/374 | 287.1 | 4.5 | 13,216 | 9 min |
| sq+pol lr1e-3 | 179.84 | 2.02% | 181.99 | 45/143/331 | 251.0 | 6.5 | 16,288 | 12 min |
| sq+pol lr2e-3 | 180.70 | 2.04% | 183.31 | 45/144/332 | 252.6 | 6.5 | 16,288 | 8 min |
| sq+pol lr4e-3 | 182.85 | 2.01% | 185.37 | 45/144/336 | 255.7 | 6.5 | 16,288 | 7 min |
| dsq+pol lr1e-3 | 170.79 | 2.00% | 172.45 | 45/137/313 | 231.8 | 22.1 | 202,912 | 12 min |
| dsq+pol lr2e-3 | 172.44 | 1.96% | 174.01 | 46/138/314 | 234.1 | 22.1 | 202,912 | 13 min |
| w2 lr1e-3 | 155.44 | 1.99% | 157.39 | 42/126/283 | 206.5 | 15.6 | 101,824 | 16 min |
| **w2+pol lr1e-3 (DONE, best)** | **148.42** | 2.02% | 150.10 | 39/119/272 | 200.8 | 17.6 | 104,896 | 21 min |

D859 (20:25): every finished design is best at lr 1e-3 (higher LR = 0.5-2% worse); lr 2e-3/4e-3 legs of w2, w2+pol, dsq+pol still
running only to confirm that. Box outputs (logs + .snar models) copied home: logs/snarch/sbx_sna|snb|snc/out (sha-verified tars).
| dw2sq+pol lr1e-3 (running, ep 9) | 156.14 | 2.05% | 158.34 | - | 220.1 | ~40 | - | - |

Per empties bucket (B slack cp / violation / children), lr 1e-3:
| model | b0 (0-5 empty) | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---|---|---|---|---|---|---|---|
| sq | 666/2.4%/69k | 275/2.3% | 202/2.2% | 170/2.1% | 156/1.8% | 135/1.6% | 184/2.2% | 731/3.7%/975 |
| sq+pol | 600/2.2% | 237/2.2% | 178/2.0% | 150/2.0% | 138/1.9% | 121/2.1% | 181/2.0% | 691/3.2% |
| dsq+pol | 538/4.2% | 222/2.7% | 169/1.9% | 141/1.9% | 133/1.8% | 125/1.5% | 180/1.6% | 514/4.0% |
| w2 | 477/3.0% | 215/2.4% | 153/2.0% | 129/2.0% | 119/1.8% | 110/1.7% | 168/1.8% | 645/3.5% |

### 3b. Screens (60k rows, paired, held-out B). D867: D849 and D852 were CAPPED at 25 epochs (every model hit the cap) = NOT results, conclusions voided; saturated reruns D867/D864
- D849 policy screen (logs/snarch/d847_polcheck_60k.log): sq 262.4, sq+pol 233.5 (-11%), w2 233.9, w2+pol 227.3; touched 4.6 / 6.6 / 15.7 / 17.7.
- D852 P2 screen (logs/snarch/d852_p2screen_60k.log; D856, 20:13), B slack vs control sq+pol 233.55 (viol 2.07%): +dis 236.64 (+1.3%), +mv 234.89 (+0.6%), +ent 237.54 (+1.7%), **+h1 219.76 (-5.9%, viol 1.94%, RMSE 318 -> 286, 0 extra rows/child: per-node dense projection of the parent's big-net h1||h2)**; all four combined FINAL (D857): 218.36 (-6.5%, viol 1.99%, touched 10.6 vs 6.6 rows/persp) = +h1 alone within noise (-0.6%); the epoch-15 value 229.2 quoted in D856 was mid-run, not a result. So dis/mv/ent add at most ~0.6% on top of h1 at 60k rows for +4 rows/child. Gain of +h1 is largest in the endgame bucket (0-1 empties 820 -> 665). Screen only (60k rows, 25-epoch cap): +dis/+mv/+ent "no gain" is NOT final until a 1.2M run; +h1 goes to P2 1.2M runs (sq+pol+h1, w2+pol+h1). Cost of +h1 = one 48x32 dense per node per perspective (not per child), NOT measured yet.

### 3c. Superseded results (kept for the record, NOT results)
- Run 1 (D834, logs/snarch/d834_run1.log): fixed 4 MSE + 16 pinball epochs, cosine LR: sq 212.6, w2 159.9/159.4, w2sq 171.5 (stopped at ep 7). Not saturated.
- Run 3 (D839, logs/snarch/d839_sat.log, box A logs/snarch/sbx_sna): saturation judged on val PINBALL (D844 bug: pinball rose from
  calibration drift while slack@tau still fell; delta restored epoch 3 = 211.7 although epoch 16 had 207.5). sq 206.6, delta 211.7, dsq 182.2.
- D846 (best slack@tau per design read from those logs, selected and measured on the same val set): w2 159.4, dw2 160.4, w2sq 171.5,
  dsq 180.3, sq 206.4, delta 207.4. Ranking information only; delta alone = no gain and dw2 ~ w2 come from here.

## 4. Theory notes
- Decision-aware loss (user D837): only errors that flip the prune decision cost anything: false prune (truth < t <= bound) or missed
  prune (bound < t <= truth), t = 57*new_depth - alpha. Over a threshold density p(t) with CDF F the expected cost is
  w_fp*(F(bound)-F(truth)) above the truth and w_mp*(F(truth)-F(bound)) below; with p uniform this is exactly the pinball loss with
  tau = w_mp/(w_fp+w_mp). Today's loss therefore assumes alpha can fall anywhere; the real thresholds are concentrated near the parent's score.
- Minimax (user D841): V(parent) = max over moves of -V(child), so the child bound = parent value + this move's regret vs the best move.
  The parent value is already an input (skip); the regret is what the policy head ranks, and the engine has every move's policy logit at the
  prune site (move ordering, logit_k(b, m, &[], kc) x pol_w). Holds exactly for search values only, so it is used as an input, not imposed.
- Old lesson (A1b, D115/D116): a tighter bound skipped 74% vs 68.6% of children but the engine was not faster: the bigger net cost what it
  saved. Every design is scored on (a) saturated held-out bound, (b) measured ns per child, (c) evals saved x big-net ns - SmallNet ns, (d) timed gates.

## 5a. USER HYPOTHESES (every one gets a measured test; user D863/D878: no idea is hand-waved; updated 2026-11-15 21:48)
RULE (user D879, 2026-11-15 21:50): every training / architecture result from before 2026-11-15 ran 1-5 epochs and is NOT evidence; only saturated runs count, and large tables (3x3) need far more epochs than 2x2.
Statement = the user's idea in short; Test = what decides it; Status with the number. Nothing here is closed by argument, only by data.
| H | Hypothesis (user) | D | Test | Status 2026-11-15 21:00 |
|---|---|---|---|---|
| H1 | Knowing what the move CHANGED vs the parent (delta) helps the bound | D826 | delta / dsq / dw2 vs controls, held-out | alone no gain (207.4 vs 206.4, old protocol); dsq+pol 170.8 vs sq+pol 179.8 (-5%) at 3.4x rows; all-in dw2sq+pol 150.83 vs w2+pol 148.42 = no gain on top of windows + policy (D866) |
| H2 | Mixture of experts by phase / move type | D827 | buckets by move type, per-phase heads | 16 buckets on w2+pol+h1 RUNNING (tit-snb, D870); move-type experts QUEUED |
| H3 | Zataxx-style 2x2 windows beat per-stone input | D829 | w2 vs sq, 1.2M saturated | CONFIRMED: 155.4 vs 203.2 (-23.5%) |
| H4 | Combine delta + full input ("just more input") | D835 | dsq/dw2/dw2sq (+pol) | dsq+pol -5% vs sq+pol; dw2sq+pol 150.83 (saturated) = no gain over w2+pol 148.42 at 2.3x rows |
| H5 | Bias the bound toward the safe direction / reward predictable error direction (alpha/beta) | D837 | decision-aware loss on real (alpha, depth) thresholds vs pinball | THEORY done (pinball = uniform-threshold case); test QUEUED P5 (needs prune-site samples from the search) |
| H6 | Minimax: best move value = position value, so the policy helps like in the policy net | D841 | +pol (move rank + log-prob) | CONFIRMED: sq+pol -11.5%, w2+pol 148.42 (-27% vs sq) |
| H7 | Combine everything into one model, then prune by ablation | D850 | all-in, then drop one group at a time | all-in DONE (150.83, no gain); ablation of the leader w2+pol+h1 RUNNING (w2+h1 = drop the policy, tit-snc) |
| H8 | Big-net knowledge as SmallNet input: disagreement head, policy, big-net features | D851 | +dis, +ent, +h1 | +h1 CONFIRMED at 1.2M saturated: w2+pol+h1 145.81 = NEW BEST (-1.8% vs w2+pol), sq+pol+h1 175.48 (-2.4%); seed-2 check RUNNING (tit-sna); +dis/+ent/+mv only capped screens = VOID (D867), retest QUEUED |
| H9 | "Artificial 3x3": cross (3x1 + 1x3) + X (2 diagonals) per centre | D861 | arch x3 vs w2, cost per child | BUILT + tested (23,814 features, 98 rows, 22 row ops vs w2 8 on the hand move); screen stopped before x3 ran (D870 focus) -> QUEUED at 1.2M |
| H10 | Full 3x3 tuples | D861 | arch w3 (information ceiling; table too big for the engine) | BUILT; screen 343 cp at epoch 34, still dropping, stopped (D870) -> QUEUED at 1.2M+ rows, to its OWN saturation (user D879: 3x3 needs exponentially more epochs than 2x2; all older 3x3 results are 1-5 epoch runs = void) |
| H11 | Move-source tuple: weakness a jump creates (own pieces around the hole, can the opponent reach it); clone = destination tuple | D861 | +src vs control | 60k SATURATED: w2+src 224.93 vs w2 220.41 (+2.0% worse), sq+pol+src 229.61 vs 207.32 (+10.8% worse): overfits at 60k (26k table rows) -> retest at 1.2M QUEUED |
| H12 | Move-relative input (user: "Timo transformer", meaning not confirmed): move's source/destination square in the tuple, every square relative to the move | D863 | +rel = (role) x (offset) x (state); +relf whole board | 60k: sq+pol+rel 200.59 at ep 48 still dropping (-3.2% vs sq+pol, the only new idea ahead); full 1.2M w2+pol+h1+rel RUNNING (tit-snb) |
| H13 | MAIN (big) net gets the prior position / prior eval ("improve evaluation by knowing what we evaluated on the prior move") | D863 | offline info test on game plies vs the 5k label, then Net2Net variant + gate | HANDLING DESIGN D875 (root: previous game position is known; first position / no history = "absent" flag + parent-dropout training; cached base eval stays position-only, parent part = separate correction term); offline test tool `ti` parent-info BEING BUILT |
| H14 | Ablation to learn what matters most | D863 | drop each group from the leader, saturated, held-out | RUNNING on w2+pol+h1 (drop pol: w2+h1; drop h1: w2+pol 148.42 known; drop windows: sq+pol+h1 175.48 known) |
| H15 | "The more precise the SmallNet, the faster the engine" | D826 | ns per child + timed gates (old lesson D115/D116: a tighter bound once did NOT make the engine faster) | QUEUED P6/P7 |
| H16 | SmallNet inputs that helped, added to the BIG net | D872/D876/D877 | each as an ADDED zero-init group on 812C2118 (step-0 parity, C14), Modal fine-tune, gate | NEVER tried before (D876). Candidates: parent eval, move policy rank+log-prob (path-dependent: D875), parent hidden layers (path-dependent), see H18-H20 |
| H17 | Ablate the big net (812C2118): which input groups / neurons / phases matter | D872 | group + per-phase + neuron ablation | GROUP DONE D873 (regret: no G1 square state 69.0 cp, no G3 ring 50.8, G2 15.7, G5 12.6, G6 6.2; all matter more than in 46FCB5A4); NEURON DONE D874 (no free edit; prune W224 +2.1%; l1 neurons 2 and 6 = 18.7% / 15.0%); per-phase + retrained ablation QUEUED |
| H18 | Big net + 2x2-window group (Zataxx), the SmallNet's biggest gain | D876 | 812C2118 + w2 group (36 x 81 rows, zero-init), fine-tune, gate | QUEUED (needs the group in engine nnfeat + hub trainer) |
| H19 | Big net + 3x3 / cross+X group | D876 | 812C2118 + w3 or x3 group, zero-init, fine-tune, gate | QUEUED (after H18). The old 3x3 runs (D482 -147 Elo, D614, D459) trained 1-5 epochs = VOID (user D879): 3x3 is completely open; 3x3 needs far more epochs than 2x2, so it is judged only after its own saturation |
| H20 | Big net gets the POSITION's OWN policy (entropy, top-1 prob, number of good moves) | D877 | added input, path-independent = safe for TT and the root; cost = a policy pass per eval | QUEUED (offline info test together with H13) |

## 5. Idea register (status 2026-11-15 20:10)
| # | Idea | Source | Status |
|---|---|---|---|
| A1 | sq: per-stone input (today) | engine | P1 running (control) |
| A2 | w2: 36 overlapping 2x2 windows x 81 states | user D829, Zataxx | P1: 155.4 (-23.5%) |
| A3 | w2sq: windows + stones | D834 | old protocol only (171.5, worse than w2); not in P1 |
| A4 | delta: changed windows, child + parent state | user D826 | old protocol: no gain alone; closed unless combined |
| A5 | dsq / dw2 / dw2sq combinations | user D835 | dsq+pol P1 170.8; dw2sq+pol P1 running (all-in, D850) |
| A6 | 3x3 windows (Jataxx 25 windows) as bound input | D611/D614 | QUEUED (never tried as a bound) |
| A7 | Mixture of experts by phase / move type | user D827 | QUEUED |
| A8 | 16 buckets / buckets by move type | D834 | QUEUED (--buckets exists) |
| A9 | Width W16-64, H8-32; SCReLU vs CReLU | D829, D115 | QUEUED, scored with ns |
| I1 | Policy log-prob + rank of the move | user D841 | DONE screen -11%; P1 sq+pol 179.8 (-11.5%), w2+pol 148.5 (-27%) |
| I2 | SEE / max opponent threat of the child | BACKLOG 0 (D744) | QUEUED |
| I3 | No hand-eval input (saves ~323 ns per uncached child) | BACKLOG 3 (D702) | QUEUED (ablation) |
| I4 | Parent disagreement value (big-net dis head) | user D851 | P2 screen running |
| I5 | Parent big-net hidden layers h1+h2 (48 values) projected into the accumulator | user D851 | P2 screen running |
| I6 | Move features: clone/jump x flips | D851 | P2 screen running |
| I7 | Parent policy entropy + top-1 log-prob | D851 | P2 screen running |
| I8 | Move ordering index + history score | D851 | QUEUED (needs search-sampled data, L5) |
| I9 | Reuse the big net's child accumulator (if its update is cheap and its head is the expensive part) | D851 | QUEUED: measure the big-net cost split first |
| I10 | SmallNet predicts its own uncertainty (per-child shift) | D851 | QUEUED after L1 |
| N1 | Per-bucket shift (calibrate tau per empties bucket; violations are uneven: 1.5%-4.2%) | 20:10 finding | QUEUED (cheap, export-time) |
| N2 | Endgame (empties 0-5: slack 477-731 cp): separate endgame head or no SmallNet below k empties | 20:10 finding | QUEUED |
| L1 | Decision-aware loss over real (alpha, depth) thresholds | user D837 | QUEUED P5 |
| L2 | Threshold-conditioned prune classifier (input t, output P(big net < t)) | D837 | QUEUED after L1 |
| L3 | tau sweep (0.5-1%) | BACKLOG 3 | QUEUED |
| L4 | Minimax consistency target from .tbst best moves | user D841 | QUEUED (optional) |
| L5 | Training samples from the search (late moves at real prune nodes) | D837 | QUEUED P5 |
| D1 | Data scaling 1.2M -> 4.8M rows (big tables overfit at 1.2M) | D842/D844 | QUEUED P4 |
| E1 | Two-sided bound (upper quantile) for RFP/leaves | old A2 | QUEUED |
| E2 | Downscale S1-S6 if the winner is too slow | D168/D169 | QUEUED (conditional) |
| E3 | Engine-style i16/AVX2 ns per child + cache footprint | A1b lesson | QUEUED P6 |
| G1 | Extend the gate of the refit D70D46DD | D831 | OWED |
| G2 | Retrain the SmallNet with every new big net | old C3 | STANDING |
| X1 | Plain margin +40/+80 | D115/D116 | CLOSED (no Elo) |

## 6. Plan (ranked, D878 2026-11-15 21:48; user D870: focus on the best designs, squeeze the SmallNet)
SMALLNET (leader w2+pol+h1 = 145.81 cp, 1.2M rows, saturated):
S1 RUNNING: w2+pol+h1+rel (H12), w2+pol+h1 16 buckets (H2), w2+h1 = ablation without policy (H14), seed 2 of w2+pol+h1 and w2+pol (noise check).
S2 data scaling on the leader: 2.4M rows (PC) and 4.8M rows (new Docker box after the ttl of the old ones) (D1).
S3 width W64 / H32 on the leader (A9), per-bucket violation shift (N1), endgame head / no SmallNet below k empties (N2).
S4 retest at 1.2M what only had capped or 60k screens: +dis, +ent, +mv (VOID screens), +src, x3, w3.
S5 decision-aware loss on search-sampled children (H5, L1/L5).  S6 ns per child (engine-style i16) + timed gates (H15, P6/P7).
BIG NET (user D863/D872/D876/D877):
B1 BUILDING: offline information test (H13/H20): do parent eval, move policy rank/log-prob, parent hidden layers, the position's own policy, 2x2 windows explain the champion's error vs the 5k label on game plies? (half A fit, half B score, per phase).
B2 add the winners as zero-init groups to 812C2118 (H16/H18/H19/H20; path-dependent ones with the D875 handling), Modal fine-tune, ti gate 5k + 100 ms + 1000 ms.
B3 per-phase + retrained big-net ablation (H17 rest).
## 7. NOT checked yet (2026-11-15 21:48)
Speed (ns per child) and Elo of ANY new SmallNet design (no engine port, no gate); seeds > 1 (running); data sizes other than 1.2M (queued);
16 buckets (running), widths, SCReLU, per-bucket shift, endgame head; +dis/+ent/+mv/+src/x3/w3 at full size; decision loss; search-sampled
children; every BIG-net input idea (H13, H16, H18-H20: never tried); big-net per-phase and retrained ablation.
## 8. Raw data and reproduction
- Code: titaniumtest/src/snarch.rs (commit 380f53f local + P2 inputs uncommitted); binaries target_snarch / target_snarch2 (Windows),
  target_linux (Linux, zig linker wrapper C:/gitProjects/toolchains/zigcc-linux, ti sha 83af20b04b7f31ac for P1 on Docker).
- Logs: PC logs/snarch/*.log (d834_run1, d839_sat, d847_polcheck_60k, d850_allin, d852_p2screen_60k); boxes /tmp/sn/out/snarch_<arch>.log
  (copy to logs/snarch/sbx_<box>/ before deletion; box A old run in logs/snarch/sbx_sna); weights data/nnue/snarch/*.snar and /tmp/sn/out/*.snar.
- Cloud: Docker tit-sna (16 vCPU), tit-snb (16), tit-snc (8), ledger logs/docker_spend.csv, ttl delete ~23:20; script scratch/sbx_snarch/run_snarch.sh.
- Decisions: D826-D853 in docs/DECISIONS.md.

## 9. Engine port (D889/D890, 2026-11-15 22:47)
Branch exp/snarch-leader (C:/gitProjects/snport-ataxx, from origin/main e33966f8, not committed yet): crates/titanium/src/smallnet2.rs, nnue.rs eval_with_dis_h, titanium-cli snbench. Log logs/snport/d889_snbench_1.log.
Same 139,697 engine self-play children (40 games, 3,000 nodes/move), truth = big net 812C2118:
| net | violation as shipped | slack as shipped | slack at 2% violation | ns per child call | ns per node setup |
|---|---|---|---|---|---|
| main 8F479ECB (sq) | 3.00% | 266.4 | 285.9 | 88.3 | 98.5 |
| leader w2+pol+h1+rel (8 buckets, shift -5.9) | 2.03% | 154.3 | **155.3 (-45.7%)** | 539.4 (6.1x) | 575 acc + 3,631 policy |
10,000 calls: main 0.88 ms, leader 5.39 ms. Per node at 2.94 calls: main 358 ns, leader 5,792 ns (+5.4 us = 1.4 big-net evals). Parity fast vs reference: max 1 cp.
NOT yet: speed work (policy reuse, i16 + register accumulator), in-search prune counts, any gate. Timings taken on a loaded PC (7-thread training running).

## 10. Results 2026-11-16 07:58 (D908; boxes tit-sne / tit-snf, all saturated, held-out B)
| design | rows | B slack cp | viol | best epoch | rows/persp | params |
|---|---|---|---|---|---|---|
| w2+pol+h1+rel (leader) | 4.8M | **133.86** | 2.03% | 45 | 36 | 127,176 |
| w2+pol+h1+rel (leader) | 1.2M | 141.54 | ~2% | - | 36 | 127,176 |
| x3+pol+h1+rel | 1.2M | 144.47 | 1.98% | 39 | 98 | 795,912 |
| w3+pol+h1+rel | 1.2M | 157.62 | 2.03% | 12 | 25 | 15,780,264 |
Reading: 4x data = -5.4% slack for the leader (data scaling not exhausted). w3 has 124x the leader's parameters and stopped at epoch 12 on 1.2M rows: data-limited, needs the 4.8M+ run before any verdict (D879). Raw: logs/snarch/sbx_sne/out, logs/snarch/sbx_snf/out.
NOT checked: w3/x3 at 4.8M rows; leader at 9.6M+; engine speed of the 4.8M leader (same shape as the 1.2M leader, so same 539 ns/call, D890); Elo.

## 11. SmallNet refit FA1FE34A in play (D914-D918, 2026-11-16)
Candidate B (exp/sn-sat b5a6a17 = cleanup + FA1FE34A, same big net + policy as main) vs main at 100 ms Botzone rules, pooled 1600 paired games: +11.7 +-16.9 Elo, LOS 91.3% (legs -9.6 / +16.5 / +20.0), nps +2.3%, depth 10 vs 9, blunders 2.5% vs 2.9%. Undecided, extending. Weak spot: empties 16-18 (b5, z -2.43). Raw: snsat-ataxx/logs/gates/2026-11-16-b5a6a178-*; reports logs/gates/reports/d914-d918_*.
