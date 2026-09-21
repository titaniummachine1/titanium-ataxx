# EXPERIMENT LEDGER

Fishtest-style record of every strength experiment. Update every row with the
gate used, game count, W-L-D, Elo estimate, and the verdict. Never delete rows
— reverted experiments stay (marked REVERTED).

Gate definitions (both engines on the same limit, never depth-capped):

| gate | meaning | command sketch |
|---|---|---|
| **5k nodes** | equal compute: `--nodes 5000` both sides, Moonbird gets `go nodes 5000` | `match --games N --nodes 5000 --opp <cmd> --opp-nodes 5000` |
| **10ms / 100ms** | equal wall-clock strength: `--time 100` both sides (10ms for quick sanity) | `match --games N --time 100 --opp <cmd> --opp-time 100` |

8 parallel shards × 4 games = 32 games ≈ 1 min per gate. Bigger runs (100 /
1000 games) when requested. Elo estimate: `−400·log10(1/s − 1)`,
`s = (W + D/2)/N`.

## Baselines (engine v0.2: negamax + ID + TT + killers + history + eval-cache)

| id | change | gate | games | W-L-D | verdict |
|---|---|---|---|---|---|
| E0 | baseline vs **Moonbird 1.1.0** | 1s/move | 50 | 1–49–0 | baseline |
| E0b | baseline vs Moonbird | 5k nodes | 32 | 0–32–0 | baseline |
| E0c | baseline vs Moonbird | 100ms | 32 | 0–32–0 | baseline |
| E0d | self-play sanity | 20k nodes | 16 | 8–8–0 | no color bias ✓ |

## Log

| id | date | change | vs | gate | games | W-L-D | Elo | verdict |
|---|---|---|---|---|---|---|---|---|
| E1 | 2026-09-16 | autaxx-style overhaul: qsearch REMOVED (capture trees are bushy in Ataxx — both reference engines dropped it), RFP {257,347,478,774}, LMP (quiet drop >27 when captures exist), universal LMR (r=2, r=3 from move 10) + PVS, NMP retuned (clone-targets ≥11, fill <0.54, R=3), IIR (TT-miss depth≥4 → −1), eval upgrade (tempo 150, PST, autaxx hole-risk table) | Moonbird 1.1.0 | 5k nodes | 32 | 0–32–0 | — | NO GAIN vs E0 (0–32). No regression; overhaul retained (node efficiency verified, enables eval iteration). Their tuned 2×2-structure eval is the remaining gap. |
| E1b | 2026-09-16 | same | Moonbird 1.1.0 | 10ms | 32 | 0–32–0 | — | same as E1 |
| E2 | 2026-09-16 | **virus-strategy eval** (user insight): endangered-stone contact penalty (−15/stone the enemy can currently convert) + clone +500 / jump −500 ordering preference | main (v0.2) | 5k nodes | 32 | **24–8–0** | **+190** | **MERGED.** Avoid contact, mass-clone first — the user's read of the game is measurably right. |
| E2b | 2026-09-16 | same | main (v0.2) | 10ms | 32 | 18–11–3 | +70 | **MERGED** (with E2) |
| E3 | 2026-09-16 | titanium-inspired tuning bundle: partial-iteration adoption, max-of-two iteration prediction, NMP eval≥beta gate, ordering-band compile asserts, per-game state reset | main (E2) | 5k nodes | 32 | 8–24–0 | −190 | **REJECTED.** Root cause: node-budget prediction makes the search STOP EARLY — main burns all 5000 nodes, E3 voluntarily under-consumes. Gate-fairness bug, not a strength idea. |
| E3b | 2026-09-16 | E3 minus the two harmful bits: partial-iteration adoption + ordering bands + per-game reset only (no node-skip, no NMP eval gate) | main (E2) | 5k nodes | 32 | **24–8–0** | **+190** | **MERGED.** Adoption converts wasted tail-of-budget into real strength at node gates. |
| E3c | 2026-09-16 | same | main (E2) | 10ms | 32 | 17–13–2 | +56 | **MERGED** (with E3b) |
| E4 | 2026-09-16 | **multi-capture exposure eval** (user idea): for every enemy-reachable empty landing square, penalty ×20 per own stone BEYOND the first that one enemy landing would convert | main (E2+E3b) | 5k nodes | 32 | **24–8–0** | **+190** | **MERGED.** No regression at shallow depth (term rarely fires), strong gain when deeper. |
| E4b | 2026-09-16 | same | main (E2+E3b) | 10ms | 32 | **25–6–1** | **+330** | **MERGED** (with E4) |
| E4c | 2026-09-16 | cumulative E2+E3b+E4 vs Moonbird | Moonbird 1.1.0 | 5k nodes | 32 | 0–32–0 | — | gap remains: their eval is self-tuned over many versions (25MB 2x2-structure weights). Next lever: eval weight tuning loop. |
| E4d | 2026-09-16 | same | Moonbird 1.1.0 | 100ms | 32 | 0–32–0 | — | same |
| S1 | 2026-09-20 | **slim-eval-speed (branch exp/slim-eval-speed, SPEED ONLY):** eval = material+PST+tempo only (holes/contact/multicap deleted). PURGE: Ordering enum deleted (lazy only), radix/insertion sorts deleted, MoveStack.scratch deleted, LMP pass deleted, make_via_lut/LUT-infect/extract8/NEIGH/CONVERT8 path deleted from tree, sperft single-variant | main | bench d8 x4 / sperft 8pos | bench 9401 nodes 4.6–8.0M nps vs main 8889 nodes 4.5–4.8M / sperft 4.11 Mnps vs main-lazy 2.40 | — | SPEED: bench noisy (jitter, NOT a win); **sperft truth: 4.11 vs 2.40 = +71% throughput.** Tree variants (ordering switch, LMP scan, radix scratch) were the tax. Main UNTOUCHED. |
| S1gate | 2026-09-20 | slim-eval-speed vs main | main | 100ms | 100 | 73–15–12 (slim–main–draw) | **+230** | **NO REGRESSION — slim stronger.** Speed (+71% sperft) beats lost eval knowledge. MERGED to main 7ab844e, pushed. |
| S2sign | 2026-09-20 | sancta lazy-incr sign bug | classical (same tree) | 100ms / 5k nodes | 20+20 / 8 | fix serving 20–0 @100ms BUT classical serving 7–1 @5k → **sign still wrong in one color path.** Fixed double-negation (forward_ready already stm-relative). | — | BUG: env-inherit serve direction confused results; flip test exposed it. NOT mergeable. |
| S2gate | 2026-09-20 | sancta lazy-incr (sign fixed) vs classical | classical | 5k nodes | 30 | 10–20 (sancta–classical) | ≈−140 | **NOT merged.** ON 5.0–5.2M vs OFF 5.6–8.4M NPS (eval swap ~free). Branch exp/sancta-incr. Retune (RFP/LMR for net scale) OR saturate-own-net next — user decision. |
| S2time | 2026-09-20 | same | classical | 100ms | 30 | 14–16 (sancta–classical) | ≈−13 | Time gate ≈ parity: speed compensates. Supports user log-law (depth beats precision). |
| S3owner | 2026-09-20 | slim(main) vs sanctaphraxx owner (titanium_sancta.exe) | sanctaphraxx | 100ms | 20 | 19–0–1 (slim–owner) | ≈+400 | Owner's brain loses to our tree: his ~300-line PVS+TT has no NMP/LMR/RFP/clone-dedup. Eval steal + our search > his search + his eval. |
| S3moon | 2026-09-20 | slim(main) vs Moonbird 1.1.0 | Moonbird | 100ms | 20 | 0–20 | — | Moonbird still untouched: tuned 2x2-structure eval + deep search. The wall. Own-net saturation is the attack. |
| S3moon1s | 2026-09-20 | slim vs Moonbird 1.1.0 | Moonbird | 1s/move | 8 | 0–8 | — | NOT a TC artifact: more time doesn't save us. Knowledge gap, not speed gap. |
| S3moonX | 2026-09-2x? | Titanium 0.4.0 (external duel harness, NOT titanium-cli match) vs Moonbird 1.1.0 — UNLOGGED until user caught it | Moonbird 1.1.0 | 5k nodes(?) / 100ms | 32+32+32+32+100 | moon2_5k 1–31 / moon2_t100 8–24 / moon3_5k **19–13 WIN** / moon3_t100 5–27 / moon4_t100 6–26 / moon5_t100 17–15 / moon5_100g **52–48 (+14, 100 games)** / moon6_t100(net-9f8e020a) 45–55 / sancmoon (Sanctaphraxx itself) 50–50 | — | MISSED: these ran OUTSIDE LEDGER protocol (duel harness, mixed node/time fairness). moon3_5k + moon5_100g are REAL parity/wins vs Moonbird and were never recorded. Root cause: only titanium-cli match results got ledger rows; duel-harness logs sat in logs/ unread. Fix: this row. |
| S4own1 | 2026-09-20 | own-net v1 (single-view, theirs=0) 25ep on 34k rows | classical | 5k nodes | 30 | 0–30 | — | FAILED: single-view can't learn enemy. Bench sane (a1a3/-45, 5.6M). Archived own_ep25.s1. |
| S4own2 | 2026-09-20 | own-net v2 (dual-view) 25ep, val out .1337/sc .0650 | classical | 5k nodes | 30 | 0–30 | — | FAILED: val loss improved (.19→.16) but gate 0–30. 34k rows too thin OR dag targets too noisy. Need: full 8.7M rows + wsum-weighting + score-only head. |
| S4v3 | 2026-09-21 | own-net v3 (147x64 FT + 128→16→1 head, exact-distill run v3exact killed ep12: tr .2602/va .2606 flat, stale loss math out ~477/sc ~109, pred ~4700 vs tgt O(1)) | classical | 5k nodes | 30 | 0–30 (own_v3.s1 @ep9) | — | FAILED same wipe pattern as v1/v2; bench d8 34k nodes vs 9.4k classical (net costs 3.6x nodes, zero knowledge). Training KILLED. |
| S4v3e | 2026-09-21 | epoch-strength check: v3 ckpts re-exported (export_epoch.py) e1/e5 gated, e12 smoked | classical | 5k nodes | 20+20 | e1 0–20, e5 0–20 (e12 bench 34k nodes/score 286) | — | NOT overfit — never strong at ANY epoch. e1 already wipes (0-9/0-8/9-0 pattern). Verdict: init/scale broken from ep0, not late overfit. |
| S5tool | 2026-09-21 | harness: --net/--opp-net preset flags replace TITANIUM_EVAL_MODE env on bench/serve/match + scripts/gate.ps1 (gitignored, local) + training/export_epoch.py | — | smoke | — | bench classical 9401/5.9M, own_v3 34043/3.5M; 4g gate classical 4–0; selfcheck ok | — | COMMITTED 4271d6b on exp/sancta-incr. Old env vars dead. |
| S6score | 2026-09-21 | score-only CP loss FINISHED ep25 (tr .0537 va .0549, pred_cp 108/112 vs tgt 120/2000 — same scale, LEARNED). Bench d8 own_v3.s1: 6856 nodes/2.5M, score 0, a1a2 (vs classical 17034/6.9M, score 150). Gate vs classical: **20–0 @5k nodes** (classical wins, real Ataxx 39-10/10-39 — no more 0-piece wipes, net plays real games now). | classical | 5k nodes | 20 | 20–0 (classical–net) | — | PROGRESS but not parity: score-only fixed scale (sane scores, real games), knowledge still short. NOT merged. |
| S6big | 2026-09-21 | USER DATA RECOVERED: big teacher attn d128/h4/L4, 844032 params, 5.7M rows cache, ep1 train .1483 val .1447 (base .2001), ep2 .1448/.1443 — only +25 Elo over classical same-nodes, classical wins only when given 4-5x nodes (us 4-5k vs it 1k). Intelligence is LOG-diminishing in Ataxx, NOT chess-like. Consequence: STOP chasing eval Elo; speed (nodes = depth) buys what intelligence can't. | Moonbird/classical | nodes | — | +25 Elo same-nodes | — | Logged from user report + big.log forensics (trainer .py gone, checkpoints big_ep1.pt/big_attn_e2.resume remain). |
| S7quality | 2026-09-21 | zero-scaffold pass BEFORE net capacity (borrowed &'a S1Net, MoveStack owned, score_into free fn, int-NMP, fallthrough-terminal, aspiration ±50/200/800 d5+, dead code pruned) | — | sperft/self | 26t + sperft x3 | 26 green, sperft 4.22–4.34M (28.2M nodes, aspiration deeper), bench d8 17034/6.9M, self-check 10–10 | — | COMMITTED 2a0a0ba. Baseline for all future quality work. |
| S8gatedef | 2026-09-21 | GATE DOCTRINE (user order): 5k nodes = quality-per-node (knowledge truth, equal compute, diagnoses eval), 100ms = realistic strength (speed×knowledge, TIME CROWNS). TIME is the crown, NODES is the diagnosis. Every future test runs BOTH gates for valid research data. | — | both | — | — | — | Standing order, not an experiment. |
| S9night | 2026-09-21 | NIGHT ALL 6 FINAL (tq=S7quality vs tb2=main/Moon/own_v3, sequential, no contention): QB5k 45–53–2 BASE (100g), QBt100 only 12g partial NO total (killed 04:09 — RERUN OWED), QMOON5k 0–100, QMOONt100 0–100, QNET5k 100–0 QUAL, QNETt100 100–0 QUAL FINAL 04:35 (night_DONE). S7 = −28 quality, Moonbird wall stands both gates, own_net still zero both gates. | main/Moonbird/own_v3 | both | 100g each (QBt100 12g partial) | QB5k 45–53–2/QMOON 0–200/QNET 200–0 QUAL | — | S7quality REGRESSES clean-main; net work continues; QBt100 rerun owed. |
| S8incrpst | 2026-09-21 | eval-saturate DONE: `pst:i32` on Board (black-white balance, delta folded into make() capture loop = free), PST moved to board.rs, evaluate() = 3 ops O(1), eval_cached DELETED (was double TT probe + depth-0 pollution), pre-existing dirty aspiration-revert finalized (tq.exe was the 28.2M-node aspiration build — re-search explosion explains S7quality 16-84). 26t green + pst-delta test. | main (tb2.exe) | 5k nodes + 100ms | 32+32 | **20–12–0 / 27–5–0** | **+130 / +270** | **WIN BOTH GATES.** bench a1b1/150 bit-identical values, 9356n 5.8M; sperft 317k nodes @5.3–5.9M vs tq 28.2M @4.15M. Committed 558ae64. Logs: logs/incrpst_5k.txt, incrpst_t100.txt. |

| S10night | 2026-09-21 | NIGHT FINAL all 6 + QBt100 rerun: tq(S7quality) vs BASE 100g@100ms = **16–84** (added to 45–53–2 @5k = REGRESSION both gates). Moonbird 0–200 both. own_net 0–200 both (QUAL 200–0). S7quality (aspiration + borrowed-net) is a decisive REGRESSION vs clean main 7ab844e. | main/Moonbird/own_v3 | both | 100g each | QB 16–84, MOON 0–200, NET 0–200 | −700+ | S7quality REGRESSION confirmed. Branch holds it; user ordered continue on branch. Movegen saturated (LUT closed 3 ways). NEXT = eval-saturate (incremental PST, O(1) eval, delete eval_cached). |

