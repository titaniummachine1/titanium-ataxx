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
