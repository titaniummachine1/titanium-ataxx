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
| E5 | 2026-09-16 | trap certificate at eval leaves (per-leaf `has_moves(opp)` check) | main (E4) | 5k/10ms | 32/64 | 24–8 / 26–34–4 | ~0/−45 | REJECTED: per-leaf cost > rare trap benefit; 10ms combined negative |
| E5b | 2026-09-16 | trap deductions at pass nodes — first version UNSOUND (ignored that enemy landings convert our stones while we are stuck; "equal=draw" and "they can't catch up" were false) | main (E4) | 5k/10ms | 32/32 | 0–32 / 22–10 | — | REJECTED (unsound → catastrophic 0–32 at 5k nodes) |
| E5c | 2026-09-16 | SOUND-only pass-node deductions: stuck side loses iff opp count > mine, or == and they have a clone available (their count non-decreasing, ours non-increasing while stuck) | main (E4) | 5k/10ms/100ms | 32 ea | 16–16 / 14–17–1 / 16–14–2 | ~0 | REJECTED (neutral-to-noise): sound but traps are rare between engines at these depths. Kept on branch `e5-trap-certificate` — likely wins at 1s+ gates where real traps occur; free to re-test.
| A1..A6 | 2026-09-16 | **EVAL INPUT ABLATION** (full vs term-removed, same binary via --mask): material 29–3, PST 29–3, holes 29–3, contact 29–3, **multicap 32–0**, **tempo 32–0** | main (self) | 5k nodes | 6×32 | see left | ALL SIX input families are critical; tempo + multicap (cluster-risk) carry the most weight. NNUE spec must include all of them; net size 147in/64w/2L per sanctaphraxx arch.
