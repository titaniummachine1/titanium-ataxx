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
| E1 | 2026-09-16 | autaxx-style overhaul: qsearch REMOVED (capture trees are bushy in Ataxx — both reference engines dropped it), RFP {257,347,478,774}, LMP (quiet drop >27 when captures exist), universal LMR (r=2, r=3 from move 10) + PVS, NMP retuned (clone-targets ≥11, fill <0.54, R=3), IIR (TT-miss depth≥4 → −1), eval upgrade (tempo 150, PST, autaxx hole-risk table) | main snapshot (v0.2) | 5k nodes | 32 | TBD | TBD | pending |
| E1b | 2026-09-16 | same | main snapshot | 10ms | 32 | TBD | TBD | pending |
