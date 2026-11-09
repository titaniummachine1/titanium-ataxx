# Titanium Ataxx: research route (human-readable summary, state 2026-11-09)

Self-contained overview of what was tried on the evaluation network, policy head and search, in order, with the measured result of each step.
It is a summary: numbers come from the gate and experiment records listed under "Sources"; every Elo figure is from a paired, seat-balanced match
against the then-current engine, with the games and time control stated. "Undecided" means the match did not reach 95% confidence either way.
Current champion: `origin/main` 285a127 (NNUE `46FCB5A4`, policy head `EE96825B`, futility small net `8F479ECB`).

## 1. Starting point
- Alpha-beta engine on 7x7 bitboards; NNUE (TNN3): 26,196 sparse features, 256-wide feature transformer, 16 -> 32 hidden, 8 output buckets by empties;
  about 208 feature rows per evaluation, about 5 us per evaluation, 110-148 knps single thread in real positions.
- Training data: 31.57M positions labelled by the engine's own 5,000-node search ("v9" set), each with search score, game outcome, a per-row weight
  and exact-proof flags for solved endgames. Later: Kata (neural engine) soft move targets for the policy head.
- Training recipe: target = blend of sigmoid(search score) and game outcome, power-2.5 loss weighted per row, Ranger optimiser, D4 augmentation.

## 2. What moved strength (measured)
| step | result | note |
|---|---|---|
| Game-result weight 0.55 in the target (late-game weighted) | +23 at 5k nodes, +47 at 1000 ms | 0.7 and 0.85 were -20/-21; less outcome weight mid-game -32 |
| Export calibration | hurt (+19 uncalibrated vs calibrated trunk) | removed |
| Policy head for move ordering | +125.7 at 100 ms (274 g) | then Kata fine-tune +21.7 at 5k (1,200 g) |
| Policy head + class-shared 3x3 pattern tables | +56.5 +-35 at 100 ms (360 g) | merged |
| Jump-ring (distance-2) policy tables | +12 at 5k, -13 at 100 ms (400 g) | reverted: the played clock decides |

## 3. What did not work (and why)
- **TSTG (8-channel pattern evaluator) as sole evaluator:** 0 points in 128 games vs the NNUE. Earlier positive gates were the NNUE evaluating with TSTG only ordering moves. It ranks sibling moves poorly (top-1 agreement 49-52%). A colour-swap bug in training explained part of the gap.
- **Pattern nets under the champion head** (one 3x3 window per stone, 58 rows vs 208): eval 1.45x faster but played nodes/s only +3.8%, and -147 +-72 at 100 ms. The speed edge shrinks as stones accumulate (1.51x at ply 10 to 0.97x at ply 70). Linear pattern/tuple evaluators cap near 54% sibling agreement with the champion.
- **Narrower nets:** W128 won at 50/100/300 ms (+60/+54/+26) but lost at 1000 ms (-38.8), and the Botzone clock is about 4 s; W224/W192 after pruning cost +0.5%/+0.9% validation error (not yet gated; W192 with an 8-bit feature table is about 23% faster eval).
- **8-bit feature table:** -9% eval time, +12% game nps, 0 Elo at 100 ms (1,318 g), -37.8 +-67 at 1000 ms (undecided).
- **Self-distillation from a stronger self (deeper labels from our own tuple net):** worse than diverse weak labels (echo chamber); outcome columns of unknown meaning poisoned training (-400).
- **Search knobs per NNUE bucket and SPSA:** all within noise; candidate "A'" -0.2 Elo over 6,000 games (neutral).

## 4. Speed work still open
- Ranked eval cache + ranked transposition table (same 96 MB): 1.25x faster at equal nodes, +17.8% played nps at 1000 ms, +40.1 at fixed nodes (LOS 94%), but time-clock gates are undecided (pooled 716 g: -9.7, P about 21%). Not merged.

## 5. Opponent study: Jataxx / PyJataxx (Botzone)
Source of the open-source bot (38 versions) was read; see `docs/RESEARCH_JATAXX.md`. Findings: 25 overlapping 3x3 windows -> 128 hidden units (squared ReLU) -> 3 output buckets by piece count, one value output, inside a best-first minimax over an NN-evaluated tree; no transposition table or evaluation cache; weights are private. A screen of that feature layout on our data (28.99M rows, game-outcome target) showed no usable outcome information beyond our search score (BCE 0.5963 -> 0.5933, accuracy down), so the strength is not explained by the 3x3 windows alone (open: self-play volume, longer think time of the Python-wrapped PyJataxx build).

## 6. Rules of evidence used throughout
Paired, seat-balanced games from a paired opening book, at least 100 games per leg; promote only at 95% probability of being better; the played time control (100 ms, 1000 ms, near the 4 s Botzone clock) decides anything touching speed or search, fixed-node results are diagnosis; every net is content-hashed and immutable; every gate is replayable from a commit.

## Sources in this repository
`LEDGER.md` (experiment ledger), `docs/RESEARCH_*.md` (per-topic write-ups), `docs/TSTG_*.md` (TSTG experiment and post-mortem), `docs/GATE_RUNS.tsv`, `docs/TSTG_MODELS.tsv`.
Raw game logs and datasets are local and not in git.

