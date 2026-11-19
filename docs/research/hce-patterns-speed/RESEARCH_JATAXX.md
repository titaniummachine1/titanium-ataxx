# RESEARCH: Jataxx (jacek's open-source Botzone Ataxx bot) - 2026-11-09

Owner jacek (`/account/65b47f9f13e80a09c4de957f`, level 29). Same author runs **PyJataxx** (Botzone Ataxx rank 6, 2166.69 on 2026-11-09, py36, LOCKED, v1, desc "kupa").
**Jataxx** (bot id `65fd2f68dc83c705baeba297`, cpp17, v37, desc "Jataxx test") is marked OPEN SOURCE, so its code is readable by anyone.
Full study folder (outside the repo): `C:\gitProjects\jataxx-study\` - `JATAXX.md` (architecture + 38-version table), `versions/vNN_<id8>.cpp` (all 38 sources), `jataxx_all_versions.txt` (combined, downloaded from the Botzone page with the user's OK on 2026-11-09), `raw/` (Botzone page JS used to find the endpoints).
Endpoints: `GET /mybots/detail/<botid>` (version list, ObjectId timestamps), `GET /mybots/viewsrc/<botid>/<versionIdx>` (source JSON). v37 SHA256 `6184BFDB7EEF3F54E2E4CFB51137DA1A7CC4B25F5550BE04D00A2F12364F7AFD` (identical to the site copy).

## What the source says (v36/v37, verified by reading; NOT run)
- Board: 7x7 u64 bitboard, bit = 7*x+y. Clone/jump movegen with floodfill masks. Game over at 400 plies / wipe-out / full / either side stuck; if one side is stuck the other gets all empties.
- **Eval = 25 overlapping 3x3 windows** (3^9 = 19683 states each, side-to-move relative, PEXT-extracted) = 492,075 sparse inputs, 25 active per position -> **128 hidden, squared-ReLU**, **3 output buckets by piece count (<10, <20, else)**, ONE tanh scalar output. int16 weights (scale 8192), int32 accumulators, AVX2. Per child only the changed windows are subtracted/added (incremental diff against the parent's accumulator). Weights ~126 MB.
- **Search = best-first minimax hybrid ("MCTS" by name)**: tree of move-edges in a 12M ring pool; all children are NN-evaluated on expansion; backup `heuristic = -max(child heuristic)` (negamax of NN values) plus exact terminal proof (win score `INF-10*rounds`); selection `alpha*heuristic + (1-alpha)*mean_result + C*sqrt(ln N/n)` with alpha 0.53, C 0.25 (0.14 root), FPU 0.19, 10% noise; final move = max `heuristic + log(games+3)`; subtree reuse across turns; ~975 ms/move.
- Weights file `data/128_RL541` is in jacek's PRIVATE Botzone storage (open source = code only). No link/host for it anywhere in the 38 sources. Not obtainable; the bot cannot be run locally. Trainer is not in v36/v37; v0/v1 contain an early `learn`/`learnRL` + softmax self-play sampling. File names: `<hidden>_RL<n>` = RL self-play generation n (RL20 -> RL541), `_SL<n>` supervised, `i` = int16 export.

## Progression (condensed; full table in jataxx-study/JATAXX.md)
2024-03-22..04-18 (v0-16): per-cell inputs 98 -> 245 -> 343 -> 2x2 windows (36x81 = 2916); hidden 128 -> 384 -> 768(dropped) -> 256; 50 ms -> 950 ms; leaky -> squared ReLU (v14).
21-month gap. 2026-01-16..21 (v17-21): 2x3 windows (30x729), hidden 192/224. 2026-01-25..02-02 (v22-34): **3x3 windows (25x19683)**, hidden 64, int16, short 64->16 second layer (v25), then 96 hidden + 3 piece-count buckets (v33), constants retuned once (v34: FPU .19 C .25 alpha .53). 2026-02-17..23 (v35-37): incremental accumulator rewrite, parent pointers removed, 128 hidden `RL541`, 975 ms. Exact revert at v18 (= v16 hash) when the 2x3 step lost.
RL generations: RL20, 79, 195, 30(?), 242, 94, 122, 170, 190, 205, 184, 253, 541 across feature sets - the generation count only grows within a feature set.

## What this changes in our KEY_FACTS (corrections, D611)
1. "PyJataxx predicts the final stone distribution (unverified)": NOT supported for Jataxx v37 - ONE tanh value + 3 piece-count buckets, no distribution head. PyJataxx source is still unknown (locked).
2. "UBFM-like": confirmed in spirit for Jataxx (negamax of NN values over a best-first tree), i.e. an NN-value best-first minimax, not rollout MCTS.
3. "Its eval sees the outcome 15-20 plies earlier (D210)": consistent with an eval trained on game outcomes through hundreds of self-play generations (hypothesis; the trainer is not public).

## Comparison with the CURRENT champion (read from KEY_FACTS/HANDOFFS/LEDGER, not the stale Oct-5 repo)
| | Jataxx v37 | Titanium main 285a127 |
|---|---|---|
| search | NN best-first minimax, ~1 s | alpha-beta + NNUE, ~110-148 knps |
| eval rows | 25 active (3x3 windows) | 208 rows, ~5 us/eval (W256) |
| training signal | RL self-play outcomes, 541 generations (inferred) | Titanium 5k-node search labels (31.57M) + result blend 0.55/0.55/0.25 by empties; Kata teacher for policy |
| output buckets | 3 (piece count) | 16 NNUE buckets |
Related findings we already have: D459/D474 compact 3x3 beats lines/2x2; k3s (one 3x3 per stone, 58 rows) = 65-69% sibling agreement but **-147 Elo at 100 ms** (D481): eval quality dominates speed; D318/D324 more game-result weight helps up to 0.55, 0.7 = -20; GATE-sat-w512 (LEDGER) = overfit/distribution collapse on the self-play manifold, "mandates external game collection".
Reading: Jataxx's strength is not the 3x3 rows themselves (we tested those) but a large volume of outcome-labelled self-play with the same net in the loop. Hypothesis only.

## Ideas (NOT started; hub/Rust only per A7, budgets per A9, decide with the user)
1. **External game data** from Botzone (`ti botzone`, D302): games of PyJataxx / KataAtaxx_mini / Moonbird vs others are outcome-labelled and off our self-play manifold. (Botzone also offers a public "Download match dataset of Botzone" link in the page footer; not downloaded; needs the user's OK.)
2. Outcome-weighted self-play loop for the champion net (play at 5k nodes, outcome-dominant labels, fresh data each generation) - compare with the 0.55 result-weight finding first.
3. Piece-count-bucket / incremental-accumulator ideas: we already have 16 buckets and incremental accumulators; nothing to copy.

## TT / eval cache / memory lessons and Botzone limits (D612, 2026-11-09)
From the Jataxx source:
- **No hash TT and no eval cache.** The tree itself is the memory: a ring pool of move-edges (`MoveMctsTTR`, ~24 B each by struct layout, estimated not measured). Transpositions are not merged; a repetition hash set exists only as commented-out code. Eval cost is cut by diffing against the parent's accumulator (`cacheScore` per expanded node), not by caching evals per position. Eviction = ring overwrite plus a cursor reset every 20th turn (`wtf%20==0 -> ccc=0`). So there is no TT/cache recipe to copy for our alpha-beta; his design is a different machine.
- **He tunes memory by the byte.** Node fields carry byte-count comments; v32 made `childrenSize` int16 and `player` int8, v35 removed `index`, v36 removed `parent`; pool size moved in 0.5M steps (v23-v37: 10.5M, 11.0M, 11.5M, 12.5M, 12.0M). Estimated footprint at v37: pool 12.0M x 24 B = 288 MB + net 126 MB (int16 25*19683*128) = ~414 MB. A C++ bot at roughly that size runs fine on Botzone, consistent with our D573 (limit >= 512 MB, Titanium ~450).
From the 5 saved Botzone matches (data/botzone/*.json, every verdict OK):
- `memory` peaks: PyJataxx 512, sotax 512, RawNN 512 (three different bots read exactly 512 and were NOT killed: the reading looks capped, the true limit is not in the logs), Titanium 436-457, v1_flash 348, hce 349.
- Time per move: PyJataxx avg 3426 ms / max 4167; Titanium avg 3.5-3.7 s, max 8.3-11.9 s (first turn / Python start) all OK; C++ bots sotax 1.2 s, hce 1.0 s, RawNN 1.3 s; Jataxx itself plays 975 ms.
- Botzone page constants (`Botzone.consts` on jacek's profile page): `lang_timeout_factor` c/cpp/cpp17 [1,1], java [3,2], cs [6,4], js [2,2], **py/py36 [6,4]**, pas [1,1], rs [1,1]; `max_upload_size` 4 (MB). Reading (NOT checked against Botzone docs): [first-turn, later-turn] multipliers, so a Python-submitted bot gets about 4x the C++ per-move time (matches ~4 s for PyJataxx/Titanium vs 1 s for C++).
- **PyJataxx identity (new, D612):** its debug JSON is `{"ccc","expansions","h","maxLevel"}` (e.g. 149,744 expansions, maxLevel 17 on move 1; avg ~95k expansions/move, maxLevel up to 308, pool cursor up to 6.7M in one game). Those are Jataxx's field names; only v19-v21 printed `ccc`, v37 prints h/expansions/maxLevel. So PyJataxx is a Jataxx-lineage engine (probably a private newer build, e.g. his locked "Test" bot v95) started from a Python wrapper, thinking ~3.4 s per move versus 975 ms for the open C++ Jataxx. Same trick as our Python driver + Rust engine. Hypothesis (not checked): the time multiplier is part of why PyJataxx ranks above what a 1 s C++ bot would.

## Not analysed / limits
PyJataxx (locked). Versions v0-v36 were compared only through extracted parameters and per-version line diffs, not run or tested. The trainer and the weights are unavailable, so "RL self-play on outcomes" is inferred from file names and v0/v1 code. Nothing was gated.

## Side experiment (obsolete): `C:\gitProjects\titanium-rl3x3`
Built on the STALE Oct-5 `titanium-main` worktree (tuple eval, not the NNUE champion). Pure-Rust prototype: 3x3-window residual net on top of the classical eval, outcome-grounded logistic training (`rlgen` / `rltrain`, env `TITANIUM_NET3`). Compiles and smoke-tests (trainer works; a position-level val split leaked and was fixed to a game-level split). CORRECTION (2026-11-09 04:55): an earlier version of this paragraph wrongly said the generation was interrupted; it finished: 8000 games at 5k nodes with the stale classical engine (black 4248 / white 3649 / draw 103), 519,907 positions in `data/rl/it0.rec`. One offline test: logistic fit of the game outcome on (classical eval + 100 x net) with a GAME-level 5% validation split (25k positions): base-only val BCE 0.6537; best net val BCE 0.6498 (hidden 32, lr 0.003, epoch 1; lr 0.001 -> 0.6505; lr 0.0003 -> 0.6521); accuracy NOT improved (57.9% base vs 57.8-58.0%); lr 0.02 memorises (train 0.39, val 0.91). Meaning: 520k positions from 8000 games are too few for a 492k-input net, the gain is within noise, no game gate was run. This says nothing about the current NNUE champion.

## Addendum 2026-11-09 (side experiment, NOT a gate, NOT hub): facts read from data/tstg_v10_31m.bin
First 1,000,000 rows: outcome_wdl loss/draw/win = 382,239 / 117,526 / 500,235 (populated, so the byte is real); mean depth 9.9; corr(score_cp, outcome) 0.851; corr(score_cp, stones us-them) 0.214. Record = 40 B: occ_us (side to move) u64, occ_them u64, blockers u64, turn, from, to, flags, score_cp i16, outcome_wdl u8, depth u8. A throwaway 1-epoch run of a 3x3-window residual net (hidden 32) on those rows moved held-out BCE 0.1834 -> 0.1813 (50k tail rows, 1M-row slice): within noise, no conclusion. These runs used a standalone Rust trainer in C:/gitProjects/titanium-rl3x3, outside the hub (breaks LAWS A7 spirit: hub first); treat as scratch. Nothing running.

## Addendum 2026-11-09 05:30: pat3 screen on train_v9 (D614; scratch analysis, not hub, not a gate)
25-window 3x3 net (hidden 32, 3 piece-count buckets, Adagrad, 3 epochs) vs the champion-labelled 5k search score cp, target = game outcome, 28,990,531 train rows / 601,957 val rows (val_v9.tdat):
| model | val BCE | val accuracy |
|---|---|---|
| cp alone (S=300) | 0.5963 | 73.16% |
| cp + 3x3 residual, lr 0.003 | 0.5933 | 71.63% |
| cp + 3x3 residual, lr 0.001 | 0.5942 | 72.55% |
| 3x3 net alone (no cp) | 0.6365 | 64.35% |
Reading: no usable outcome information from 3x3 windows beyond the search score on this data; alone the small net is ~9 accuracy points below cp. So the 3x3 features are not what makes Jataxx strong (consistent with D611: outcome-loop volume and the 3.5x time multiplier are the open explanations). NOT tested: hidden >32, regularisation/tuning, RL with the net in the loop, game-level val split verification, more epochs. Logs logs/pat3_screen/R1-R3.log.
