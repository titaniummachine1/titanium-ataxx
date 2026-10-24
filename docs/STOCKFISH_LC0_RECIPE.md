# Stockfish NNUE recipe: Stockfish data first, Lc0 data second

Source: official-stockfish/nnue-pytorch wiki (Training datasets, Basic training
procedure, easy_train.py) and the Stockfish release notes.

## The recipe

1. Generate a base dataset with Stockfish selfplay:
   - `generate_training_data depth 9 count 18000000000 random_multi_pv 4
     random_multi_pv_diff 100 set_recommended_uci_options data_format binpack`
   - opening book: `noob_3moves.epd`
   - target: search score + game result, `lambda` mixes them
     (`lambda=1` = score only, `lambda=0` = outcome only).
2. Train a network from scratch on that Stockfish-generated base data.
3. Get Lc0 training data (Leela selfplay / Lc0 evaluation data) and convert or
   rescore it into the NNUE `.plain` / `.binpack` format.
4. Fine-tune the already-trained network on the Lc0-derived data.
   - The wiki states plainly: high-quality Lc0 data works better when used to
     retrain a network already trained on lower-quality Stockfish data than
     when training from Lc0 data alone.
   - Training solely on Lc0 data does not produce as good a net, likely because
     of position coverage and because the concepts are too hard from scratch.
5. For Lc0 data, optionally filter rows where the evaluation does not
   correlate with the game result.
6. Modern Stockfish keeps using Lc0 game databases; the release notes mention
   training on 100B+ Lc0 evaluation positions and Lc0 games contributing Elo.

## Titanium mapping

- Base stage (Stockfish-equivalent selfplay):
  - own cache positions (`titanium_cache12m`) with Titanium search labels
    (5k nodes) + game outcome.
  - This is stage 1: train from scratch / from the current net on our own
    positions first.
- Teacher stage (Lc0-equivalent):
  - KataGo Ataxx 202604 positions (21.8M) with their value/outcome labels.
  - existing deep labels: `lab2m` 20k nodes, `golden.npz` 500k nodes,
    `deep_labels.json` 100k nodes.
  - Ataxx_CPP MCTS/teacher labels if/when generated.
  - This is stage 2: fine-tune the stage-1 net on the teacher data; never
    train a fresh net from teacher data only.
- Per-position columns for the trainer:
  - `score_cp` = our search eval or deep search eval,
  - `outcome` = game result,
  - `lambda` / outcome weight controls the mix,
  - `source` = `base` or `teacher` so the two stages are exported separately.

## Current Titanium pipeline status

The running pipeline currently blends engine + outcome into one `target_cp`
for all sources. That is **not** the Stockfish recipe. The recipe needs:

- `stage1_base.npz`: own-cache positions, Titanium 5k + outcome.
- `stage2_teacher.npz`: KataGo/deep positions, deep/teacher + outcome.
- Stage 1 trains first, stage 2 fine-tunes that checkpoint.
