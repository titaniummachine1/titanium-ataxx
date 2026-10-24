@echo off
REM ============================================================================
REM A/B: one-pass teacher training vs Stockfish-style two-stage (base -> teacher)
REM Same total epochs (4), same data, same sidecars, same seed/optimizer/mix.
REM val_mse is NOT comparable across arms (D39) -> winner is decided by GAMES.
REM DO NOT RUN without explicit user order. Est: ~4h training (arms 1-3).
REM ============================================================================
set ROOT=C:\gitProjects\titanium-ataxx
set CACHE=data\nnue\final\final_train.npz
set SC=data\nnue\final\sidecar
set COMMON=--cache %CACHE% --sidecar-dir %SC% --dirs 0,1,2,3 --chunk-rows 800000 --outcome-col outcome_p --outcome-weight 0.25 --wdl-loss --wdl-scale 400 --opt adam --opt-lr 0.001 --conv-lr 0.003 --wd 0 --freq-decay 0.0 --seed 7 --threads 4 --resume
cd /d %ROOT%

REM --- ARM 1: one-pass teacher target, 4 epochs -------------------------------
REM needs --teacher-prob-col teacher_p (calibrated probabilities)
python training\train_stream.py %COMMON% ^
  --teacher-prob-col teacher_p --tcol teacher ^
  --init data\nnue\tupS55tree.tup ^
  --epochs 4 --max-epochs 4 --plateau 0 ^
  --ckpt data\nnue\ab1_ep{ep}.tup --out data\nnue\ab1_onepass.tup ^
  > logs\ab1_onepass.txt 2>&1

REM --- ARM 2, STEP 1: base stage on Titanium 5k labels, 2 epochs --------------
REM no --teacher-prob-col: target is sigmoid(clip(titanium_cp)/400) + 0.25 outcome
python training\train_stream.py %COMMON% ^
  --tcol titanium_cp ^
  --init data\nnue\tupS55tree.tup ^
  --epochs 2 --max-epochs 2 --plateau 0 ^
  --ckpt data\nnue\ab2s1_ep{ep}.tup --out data\nnue\ab2_stage1.tup ^
  > logs\ab2_stage1.txt 2>&1

REM --- ARM 2, STEP 2: fine-tune on teacher target, 2 epochs -------------------
python training\train_stream.py %COMMON% ^
  --teacher-prob-col teacher_p --tcol teacher ^
  --init data\nnue\ab2_stage1.tup ^
  --epochs 2 --max-epochs 2 --plateau 0 ^
  --ckpt data\nnue\ab2s2_ep{ep}.tup --out data\nnue\ab2_twostage.tup ^
  > logs\ab2_stage2.txt 2>&1

echo ARMS DONE. Gate ab1_onepass.tup vs ab2_twostage.tup (paired book, 5k nodes, 100g+).
