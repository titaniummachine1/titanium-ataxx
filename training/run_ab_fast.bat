@echo off
REM ============================================================================
REM FAST A/B: one-pass teacher vs Stockfish-style two-stage. 1 epoch per stage.
REM Same data/sidecars/seed/optimizer/mix. Only target column + init differ.
REM val_mse NOT comparable across arms (D39) -> gate by games vs champion.
REM Est: ~105 min (3 runs). Every stage writes its own .tup; nothing deleted.
REM ============================================================================
set ROOT=C:\gitProjects\titanium-ataxx
cd /d %ROOT%
set CACHE=data\nnue\final\final_train.npz
set SC=data\nnue\final\sidecar
set COMMON=--cache %CACHE% --sidecar-dir %SC% --dirs 0,1,2,3 --chunk-rows 800000 --outcome-col outcome_p --outcome-weight 0.25 --wdl-loss --wdl-scale 400 --opt adam --opt-lr 0.001 --conv-lr 0.003 --wd 0 --freq-decay 0.0 --seed 7 --threads 4 --resume

echo === ARM A: one-pass teacher, 1 epoch ===
python training\train_stream.py %COMMON% ^
  --tcol teacher --teacher-prob-col teacher_p ^
  --init data\nnue\tupS55tree.tup ^
  --epochs 1 --max-epochs 1 --plateau 0 ^
  --ckpt data\nnue\abA_ep{ep}.tup --out data\nnue\abA_onepass.tup ^
  > logs\abA_onepass.txt 2>&1

echo === ARM B step 1: base (Titanium 5k labels), 1 epoch ===
python training\train_stream.py %COMMON% ^
  --tcol titanium_cp ^
  --init data\nnue\tupS55tree.tup ^
  --epochs 1 --max-epochs 1 --plateau 0 ^
  --ckpt data\nnue\abB1_ep{ep}.tup --out data\nnue\abB_stage1.tup ^
  > logs\abB_stage1.txt 2>&1

echo === ARM B step 2: teacher fine-tune, 1 epoch ===
python training\train_stream.py %COMMON% ^
  --tcol teacher --teacher-prob-col teacher_p ^
  --init data\nnue\abB_stage1.tup ^
  --epochs 1 --max-epochs 1 --plateau 0 ^
  --ckpt data\nnue\abB2_ep{ep}.tup --out data\nnue\abB_twostage.tup ^
  > logs\abB_stage2.txt 2>&1

echo FAST A/B DONE.
