@echo off
REM v2 WDL training: Ataxx-relabeled teacher (corr .49 vs outcome) on the
REM KataGo-position rows; same sidecars (row order preserved), warm start = champ.
REM DO NOT RUN without explicit user order after audit ok=true.
REM Kill-test: launch -> wait for 1 banked chunk in the .resume manifest ->
REM kill python -> relaunch must print "resumed weights"; max cost = 1 chunk.
cd /d C:\gitProjects\titanium-ataxx
python training\train_stream.py ^
  --cache data\nnue\final\final_train_v2.npz ^
  --sidecar-dir data\nnue\final\sidecar ^
  --wdl-loss --wdl-scale 400 ^
  --outcome-col outcome_p --teacher-prob-col teacher_p ^
  --tcol teacher --blend 1.0 --outcome-weight 0.25 ^
  --dirs 0,1,2,3 --chunk-rows 800000 ^
  --opt adam --opt-lr 0.001 --conv-lr 0.003 --wd 0 --freq-decay 0.0 ^
  --init data\nnue\tupS56twostage.tup ^
  --epochs 20 --max-epochs 20 --plateau 1e-4 ^
  --seed 7 --threads 4 --min-free-mb 500 --resume ^
  --ckpt data\nnue\final_v2_wdl_ep{ep}.tup ^
  --out data\nnue\final_v2_wdl.tup ^
  > logs\train_v2_wdl.txt 2>&1
