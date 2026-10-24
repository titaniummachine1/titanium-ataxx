@echo off
REM FINAL WDL training launch (sidecar-fed). DO NOT RUN without explicit user order.
REM Requires: dataset audit ok=true (data/nnue/final/final_audit.json) + user says "train".
REM Kill-test protocol (execute only as part of an ordered run):
REM   1. Launch this bat. 2. Wait until *.resume_pt manifest shows 1 chunk banked.
REM   3. Kill the python process. 4. Relaunch -> must print "resumed weights at ep 1
REM      (1/N chunks already banked)". 5. A kill may never cost more than 1 chunk.
cd /d C:\gitProjects\titanium-ataxx
python training\train_stream.py ^
  --cache data\nnue\final\final_train.npz ^
  --sidecar-dir data\nnue\final\sidecar ^
  --wdl-loss --wdl-scale 400 ^
  --outcome-col outcome_p --teacher-prob-col teacher_p ^
  --tcol teacher --blend 1.0 --outcome-weight 0.25 ^
  --dirs 0,1,2,3 --chunk-rows 800000 ^
  --opt adam --opt-lr 0.001 --conv-lr 0.003 --wd 0 --freq-decay 0.0 ^
  --init data\nnue\tupS55tree.tup ^
  --epochs 4 --max-epochs 8 --plateau 1e-4 ^
  --seed 7 --threads 4 --resume ^
  --ckpt data\nnue\final_wdl_ep{ep}.tup ^
  --out data\nnue\final_wdl.tup ^
  > logs\train_final_wdl.txt 2>&1
