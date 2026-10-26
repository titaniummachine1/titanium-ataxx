#!/bin/bash
# Stage 1 (base) retrain on the 2026-10-25 relabel (final_*_v4.npz), seeded with main's net (nnue_teacher_v1_ep4_ema.pt
# = c4829350), trained to SATURATION: per-epoch val-driven auto-lr with warm restarts (D69) + per-step hypergradient LR
# (D72), block-mixed chunks (--block 1024). The trainer exits after its RAM guard saved a resume file; this loop restarts
# it with the SAME command (it resumes) until the log says it finished, saturated or crashed. Stage 2 is NOT started here.
# Usage: bash training/run_base_v4s.sh   (log: logs/train_nnue_base_v4s.txt, runner log: logs/run_base_v4s.txt)
cd /c/gitProjects/titanium-ataxx
LOG=logs/train_nnue_base_v4s.txt; RL=logs/run_base_v4s.txt
CMD="python -I -u training/train_nnue.py --stage base --init data/nnue/nnue_teacher_v1_ep4_ema.pt --out data/nnue/nnue_base_v4s
 --epochs 60 --auto-lr --auto-restarts 2 --min-rel 0.002 --metric mse_target --lr 4e-3 --lr-min 2e-5 --hypergrad
 --bs 4096 --width 256 --opt ranger --wd 3e-7 --la-k 8 --la-alpha 0.56 --ema 0.9995 --block 1024 --threads 6 --min-free-gb 1.5"
for attempt in $(seq 1 20); do
  echo "$(date '+%F %H:%M') start attempt $attempt" >> $RL
  $CMD >> $LOG 2>&1
  rc=$?
  last=$(tail -n 5 $LOG)
  echo "$(date '+%F %H:%M') exit rc=$rc" >> $RL
  if echo "$last" | grep -q "RAM GUARD"; then echo "  RAM guard - waiting 3 min, then resume" >> $RL; sleep 180; continue; fi
  if echo "$last" | grep -q "^done ->"; then echo "  finished" >> $RL; break; fi
  echo "  stopped for another reason (crash or saturation message) - not restarting; see $LOG" >> $RL; break
done
