#!/bin/sh
# D1069 (user: sync every helper to the new big net): refit the futility SmallNet on nn_v11bsat F8BD53F0, same recipe as
# D904 (FA1FE34A), data train_v11b / val_v11_clean, init = the shipped FA1FE34A (C14), saturation (--patience default).
# Waits for the D1067 eval-gate runs (lab.exe) so only one job runs.
cd /c/gitProjects/titanium-ataxx
until grep -q "^FINAL" logs/evalgate/d1067_v11bsat_vs_812C_d3.log 2>/dev/null; do sleep 30; done
while tasklist //FI "IMAGENAME eq lab.exe" | grep -qi "lab.exe"; do sleep 30; done
echo "smallnet start $(date +%H:%M)"
/c/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe train smallnet \
  --net C:/gitProjects/titanium-ataxx/data/nnue/proxy/nn_v11bsat_nocal.tnn \
  --train data/nnue/packed/train_v11b --val data/datasets/v11/val/val_v11_clean \
  --init C:/gitProjects/mainpromo-ataxx/data/nnue/fut_small.tsn --rows 2400000 --tau 0.02 --lr 0.002 --threads 7 \
  --out data/nnue/proxy/fut_v11bsat_d1069.tsn > logs/smallnet/d1069_sn_v11bsat.log 2>&1
echo "exit $? $(date +%H:%M)"
grep -E "BEFORE|AFTER|saturat|parity|wrote|error" logs/smallnet/d1069_sn_v11bsat.log | tail -6
