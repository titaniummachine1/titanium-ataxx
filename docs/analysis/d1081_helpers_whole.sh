#!/bin/sh
# D1081 (user D1076/D1077/D1079: sync EVERY helper to net F8BD53F0 on the WHOLE dataset, then gate the bundle).
# Runs after the D1075 gate script prints "done" (one long job per machine). Steps (sequential, each logs its own file):
#  1. dis head: ti train export of the trained net (A2FA1B4C) with the dis head refit on ALL train_v11b rows (--rows 6e7 >
#     52M = every block); value trunk unchanged (re-export is deterministic); nocal recipe as before (D318).
#  2. policy: ti policy2 init 380F609C (SEE block dropped = play parity, D893), labels = engine BEST move of every
#     position of all main self-play games (286 .tgam + .tbst: v10 + v11 + pc_run8-10, 0 duplicates, all with labels)
#     + all 3 Kata shards; stop on games + Kata val (sum), saturation (patience 4).
#  3. SmallNet: ti train smallnet on ALL 52M train_v11b rows, every 24th child (~96M samples, ~3.1 GB; every 3rd = ~765M
#     = 24.5 GB does not fit 16 GB RAM), init = the D1074 refit 46117B2A, saturation.
TI=C:/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
GATE_OUT="C:/Users/TERMIN~1/AppData/Local/Temp/claude/eff9487f3b6f3868/d4c2cbcc-d66a-4bf0-9abe-1f2fe101f565/tasks/blzhytww5.output"
cd /c/gitProjects/titanium-ataxx || exit 1
until grep -q "^done" "$GATE_OUT" 2>/dev/null && ! tasklist | grep -qi "ti.exe"; do sleep 60; done
echo "start dis $(date +%H:%M)"
OUTN=data/nnue/proxy/nn_v11bsat_nocal_disall.tnn
$TI train export --net runs/nn_v11bsat/nn_v11bsat/best.tnn --data data/nnue/packed/train_v11b --val-dir data/datasets/v11/val \
  --rows 60000000 --out $OUTN > logs/d1081_dis_export.log 2>&1
if [ $? -ne 0 ]; then
  echo "export retry with the run's result-weight target $(date +%H:%M)"
  $TI train export --net runs/nn_v11bsat/nn_v11bsat/best.tnn --data data/nnue/packed/train_v11b --val-dir data/datasets/v11/val \
    --rows 60000000 --result-weight-phase 0.55,0.55,0.25 --out $OUTN > logs/d1081_dis_export_rw.log 2>&1
fi
echo "dis exit $? $(date +%H:%M)"
echo "start policy $(date +%H:%M)"
GAMES=$(paste -sd, logs/policy/d1081_games_list.txt)
$TI policy2 --net data/nnue/proxy/nn_v11bsat_nocal.tnn --games "$GAMES" --kata data/kata_raw/s3413,data/kata_raw/s1835,data/kata_raw/s1914 \
  --pat k3c --init data/nnue/policy_store/pol_380F609C.bin --kata-w 1 --games-w 1 --threads 7 --stop-on sum \
  --out-bin nets/pol_v11all_d1081.bin > logs/policy/d1081_pol_v11all.log 2>&1
echo "policy exit $? $(date +%H:%M)"
echo "start smallnet $(date +%H:%M)"
$TI train smallnet --net data/nnue/proxy/nn_v11bsat_nocal.tnn --train data/nnue/packed/train_v11b --val data/datasets/v11/val/val_v11_clean \
  --init data/nnue/proxy/fut_v11bsat_d1069.tsn --rows 60000000 --child-every 24 --tau 0.02 --lr 0.002 --threads 7 \
  --out data/nnue/proxy/fut_v11ball_d1081.tsn > logs/smallnet/d1081_sn_v11ball.log 2>&1
echo "smallnet exit $? $(date +%H:%M)"
echo "done $(date +%H:%M)"
