#!/bin/sh
# D1067 (user idea): net-only strength under a PLAIN alpha-beta (lab eval-gate). Waits for the D1066 ti gate to end
# (one job per machine), then: new net nn_v11bsat F8BD53F0 (A) vs champion 812C2118 (B), 400 pairs each:
# (1) 5,000 nodes/move (user's setting), (2) fixed depth 3 (pure: only the eval decides).
LAB=/c/gitProjects/titaniumtest/lab/target_ext/x86_64-pc-windows-gnu/release/lab.exe
A=C:/gitProjects/titanium-ataxx/data/nnue/proxy/nn_v11bsat_nocal.tnn
B=C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn
L=/c/gitProjects/titanium-ataxx/logs/evalgate
while tasklist //FI "IMAGENAME eq ti.exe" | grep -qi "ti.exe"; do sleep 30; done
echo "start 5k $(date +%H:%M)"
"$LAB" eval-gate --a $A --b $B --nodes 5000 --pairs 400 --threads 6 > $L/d1067_v11bsat_vs_812C_5k.log 2>&1
tail -1 $L/d1067_v11bsat_vs_812C_5k.log
echo "start depth3 $(date +%H:%M)"
"$LAB" eval-gate --a $A --b $B --depth 3 --pairs 400 --threads 6 > $L/d1067_v11bsat_vs_812C_d3.log 2>&1
tail -1 $L/d1067_v11bsat_vs_812C_d3.log
echo "done $(date +%H:%M)"
