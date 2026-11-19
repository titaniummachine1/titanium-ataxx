#!/bin/sh
# D1075: gate exp/v11b-sync 03d4738 (net F8BD53F0 + SmallNet 46117B2A refit to it, policy 380F609C) vs main 0120fd12.
# Leg 1: 5,000 nodes (diagnosis, eval), leg 2: 100 ms (decides, B9). Botzone rules, --bz --tt-bits 22 both sides.
# Each leg ends with ti insight + ti analyze (LAWS B12) into logs/gates/reports/.
TI=C:/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
L=C:/gitProjects/logs
cd /c/gitProjects/v11bsync-ataxx || exit 1
echo "start 5k $(date +%H:%M)"
$TI gate --test 03d4738 --base main --kind eval --rules botzone --nodes 5000 --max-games 800 --elo1 10 --threads 6 \
  --test-args "--bz --tt-bits 22" --base-args "--bz --tt-bits 22" > $L/gate_d1075_v11bsync_5k.log 2>&1
echo "5k exit $? $(date +%H:%M)"
echo "start 100ms $(date +%H:%M)"
$TI gate --test 03d4738 --base main --kind search --rules botzone --time-ms 100 --max-games 400 --elo1 10 --threads 4 \
  --test-args "--bz --tt-bits 22" --base-args "--bz --tt-bits 22" > $L/gate_d1075_v11bsync_100ms.log 2>&1
echo "100ms exit $? $(date +%H:%M)"
R=/c/gitProjects/titanium-ataxx/logs/gates/reports
for leg in nodes5000 ms100; do
  G=$(ls -td /c/gitProjects/v11bsync-ataxx/logs/gates/*03d4738*-$leg 2>/dev/null | head -1)
  [ -n "$G" ] || continue
  J=$(ls $G/*.jsonl 2>/dev/null | head -1)
  $TI insight --games "$J" --by-bucket > $R/d1075_v11bsync_${leg}_insight.txt 2>&1
  $TI analyze --games "$J" > $R/d1075_v11bsync_${leg}_analyze.txt 2>&1
done
echo "done $(date +%H:%M)"
