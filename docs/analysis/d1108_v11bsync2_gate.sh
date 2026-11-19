#!/bin/sh
# D1108: gate exp/v11b-sync2 6668d59 (net BB9CECE8 + policy 8AA2963C + SmallNet 46117B2A) vs main 0120fd12.
# Leg 1: 5,000 nodes (diagnosis), leg 2: 100 ms (decides, B9), leg 3: 1000 ms only when leg 2 is ahead (Elo > 0).
# Botzone rules, --bz --tt-bits 22 both sides. Each leg ends with ti insight + ti analyze (LAWS B12).
TI=C:/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
L=C:/gitProjects/logs
C=6668d59
R=/c/gitProjects/titanium-ataxx/logs/gates/reports
cd /c/gitProjects/v11bsync2-ataxx || exit 1
report() {
  G=$(ls -td /c/gitProjects/v11bsync2-ataxx/logs/gates/*${C}*-$1 2>/dev/null | head -1)
  [ -n "$G" ] || return
  J=$(ls $G/*.jsonl 2>/dev/null | head -1)
  $TI insight --games "$J" --by-bucket > $R/d1108_v11bsync2_$1_insight.txt 2>&1
  $TI analyze --games "$J" > $R/d1108_v11bsync2_$1_analyze.txt 2>&1
}
echo "start 5k $(date +%H:%M)"
$TI gate --test $C --base main --kind eval --rules botzone --nodes 5000 --max-games 800 --elo1 10 --threads 6 \
  --test-args "--bz --tt-bits 22" --base-args "--bz --tt-bits 22" > $L/gate_d1108_v11bsync2_5k.log 2>&1
echo "5k exit $? $(date +%H:%M)"; report nodes5000
echo "start 100ms $(date +%H:%M)"
$TI gate --test $C --base main --kind search --rules botzone --time-ms 100 --max-games 400 --elo1 10 --threads 4 \
  --test-args "--bz --tt-bits 22" --base-args "--bz --tt-bits 22" > $L/gate_d1108_v11bsync2_100ms.log 2>&1
echo "100ms exit $? $(date +%H:%M)"; report ms100
E=$(grep -o 'Elo [-+][0-9.]*' $L/gate_d1108_v11bsync2_100ms.log | tail -1 | awk '{print $2}')
echo "100ms Elo $E"
case "$E" in
  +*) echo "start 1000ms $(date +%H:%M)"
      $TI gate --test $C --base main --kind search --rules botzone --time-ms 1000 --max-games 200 --elo1 10 --threads 4 \
        --test-args "--bz --tt-bits 22" --base-args "--bz --tt-bits 22" > $L/gate_d1108_v11bsync2_1000ms.log 2>&1
      echo "1000ms exit $? $(date +%H:%M)"; report ms1000 ;;
  *) echo "1000ms skipped (100 ms not ahead)" ;;
esac
echo "done $(date +%H:%M)"
