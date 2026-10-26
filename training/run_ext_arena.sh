#!/bin/bash
# D103: main (exe + default NNUE e11827d1) vs Ataxx_CPP, Moonbird 1.1.0, KataAtaxx (CPU, 1 thread) - Botzone end rules,
# paired book (100 games = 50 lines x both seats), 4 games in parallel (8 threads: one per engine). Node legs first
# (load-independent), timed legs only when no other gate runs (fair clock).
cd /c/gitProjects/titanium-ataxx
M="C:/gitProjects/titanium-ataxx/scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe"
A="python -I training/ax_vs_ti.py --games 100 --workers 4"
run() { echo "$(date '+%F %H:%M') start $1" >> logs/arena_ext_runner.txt; shift; "$@" >> logs/arena_ext_runner.txt 2>&1; echo "$(date '+%F %H:%M') done" >> logs/arena_ext_runner.txt; }
run "cpp 5k evals"  $A --opp cpp --ti-nodes 5000 --cpp-evals 5000 --book-start 8000 --out logs/arena_cpp_5k.txt
run "moon 5k nodes" $A --opp uai --opp-cmd "$M" --opp-go "go nodes 5000" --ti-nodes 5000 --book-start 8000 --out logs/arena_moon_5k.txt
# wait until no fixed-node gate (gate_net_vs_main / match shards) is running before any timed leg
while tasklist //FI "IMAGENAME eq titanium-cli.exe" 2>/dev/null | grep -q titanium-cli; do sleep 60; done
run "cpp 100ms"     $A --opp cpp --ti-ms 100 --cpp-ms 100 --book-start 8100 --out logs/arena_cpp_100ms.txt
run "moon 100ms"    $A --opp uai --opp-cmd "$M" --opp-go "go movetime 100" --ti-ms 100 --book-start 8100 --out logs/arena_moon_100ms.txt
run "kata 100ms"    $A --opp kata --kata-ms 100 --ti-ms 100 --book-start 8100 --out logs/arena_kata_100ms.txt
run "cpp 1s"        $A --opp cpp --ti-ms 1000 --cpp-ms 1000 --book-start 8200 --out logs/arena_cpp_1s.txt
run "moon 1s"       $A --opp uai --opp-cmd "$M" --opp-go "go movetime 1000" --ti-ms 1000 --book-start 8200 --out logs/arena_moon_1s.txt
run "kata 1s"       $A --opp kata --kata-ms 1000 --ti-ms 1000 --book-start 8200 --out logs/arena_kata_1s.txt
echo "$(date '+%F %H:%M') ALL DONE" >> logs/arena_ext_runner.txt
