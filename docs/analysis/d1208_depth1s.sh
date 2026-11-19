#!/bin/sh
# D1208: depth reached in 1 s (movetime 1000), Titanium main 0120fd12 vs Zataxx, same 60 positions (12 per empties band), one fresh process per position, single thread, engine default hash.
T=/c/gitProjects/_gate/exes/titanium-0120fd12.exe
Z=/c/gitProjects/Zataxx/target/release/zataxx.exe
O=logs/depth1s/d1208_results.tsv
echo -e "empties\tengine\tdepth\tnodes\ttime\tbestmove\tfen" > $O
while IFS="$(printf '\t')" read e f; do
  for eng in titanium zataxx; do
    if [ $eng = titanium ]; then out=$(printf "uai\nuainewgame\nposition fen $f\ngo movetime 1000\nquit\n" | timeout 20 $T serve 2>/dev/null)
    else out=$(printf "uai\nuainewgame\nposition fen $f\ngo movetime 1000\nquit\n" | timeout 20 $Z 2>/dev/null); fi
    last=$(echo "$out" | grep '^info depth' | tail -1)
    d=$(echo "$last" | sed -n 's/.*info depth \([0-9]*\).*/\1/p'); n=$(echo "$last" | sed -n 's/.* nodes \([0-9]*\).*/\1/p'); t=$(echo "$last" | sed -n 's/.* time \([0-9]*\).*/\1/p')
    bm=$(echo "$out" | sed -n 's/^bestmove \(.*\)/\1/p' | tr -d '\r')
    echo -e "$e\t$eng\t$d\t$n\t$t\t$bm\t$f" >> $O
  done
done < logs/depth1s/positions.tsv
