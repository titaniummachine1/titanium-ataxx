#!/bin/sh
# D1604 proof hunt over every never-hunted game (sp16 = data/gen/pc_run16_15k, sp15 = datagenpc_run15_5k): ti proof-lines
# per batch (hub b3882a2, zataxx-bz 5bbad56), output data/proofs/hunt_v16/<run>_<batch>.{tgam,tbst,report.txt,conflicts.*}.
# RESUMABLE: a batch with a .report.txt is done and skipped; an unfinished one is moved to .partial_<time> and redone.
# STOP: C:/gitProjects/STOP_ALL (the running batch stops within seconds, the script starts nothing more).
TI=/c/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
cd /c/gitProjects/titanium-ataxx
OUT=data/proofs/hunt_v16
for f in data/gen/pc_run16_15k/batch_*.tgam datagenpc_run15_5k/batch_*.tgam; do
  [ -f /c/gitProjects/STOP_ALL ] && { echo "=== STOP_ALL, exit $(date)"; exit 0; }
  run=$(basename $(dirname $f)); b=$(basename $f .tgam); o=$OUT/${run}_$b
  [ -f $o.report.txt ] && { echo "=== $o done, skipped"; continue; }
  if [ -f $o.tgam ]; then ts=$(date +%Y%m%d_%H%M%S); for x in $o.tgam $o.tbst; do [ -f $x ] && mv $x $x.partial_$ts; done; echo "=== $o unfinished output moved to .partial_$ts"; fi
  echo "=== $o start $(date)"
  $TI proof-lines --tgam $f --out $o --threads 7 --max-ms 1000 2>&1 | grep -v "^proof-lines: [0-9]*/" ; echo "=== $o end rc $? $(date)"
done
echo "=== ALL DONE $(date)"
