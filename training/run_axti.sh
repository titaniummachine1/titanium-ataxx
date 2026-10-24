#!/bin/bash
cd /c/gitProjects/titanium-ataxx
python training/ax_vs_ti.py --games 100 --workers 4 --ti-ms 100 --cpp-ms 100 --book-start 1000 --out logs/axti_100ms.txt
for N in 1 3 10 30 100 300; do
  python training/ax_vs_ti.py --games 40 --workers 4 --ti-nodes $N --cpp-evals 1 --book-start 1100 --out logs/axti_ladder_n$N.txt
done
echo ALL_AXTI_DONE
