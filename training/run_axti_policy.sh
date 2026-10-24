#!/bin/bash
cd /c/gitProjects/titanium-ataxx
for N in 1 3 10 30 100 300 1000; do
  python training/ax_vs_ti.py --games 40 --workers 4 --ti-nodes $N --cpp-policy --book-start 1500 --out logs/axti_policy_n$N.txt
done
echo ALL_POLICY_DONE
