#!/bin/bash
cd /c/gitProjects/titanium-ataxx
for N in 3000 10000 30000 100000 300000; do
  python training/ax_vs_ti.py --games 40 --workers 4 --ti-nodes $N --cpp-policy --book-start 1500 --out logs/axti_policy_n$N.txt
done
echo ALL_POLICY2_DONE
