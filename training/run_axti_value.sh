#!/bin/bash
cd /c/gitProjects/titanium-ataxx
while ! grep -q ALL_POLICY2_DONE logs/axti_policy2_run.txt; do sleep 5; done
export CPP_EXE=C:/gitProjects/titanium-ataxx/data/engines/ataxx_cpp/build/engine-ctl2.exe
for N in 1 10 100 1000 10000; do
  python training/ax_vs_ti.py --games 40 --workers 4 --ti-nodes $N --cpp-valuegreedy --book-start 1500 --out logs/axti_value_n$N.txt
done
echo ALL_VALUE_DONE
