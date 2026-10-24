#!/bin/bash
cd /c/gitProjects/titanium-ataxx
while ! grep -q ALL_EXT_DONE logs/ext_gates_run.txt; do sleep 10; done
python training/ext_gate.py --tag ext_autaxx_100ms_clean --opp-cmd "python C:/gitProjects/titanium-ataxx/training/autaxx_nnue_bridge.py" --time 100 --games 100 --book-base 3500
python C:/Users/TERMIN~1/AppData/Local/Temp/claude/C--gitProjects-titanium-ataxx/7e3ffd33-1f98-40e2-9e9e-158a76f7fbbb/scratchpad/ort_bench.py
echo ALL_AFTER_DONE
