#!/bin/bash
cd /c/gitProjects/titanium-ataxx
MOON="C:/gitProjects/titanium-ataxx/scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe"
AUT="python C:/gitProjects/titanium-ataxx/training/autaxx_nnue_bridge.py"
python training/ext_gate.py --tag ext_moon_100ms --opp-cmd "$MOON" --time 100 --games 100 --book-base 3100
python training/ext_gate.py --tag ext_autaxx_100ms --opp-cmd "$AUT" --time 100 --games 100 --book-base 3100
python training/ext_gate.py --tag ext_moon_1s --opp-cmd "$MOON" --time 1000 --games 40 --book-base 3300
python training/ext_gate.py --tag ext_autaxx_1s --opp-cmd "$AUT" --time 1000 --games 40 --book-base 3300
echo ALL_EXT_DONE
