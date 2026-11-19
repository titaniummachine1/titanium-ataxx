#!/bin/sh
# D1090: fetch the annealed run nn_v11banneal, export it (nocal, the run's result-weight target for the provenance check)
# with the disagreement head refit on ALL train_v11b rows (--rows 6e7 > 52M = every block).
TI=C:/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
cd /c/gitProjects/titanium-ataxx || exit 1
echo "fetch $(date +%H:%M)"
$TI train modal fetch --tag nn_v11banneal > logs/d1090_fetch.log 2>&1
echo "fetch exit $? $(date +%H:%M)"
B=$(ls runs/nn_v11banneal/*/best.tnn runs/nn_v11banneal/best.tnn 2>/dev/null | head -1)
echo "best = $B"
$TI train export --net "$B" --data data/nnue/packed/train_v11b --val-dir data/datasets/v11/val --rows 60000000 \
  --result-weight-phase 0.55,0.55,0.25 --out data/nnue/proxy/nn_v11banneal_nocal.tnn > logs/d1090_export.log 2>&1
echo "export exit $? $(date +%H:%M)"
