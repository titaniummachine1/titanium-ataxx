# D1060b: rerun of nn_v11sat with a validation set from the NEW games (val_v11_clean, position-disjoint) and train_v11b
# (train_v10 + the other new games). Uploads ~0.72 GB, assembles train_v11b on Modal, launches nn_v11bsat detached.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
$b = 'data\datasets\v11\selfplay_b'; $v = 'data\datasets\v11\val'
"put start $(Get-Date -Format HH:mm)"
& $ti train modal put --files "$b\sp11b_rows.tdat,$b\sp11b_rows.manifest.json,$b\sp11b_rows.chains,$v\val_v11_clean.tdat,$v\val_v11_clean.manifest.json" 2>&1 | ForEach-Object { "$_" }
"append start $(Get-Date -Format HH:mm)"
& $ti train modal launch --tag v11b_append --args 'data append --base /vol/data/train_v10 --add /vol/data/sp11b_rows --out /vol/data/train_v11b' --timeout-min 15 --max-usd 0.5 2>&1 | Select-String 'append:|exit code|refused' | ForEach-Object { $_.Line }
"train launch $(Get-Date -Format HH:mm)"
$a = 'train run --backend cuda --data /vol/data/train_v11b --val /vol/data/val_v11_clean --init /vol/nets/nn_v10sat_nocal.tnn --out /vol/runs/nn_v11bsat --steps 280000 --eval-every 250 --bs 16384 --lr 5e-5 --lr-min 1e-6 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 0 --patience 52 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
& $ti train modal launch --tag nn_v11bsat --args $a --timeout-min 110 --max-usd 3 2>&1 | ForEach-Object { "$_" }
"done $(Get-Date -Format HH:mm)"
