# D1192: standard anneal continuation of nn_v13sat (D1086/D1090 recipe): from its best.tnn, one val check per epoch
# (3,337 steps = 54,680,997 rows / 16,384), lr 5e-5 constant, halve on 4 epochs without gain, stop after 3 useless drops.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
"train launch $(Get-Date -Format HH:mm)"
$a = 'train run --backend cuda --data /vol/data/train_v13h --val /vol/data/val_v13tail --init /vol/runs/nn_v13sat/best.tnn --out /vol/runs/nn_v13anneal --steps 100000000 --eval-every 3337 --bs 16384 --lr 5e-5 --lr-min 5e-5 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 4 --patience 4 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
& $ti train modal launch --tag nn_v13anneal --args $a --timeout-min 120 --max-usd 3.3 2>&1 | ForEach-Object { "$_" }
"done $(Get-Date -Format HH:mm)"
