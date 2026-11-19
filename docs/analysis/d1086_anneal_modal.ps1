# D1086: continuation of nn_v11bsat (its patience stop fired at 25% of the cosine LR horizon, LR still ~86% of peak = never
# annealed). Init = its best.tnn (C14), same data / val / target as D1060b. Saturation per EPOCH (user rule): one val
# check per epoch (eval-every 3188 steps = 52,238,311 rows / 16,384), LR 4.3e-5 constant base (lr-min = lr, steps 1e8 =
# no cosine horizon), halve after 4 epochs without a 2e-6 gain, stop after 3 halvings that did not help.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
"train launch $(Get-Date -Format HH:mm)"
$a = 'train run --backend cuda --data /vol/data/train_v11b --val /vol/data/val_v11_clean --init /vol/runs/nn_v11bsat/best.tnn --out /vol/runs/nn_v11banneal --steps 100000000 --eval-every 3188 --bs 16384 --lr 4.3e-5 --lr-min 4.3e-5 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 4 --patience 4 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
& $ti train modal launch --tag nn_v11banneal --args $a --timeout-min 120 --max-usd 3.3 2>&1 | ForEach-Object { "$_" }
"done $(Get-Date -Format HH:mm)"
