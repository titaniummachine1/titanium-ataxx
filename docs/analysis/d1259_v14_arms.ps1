# D1259: two arms on train_v14h (proofs + 0.9^n back-propagation + pooled WDL, D1253), both from nn_v13anneal best.tnn
# (59B7B1FD, the best-fitted net, D1241), recipe = nn_v13anneal's (D1192) except the optimiser:
#   control nn_v14anneal = Ranger (the champion recipe default), test nn_v14muon = --opt muon (Muon on l1w/l2w, Ranger on the rest).
# Validation = val_v13tail (identical bytes to val_v14tail): comparable with nn_v13anneal 0.030208.
param([ValidateSet('control','muon')][string]$Arm)
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
$base = 'train run --backend cuda --data /vol/data/train_v14h --val /vol/data/val_v13tail --init /vol/runs/nn_v13anneal/best.tnn --steps 100000000 --eval-every 3337 --bs 16384 --lr 5e-5 --lr-min 5e-5 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 4 --patience 4 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
if ($Arm -eq 'control') { $tag = 'nn_v14anneal'; $a = "$base --out /vol/runs/nn_v14anneal" } else { $tag = 'nn_v14muon'; $a = "$base --out /vol/runs/nn_v14muon --opt muon" }
"launch $tag $(Get-Date -Format HH:mm:ss)"
& $ti train modal launch --tag $tag --args $a --timeout-min 120 --max-usd 3.3 2>&1 | ForEach-Object { "$_" }
"done $tag $(Get-Date -Format HH:mm:ss)"
