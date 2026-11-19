# D1282: Modal MuonAdamW saturation run nn_v15muon from ws_v14muon_resurrect.tnn (39 revived dead L2, 0.000000000 step-0 parity)
# using L4 16-bucket Elo-correlated result weights + sqrt-damped Elo-concordance loss weights on /vol/data/train_v14h.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
$rw = '0.40,0.70,0.95,0.70,0.40,0.40,0.40,0.40,0.40,0.30,0.20,0.10,0.10,0.10,0.10,0.25'
$lw = '1.20,1.25,1.40,1.25,1.02,1.02,1.05,1.05,1.05,1.00,0.82,0.75,0.75,0.75,0.75,0.95'
$a = "train run --backend cuda --data /vol/data/train_v14h --val /vol/data/val_v13tail --init /vol/nets/ws_v14muon_resurrect.tnn --out /vol/runs/nn_v15muon --steps 100000000 --eval-every 3337 --bs 16384 --lr 5e-5 --lr-min 5e-5 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 4 --patience 4 --lr-factor 0.5 --max-drops 3 --result-weight-buckets $rw --loss-weight-buckets $lw --opt muon"
"launch nn_v15muon $(Get-Date -Format HH:mm:ss)"
& $ti train modal launch --tag nn_v15muon --args $a --timeout-min 120 --max-usd 3.3 2>&1 | ForEach-Object { "$_" }
"done nn_v15muon $(Get-Date -Format HH:mm:ss) exit=$LASTEXITCODE"
