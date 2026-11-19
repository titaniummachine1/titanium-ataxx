# D1177 (user: 'train a new champion on the whole dataset and see what happens at saturation'): champion recipe
# (nn_v11bsat flags, D1060b) on train_v13 minus its held-out tail; validation = the tail (the lab-standard held-out games).
# Built ON MODAL from train_v11 already there + the two new row files (~154 MB upload); fingerprints must equal the PC's
# (train_v13h 53B85DD691F06389, val_v13tail A457C7F054ECD769, logs/d1177_train_v13.log).
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
$s12 = 'data\datasets\v12\selfplay'; $s13 = 'data\datasets\v13\selfplay'
"put start $(Get-Date -Format HH:mm)"
& $ti train modal put --files "$s12\sp12_rows.tdat,$s12\sp12_rows.manifest.json,$s13\sp13_rows.tdat,$s13\sp13_rows.manifest.json" 2>&1 | ForEach-Object { "$_" }
"build start $(Get-Date -Format HH:mm)"
foreach ($c in @(
  @('v13_split',  'data split --base /vol/data/train_v11 --tail 1528091 --head-out /vol/data/v11head --tail-out /vol/data/val_v13tail'),
  @('v13_app12',  'data append --base /vol/data/v11head --add /vol/data/sp12_rows --out /vol/data/t12head'),
  @('v13_app13',  'data append --base /vol/data/t12head --add /vol/data/sp13_rows --out /vol/data/train_v13h'),
  @('v13_fp',     'data fp --tdat /vol/data/train_v13h.tdat'))) {
  & $ti train modal launch --tag $c[0] --args $c[1] --timeout-min 15 --max-usd 0.5 2>&1 | Select-String 'split:|append:|fp:|exit code|refused|error' | ForEach-Object { $_.Line }
}
"train launch $(Get-Date -Format HH:mm)"
$a = 'train run --backend cuda --data /vol/data/train_v13h --val /vol/data/val_v13tail --init /vol/nets/nn_v10sat_nocal.tnn --out /vol/runs/nn_v13sat --steps 100000000 --eval-every 250 --bs 16384 --lr 5e-5 --lr-min 1e-6 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 0 --patience 54 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
& $ti train modal launch --tag nn_v13sat --args $a --timeout-min 140 --max-usd 3.8 2>&1 | ForEach-Object { "$_" }
"done $(Get-Date -Format HH:mm)"
