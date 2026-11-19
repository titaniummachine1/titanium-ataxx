# D1058 (user 2026-11-17: publish the dataset; if that is slow, train on Modal): (1) upload the new rows + the champion
# net to the Modal volume, (2) assemble train_v11 there (ti data append), (3) launch nn_v11sat detached (champion
# recipe, only the data changed), (4) THEN the slow ModelScope upload of the full train_v11 (0.5 MB/s, ~2.3 h).
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$env:PYTHONIOENCODING = 'utf-8'
$s = 'data\datasets\v11\selfplay'
"put start $(Get-Date -Format HH:mm)"
& $ti train modal put --files "$s\sp11_rows.tdat,$s\sp11_rows.manifest.json,$s\sp11_rows.chains" 2>&1 | ForEach-Object { "$_" }
& $ti train modal put --files 'data\nnue\proxy\nn_v10sat_nocal.tnn' --dest nets 2>&1 | ForEach-Object { "$_" }
"append start $(Get-Date -Format HH:mm)"
& $ti train modal launch --tag v11_append --args 'data append --base /vol/data/train_v10 --add /vol/data/sp11_rows --out /vol/data/train_v11' --timeout-min 15 --max-usd 0.5 2>&1 | Select-String 'append|exit|RUN|refused|error' | ForEach-Object { $_.Line }
python -m modal volume ls ti-data /data --json 2>&1 | Select-String 'train_v11' | ForEach-Object { $_.Line }
"train launch $(Get-Date -Format HH:mm)"
$args11 = 'train run --backend cuda --data /vol/data/train_v11 --val /vol/data/val_v9_clean --init /vol/nets/nn_v10sat_nocal.tnn --out /vol/runs/nn_v11sat --steps 280000 --eval-every 250 --bs 16384 --lr 5e-5 --lr-min 1e-6 --warmup 500 --ram --guard 1.15 --ema 0.9 --val-rows 131072 --proof-blend 0.5 --plateau 0 --patience 52 --lr-factor 0.5 --max-drops 3 --result-weight-phase 0.55,0.55,0.25'
& $ti train modal launch --tag nn_v11sat --args $args11 --timeout-min 110 --max-usd 3 --detach 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath C:\gitProjects\logs\modal_nn_v11sat_launch.log
"modelscope upload start $(Get-Date -Format HH:mm)"
python -I -c "
from modelscope.hub.api import HubApi
import time, os
api = HubApi()
p = 'C:/gitProjects/titanium-ataxx/data/nnue/packed/'
for f in ['train_v11.tdat', 'train_v11.kata', 'train_v11.aux']:
    t = time.time()
    api.upload_file(path_or_fileobj=p + f, path_in_repo=f, repo_id='titaniummachine1/titanium_ataxx', repo_type='dataset', commit_message='train_v11 (D1049/D1058): ' + f)
    print(f, os.path.getsize(p + f), 'bytes in', round(time.time() - t), 's', flush=True)
" 2>&1 | Select-String -NotMatch 'Uploading\]' | ForEach-Object { $_.Line }
"done $(Get-Date -Format HH:mm)"
