# D1269: Sequential PC evaluation queue (one job at a time; PC 100% idle for Leg 3 timed gate):
#   Leg 1: eval-gate 5k (400p @0, 7t) nn_v14muon vs Champion nnue_main.tnn
#   Leg 2: eval-gate 5k (400p @0, 7t) nn_v14muon vs nn_v13anneal
#   Leg 3: clean idle-PC R2 (exp/nonpv-re2 f4a3ed4 vs main 0120fd12) ti gate 100 ms Botzone (--book0 400, <=800g, 4t) + B12 reports
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_ws\x86_64-pc-windows-gnu\release\lab.exe'
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$net = 'C:\gitProjects\titanium-ataxx\runs\nn_v14muon\nn_v14muon\best.tnn'
$v13 = 'C:\gitProjects\titanium-ataxx\runs\nn_v13anneal\nn_v13anneal\best.tnn'
$chp = 'C:\gitProjects\mainpromo-ataxx\data\nnue\nnue_main.tnn'
$bk  = 'C:\gitProjects\titanium-ataxx\data\nnue\book50k_std5k.txt'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
New-Item -ItemType Directory -Force -Path $rep | Out-Null

"=== Leg 1: eval-gate 5k nn_v14muon vs Champion start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $lab eval-gate --a $net --b $chp --nodes 5000 --book $bk --book0 0 --pairs 400 --threads 7 | Tee-Object -FilePath C:\gitProjects\titanium-ataxx\logs\evalgate\d1269_v14muon_vs_812C_n5000.log

"=== Leg 2: eval-gate 5k nn_v14muon vs nn_v13anneal start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $lab eval-gate --a $net --b $v13 --nodes 5000 --book $bk --book0 0 --pairs 400 --threads 7 | Tee-Object -FilePath C:\gitProjects\titanium-ataxx\logs\evalgate\d1269_v14muon_vs_v13anneal_n5000.log

"=== Leg 3: clean R2 (f4a3ed4 vs main) ti gate 100ms @book0=400 start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$t0 = Get-Date
& $ti gate --test f4a3ed4 --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --book0 400 --max-games 800 --elo1 10 --threads 4 2>&1 | Tee-Object -FilePath C:\gitProjects\logs\d1269_r2_gate_100ms_b400.log
$gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($gj) {
  & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1269_r2_100ms_b400_insight.txt"
  & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1269_r2_100ms_b400_analyze.txt"
  "R2 reports written from $($gj.FullName)"
} else {
  "WARNING: no games.jsonl found for R2 gate"
}
"=== queue done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
