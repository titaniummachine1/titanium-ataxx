# D1283: Step 5 sequential 5k plain-AB eval-gate matches (LAWS B11b, 400p @0, 7 threads)
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_ws\x86_64-pc-windows-gnu\release\lab.exe'
$v15 = 'C:\gitProjects\titanium-ataxx\runs\nn_v15muon\nn_v15muon\best.tnn'
$v14 = 'C:\gitProjects\titanium-ataxx\runs\nn_v14muon\nn_v14muon\best.tnn'
$chp = 'C:\gitProjects\mainpromo-ataxx\data\nnue\nnue_main.tnn'
$bk  = 'C:\gitProjects\titanium-ataxx\data\nnue\book50k_std5k.txt'

"=== Leg 1: eval-gate 5k nn_v15muon vs Champion start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $lab eval-gate --a $v15 --b $chp --nodes 5000 --book $bk --book0 0 --pairs 400 --threads 7 *>&1 | Tee-Object -FilePath C:\gitProjects\titanium-ataxx\logs\evalgate\d1283_v15muon_vs_812C_n5000.log

"=== Leg 2: eval-gate 5k nn_v15muon vs nn_v14muon start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $lab eval-gate --a $v15 --b $v14 --nodes 5000 --book $bk --book0 0 --pairs 400 --threads 7 *>&1 | Tee-Object -FilePath C:\gitProjects\titanium-ataxx\logs\evalgate\d1283_v15muon_vs_v14muon_n5000.log
"=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
