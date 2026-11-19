# D1284: Clean idle-PC 100ms Botzone gate of exp/mcgs-puct (fe24a8d --mcgs, nn_v14muon 32D573EA + dis + 380F609C) vs Champion main (0120fd12)
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
New-Item -ItemType Directory -Force -Path $rep | Out-Null

"=== PUCT+D4-MCGS fe24a8d vs main 0120fd12 (100ms Botzone, 400g @0) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$t0 = Get-Date
& $ti gate --test fe24a8d --kind search --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22 --mcgs" --time-ms 100 --book0 0 --max-games 400 --elo1 10 --threads 4 2>&1 | Tee-Object -FilePath C:\gitProjects\logs\d1284_mcgs_vs_main_100ms.log
$ec = $LASTEXITCODE
"gate exit=$ec"
$gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($gj) {
  & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1284_mcgs_vs_main_100ms_insight.txt"
  & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1284_mcgs_vs_main_100ms_analyze.txt"
  "reports written from $($gj.FullName)"
} else {
  "WARNING: no games.jsonl found"
}
"=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
