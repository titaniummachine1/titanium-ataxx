# D1283: Idle-PC 1000ms Botzone confirmation gate for exp/v14muon-eval (87b6b671) vs main (0120fd12).
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
New-Item -ItemType Directory -Force -Path $rep | Out-Null

"=== Leg 2: 1000ms Botzone confirmation gate 87b6b671 vs main start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$t0 = Get-Date
& $ti gate --test 87b6b67 --kind eval --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 1000 --book0 400 --max-games 200 --elo1 10 --threads 4 2>&1 | Tee-Object -FilePath C:\gitProjects\logs\d1283_v14muon_gate_1000ms.log
$ec = $LASTEXITCODE
"gate exit=$ec"
$gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($gj) {
  & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1283_v14muon_1000ms_insight.txt"
  & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1283_v14muon_1000ms_analyze.txt"
  "reports written from $($gj.FullName)"
} else {
  "WARNING: no games.jsonl found"
}
"=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
