# D1284 Step 4: Clean idle-PC 100ms --non-main-ab gate of C2 (f819cea = pol_v15_d1284 F0BB257A) vs 87b6b67 (380F609C)
$ErrorActionPreference = 'Continue'
Set-Location C:\gitProjects\titanium-ataxx
$ti  = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$rep = 'C:\gitProjects\titanium-ataxx\logs\gates\reports'
New-Item -ItemType Directory -Force -Path $rep | Out-Null

"=== Step 4: 100ms Botzone A/B f819cea (pol_v15_d1284) vs 87b6b67 (380F609C) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$t0 = Get-Date
& $ti gate --test f819cea --base 87b6b67 --non-main-ab --kind eval --rules botzone --base-args "--bz --tt-bits 22" --test-args "--bz --tt-bits 22" --time-ms 100 --book0 0 --max-games 800 --elo1 10 --threads 4 2>&1 | Tee-Object -FilePath C:\gitProjects\logs\d1284_policy_ab_100ms.log
$ec = $LASTEXITCODE
"gate exit=$ec"
$gj = Get-ChildItem -Recurse C:\gitProjects\titanium-ataxx\logs\gates -Filter *.games.jsonl -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($gj) {
  & $ti insight $gj.FullName --by-bucket 2>&1 | Out-File -Encoding utf8 "$rep\d1284_policy_ab_100ms_insight.txt"
  & $ti analyze $gj.FullName 2>&1 | Out-File -Encoding utf8 "$rep\d1284_policy_ab_100ms_analyze.txt"
  "reports written from $($gj.FullName)"
} else {
  "WARNING: no games.jsonl found"
}
"=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
