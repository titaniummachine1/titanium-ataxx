# D1024 (user 2026-11-17: 'what happens if you replace Zataxx's eval with our best-fit HCE'):
# Zataxx search + our D991 HCE (C:/gitProjects/zataxx-ab/hce, lab lib titanium_lab::hceeval) vs Zataxx shipped, 100 ms.
# Waits for the running gate (ti.exe) to finish first (one PC job at a time).
while (Get-Process ti -ErrorAction SilentlyContinue) { Start-Sleep 15 }
$ab = 'C:\gitProjects\zataxx-ab'
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
Set-Location C:\gitProjects\titanium-ataxx
"gate start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate "$ab\hce\target\x86_64-pc-windows-gnu\release\zataxx.exe" --baseline "$ab\default\target\x86_64-pc-windows-gnu\release\zataxx.exe" --non-main-ab --rules libataxx --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\zat_hce_vs_default_100ms.log 2>&1 | Select-Object -Last 22
"gate done $(Get-Date -Format HH:mm)"
