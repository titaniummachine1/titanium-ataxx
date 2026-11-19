# D1028 (user 2026-11-17: 'is that HCE any better than gataxx's HCE?'): same Zataxx search, two evals head to head:
# candidate = our best HCE (D991, C:/gitProjects/zataxx-ab/hce), baseline = gataxx 2.0.0 default eval (C:/gitProjects/zataxx-ab/gx).
# Leg 1: 5,000 nodes per move (eval quality per node); leg 2: 100 ms (speed included). Max 400 games each.
$ab = 'C:\gitProjects\zataxx-ab'
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$c = "$ab\hce\target\x86_64-pc-windows-gnu\release\zataxx.exe"
$b = "$ab\gx\target\x86_64-pc-windows-gnu\release\zataxx.exe"
Set-Location C:\gitProjects\titanium-ataxx
"5k start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $c --baseline $b --non-main-ab --rules libataxx --nodes 5000 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\zat_hce_vs_gx_5k.log 2>&1 | Select-Object -Last 22
"100ms start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $c --baseline $b --non-main-ab --rules libataxx --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\zat_hce_vs_gx_100ms.log 2>&1 | Select-Object -Last 22
"done $(Get-Date -Format HH:mm)"
