# D1029 (user 2026-11-17: 'in gataxx check our best HCE vs its default'): gataxx 2.0.0's own search, two evals:
# candidate = gataxx + Titanium D991 HCE through titanium_lab.dll (C:/gitProjects/gataxx-ab, RustHceEvaluator, parity 200/200)
# baseline  = gataxx 2.0.0 default (FeatureEvaluator + built-in pair tables, original build/classes).
# Leg 1: 5,000 nodes per move; leg 2: 100 ms. Max 400 games each, 4 parallel.
$jdk = 'C:\gitProjects\toolchains\jdk21\jdk-21.0.13+9\bin\java.exe'
$w = 'C:\gitProjects\titanium-ataxx\logs\hce\d991_hcetrain_b32_pairs_weights.json'
$cand = "$jdk --enable-preview --enable-native-access=ALL-UNNAMED -Dgataxx.rusthce=C:\gitProjects\gataxx-ab\lib\titanium_lab.dll -Dgataxx.rusthce.weights=$w -cp C:\gitProjects\gataxx-ab\classes spamalot.gataxx.Main"
$base = "$jdk -cp C:\gitProjects\engines-src\aoflaofl__gataxx\build\classes spamalot.gataxx.Main"
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
Set-Location C:\gitProjects\titanium-ataxx
"5k start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $cand --baseline $base --non-main-ab --rules libataxx --nodes 5000 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\gx_rusthce_vs_default_5k.log 2>&1 | Select-Object -Last 22
"100ms start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $cand --baseline $base --non-main-ab --rules libataxx --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\gx_rusthce_vs_default_100ms.log 2>&1 | Select-Object -Last 22
"done $(Get-Date -Format HH:mm)"
