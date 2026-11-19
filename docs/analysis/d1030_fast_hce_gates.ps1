# D1030: re-gate our HCE with the fast eval path (lab hceeval.rs, 437 ns vs 2,063) + the best-fit trained Zataxx net at 5k nodes.
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$ab = 'C:\gitProjects\zataxx-ab'
$z = { param($v) "$ab\$v\target\x86_64-pc-windows-gnu\release\zataxx.exe" }
$L = 'C:\gitProjects\titanium-ataxx\logs\ext'
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
Set-Location C:\gitProjects\titanium-ataxx
function Gate($c, $b, $lim, $out) { "$out start $(Get-Date -Format HH:mm)"; & $ti sprt --candidate $c --baseline $b --non-main-ab --rules libataxx @lim --elo1 20 --max-games 400 --threads 4 --out "$L\$out.log" 2>&1 | Select-Object -Last 4 }
# (1) best-fit trained Zataxx net (all-8 export 1599439F at SCALE 698) vs shipped, 5k nodes
Gate (& $z 'trained_k') (& $z 'default') @('--nodes', '5000') 'zat_trainedk_vs_default_5k'
# (2) rebuild Zataxx-HCE (path dep on the lab -> fast eval) and re-gate at 100 ms
Push-Location "$ab\hce"; $env:CARGO_TARGET_DIR = "$ab\hce\target"; cargo build --release --offline 2>&1 | Select-Object -Last 1; Pop-Location; Remove-Item Env:CARGO_TARGET_DIR
"hce exe $((Get-FileHash (& $z 'hce')).Hash.Substring(0,8))"
Gate (& $z 'hce') (& $z 'gx') @('--time-ms', '100') 'zat_hcefast_vs_gx_100ms'
Gate (& $z 'hce') (& $z 'default') @('--time-ms', '100') 'zat_hcefast_vs_default_100ms'
# (3) rebuild the DLL (fast eval) into a NEW file and re-gate gataxx-ours vs gataxx default at 100 ms
Push-Location C:\gitProjects\titaniumtest\lab; $env:CARGO_TARGET_DIR = 'C:\gitProjects\titaniumtest\lab\target_dll'; cargo build --release --lib 2>&1 | Select-Object -Last 1; Pop-Location; Remove-Item Env:CARGO_TARGET_DIR
Copy-Item C:\gitProjects\titaniumtest\lab\target_dll\x86_64-pc-windows-gnu\release\titanium_lab.dll C:\gitProjects\gataxx-ab\lib\titanium_lab_fast.dll -Force
"dll $((Get-FileHash C:\gitProjects\gataxx-ab\lib\titanium_lab_fast.dll).Hash.Substring(0,8))"
$jdk = 'C:\gitProjects\toolchains\jdk21\jdk-21.0.13+9\bin\java.exe'
$w = 'C:\gitProjects\titanium-ataxx\logs\hce\d991_hcetrain_b32_pairs_weights.json'
$gc = "$jdk --enable-preview --enable-native-access=ALL-UNNAMED -Dgataxx.rusthce=C:\gitProjects\gataxx-ab\lib\titanium_lab_fast.dll -Dgataxx.rusthce.weights=$w -cp C:\gitProjects\gataxx-ab\classes spamalot.gataxx.Main"
$gb = "$jdk -cp C:\gitProjects\engines-src\aoflaofl__gataxx\build\classes spamalot.gataxx.Main"
Gate $gc $gb @('--time-ms', '100') 'gx_rusthcefast_vs_default_100ms'
"done $(Get-Date -Format HH:mm)"
