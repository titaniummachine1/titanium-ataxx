# D1019 (user 2026-11-17: 'maybe try the net that was trained before resizing'): gate the D999 net
# (D4-trained, quantised val 0.009928, NO 0.92 rescale fold) inside Zataxx (SCALE 400, as trained) vs shipped Zataxx.
# Waits for the running trained_k gate (ti.exe) to finish first (one PC job at a time).
while (Get-Process ti -ErrorAction SilentlyContinue) { Start-Sleep 20 }
$ab = 'C:\gitProjects\zataxx-ab'
$v = "$ab\trained_d999"
New-Item -ItemType Directory -Force $v | Out-Null
Copy-Item "$ab\trained\src", "$ab\trained\Cargo.toml", "$ab\trained\Cargo.lock" $v -Recurse -Force
Copy-Item 'C:\gitProjects\titanium-ataxx\logs\hce\d999_zattrain_d4_net.bin' "$v\src\net6.bin" -Force
"d999 net sha $((Get-FileHash "$v\src\net6.bin").Hash.Substring(0,8)); SCALE line: $((Select-String -Path "$v\src\nn.rs" -Pattern 'SCALE: i32').Line)"
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
Set-Location $v; $env:CARGO_TARGET_DIR = "$v\target"
cargo build --release --offline 2>&1 | Select-Object -Last 1
Remove-Item Env:CARGO_TARGET_DIR
$cand = "$v\target\x86_64-pc-windows-gnu\release\zataxx.exe"
if (-not (Test-Path $cand)) { "BUILD FAILED"; exit 1 }
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
Set-Location C:\gitProjects\titanium-ataxx
"gate start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $cand --baseline "$ab\default\target\x86_64-pc-windows-gnu\release\zataxx.exe" --non-main-ab --rules libataxx --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\zat_d999_vs_default_100ms.log 2>&1 | Select-Object -Last 20
"gate done $(Get-Date -Format HH:mm)"
