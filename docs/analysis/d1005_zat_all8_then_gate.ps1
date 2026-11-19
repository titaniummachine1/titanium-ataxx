# D1004/D1005: after the D1002 HCE continuation: (1) Zataxx all-8-symmetries-per-epoch continuation from the D999 best,
# (2) build Zataxx with the lower-val-loss net, (3) gate it vs Zataxx-default at 100 ms
$hceOut = 'C:\Users\TERMIN~1\AppData\Local\Temp\claude\eff9487f3b6f3868\d4c2cbcc-d66a-4bf0-9abe-1f2fe101f565\tasks\b7wa1qxtl.output'
while (-not ((Test-Path $hceOut) -and (Select-String -Path $hceOut -Pattern 'hce done' -Quiet))) { Start-Sleep 30 }
$h = 'C:\gitProjects\titanium-ataxx\logs\hce'
$d = 'C:\gitProjects\titanium-ataxx\data\nnue\packed'
$net1 = "$h\d999_zattrain_d4_net.bin"
if (-not (Test-Path $net1)) { "NO D999 NET"; exit 1 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
"all8 start $(Get-Date -Format HH:mm)"
& $lab zataxx-train --data "$d\train_v10.tdat" --val "$d\val_v9_clean.tdat" --net $net1 --sym-per-epoch 8 --lr 0.001 --threads 7 --out "$h\d1005_zattrain_all8" 2>&1 | Out-Null
"all8 done $(Get-Date -Format HH:mm)"
function QLoss($log) { $m = Select-String -Path $log -Pattern 'QUANTISED val loss ([0-9.]+)' | Select-Object -Last 1; if ($m) { [double]$m.Matches[0].Groups[1].Value } else { 9.0 } }
$l1 = QLoss "$h\d999_zattrain_d4.log"; $l2 = QLoss "$h\d1005_zattrain_all8.log"
$net = if ($l2 -lt $l1 -and (Test-Path "$h\d1005_zattrain_all8_net.bin")) { "$h\d1005_zattrain_all8_net.bin" } else { $net1 }
"quantised val: d999 $l1, d1005 all8 $l2 -> gate net $net"
$ab = 'C:\gitProjects\zataxx-ab'
Copy-Item $net "$ab\trained\src\net6.bin" -Force
"trained net sha $((Get-FileHash "$ab\trained\src\net6.bin").Hash.Substring(0,8))"
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
Set-Location "$ab\trained"; $env:CARGO_TARGET_DIR = "$ab\trained\target"
cargo build --release --offline 2>&1 | Select-Object -Last 1
$cand = "$ab\trained\target\x86_64-pc-windows-gnu\release\zataxx.exe"
$base = "$ab\default\target\x86_64-pc-windows-gnu\release\zataxx.exe"
if (-not (Test-Path $cand)) { "BUILD FAILED"; exit 1 }
Remove-Item Env:CARGO_TARGET_DIR
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
Set-Location C:\gitProjects\titanium-ataxx
"gate start $(Get-Date -Format HH:mm)"
& $ti sprt --candidate $cand --baseline $base --non-main-ab --rules libataxx --time-ms 100 --elo1 20 --max-games 400 --threads 4 --out C:\gitProjects\titanium-ataxx\logs\ext\zat_trained_vs_default_100ms.log 2>&1 | Select-Object -Last 15
"gate done $(Get-Date -Format HH:mm)"
