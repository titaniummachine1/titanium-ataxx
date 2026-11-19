# D1191: after the D1185 cluster residual (PID 21256) ends: paired tests of s10 erosion + w3 windows vs calibration
# (lab residual-pair, D1188), then self-play generation on the PC (user: more data to measure bigger nets).
Set-Location C:\gitProjects\titanium-ataxx
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
while (Get-Process -Id 21256 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_pair\x86_64-pc-windows-gnu\release\lab.exe'
$h = 'logs\hce'
foreach ($b in 'd1185_res_s10_erosion', 'd1185_res_w3_windows') {
  if (Test-Path "$h\${b}_weights.json") {
    Say "residual-pair gB vs $b"
    & $lab residual-pair --a "$h\d1176_res_gB_calibration_weights.json" --b "$h\${b}_weights.json" 2>&1 | Select-String 'rows|A =|B =' | ForEach-Object { Say $_.Line }
  } else { Say "MISSING $b weights" }
}
Say 'starting self-play generation (gen_pc_loop.ps1, data\gen\pc_run16_5k, 7 threads, 5000 nodes, clean STOP)'
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir data\gen\pc_run16_5k -Threads 7 -Batches 60
Say 'queue end'
