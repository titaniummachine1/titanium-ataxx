# D1203 (user: 'attention is all you need'): after the D1193 lever-screen queue (powershell PID 16120) finishes its 11
# lever runs (its generation step leaves at once: data\gen\pc_run16_5k\STOP was created for that), run LEVER ATTENTION v1
# (lab levatt-train: attention arm, then the control arm --no-attn, D1202), then remove the STOP file and start self-play.
Set-Location C:\gitProjects\titanium-ataxx
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
while (Get-Process -Id 16120 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
$h = 'logs\hce'
$lev = 'C:\gitProjects\titaniumtest\lab\target_lev\x86_64-pc-windows-gnu\release\lab.exe'
foreach ($arm in @(@('d1203_levatt_attn', @()), @('d1203_levatt_ctrl', @('--no-attn')))) {
  Say "levatt-train $($arm[0]) start"
  & $lev levatt-train @($arm[1]) --threads 7 --out "$h\$($arm[0])" 2>&1 | Select-String 'step 0|epoch [0-9]+:|SATURATED|RESTART' | ForEach-Object { Say ("$($arm[0]): " + $_.Line) }
}
Remove-Item data\gen\pc_run16_5k\STOP -ErrorAction SilentlyContinue
Say 'starting self-play generation (gen_pc_loop.ps1, data\gen\pc_run16_5k, 7 threads, 5000 nodes, clean STOP)'
& powershell -NoProfile -ExecutionPolicy Bypass -File docs\analysis\gen_pc_loop.ps1 -RunDir data\gen\pc_run16_5k -Threads 7 -Batches 60
Say 'queue end'
