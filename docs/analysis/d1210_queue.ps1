# D1210 (user: 'set up the proof hunt in the hub so the PC runs a day or two; proof = highest quality data').
# Order on the PC (one long job per machine): the d1204 queue's lever-attention arms (attention, control) -> PROOF HUNT
# (ti proof-hunt, up to -Hours, clean STOP) -> the 10 deferred lever screens (moved here from d1204) -> self-play generation.
# How the d1204 queue (PID 20088) is steered without stopping it (its lab child writes into its pipe):
#   * its lever loop skips a run whose log exists: placeholder logs 'DEFERRED D1210' are written for the 10 runs;
#   * its last step starts gen_pc_loop.ps1, which now waits while data\gen\pc_run16_5k\HOLD exists (created here).
# This queue waits for the control arm to end, runs the hunt, removes the placeholders and runs the screens itself,
# then removes HOLD (the waiting generation starts).
param([string]$Engine = 'zataxx', [int]$Threads = 7, [double]$Hours = 48, [int]$Nodes = 5000000)
Set-Location C:\gitProjects\titanium-ataxx
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" }
$h = 'logs\hce'
$ti = 'C:\gitProjects\titaniumtest\target_proof\x86_64-pc-windows-gnu\release\ti.exe'
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
$pair = 'C:\gitProjects\titaniumtest\lab\target_pair\x86_64-pc-windows-gnu\release\lab.exe'
$gb = "$h\d1176_res_gB_calibration_weights.json"
$out = 'data\proofs\v13_hunt'
$runs = @(
  @('l2_raw_pst',       @('--set','2')),
  @('l3_ours_all',      @('--set','3')),
  @('l6_ours_scalars',  @('--set','6')),
  @('l7_dynamic_pst',   @('--set','7')),
  @('l8_soft_weak',     @('--set','8')),
  @('l_win2',           @('--set','5','--win','2')),
  @('l_gataxx_pairs',   @('--set','5','--pairs','1','--no-mob','1')),
  @('l_spairs_d1',      @('--set','5','--spairs','1')),
  @('l_spairs_d2',      @('--set','5','--spairs','2')),
  @('l_spairs_d3',      @('--set','5','--spairs','3'))
)
New-Item -ItemType File -Force data\gen\pc_run16_5k\HOLD | Out-Null
foreach ($r in $runs) {
  $stem = "$h\d1193_res_$($r[0])"
  if (-not (Test-Path "$stem.log")) { "DEFERRED D1210: this lever run is done by docs\analysis\d1210_queue.ps1 after the proof hunt" | Out-File "$stem.log" -Encoding ascii }
}
Say "HOLD + placeholders set; waiting for the control arm (logs\hce\d1203_levatt_ctrl.log) to end"
while (-not ((Test-Path "$h\d1203_levatt_ctrl.log") -and (Select-String -Path "$h\d1203_levatt_ctrl.log" -Pattern 'SATURATED' -Quiet))) { Start-Sleep 60 }
while (Get-CimInstance Win32_Process -Filter "Name='lab.exe'" | Where-Object { $_.CommandLine -match 'levatt-train' }) { Start-Sleep 30 }
Say "control arm ended; PROOF HUNT engine=$Engine threads=$Threads nodes=$Nodes for up to $Hours h -> $out"
Remove-Item "$out\STOP" -ErrorAction SilentlyContinue
$hunt = Start-Process -FilePath $ti -ArgumentList @('proof-hunt','--tdat','data\nnue\packed\train_v13','--out',$out,'--engine',$Engine,'--threads',"$Threads",'--nodes',"$Nodes") -NoNewWindow -PassThru -RedirectStandardError "$out\hunt_stderr.log" -RedirectStandardOutput "$out\hunt_stdout.log"
Say "proof hunt PID $($hunt.Id)"
$deadline = (Get-Date).AddHours($Hours)
while (-not $hunt.HasExited) {
  if ((Get-Date) -gt $deadline -and -not (Test-Path "$out\STOP")) { New-Item -ItemType File "$out\STOP" | Out-Null; Say "time budget reached: STOP file created (positions in flight finish)" }
  Start-Sleep 60
}
Say "proof hunt exited (code $($hunt.ExitCode)); last lines:"; Get-Content "$out\hunt_stderr.log" -Tail 3 | ForEach-Object { Say "  $_" }
foreach ($r in $runs) {
  $stem = "$h\d1193_res_$($r[0])"
  if ((Test-Path "$stem.log") -and (Select-String -Path "$stem.log" -Pattern 'DEFERRED D1210' -Quiet)) { Remove-Item "$stem.log" } else { continue }
  Say "$($r[0]) start"
  & $lab hce-train @($r[1]) --buckets 16 --on-nnue 1 --nnue-scale 1 --lr 0.1 --threads 7 --out $stem 2>&1 | Select-String 'SATURATED' | ForEach-Object { Say ("$($r[0]): " + $_.Line) }
  if (Test-Path "${stem}_weights.json") { & $pair residual-pair --a $gb --b "${stem}_weights.json" 2>&1 | Select-String 'rows' | ForEach-Object { Say ("PAIR $($r[0]): " + $_.Line) } } else { Say "MISSING ${stem}_weights.json" }
}
Remove-Item data\gen\pc_run16_5k\HOLD -ErrorAction SilentlyContinue
Say 'HOLD removed: the d1204 queue starts self-play generation; d1210 queue end'
