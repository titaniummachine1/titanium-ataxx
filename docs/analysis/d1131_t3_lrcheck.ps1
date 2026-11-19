# D1131: last step of the 3x3 standard saturation check. d1130_t3_std.ps1 had a bug (Run returned its log lines with
# the value, so the restart loop could never stop); by the logs the restart rule was already met: r1 and r2 both gained
# exactly 0 over run 1 (best epoch 0, val 0.019011). This runs the two start-lr checks from the run-1 weights, under the
# lab standard (guarded binary, no data flags). Values go to the log with Add-Content; Run returns only the number.
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
$log = 'C:\gitProjects\logs\d1131_t3_lrcheck.log'
$common = @('--buckets', '16', '--threads', '7', '--set', '5', '--win', '3')
function Say($s) { Add-Content -Path $log -Value $s -Encoding utf8 }
function Run($name, $lr, $init) {
  Say "$name start $(Get-Date -Format HH:mm:ss) lr $lr init $init"
  $o = & $lab hce-train @common --lr $lr --init $init --out "logs\hce\$name" 2>&1
  foreach ($x in ($o | Select-String '^STANDARD|^EXPERIMENT|refused|SATURATED')) { Say $x.Line }
  $line = ($o | Select-String 'SATURATED').Line
  if ($line -match 'val loss ([0-9.]+)') { return [double]$Matches[1] }
  return [double]::NaN
}
[double]$best = 0.019011
$cur = 'logs\hce\d1130_t3_weights.json'
foreach ($lr in @(0.15, 2.0)) {
  $n = "d1131_t3_lr$lr"
  [double]$v = Run $n $lr $cur
  if (-not [double]::IsNaN($v) -and $v -lt $best - 1e-6) { Say "lr $lr improved: $v (was $best)"; $best = $v; $cur = "logs\hce\${n}_weights.json" } else { Say "lr $lr no gain ($v vs $best)" }
}
Say "FINAL 3x3 standard saturation: best val loss $best weights $cur (champion 0.012118)"
Say "done $(Get-Date -Format HH:mm:ss)"
