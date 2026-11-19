# D1130: the standalone 3x3 tuple under the lab STANDARD (lab/STANDARD_TRAIN.txt: whole train_v11, 53,004,489 rows,
# last 1,528,091 held out, champion target). The guarded binary refuses any data / target flag, so this script passes
# none. Run 1 = standard init (C14), lr 1.0; then fresh-lr restarts from the best weights (--init) until two restarts in
# a row gain < 1e-6 val loss; then start lrs 0.15 and 2.0. No epoch limit (LAWS C15).
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_std\x86_64-pc-windows-gnu\release\lab.exe'
$common = @('--buckets', '16', '--threads', '7', '--set', '5', '--win', '3')
function Run($name, $lr, $init) {
  "$name start $(Get-Date -Format HH:mm:ss) lr $lr init $init"
  if ($init) { $o = & $lab hce-train @common --lr $lr --init $init --out "logs\hce\$name" 2>&1 } else { $o = & $lab hce-train @common --lr $lr --out "logs\hce\$name" 2>&1 }
  ($o | Select-String '^STANDARD|^EXPERIMENT|refused|is not the standard').Line
  $line = ($o | Select-String 'SATURATED').Line
  $line
  if ($line -match 'val loss ([0-9.]+)') { return [double]$Matches[1] } else { return [double]::NaN }
}
$best = Run 'd1130_t3' 1.0 $null
if ([double]::IsNaN($best)) { "first run failed"; exit 1 }
$cur = 'logs\hce\d1130_t3_weights.json'
$noGain = 0; $i = 0
while ($noGain -lt 2) {
  $i++
  $n = "d1130_t3_r$i"
  $v = Run $n 0.5 $cur
  if ([double]::IsNaN($v)) { "run $n failed"; break }
  if ($best - $v -lt 1e-6) { $noGain++ } else { $noGain = 0 }
  if ($v -lt $best) { $best = $v; $cur = "logs\hce\${n}_weights.json" }
  "after $n : best $best (noGain $noGain)"
}
foreach ($lr in @(0.15, 2.0)) {
  $n = "d1130_t3_lr$lr"
  $v = Run $n $lr $cur
  if (-not [double]::IsNaN($v) -and $v -lt $best) { "lr $lr improved: $v (was $best)"; $best = $v; $cur = "logs\hce\${n}_weights.json" } else { "lr $lr no gain ($v)" }
}
"FINAL 3x3 standard saturation: best val loss $best weights $cur (champion 0.012118)"
"done $(Get-Date -Format HH:mm:ss)"
