# D1126 (user order): finish the saturation of the standalone 3x3 tuple, machine entirely on it.
# Why: the lab's stop rule halves lr after every non-improving epoch and ends after 4 bad epochs, so the first run
# (d1119_t3, best 0.018500 at epoch 20, lr already ~0.008) can end because the lr died, not because the model converged.
# Test: fresh-lr restarts from the finished weights (--init, D988) until restarts stop improving; plus restarts at
# other start lrs (0.3 / 2.0) to see whether another lr reaches a lower loss. Saturated = two restarts in a row gain
# less than 1e-6 val loss. No epoch limit anywhere (LAWS C15); each run ends by the lab's own rule.
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_win\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
$common = @('--data', $tr, '--val', $va, '--holdout', '0.1', '--buckets', '16', '--val-target', '1', '--threads', '7', '--set', '5', '--win', '3')
function Run($name, $lr, $init) {
  "$name start $(Get-Date -Format HH:mm:ss) lr $lr init $init"
  $o = & $lab hce-train @common --lr $lr --init $init --out "logs\hce\$name" 2>&1
  $line = ($o | Select-String 'SATURATED').Line
  $line
  if ($line -match 'val loss ([0-9.]+)') { return [double]$Matches[1] } else { return [double]::NaN }
}
$best = 0.018500   # d1119_t3
$cur = 'logs\hce\d1119_t3_weights.json'
$noGain = 0
$i = 0
while ($noGain -lt 2) {
  $i++
  $n = "d1126_t3_r$i"
  $v = Run $n 0.5 $cur
  if ([double]::IsNaN($v)) { "run $n failed"; break }
  if ($best - $v -lt 1e-6) { $noGain++ } else { $noGain = 0 }
  if ($v -lt $best) { $best = $v; $cur = "logs\hce\${n}_weights.json" }
  "after $n : best $best (noGain $noGain)"
}
# other start lrs from the best weights
foreach ($lr in @(0.15, 2.0)) {
  $n = "d1126_t3_lr$lr"
  $v = Run $n $lr $cur
  if (-not [double]::IsNaN($v) -and $v -lt $best) { "lr $lr improved: $v (was $best)"; $best = $v; $cur = "logs\hce\${n}_weights.json" } else { "lr $lr no gain ($v)" }
}
"FINAL 3x3 saturation: best val loss $best weights $cur (first run 0.018500, champion 0.012118)"
"done $(Get-Date -Format HH:mm:ss)"
