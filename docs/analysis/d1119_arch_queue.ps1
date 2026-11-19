# D1119 ARCH HUNT queue (replaces the stopped D1117 waiter; same content plus the user's priority test first).
# User D1119: "try the most basic 3x3 tuple and check whether Jataxx was right".
# Waits for the D1111 stone-pair screens, then (one job at a time):
# (A) STANDALONE linear tuple evals (no champion base, no hidden layer: bias per empties bucket + D4-tied window
#     tables), same rows / held-out games / target as the residual screens (champion alone = 0.012118 there):
#     t3 = 3x3 windows, t2 = 2x2 windows, t23 = both, hce = our best HCE (set 4 + gataxx pair tables, D991);
# (B) plain alpha-beta (lab eval-gate, no margins / TT / pruning / policy / SmallNet), 400 book pairs:
#     t3 vs t2, t3 vs hce, t3 vs champion (depth 4 and 1k nodes);
# (C) window residuals on the frozen champion (--win 2 / 3 / 2,3) and the D1117 eval-gate list.
Set-Location C:\gitProjects\titanium-ataxx
$q = 'C:\gitProjects\logs\d1111_spairs.log'
while (-not ((Test-Path $q) -and (Select-String -Path $q -Pattern '^done' -Quiet))) { Start-Sleep -Seconds 60 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_win\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
$common = @('--data', $tr, '--val', $va, '--holdout', '0.1', '--buckets', '16', '--val-target', '1', '--threads', '7')
function Train($name, $extra) {
  "$name start $(Get-Date -Format HH:mm)"
  & $lab hce-train @common @extra --out "logs\hce\$name" 2>&1 | Select-String 'reference|SATURATED|agreement' | ForEach-Object { $_.Line }
}
# (A) standalone
Train 'd1119_t3' @('--set', '5', '--win', '3', '--lr', '1.0')
Train 'd1119_t2' @('--set', '5', '--win', '2', '--lr', '1.0')
Train 'd1119_t23' @('--set', '5', '--win', '2,3', '--lr', '1.0')
Train 'd1119_hce' @('--set', '4', '--pairs', '1', '--lr', '2.0')
# (B) games
$h = 'logs\hce'
$champ = 'C:\gitProjects\mainpromo-ataxx\data\nnue\nnue_main.tnn'
function Gate($name, $a, $b, $mode, $v) {
  if (-not (Test-Path $a) -or -not (Test-Path $b)) { "skip $name (missing model)"; return }
  "$name $mode $v start $(Get-Date -Format HH:mm)"
  & $lab eval-gate --a $a --b $b $mode $v --pairs 400 --threads 7 2>&1 | Select-String 'eval-gate:|FINAL' | ForEach-Object { $_.Line }
}
Gate 't3 vs t2' "$h\d1119_t3_weights.json" "$h\d1119_t2_weights.json" '--depth' 4
Gate 't3 vs hce' "$h\d1119_t3_weights.json" "$h\d1119_hce_weights.json" '--depth' 4
Gate 't3 vs champion' "$h\d1119_t3_weights.json" $champ '--depth' 4
Gate 't3 vs champion' "$h\d1119_t3_weights.json" $champ '--nodes' 1000
Gate 't23 vs champion' "$h\d1119_t23_weights.json" $champ '--depth' 4
# (C) residual windows + earlier list
foreach ($w in @('2', '3', '2,3')) { Train ('d1117_res_win' + ($w -replace ',', '')) @('--on-nnue', '1', '--nnue-scale', '1', '--lr', '0.1', '--set', '5', '--win', $w) }
$gB = "$h\d1053_res_gB_calibration_weights.json"
$new = 'data\nnue\proxy\nn_v11banneal_nocal.tnn'
Gate 'new net vs champion' $new $champ '--nodes' 1000
Gate 'new net vs champion' $new $champ '--nodes' 5000
Gate 'new net vs champion' $new $champ '--depth' 4
Gate 'g4 pairs vs gB' "$h\d1053_res_g4_pairs_weights.json" $gB '--depth' 4
Gate 'win3 res vs gB' "$h\d1117_res_win3_weights.json" $gB '--depth' 4
Gate 'sp1234 vs gB' "$h\d1111_res_sp1234_weights.json" $gB '--depth' 4
Gate 'g5 all vs gB' "$h\d1053_res_g5_all_weights.json" $gB '--depth' 4
"done $(Get-Date -Format HH:mm)"
