# D1123 (user order): train the standalone 3x3 tuple NOW. Same content as d1119_arch_queue.ps1 without the wait for the
# stone-pair screens; the stopped sp4 + sp1234 screens are re-queued at the END.
# (A) STANDALONE linear tuple evals (bias per empties bucket + D4-tied window tables, no champion base), same rows /
#     held-out games / target as the residual screens (champion alone = 0.012118): t3 = 3x3, t2 = 2x2, t23 = both, hce.
# (B) plain alpha-beta games (lab eval-gate, no margins / TT / pruning / policy / SmallNet), 400 book pairs.
# (C) window residuals on the frozen champion, then the earlier eval-gate list, then the stone-pair screens sp4 / sp1234.
Set-Location C:\gitProjects\titanium-ataxx
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
Gate 'g5 all vs gB' "$h\d1053_res_g5_all_weights.json" $gB '--depth' 4
# stone-pair screens stopped by D1123
foreach ($b in @('4', '1,2,3,4')) {
  $n = 'sp' + ($b -replace ',', '')
  "$n start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data $tr --val $va --holdout 0.1 --buckets 16 --on-nnue 1 --nnue-scale 1 --val-target 1 --lr 0.1 --threads 7 --set 5 --spairs $b --out "logs\hce\d1111_res_$n" 2>&1 | Select-String 'reference|SATURATED|agreement' | ForEach-Object { $_.Line }
}
Gate 'sp1234 vs gB' "$h\d1111_res_sp1234_weights.json" $gB '--depth' 4
"done $(Get-Date -Format HH:mm)"
