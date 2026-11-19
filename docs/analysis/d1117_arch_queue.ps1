# D1117 ARCH HUNT queue (replaces the stopped D1113 / D1116 waiters; lab binary target_win = window tuples + fixed
# quoted-field parsing). Waits for the D1111 stone-pair screens, then:
# (A) window tuples as a residual on the frozen champion (user D1117: "maybe 2x2 tuples could replace the feature
#     collection"), D1053 protocol: --win 2 / 3 / 2,3;
# (B) plain alpha-beta (lab eval-gate, no margins / TT / pruning / policy / SmallNet), 400 book pairs, vs the
#     calibration model gB: residual models at depth 4 and at 1k nodes (D1113, D1116);
# (C) new net BB9CECE8 vs champion at 1k / 5k nodes + depth 4 (D1115: margins vs eval quality).
Set-Location C:\gitProjects\titanium-ataxx
$q = 'C:\gitProjects\logs\d1111_spairs.log'
while (-not ((Test-Path $q) -and (Select-String -Path $q -Pattern '^done' -Quiet))) { Start-Sleep -Seconds 60 }
$lab = 'C:\gitProjects\titaniumtest\lab\target_win\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
foreach ($w in @('2', '3', '2,3')) {
  $n = 'win' + ($w -replace ',', '')
  "$n start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data $tr --val $va --holdout 0.1 --buckets 16 --on-nnue 1 --nnue-scale 1 --val-target 1 --lr 0.1 --threads 7 --set 5 --win $w --out "logs\hce\d1117_res_$n" 2>&1 | Select-String 'reference|SATURATED|agreement' | ForEach-Object { $_.Line }
}
$gB = 'logs\hce\d1053_res_gB_calibration_weights.json'
$new = 'data\nnue\proxy\nn_v11banneal_nocal.tnn'
$champ = 'C:\gitProjects\mainpromo-ataxx\data\nnue\nnue_main.tnn'
$h = 'logs\hce'
$runs = @(
  @('new net vs champion', $new, $champ, '--nodes', 1000),
  @('new net vs champion', $new, $champ, '--nodes', 5000),
  @('new net vs champion', $new, $champ, '--depth', 4),
  @('g4 pairs vs gB', "$h\d1053_res_g4_pairs_weights.json", $gB, '--depth', 4),
  @('g4 pairs vs gB', "$h\d1053_res_g4_pairs_weights.json", $gB, '--nodes', 1000),
  @('win2 vs gB', "$h\d1117_res_win2_weights.json", $gB, '--depth', 4),
  @('win23 vs gB', "$h\d1117_res_win23_weights.json", $gB, '--depth', 4),
  @('sp1234 vs gB', "$h\d1111_res_sp1234_weights.json", $gB, '--depth', 4),
  @('g5 all vs gB', "$h\d1053_res_g5_all_weights.json", $gB, '--depth', 4),
  @('g3 ours vs gB', "$h\d1053_res_g3_ours_dynpst_weights.json", $gB, '--depth', 4)
)
foreach ($r in $runs) {
  if (-not (Test-Path $r[1])) { "skip $($r[0]) (no model)"; continue }
  "$($r[0]) $($r[3]) $($r[4]) start $(Get-Date -Format HH:mm)"
  & $lab eval-gate --a $r[1] --b $r[2] $r[3] $r[4] --pairs 400 --threads 7 2>&1 | Select-String 'eval-gate:|FINAL' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
