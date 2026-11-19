# D1053 (user: "research what the networks lack as input / how to improve the architecture with HCE features"):
# frozen champion NNUE (812C2118) + a linear correction from ONE HCE feature group at a time (lab hce-train --on-nnue 1,
# init 0 = exact champion, C14). Rows = data/datasets/v11/selfplay/sp11_rows (15.3M rows the champion was NEVER trained
# on); --holdout 0.1 = the last 10% of the rows (whole games, file in game order) are the validation set, so train and
# val share one label distribution (val_v9's older labels differ systematically, first try D1053). --val-target 1 (the
# information, not the target gap). Every run has --nnue-scale 1 (a learned scale on the champion cp per bucket) and a
# bias, so gB (= bias + scale only) is the CALIBRATION baseline: a group's drop BELOW gB = information the net lacks.
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_ext\x86_64-pc-windows-gnu\release\lab.exe'
$tr = 'data\datasets\v11\selfplay\sp11_rows.tdat'
$va = 'data\nnue\packed\val_v9_clean.tdat'
$runs = @(
  @{ n = 'gB_calibration';    a = @('--set', '5') },
  @{ n = 'g0_gataxx_terms';   a = @('--set', '0') },
  @{ n = 'g1_funes_regions';  a = @('--set', '1') },
  @{ n = 'g2_raw_pst';        a = @('--set', '2') },
  @{ n = 'g3_ours_dynpst';    a = @('--set', '3') },
  @{ n = 'g4_pairs';          a = @('--set', '5', '--pairs', '1', '--no-mob', '1') },
  @{ n = 'g5_all';            a = @('--set', '4', '--pairs', '1', '--no-mob', '1') }
)
foreach ($r in $runs) {
  "$($r.n) start $(Get-Date -Format HH:mm)"
  & $lab hce-train --data $tr --val $va --holdout 0.1 --buckets 16 --on-nnue 1 --nnue-scale 1 --val-target 1 --lr 0.1 --threads 7 --out "logs\hce\d1053_res_$($r.n)" @($r.a) 2>&1 | Select-String 'reference|SATURATED' | ForEach-Object { $_.Line }
}
"done $(Get-Date -Format HH:mm)"
