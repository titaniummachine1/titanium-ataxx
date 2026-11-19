# D1275: confirm + ablate the D1274 SPSA theta* on FRESH openings (book0=11000, 400 pairs @5k nodes), head-refit proxy on frozen nn_v14muon.
# Every leg: A (refit with theta_A) vs B (refit with BASE_HEAD = train_v14 blend). Same openings for every leg => comparable.
Set-Location C:\gitProjects\titanium-ataxx
$lab = 'C:\gitProjects\titaniumtest\lab\target_ws\x86_64-pc-windows-gnu\release\lab.exe'
$common = @('spsa-blend','--net','C:\gitProjects\titanium-ataxx\runs\nn_v14muon\nn_v14muon\best.tnn','--train-raw','C:\gitProjects\titanium-ataxx\data\nnue\packed\train_v14h_a.tdat','--chains','C:\gitProjects\titanium-ataxx\data\nnue\packed\train_v14.chains','--n-train','400000','--val-pairs','400','--val-nodes','5000','--threads','7','--book0','11000')
$base_w = '0.55,0.55,0.55,0.55,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25'
$star_w = '0.4895,0.5289,0.5681,0.5649,0.3860,0.3888,0.3296,0.2538,0.3071,0.3367,0.2712,0.2344,0.2196,0.2100,0.1677,0.2381'
$s1_w   = '0.40,0.70,0.95,0.70,0.40,0.40,0.40,0.40,0.40,0.30,0.20,0.10,0.10,0.10,0.10,0.25'
$base_p = '0.90,0,0,0,0.50'
$star_p = '0.6549,0.6140,0.3308,0.2778,0.5929'
$legs = @(
  @('L1_star_all',      "$star_w,$star_p", '--vs-orig'),
  @('L2_star_wres_only',"$star_w,$base_p", ''),
  @('L3_star_proof_only',"$base_w,$star_p", ''),
  @('L4_stage1_wres_only',"$s1_w,$base_p", '')
)
foreach ($l in $legs) {
  "=== $($l[0]) $(Get-Date -Format HH:mm:ss) ==="
  $a = $common + @('--match-a', $l[1])
  if ($l[2] -ne '') { $a += $l[2] }
  & $lab @a 2>&1 | ForEach-Object { "$_" }
}
"=== DONE $(Get-Date -Format HH:mm:ss) ==="
