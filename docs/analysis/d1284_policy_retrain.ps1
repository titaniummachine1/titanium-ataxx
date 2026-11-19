# D1284 Step 3: Saturation retrain of TPOL3 no-SEE policy head (`ti policy2 --stop-on sum --patience 4`)
$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\gitProjects\toolchains\llvm-mingw-20261006-ucrt-x86_64\bin;' + $env:PATH
$root = 'C:\gitProjects\titanium-ataxx'
Set-Location $root

New-Item -ItemType Directory -Force -Path "$root\logs\policy", "$root\data\nnue\policy_store" | Out-Null

$files = New-Object System.Collections.Generic.List[string]
# 1. pre-D806 merged runs
$files.Add('data/gen/bin/all_runs_D806.tgam')
# 2. v11 selfplay_d921 (gen_nov16 + pc_run9/10)
Get-ChildItem "$root\data\datasets\v11\selfplay_d921\games" -Recurse -Filter '*.tgam' | ForEach-Object {
  $files.Add($_.FullName.Substring($root.Length + 1).Replace('\', '/'))
}
# 3. d1049_merge_files.txt (pc_run11..13 + docker_import tit-gen1/2)
Get-Content "$root\data\gen\d1049_merge_files.txt" | Where-Object { $_.Trim() -ne '' } | ForEach-Object {
  $files.Add($_.Trim().Substring($root.Length + 1).Replace('\', '/'))
}
# 4. pc_run15_5k
Get-ChildItem "$root\data\gen\pc_run15_5k" -Filter '*.tgam' | ForEach-Object {
  $files.Add($_.FullName.Substring($root.Length + 1).Replace('\', '/'))
}
# 5. gate runs (exclude reports/*pooled*)
Get-ChildItem "$root\logs\gates" -Directory | Where-Object { $_.Name -ne 'reports' } | ForEach-Object {
  Get-ChildItem $_.FullName -Filter '*.tgam' | ForEach-Object {
    if (Test-Path ($_.FullName -replace '\.tgam$', '.tbst')) {
      $files.Add($_.FullName.Substring($root.Length + 1).Replace('\', '/'))
    }
  }
}

$unique = @($files | Select-Object -Unique)
$missing = @($unique | Where-Object {
  -not (Test-Path "$root\$_") -or -not (Test-Path ("$root\$_" -replace '\.tgam$', '.tbst'))
})
$games = $unique -join ','
Write-Host ("Unique .tgam+.tbst files: {0}, Missing: {1}, Arg length: {2} chars" -f $unique.Count, $missing.Count, $games.Length)
if ($missing.Count -gt 0) {
  throw "Missing files or .tbst companions: $($missing -join '; ')"
}
if ($games.Length -ge 30000) {
  throw "Games arg length $($games.Length) >= 30000 Windows CreateProcess limit!"
}
Set-Content -Path "$root\logs\policy\d1284_games_list.txt" -Value $games -NoNewline

$ErrorActionPreference = 'Continue'
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
$log = "$root\logs\policy\d1284_pol_v15.log"
"=== Step 3: ti policy2 start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ($($unique.Count) files) ===" | Tee-Object -FilePath $log
& $ti policy2 `
  --net data/nnue/proxy/nn_v14muon_nocal_keepdis.tnn `
  --games $games `
  --kata data/kata_raw/s3413,data/kata_raw/s1835,data/kata_raw/s1914 `
  --pat k3c `
  --init data/nnue/policy_store/pol_380F609C.bin `
  --kata-w 1 --games-w 1 --stop-on sum --patience 4 --threads 6 `
  --out-bin data/nnue/policy_store/pol_v15_d1284.bin *>&1 | Tee-Object -FilePath $log -Append
$ec = $LASTEXITCODE
"policy2 exit=$ec at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $log -Append
exit $ec
