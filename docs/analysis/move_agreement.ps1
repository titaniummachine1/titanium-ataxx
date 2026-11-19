# D667: MOVE AGREEMENT WITH A REFERENCE SEARCH (a sharper test than games for end-of-search effort changes).
# For N sampled positions (from real games, plies spread over the game): the REFERENCE = the best move of a search with RefMs (e.g. 10x the time);
# every variant searches the same positions with Ms. Reports, per variant, how often its move equals the reference move, and a paired comparison
# against the first variant (baseline): moves where only one of the two agrees (McNemar), so a +2% agreement difference is resolvable.
# usage: powershell -File move_agreement.ps1 -Ms 100 -RefMs 1000 -N 600 -Procs 4 -Out report.txt   (variants edited in the table below)
param(
  [int]$Ms = 100, [int]$RefMs = 1000, [string]$Preset = "", [int]$N = 600, [int]$Procs = 4, [string]$Out = "$env:TEMP\move_agreement.txt",
  [string]$Games = "C:\gitProjects\mainpromo-ataxx\logs\gates\2026-11-09-ac2a8ce8-aB6C5A149-BB6C5A149-vs-0e9efb08-ms100-b40000\sprt.log.games.jsonl"
)
$variants = [ordered]@{
  'baseline (main)'   = @('C:\gitProjects\_gate\exes\titanium-81239ee5.exe', '--tt-bits 22')
  'root-sort'         = @('C:\gitProjects\sprint-ataxx\target_s2\x86_64-pc-windows-gnu\release\titanium-cli.exe', '--tt-bits 22 --root-sort')
}
if ($Preset -eq 'timefix') { $variants = [ordered]@{ 'old main 0e9efb08 (early stop)' = @('C:\gitProjects\_gate\exes\titanium-0e9efb08.exe', '--tt-bits 22'); 'new main 81239ee5 (to the deadline)' = @('C:\gitProjects\_gate\exes\titanium-81239ee5.exe', '--tt-bits 22') } }
if ($Preset -eq 'finalext') { $variants = [ordered]@{ 'baseline (main 81239ee5)' = @('C:\gitProjects\_gate\exes\titanium-81239ee5.exe', '--tt-bits 22'); 'final-ext (D668)' = @('C:\gitProjects\sprint-ataxx\target_s2\x86_64-pc-windows-gnu\release\titanium-cli.exe', '--tt-bits 22 --final-ext') } }
if ($Preset -eq 'fesweep') { $fe = 'C:\gitProjects\sprint-ataxx\target_s2\x86_64-pc-windows-gnu\release\titanium-cli.exe'; $variants = [ordered]@{ 'baseline (main)' = @('C:\gitProjects\_gate\exes\titanium-81239ee5.exe', '--tt-bits 22'); 'K2 pv2 alt1 trig2' = @($fe, '--tt-bits 22 --final-ext', @{TI_FE_K='2';TI_FE_PV='2';TI_FE_ALT='1';TI_FE_TRIG='2.0'}); 'K10 pv2 alt1 trig2' = @($fe, '--tt-bits 22 --final-ext', @{TI_FE_K='10';TI_FE_PV='2';TI_FE_ALT='1';TI_FE_TRIG='2.0'}); 'K10 pv2 alt1 trig1' = @($fe, '--tt-bits 22 --final-ext', @{TI_FE_K='10';TI_FE_PV='2';TI_FE_ALT='1';TI_FE_TRIG='1.0'}); 'K10 pv1 alt1 trig1' = @($fe, '--tt-bits 22 --final-ext', @{TI_FE_K='10';TI_FE_PV='1';TI_FE_ALT='1';TI_FE_TRIG='1.0'}); 'K2 pv2 alt1 trig0.5' = @($fe, '--tt-bits 22 --final-ext', @{TI_FE_K='2';TI_FE_PV='2';TI_FE_ALT='1';TI_FE_TRIG='0.5'}) } }
$refExe = @('C:\gitProjects\_gate\exes\titanium-81239ee5.exe', '--tt-bits 22')
$pos = New-Object System.Collections.Generic.List[string]
foreach ($l in (Get-Content $Games -TotalCount 400)) {
  if ($pos.Count -ge $N) { break }
  $g = $l | ConvertFrom-Json; $mv = @($g.moves | ForEach-Object { $_.move })
  if (@($mv | Where-Object { $_ -notmatch '^[a-g][1-7]([a-g][1-7])?$' }).Count -gt 0) { continue }
  foreach ($p in 8, 20, 32, 44, 56, 68) { if ($p -lt $mv.Count -and $pos.Count -lt $N) { $pos.Add("position fen $($g.start) moves " + ($mv[0..($p - 1)] -join ' ')) } }
}
$n = $pos.Count
$chunkFiles = @(); $chunkIds = @()
function Build($ms) {
  $script:chunkFiles = @(); $script:chunkIds = @()
  for ($c = 0; $c -lt $Procs; $c++) {
    $txt = New-Object System.Collections.Generic.List[string]; $ids = New-Object System.Collections.Generic.List[int]
    for ($i = $c; $i -lt $n; $i += $Procs) { $txt.Add('uainewgame'); $txt.Add($pos[$i]); $txt.Add("go movetime $ms"); $ids.Add($i) }
    $txt.Add('quit'); $f = "$env:TEMP\ma_chunk_${ms}_$c.uai"; $txt | Set-Content $f -Encoding ascii; $script:chunkFiles += $f; $script:chunkIds += ,@($ids)
  }
}
$worker = { param($exe, $argstr, $file) $a = @('serve') + ($argstr -split ' '); $r = New-Object System.Collections.Generic.List[string]; foreach ($line in (Get-Content $file | & $exe @a 2>$null)) { if ($line -like 'bestmove*') { $r.Add($line.Split(' ')[1]) } }; return ,$r }
function RunAll($exe, $argstr) {
  $jobs = for ($c = 0; $c -lt $Procs; $c++) { Start-Job -ScriptBlock $worker -ArgumentList $exe, $argstr, $chunkFiles[$c] }
  $null = $jobs | Wait-Job; $res = New-Object string[] $n
  for ($c = 0; $c -lt $Procs; $c++) { $r = @($jobs[$c] | Receive-Job); if ($r.Count -eq 1 -and $r[0] -is [System.Collections.IEnumerable] -and $r[0] -isnot [string]) { $r = @($r[0]) }; for ($i = 0; $i -lt [Math]::Min($r.Count, $chunkIds[$c].Count); $i++) { $res[$chunkIds[$c][$i]] = $r[$i] } }
  $jobs | Remove-Job -Force; return ,$res
}
"# move agreement $(Get-Date -Format 'yyyy-MM-dd HH:mm'): $n positions, variants at $Ms ms vs reference at $RefMs ms, $Procs processes" | Set-Content $Out
$refCache = "$env:TEMP\ma_ref_${n}_${RefMs}_$($pos[0].GetHashCode()).txt"; if (Test-Path $refCache) { $ref = @(Get-Content $refCache); "reference loaded from cache" } else { Build $RefMs; $ref = RunAll $refExe[0] $refExe[1]; $ref | Set-Content $refCache; "reference done ($(@($ref | Where-Object { $_ }).Count)/$n)" }
Build $Ms
$moves = @{}; foreach ($k in $variants.Keys) { $envs = if ($variants[$k].Count -gt 2) { $variants[$k][2] } else { @{} }; foreach ($e in $envs.Keys) { Set-Item "Env:$e" $envs[$e] }; $moves[$k] = RunAll $variants[$k][0] $variants[$k][1]; foreach ($e in $envs.Keys) { Remove-Item "Env:$e" -ErrorAction SilentlyContinue }; "ran $k" }
$keys = @($variants.Keys); $base = $keys[0]
foreach ($k in $keys) {
  $agree = 0; $tot = 0; $b10 = 0; $b01 = 0
  for ($i = 0; $i -lt $n; $i++) { if (-not $ref[$i] -or -not $moves[$k][$i] -or -not $moves[$base][$i]) { continue }; $tot++; $a = $moves[$k][$i] -eq $ref[$i]; $b = $moves[$base][$i] -eq $ref[$i]; if ($a) { $agree++ }; if ($a -and -not $b) { $b10++ }; if ($b -and -not $a) { $b01++ } }
  $z = if (($b10 + $b01) -gt 0) { ($b10 - $b01) / [Math]::Sqrt($b10 + $b01) } else { 0 }
  $line = "{0,-18} agreement with the {1} ms reference: {2}/{3} = {4:P1} | vs baseline: only-this-agrees {5}, only-baseline-agrees {6}, McNemar z {7:+0.00;-0.00}" -f $k, $RefMs, $agree, $tot, ($agree / [Math]::Max(1, $tot)), $b10, $b01, $z
  $line; Add-Content $Out $line
}
