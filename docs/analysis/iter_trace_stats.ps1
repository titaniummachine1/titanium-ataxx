# D667: STRUCTURE OF THE LAST ITERATIONS (the "final sprint" question): where does the clock go in the iterative-deepening loop?
# Runs an engine built with TI_ITER_TRACE (exp/sprint-analysis) on sampled positions with `go movetime T` and parses the per-iteration lines
#   ITER d= t0= t1= t2= done=k/n partial= changed= score= first_score= first= best=
# usage: powershell -File iter_trace_stats.ps1 -Exe <engine exe> [-Ms 300] [-N 120] [-Out report.txt]
param(
  [Parameter(Mandatory = $true)][string]$Exe, [int]$Ms = 300, [int]$N = 120, [string]$Out = "$env:TEMP\iter_trace_stats.txt",
  [string]$Games = "C:\gitProjects\mainpromo-ataxx\logs\gates\2026-11-09-ac2a8ce8-aB6C5A149-BB6C5A149-vs-0e9efb08-ms100-b40000\sprt.log.games.jsonl"
)
$pos = New-Object System.Collections.Generic.List[string]; $used = 0
foreach ($l in (Get-Content $Games -TotalCount 160)) {
  if ($pos.Count -ge $N) { break }
  $g = $l | ConvertFrom-Json; $mv = @($g.moves | ForEach-Object { $_.move })
  if (@($mv | Where-Object { $_ -notmatch '^[a-g][1-7]([a-g][1-7])?$' }).Count -gt 0) { continue }
  foreach ($p in 8, 24, 40, 56) { if ($p -lt $mv.Count -and $pos.Count -lt $N) { $pos.Add("position fen $($g.start) moves " + ($mv[0..($p - 1)] -join ' ')) } }
}
$err = "$env:TEMP\iter_trace_err.txt"; if (Test-Path $err) { Remove-Item $err }
$uai = @(); foreach ($p in $pos) { $uai += 'uainewgame'; $uai += $p; $uai += "go movetime $Ms" }; $uai += 'quit'
$env:TI_ITER_TRACE = '1'
$psi = New-Object Diagnostics.ProcessStartInfo; $psi.FileName = 'cmd.exe'; $psi.Arguments = "/c `"`"$Exe`" serve 2> `"$err`"`""; $psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
$p = [Diagnostics.Process]::Start($psi); $outTask = $p.StandardOutput.ReadToEndAsync()
foreach ($l in $uai) { $p.StandardInput.WriteLine($l) }; $p.StandardInput.Flush(); $null = $p.WaitForExit(600000)
Remove-Item Env:\TI_ITER_TRACE
# parse: a new search starts at d=1
$searches = New-Object System.Collections.Generic.List[object]; $cur = $null
foreach ($line in (Get-Content $err)) {
  if (-not ($line -match '^ITER d=(\d+) t0=([\d.]+) t1=([\d.]+) t2=([\d.]+) done=(\d+)/(\d+) partial=(\d) changed=(\d) score=(-?\d+) first_score=(-?\d+) first=(\S+) best=(\S+)')) { continue }
  $it = [pscustomobject]@{ d = [int]$matches[1]; t0 = [double]$matches[2]; t1 = [double]$matches[3]; t2 = [double]$matches[4]; done = [int]$matches[5]; n = [int]$matches[6]; partial = [int]$matches[7]; changed = [int]$matches[8]; score = [int64]$matches[9]; fscore = [int64]$matches[10]; first = $matches[11]; best = $matches[12] }
  if ($it.d -eq 1) { $cur = New-Object System.Collections.Generic.List[object]; $searches.Add($cur) }
  if ($null -ne $cur) { $cur.Add($it) }
}
$bm = @(($outTask.Result -split "`n") | Where-Object { $_ -like 'bestmove*' })
"# iteration structure $(Get-Date -Format 'yyyy-MM-dd HH:mm') exe $Exe movetime $Ms ms, $($searches.Count) searches parsed ($($bm.Count) bestmoves)" | Set-Content $Out
function W($s) { $s; Add-Content $Out $s }
# completed iterations at depth >= 8: first-move share, growth
$f = New-Object System.Collections.Generic.List[double]; $g = New-Object System.Collections.Generic.List[double]; $chg = 0; $cn = 0
foreach ($s in $searches) {
  $comp = @($s | Where-Object { $_.partial -eq 0 -and $_.d -ge 8 -and $_.t2 -gt $_.t0 })
  foreach ($it in $comp) { $f.Add(($it.t1 - $it.t0) / ($it.t2 - $it.t0)); $cn++; if ($it.changed) { $chg++ } }
  for ($i = 1; $i -lt $comp.Count; $i++) { if ($comp[$i].d -eq $comp[$i - 1].d + 1) { $g.Add(($comp[$i].t2 - $comp[$i].t0) / ($comp[$i - 1].t2 - $comp[$i - 1].t0)) } }
}
W ("completed iterations (depth >= 8): n={0}; first root move takes {1:P0} of the iteration (median {2:P0}); later moves {3:P0}; the best move CHANGES in {4:P1} of them; iteration cost growth d->d+1: median x{5:N2} (p25 x{6:N2}, p75 x{7:N2})" -f $cn, (($f | Measure-Object -Average).Average), (($f | Sort-Object)[[int]($f.Count / 2)]), (1 - ($f | Measure-Object -Average).Average), ($chg / [Math]::Max(1, $cn)), (($g | Sort-Object)[[int]($g.Count / 2)]), (($g | Sort-Object)[[int]($g.Count / 4)]), (($g | Sort-Object)[[int]($g.Count * 3 / 4)]))
# the final partial iteration
$lastp = @(); foreach ($s in $searches) { $l = $s[$s.Count - 1]; if ($l.partial -eq 1) { $lastp += $l } }
$none = @($lastp | Where-Object { $_.done -eq 0 }); $one = @($lastp | Where-Object { $_.done -eq 1 }); $many = @($lastp | Where-Object { $_.done -ge 2 })
W ("searches that end in a PARTIAL iteration: {0}/{1}. root moves finished in it: 0 -> {2}, 1 -> {3}, >=2 -> {4}" -f $lastp.Count, $searches.Count, $none.Count, $one.Count, $many.Count)
$wasteNone = ($none | ForEach-Object { ($_.t2 - $_.t0) / $Ms } | Measure-Object -Average).Average
W ("  time spent in a final partial iteration that finished NO root move (pure waste): mean {0:P1} of the move time (over those searches), overall {1:P1} of all clock" -f $wasteNone, (($none | ForEach-Object { ($_.t2 - $_.t0) / $Ms } | Measure-Object -Sum).Sum / [Math]::Max(1, $searches.Count)))
$wasteOne = ($one | ForEach-Object { ($_.t2 - $_.t0) / $Ms } | Measure-Object -Average).Average
W ("  final partial iterations with exactly 1 finished move (= the previous best re-searched deeper, the move cannot change): mean {0:P1} of the move time" -f $wasteOne)
if ($many.Count) { W ("  final partial iterations with >= 2 finished moves: the played move CHANGED in {0}/{1} = {2:P1}; they used {3:P1} of the move time" -f @($many | Where-Object { $_.changed }).Count, $many.Count, (@($many | Where-Object { $_.changed }).Count / $many.Count), (($many | ForEach-Object { ($_.t2 - $_.t0) / $Ms } | Measure-Object -Average).Average)) }
# what a sprint would need: time left when the last completed iteration ended vs the cost of the next iteration's first move
$left = New-Object System.Collections.Generic.List[double]; $need = New-Object System.Collections.Generic.List[double]
foreach ($s in $searches) { $comp = @($s | Where-Object { $_.partial -eq 0 }); if ($comp.Count -lt 2) { continue }; $lc = $comp[$comp.Count - 1]; $left.Add($Ms - $lc.t2); $need.Add(($lc.t2 - $lc.t0) * 0.8 * (($g | Sort-Object)[[int]($g.Count / 2)])) }
if ($left.Count) { W ("time left after the last COMPLETED iteration: median {0:N0} ms ({1:P0} of the move time); estimated cost of the next iteration's FIRST move (~80% of last iteration x median growth): median {2:N0} ms; first move of the next iteration would have fit in {3:P0} of the searches" -f (($left | Sort-Object)[[int]($left.Count / 2)]), ((($left | Sort-Object)[[int]($left.Count / 2)]) / $Ms), (($need | Sort-Object)[[int]($need.Count / 2)]), ((0..($left.Count - 1) | Where-Object { $left[$_] -ge $need[$_] }).Count / $left.Count)) }
