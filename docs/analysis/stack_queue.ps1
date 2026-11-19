# D595 (user: accumulate small Elo gains like Stockfish, never park them): extension blocks of 600 games at 5k nodes on
# fresh book lines per candidate, pentanomial pooling with the earlier blocks, stop at P(better) >= 95% (ACCEPT -> stack)
# or <= 5% (OUT) or the game cap; the accepted knob lines are merged into data/knobs/stack_current.txt at the end.
# Gates only through the hub (ti gate on a commit, B14). Run when no timed gate runs.
# usage: powershell -File docs\analysis\stack_queue.ps1
$ti = "C:/gitProjects/titaniumtest/target_tt/release/ti.exe"
$commit = "db46165"
$kn = "C:/gitProjects/titanium-ataxx/data/knobs"
$log = "C:\gitProjects\titanium-ataxx\logs\gates\d595_stack_queue.log"
Set-Location C:\gitProjects\titanium-ataxx
# wait for the running timed gate (D594) so the time leg stays on an idle machine
while (-not (Select-String -Path logs\gates\d594_caches_1000ms.out -Pattern "GATE RESULT" -Quiet)) { Start-Sleep 30 }
# name, knob file, earlier block outputs to pool (same candidate, same base = main), next book0
$cands = @(
    @{ n = "B_lmr_b0"; f = "knobs_E9719893.txt"; prior = @("logs\gates\d561_lmrB_5k.out"); book0 = 300 },
    @{ n = "A2_nmp_b9_11"; f = ""; prior = @(); book0 = 0 },
    @{ n = "E_killcanon"; f = "knobs_C4B7649B.txt"; prior = @("logs\gates\d565_killE_5k.out"); book0 = 300 }
)
# A' = null margin 200 only where the per-bucket insight showed the gain (b9-b11, D578)
$a2 = "# D595 A': null-move margin 200 only in NNUE buckets 9-11 (per-bucket gain, D578)`nnmp_margin = -1000000 -1000000 -1000000 -1000000 -1000000 -1000000 -1000000 -1000000 -1000000 200 200 200 -1000000 -1000000 -1000000 -1000000`n"
$tmp = "$kn/tmp.txt"; [IO.File]::WriteAllText($tmp, $a2); $h = (Get-FileHash $tmp -Algorithm SHA256).Hash.Substring(0, 8)
Move-Item $tmp "$kn/knobs_$h.txt" -Force; $cands[1].f = "knobs_$h.txt"

function Penta($file) {
    $t = Get-Content $file -ErrorAction SilentlyContinue | Select-String "^Penta : LL=(\d+) LD=(\d+) DD\+WL=(\d+) WD=(\d+) WW=(\d+)" | Select-Object -Last 1
    if (-not $t) { return $null }
    return @(1..5 | ForEach-Object { [int]$t.Matches[0].Groups[$_].Value })
}
function Pool($files) {
    $p = @(0, 0, 0, 0, 0)
    foreach ($f in $files) { $x = Penta $f; if ($x) { for ($i = 0; $i -lt 5; $i++) { $p[$i] += $x[$i] } } }
    $n = ($p | Measure-Object -Sum).Sum
    if ($n -lt 10) { return @{ n = $n; P = 0.5; elo = 0 } }
    $w = @(0, 0.25, 0.5, 0.75, 1.0)
    $m = 0; $m2 = 0; for ($i = 0; $i -lt 5; $i++) { $m += $p[$i] * $w[$i] / $n; $m2 += $p[$i] * $w[$i] * $w[$i] / $n }
    $se = [Math]::Sqrt([Math]::Max(1e-9, $m2 - $m * $m) / $n)
    $z = ($m - 0.5) / $se
    $P = 0.5 * (1 + [Math]::Sign($z) * [Math]::Sqrt(1 - [Math]::Exp(-2 * $z * $z / [Math]::PI)))   # normal CDF approx
    $elo = -400 * [Math]::Log10(1 / [Math]::Min(0.999, [Math]::Max(0.001, $m)) - 1)
    return @{ n = 2 * $n; P = $P; elo = $elo; penta = ($p -join ",") }
}
"# stack queue $(Get-Date -Format 'yyyy-MM-dd HH:mm') commit $commit" | Add-Content $log
$accepted = @()
foreach ($c in $cands) {
    $files = @($c.prior)
    $b0 = $c.book0
    while ($true) {
        $r = Pool $files
        "$($c.n): pooled $($r.n) g, penta [$($r.penta)], Elo {0:N1}, P(better) {1:N3}" -f $r.elo, $r.P | Tee-Object -FilePath $log -Append
        if ($r.P -ge 0.95) { "$($c.n): ACCEPT -> stack" | Tee-Object -FilePath $log -Append; $accepted += $c; break }
        if ($r.n -ge 600 -and $r.P -le 0.05) { "$($c.n): OUT (P <= 5%)" | Tee-Object -FilePath $log -Append; break }
        if ($r.n -ge 3000) { "$($c.n): CAP 3000 games, undecided -> report to the user" | Tee-Object -FilePath $log -Append; break }
        $out = "logs\gates\d595_$($c.n)_b$b0.out"
        & $ti gate --test $commit --base main --test-args "--knobs $kn/$($c.f)" --kind search --nodes 5000 --threads 4 --max-games 600 --book0 $b0 --allow-busy *> $out
        $files += $out
        $b0 += 300
    }
}
# merge accepted knob lines (each candidate touches its own knob) into the stack file
$stack = "# D595 stack (accepted at P >= 95% vs main, 5k nodes): $(($accepted | ForEach-Object { $_.n }) -join ', ')`n"
foreach ($c in $accepted) { $stack += (Get-Content "$kn/$($c.f)" | Where-Object { $_ -notmatch "^#" }) -join "`n"; $stack += "`n" }
[IO.File]::WriteAllText("$kn/stack_current.txt", $stack)
"stack written: $kn/stack_current.txt ($(($accepted | ForEach-Object { $_.n }) -join ', '))" | Tee-Object -FilePath $log -Append
"done" | Set-Content logs\gates\d595_queue.done
