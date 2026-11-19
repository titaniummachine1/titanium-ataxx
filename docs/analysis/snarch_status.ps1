# D840: one-shot status of the SmallNet architecture research (PC run 3 + Docker boxes). Prints the last epoch line and any
# plateau / SATURATED / DONE lines per design.
param([string]$PcLog = 'C:\gitProjects\titanium-ataxx\logs\snarch\d850_allin.log')
$sbx = "$env:LOCALAPPDATA\DockerSandboxes\bin\sbx.exe"
Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"== PC ($PcLog)"
$l = Get-Content $PcLog -ErrorAction SilentlyContinue
$l | Select-String 'plateau|SATURATED|DONE|WARNING' | ForEach-Object Line
$l | Select-Object -Last 1
foreach ($n in 'tit-sna', 'tit-snb', 'tit-snc') {
    "== $n"
    & $sbx --cloud exec $n bash -c "cd /tmp/sn/out 2>/dev/null && cat status.txt; for f in snarch_*.log; do grep -E 'plateau|SATURATED|DONE|WARNING' `$f; tail -1 `$f; done" 2>$null
}
