# Unattended self-play generation on this PC (D775; D777: default depth = 5,000 nodes per move, the dataset's label depth). Resumable: start it again and it continues with the next
# batch number. Stop it cleanly by creating the file  <RunDir>\STOP  (D1174: ti gen sees it before every new game: no new game starts, games in flight finish and are written, exit 0; the loop then leaves), or kill
# the process "ti_gen" (the batch file keeps every game finished so far).
#
#   powershell -NoProfile -File docs\analysis\gen_pc_loop.ps1 [-RunDir data\gen\pc_run1] [-Batches 60]
#       [-Games 1000] [-TimeMs 100] [-Threads 4] [-Novelty 8]
#
# Each batch: ti gen -> <RunDir>\batch_NNNN.tgam + batch_NNNN.tbst (BINARY only, D806: no JSON for game data; .tgam = standard
# game file, .tbst = engine best move + score per ply + DAG game id), every game also written to the sequence DAG
# (data\dag\seq_dag.db) so later games avoid sequences already played.
# Check a batch:  ti dag tbst-check --db data\dag\seq_dag.db --tgam <RunDir>\batch_NNNN.tgam
param(
    [string]$RunDir = 'data\gen\pc_run2_5k',
    [int]$Batches = 60,
    [int]$Games = 2000,
    [int]$Nodes = 5000,
    [string]$Book = '',   # default: NO book, pure novelty from the standard start (user D787); a path here is an explicit override
    [int]$TimeMs = 100,
    [int]$Threads = 0,
    [string]$Dag = 'data\dag\seq_dag.db',   # the SEQUENCE DAG (D804); the old position DAGs selfplay_dag.db / games_dag.db are never touched
    [string]$GateRoots = 'C:\gitProjects\mainpromo-ataxx\logs\gates,C:\gitProjects\titanium-ataxx\logs\gates',   # gate logs imported before each batch (D830)
    [string]$GenArgs = '',   # extra ti gen arguments, e.g. '--shard 0/3' while Docker boxes play the other shards (D1041)
    [string]$Inbox = 'data\gen\docker_inbox'   # D1041 auto-sync: finished box batches pulled here are imported into the DAG before each batch
)
Set-Location C:\gitProjects\titanium-ataxx
New-Item -ItemType Directory -Force $RunDir | Out-Null
$log = Join-Path $RunDir 'run.log'
function Write-Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Out-File $log -Append -Encoding ascii }
# D1210: <RunDir>\HOLD = wait before the first batch (another long PC job, e.g. the proof hunt, owns the CPU; one long job per machine).
if (Test-Path (Join-Path $RunDir 'HOLD')) { Write-Log 'HOLD file present: waiting (D1210)'; while (Test-Path (Join-Path $RunDir 'HOLD')) { Start-Sleep 60 }; Write-Log 'HOLD removed: starting' }

# Keep the PC awake while this script runs (a request by this process only; it ends with the process).
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483649)   # 0x80000001 = ES_CONTINUOUS | ES_SYSTEM_REQUIRED (PS 5.1 reads the hex literal as a negative Int32)

# A fixed copy of the hub binary: later rebuilds of ti.exe cannot disturb the run.
$exe = Join-Path $RunDir 'ti_gen.exe'
if (-not (Test-Path $exe)) {
    # canonical hub binary (D930: target_main; the old target\ copy was built 3 h earlier, 2026-11-16 15:35 vs 18:23)
    Copy-Item 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe' $exe
    Write-Log ("binary copied, sha256 " + (Get-FileHash $exe -Algorithm SHA256).Hash)
}
$extra = ''; $bookOn = ($Book -ne '') -and (Test-Path $Book); if ($Threads -gt 0) { $extra += " --threads $Threads" }
if ($GenArgs) { $extra += " $GenArgs" }
$limit = if ($Nodes -gt 0) { "--nodes $Nodes" } else { "--time-ms $TimeMs" }
Write-Log "loop start: batches $Batches, games/batch $Games, limit $limit per move, threads $Threads (0 = auto), dag $Dag"
$fails = 0
for ($i = 0; $i -lt $Batches; $i++) {
    if (Test-Path (Join-Path $RunDir 'STOP')) { Write-Log 'STOP file found, leaving'; break }
    $free = [int]((Get-PSDrive C).Free / 1GB)
    if ($free -lt 20) { Write-Log "only $free GB free on C:, leaving"; break }
    $n = 1 + (Get-ChildItem $RunDir -Filter 'batch_*.tgam' | Measure-Object).Count
    $name = 'batch_{0:D4}' -f $n
    while (Test-Path (Join-Path $RunDir "$name.tgam")) { $n++; $name = 'batch_{0:D4}' -f $n }
    $out = Join-Path $RunDir $name   # file stem: ti gen writes $out.tgam + $out.tbst (BINARY only, D806: no JSON for game data)
    # D830: gate games join the dataset too: import every gate log not yet imported (binary .tgam/.tbst + sequence DAG)
    # while no ti_gen runs (one DAG writer at a time); a gate that ran beside generation is picked up here
    $gi = & $exe dag gate-import --db $Dag --roots $GateRoots 2>&1 | Out-String
    Write-Log ("gate import: " + $gi.Trim())
    # D1041 auto-sync (user): games made on Docker boxes join this DAG between batches (one DAG writer at a time).
    # Each inbox file pair BOX/batch_N.tgam+.tbst is read only; new sequences go to data\gen\docker_import\BOX\pcdag_batch_N
    # with this DAG's game ids; a marker file .imported prevents a second import.
    if ($Inbox -and (Test-Path $Inbox)) {
        foreach ($g in Get-ChildItem $Inbox -Recurse -Filter 'batch_*.tgam') {
            if (Test-Path "$($g.FullName).imported") { continue }
            if (-not (Test-Path ($g.FullName -replace '\.tgam$', '.tbst'))) { continue }
            $boxd = Join-Path 'data\gen\docker_import' $g.Directory.Name
            New-Item -ItemType Directory -Force $boxd | Out-Null
            $o = Join-Path $boxd ('pcdag_' + $g.BaseName)
            $r = & $exe dag seq-add --db $Dag --games $g.FullName --out $o --src "docker_$($g.Directory.Name)_$($g.BaseName)" 2>&1 | Out-String
            Write-Log ("box import: " + $r.Trim())
            if ($LASTEXITCODE -eq 0) { Set-Content "$($g.FullName).imported" $r.Trim() }
        }
    }
    $bk = ''; if ($bookOn) { $bk = " --book ""$Book"" --book0 $((($n - 1) * $Games) % 50000)" }
    Write-Log "$name start"
    cmd /c "`"$exe`" gen --out `"$out`" --games $Games $limit$extra$bk --seq-dag `"$Dag`" --src $(Split-Path $RunDir -Leaf)_$name >> `"$(Join-Path $RunDir "$name.log")`" 2>&1"
    $code = $LASTEXITCODE
    $tot = 0; if (Test-Path "$out.tgam") { $tot = [BitConverter]::ToUInt64([byte[]](Get-Content "$out.tgam" -Encoding Byte -TotalCount 16), 8) }   # game count in the .tgam header
    Write-Log "$name end: exit $code, $tot games in the file"
    # D1041: one failed batch (e.g. a transient DAG lock) must not end an unattended night; the finished games stay in
    # the batch file. Leave only after 3 failures in a row.
    if ($code -ne 0) {
        $fails++
        if ($fails -ge 3) { Write-Log 'generator failed 3 times in a row, leaving (see the batch logs)'; break }
        Write-Log "generator failed ($fails in a row), next batch in 30 s (see the batch log)"; Start-Sleep 30; continue
    }
    $fails = 0
    if (Test-Path (Join-Path $RunDir 'STOP')) { Write-Log "STOP file found after $name ($tot games kept), leaving"; break }   # D1174 clean stop
    # Guard (D797/D804): only NEW sequences are written to the files (a repeated sequence only updates the DAG counts), so under 90% of the requested games means the generator replays known lines.
    if ($tot -lt 0.9 * $Games) { Write-Log "REPEAT GUARD: only $tot of $Games requested games were new sequences, leaving"; break }
}
Write-Log 'loop end'
[void][Win.Power]::SetThreadExecutionState([uint32]2147483648)   # 0x80000000 = ES_CONTINUOUS: release the request





