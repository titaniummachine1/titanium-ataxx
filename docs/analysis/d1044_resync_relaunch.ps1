# D1044 recovery: (1) import the inbox box games into the PC DAG (no generator running), (2) new DAG snapshot for the
# boxes, (3) restart the local sharded loop, (4) relaunch the Docker orchestrator. Run only while no ti_gen runs.
Set-Location C:\gitProjects\titanium-ataxx
if (Get-Process ti_gen -ErrorAction SilentlyContinue) { 'ti_gen still running: refusing'; exit 1 }
$ti = 'C:\gitProjects\titaniumtest\target_main\x86_64-pc-windows-gnu\release\ti.exe'
foreach ($g in Get-ChildItem data\gen\docker_inbox -Recurse -Filter 'batch_*.tgam') {
    if (Test-Path "$($g.FullName).imported") { continue }
    $boxd = Join-Path 'data\gen\docker_import' $g.Directory.Name
    New-Item -ItemType Directory -Force $boxd | Out-Null
    $o = Join-Path $boxd ('pcdag_' + $g.BaseName)
    $r = (& $ti dag seq-add --db data\dag\seq_dag.db --games $g.FullName --out $o --src "docker_$($g.Directory.Name)_$($g.BaseName)" 2>&1) -join ' '
    "import: $r"
    if ($LASTEXITCODE -eq 0) { Set-Content "$($g.FullName).imported" $r }
}
$st = 'data\gen\docker_stage'
Remove-Item "$st\seq_dag.tgz" -ErrorAction SilentlyContinue
Push-Location data\dag; tar -czf "..\gen\docker_stage\seq_dag.tgz" (Get-ChildItem seq_dag.db* | ForEach-Object Name); Pop-Location
"snapshot: $((Get-Item "$st\seq_dag.tgz").Length) B"
Remove-Item data\gen\pc_run12_5k\STOP -ErrorAction SilentlyContinue
"ready"
