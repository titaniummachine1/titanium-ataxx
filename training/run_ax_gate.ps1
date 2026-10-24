# Gate: Titanium + Ataxx_CPP transformer eval (exp/ataxxcpp-eval) vs the SAME exe with another net.
# Both sides = identical exe; only the eval differs. Paired book, 50/50 seats per worker.
# Candidate = this process (AXNET_DIR set). Opponent child = "<exe> serve" (match strips AXNET_DIR for it).
param(
    [int]$Workers = 8,
    [int]$GamesPerWorker = 14,
    [int]$Nodes = 5000,
    [string]$Tag = "ax5k",
    [string]$OppTuple = "data\nnue\final_v2_wdl_ep8.tup",
    [string]$Scale = "800",
    [string]$Exe = "C:\gitProjects\ataxxeval-ataxx\target\release\titanium-cli.exe",
    [int]$BookBase = 0,
    [string]$Mat = "0"
)
$ErrorActionPreference = "Stop"
$root = "C:\gitProjects\titanium-ataxx"
$net  = "data\nnue\final_v2_wdl_ep8.tup"
Set-Location $root
$env:AXNET_DIR = "$root\data\engines\ataxx_cpp\weights"
$env:AXNET_SCALE = $Scale
$env:AXNET_MAT = $Mat
"exe sha  : " + (Get-FileHash $Exe -Algorithm SHA256).Hash.Substring(0,8)
"S57 net  : " + (Get-FileHash $net -Algorithm SHA256).Hash.Substring(0,8)
"opp tuple: $OppTuple   AXNET_SCALE: $Scale   nodes: $Nodes   tag: $Tag"
$lines = [int]($GamesPerWorker / 2)
$procs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $start = $BookBase + $i * $lines
    $out = "logs\gate_${Tag}_p$i.txt"
    $a = @("match","--games",$GamesPerWorker,"--nodes",$Nodes,"--tuple",$net,
           "--opp",('"' + $Exe + ' serve"'),"--opp-tuple",$OppTuple,"--opp-nodes",$Nodes,
           "--book","data\nnue\book50k.txt","--book-start",$start,"--start-game",1,"--out",$out)
    $procs += Start-Process -FilePath $Exe -ArgumentList $a -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "logs\gate_${Tag}_p$i.stdout" -RedirectStandardError "logs\gate_${Tag}_p$i.stderr"
}
"launched " + $procs.Count + " workers: " + (($procs | ForEach-Object Id) -join ",")
