# Production policy-head recipe (D744: 91 game logs + 3 Kata shards, --pat k3c --see 2d8, stop on the Kata loss),
# with optional extra flags. Used for D772 (plain head re-scored, advantage-weighted head trained).
#   powershell -NoProfile -File docs\analysis\policy_full_recipe.ps1 -Name adv09 -Extra "--adv-lambda 0.9"
#   powershell -NoProfile -File docs\analysis\policy_full_recipe.ps1 -Name base380_eval -Init data/nnue/policy_store/pol_380F609C.bin -Epochs 0
# Output: logs\policy_adv\pol_<Name>.bin and logs\policy_adv\d772_<Name>.log. The game-log list is the one the
# D744 run used, saved in logs\policy_adv\d744_games_list.txt.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Extra = '',
    [string]$Init = 'data/nnue/policy_store/pol_EE96825B.bin',
    [int]$Epochs = 10
)
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target\x86_64-pc-windows-gnu\release\ti.exe'
$games = (Get-Content logs\policy_adv\d744_games_list.txt) -join ','
$kata = 'data/kata_raw/s3413,data/kata_raw/s1835,data/kata_raw/s1914'
$log = "logs\policy_adv\d772_$Name.log"
"start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  name $Name  init $Init  epochs $Epochs  extra '$Extra'" | Out-File $log -Encoding ascii
cmd /c "`"$ti`" policy2 --net data/nnue/nnue_main.tnn --games $games --kata $kata --pat k3c --see 2d8 --init $Init --kata-w 1 --games-w 1 --max-epochs $Epochs --patience 1 --threads 6 --stop-on kata --out-bin logs/policy_adv/pol_$Name.bin $Extra >> `"$log`" 2>&1"
"end $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  exit $LASTEXITCODE" | Out-File $log -Append -Encoding ascii
Get-Content $log | Select-String -Pattern 'advantage weights|init from|val init|^epoch|best epoch|wrote|early stop|^end ' | ForEach-Object { $_.Line }
