# D893: retrain the policy head WITHOUT SEE, the way main plays it (pol_see_min 99 = SEE never read).
# Init = main's 380F609C; policy2 drops its SEE block on load into a --see none run (step-0 parity with play, C14).
# Same data as D744 (91 game files + Kata s3413/s1835/s1914). No epoch cap (C15): stops after 4 epochs without a Kata val gain.
Set-Location C:\gitProjects\titanium-ataxx
$ti = 'C:\gitProjects\titaniumtest\target_pol\x86_64-pc-windows-gnu\release\ti.exe'
$games = Get-Content logs\policy\d893_games_list.txt -Raw
$kata = 'data/kata_raw/s3413,data/kata_raw/s1835,data/kata_raw/s1914'
& $ti policy2 --net C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn --games $games --kata $kata --pat k3c `
  --init data/nnue/policy_store/pol_380F609C.bin --kata-w 1 --games-w 1 --threads 6 --stop-on kata `
  --out-bin nets/pol_nosee_d893.bin *> logs/policy/d893_pol_nosee.log
"exit=$LASTEXITCODE"
Get-Content logs/policy/d893_pol_nosee.log | Select-String "init|val|epoch|best|wrote|stop"
