@echo off
cd /d C:\gitProjects\titanium-ataxx
for /L %%i in (1,1,8) do python training\relabel_katago5k_main.py --workers 4 --chunk 10000 >> logs\relabel_katago5k_main_w4_sched.log 2>&1
