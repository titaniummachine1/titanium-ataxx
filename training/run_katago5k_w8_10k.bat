@echo off
cd /d C:\gitProjects\titanium-ataxx
for /L %%i in (1,1,2) do python training\relabel_katago5k_main.py --workers 8 --chunk 10000 >> logs\relabel_katago5k_w8_10k_sched.log 2>&1
