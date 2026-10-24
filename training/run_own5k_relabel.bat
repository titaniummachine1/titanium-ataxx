@echo off
cd /d C:\gitProjects\titanium-ataxx
for /L %%i in (1,1,4) do python training\relabel_own5k_main.py --workers 8 --chunk 50000 >> logs\relabel_own5k_main_w8_sched.log 2>&1
