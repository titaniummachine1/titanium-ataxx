@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\optuna_stage1.py >> logs\optuna_stage1_run.log 2>&1
