@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\relabel_katago5k_main.py --workers 4 --chunk 10000 >> logs\relabel_db_final.log 2>&1
