@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\import_outcomes_db.py >> logs\outcomes_import.log 2>&1
