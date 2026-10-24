@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\finalize_db.py >> logs\finalize_final.log 2>&1
python training\export_training_npz.py >> logs\export_final.log 2>&1
python training\audit_dataset.py >> logs\audit_final.log 2>&1
