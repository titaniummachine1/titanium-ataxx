@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\build_training_db.py >> logs\master_build.log 2>&1
python training\import_deep_labels.py >> logs\master_deep.log 2>&1
python training\relabel_katago5k_main.py --workers 8 --chunk 10000 >> logs\master_relabel.log 2>&1
python training\finalize_db.py >> logs\master_finalize.log 2>&1
python training\export_training_npz.py >> logs\master_export.log 2>&1
python training\audit_dataset.py >> logs\master_audit.log 2>&1
