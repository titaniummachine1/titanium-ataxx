@echo off
cd /d C:\gitProjects\titanium-ataxx
python -u training\train_stream.py --cache data\nnue\final\final_train.npz --val-probe data\nnue\final\final_val.npz --init data\nnue\tupS55tree.tup --out data\nnue\final_stage2.tup --ckpt data\nnue\final_stage2_ep{ep}.tup --epochs 4 --max-epochs 4 --plateau 0 --opt adamw --opt-lr 0.0002 --conv-lr 0.0002 --lookahead --lookahead-k 5 --lookahead-alpha 0.5 --tcol teacher --outcome-weight 0.5 --outcome-col outcome_p --teacher-prob-col teacher_p --wdl-loss --wdl-scale 400 --freq-decay 0 --threads 4 >> logs\final_train_v2.log 2>&1
