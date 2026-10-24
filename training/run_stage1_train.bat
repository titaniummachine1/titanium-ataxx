@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\train_stream.py --cache data\nnue\stage1_base_train.npz --val-probe data\nnue\stage1_base_val.npz --init data\nnue\tupS55tree.tup --out data\nnue\stage1_main.tup --ckpt data\nnue\stage1_ep{ep}.tup --epochs 6 --max-epochs 8 --opt adamw --opt-lr 0.0002 --conv-lr 0.0002 --tcol teacher --outcome-weight 0.25 --threads 4 >> logs\stage1_train.log 2>&1
