@echo off
cd /d C:\gitProjects\titanium-ataxx
python training\train_stream.py --cache data\nnue\stage2_katago\stage2_katago_train.npz --val-probe data\nnue\stage2_katago\stage2_katago_val.npz --init data\nnue\tupS55tree.tup --out data\nnue\stage2_katago.tup --ckpt data\nnue\stage2_katago_ep{ep}.tup --epochs 4 --max-epochs 4 --plateau 0 --opt adamw --opt-lr 0.0002 --conv-lr 0.0002 --lookahead --lookahead-k 5 --lookahead-alpha 0.5 --tcol teacher --outcome-weight 0.5 --wdl-loss --wdl-scale 400 --freq-decay 0 --threads 4 >> logs\stage2_katago_train.log 2>&1
