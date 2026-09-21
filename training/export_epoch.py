"""Re-export a .s1 from a saved v3 checkpoint (for epoch-strength checks).

Usage: python3 training/export_epoch.py --ckpt data/nnue/v3_ckpt_e5.pt --out data/nnue/v3_e5.s1
Layout: S1N_V3 i16LE, ALL i16 direct (matches export_v3_blob in train_v3.py
and sancta.rs from_bytes — no hidden rescale).
"""
import argparse
import struct

import torch

L1 = 64
H2 = 16
NFEAT = 147
S1N_V3 = (147 * 64 + 64 + 128 * H2 + H2 + H2 + 1) * 2


def export_v3_blob(path, ft_w, ft_b, w1, b1, w2, b2):
    vals = []
    for f in range(NFEAT):
        vals += [int(max(-32768, min(32767, round(v)))) for v in ft_w[f]]
    vals += [int(max(-32768, min(32767, round(v)))) for v in ft_b]
    for i in range(2 * L1):
        vals += [int(max(-32768, min(32767, round(v)))) for v in w1[i]]
    vals += [int(max(-32768, min(32767, round(v)))) for v in b1]
    vals += [int(max(-32768, min(32767, round(v)))) for v in w2]
    vals += [int(max(-32768, min(32767, round(b2))))]
    assert len(vals) == S1N_V3 // 2, len(vals)
    with open(path, 'wb') as f:
        f.write(struct.pack('<%dh' % len(vals), *[max(-32768, min(32767, v)) for v in vals]))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ckpt', required=True)
    a.add_argument('--out', required=True)
    a = a.parse_args()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    ft_w = sd['ft.weight'].tolist()
    ft_b = sd['_bias'].tolist()
    w1t = sd['w1.weight'].t().tolist()
    b1 = sd['w1.bias'].tolist()
    w2 = sd['w2.weight'][0].tolist()
    b2 = float(sd['w2.bias'][0])
    export_v3_blob(a.out, ft_w, ft_b, w1t, b1, w2, b2)
    print('exported', a.out)


if __name__ == '__main__':
    main()
