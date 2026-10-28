"""Net2Net Channel-Merge Downscaler: W=512 -> W=256.

Preserves the learned representation of the W=512 network when targeting AVX2
systems without AVX-512 vector units.

For each twin channel i in [0..255] and i+256 in [256..511]:
  FT_down[i] = 0.5 * (FT[i] + FT[i+256])
  FTB_down[i] = 0.5 * (FTB[i] + FTB[i+256])
  L1_down[:, i] = L1[:, i] + L1[:, i+256]           (white perspective)
  L1_down[:, i+256] = L1[:, i+512] + L1[:, i+768]   (black perspective)
"""

import os
import sys
import torch

def downscale_w512_to_w256(in_pt_path: str, out_pt_path: str):
    ckpt = torch.load(in_pt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    w_old = state["ft.weight"].shape[1]
    assert w_old == 512, f"Expected input width 512, got {w_old}"

    new_state = {}
    # 1. Feature transformer
    ft_w = state["ft.weight"]
    new_ft_w = 0.5 * (ft_w[:, :256] + ft_w[:, 256:512])
    new_state["ft.weight"] = new_ft_w

    ft_b = state["ftb"]
    new_ft_b = 0.5 * (ft_b[:256] + ft_b[256:512])
    new_state["ftb"] = new_ft_b

    # 2. PSQT (identical)
    new_state["psqt.weight"] = state["psqt.weight"].clone()

    # 3. Layer 1: merge white twins and black twins
    l1_w = state["l1.weight"] # shape [128, 1024]
    # white: 0..255 and 256..511 -> new 0..255
    l1_white = l1_w[:, :256] + l1_w[:, 256:512]
    # black: 512..767 and 768..1023 -> new 256..511
    l1_black = l1_w[:, 512:768] + l1_w[:, 768:1024]
    new_l1_w = torch.cat([l1_white, l1_black], dim=1) # shape [128, 512]
    new_state["l1.weight"] = new_l1_w
    new_state["l1.bias"] = state["l1.bias"].clone()

    # 4. Layer 2 and Out (identical)
    new_state["l2.weight"] = state["l2.weight"].clone()
    new_state["l2.bias"] = state["l2.bias"].clone()
    new_state["out.weight"] = state["out.weight"].clone()
    new_state["out.bias"] = state["out.bias"].clone()

    out_ckpt = dict(ckpt)
    out_ckpt["model"] = new_state
    if "args" in out_ckpt and hasattr(out_ckpt["args"], "width"):
        out_ckpt["args"].width = 256

    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    torch.save(out_ckpt, out_pt_path)
    print(f"Successfully downscaled {in_pt_path} (W=512) -> {out_pt_path} (W=256)")
    print(f"  FT: {new_ft_w.shape}, L1: {new_l1_w.shape}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\gitProjects\titanium-ataxx\data\nnue\modal_ckpt\w512_net2net_uniform_dir\net_ep2_ema.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else r"C:\gitProjects\titanium-ataxx\data\nnue\modal_ckpt\w512_net2net_uniform_dir\downscaled_w256.pt"
    downscale_w512_to_w256(src, dst)
