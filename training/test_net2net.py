import torch
import torch.nn.functional as F

p = 'C:/gitProjects/titanium-ataxx/data/nnue/modal_ckpt/s1c_hg/net.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
m_old = ckpt['model']

# Build m_new via Net2Net
m_new = {}
m_new['ftb'] = torch.cat([m_old['ftb'], m_old['ftb']], dim=0)
m_new['ft.weight'] = torch.cat([m_old['ft.weight'], m_old['ft.weight']], dim=1)
m_new['psqt.weight'] = m_old['psqt.weight'].clone()

W_l1_old = m_old['l1.weight']
W0 = W_l1_old[:, :256] * 0.5
W1 = W_l1_old[:, 256:] * 0.5
m_new['l1.weight'] = torch.cat([W0, W0, W1, W1], dim=1)
m_new['l1.bias'] = m_old['l1.bias'].clone()
m_new['l2.weight'] = m_old['l2.weight'].clone()
m_new['l2.bias'] = m_old['l2.bias'].clone()
m_new['out.weight'] = m_old['out.weight'].clone()
m_new['out.bias'] = m_old['out.bias'].clone()

# Let's test forward equivalence on a synthetic feature input
torch.manual_seed(42)
feat_indices_0 = torch.randint(0, 26196, (10, 15))
feat_indices_1 = torch.randint(0, 26196, (10, 15))

def forward_net(m, W, f0, f1, bucket):
    # accumulator 0 and 1
    # f0: [B, K]
    B, K = f0.shape
    acc0 = m['ftb'].repeat(B, 1)
    acc1 = m['ftb'].repeat(B, 1)
    for b in range(B):
        acc0[b] += m['ft.weight'][f0[b]].sum(dim=0)
        acc1[b] += m['ft.weight'][f1[b]].sum(dim=0)
    
    # concat clipped ReLU
    x = torch.cat([torch.clamp(acc0, 0, 1), torch.clamp(acc1, 0, 1)], dim=1)
    
    # Layer 1
    W_l1 = m['l1.weight'][bucket*16 : (bucket+1)*16]
    b_l1 = m['l1.bias'][bucket*16 : (bucket+1)*16]
    h1 = torch.clamp(F.linear(x, W_l1, b_l1), 0, 1)
    
    # Layer 2
    W_l2 = m['l2.weight'][bucket*32 : (bucket+1)*32]
    b_l2 = m['l2.bias'][bucket*32 : (bucket+1)*32]
    h2 = torch.clamp(F.linear(h1, W_l2, b_l2), 0, 1)
    
    # Out
    W_out = m['out.weight'][bucket : bucket+1]
    b_out = m['out.bias'][bucket : bucket+1]
    out = F.linear(h2, W_out, b_out)
    return out

out_old = forward_net(m_old, 256, feat_indices_0, feat_indices_1, bucket=3)
out_new = forward_net(m_new, 512, feat_indices_0, feat_indices_1, bucket=3)

diff = (out_old - out_new).abs().max().item()
print(f"Max absolute difference between Champion (W=256) and Net2Net-Upscaled (W=512): {diff:.8e}")
assert diff < 1e-6, f"Difference too high: {diff}"
print("EXACT FUNCTION-PRESERVING UP-SCALING PROVEN MATHEMATICALLY AND EMPIRICALLY!")
