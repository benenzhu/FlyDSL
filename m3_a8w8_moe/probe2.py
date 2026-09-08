import sys, torch
sys.argv = [sys.argv[0]]
from m3_a8w8_moe.test_prefill_gemm2 import *
dev = torch.device("cuda")
raw, shuffled = make_weights(dev)
_, _, w2_q, w2_s = raw
m = 4096
torch.manual_seed(m)
x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
ids, w = routing(m, dev)
bm = block_m_for(m)
bufs, a_q, a_s, h_q, h_s, nmb = stage1(x, shuffled, ids, w)
torch.cuda.synchronize()
rows_all = h_q.shape[0]
h_q.zero_()
h_q[torch.arange(rows_all, device=dev), torch.arange(rows_all, device=dev) % INTER] = 0x38
h_s.fill_(127)
p1 = run_gemm2(bufs, h_q, h_s, nmb, shuffled, m, bm)
torch.cuda.synchronize()
n_valid = int(bufs.num_valid_ids[0].item())
sorted_ids = bufs.sorted_ids[:n_valid].cpu(); eids = bufs.sorted_expert_ids.cpu(); sw = bufs.sorted_weights[:n_valid].cpu()
valid_rows = [r for r in range(n_valid) if int(sorted_ids[r]) & 0xFFFFFF < m]
by_c = {}
for r in valid_rows:
    by_c.setdefault(r % INTER, r)
for c in [0, 17, 64, 130, 256, 400, 512, 700]:
    r = by_c[c]; sid = int(sorted_ids[r]); tok, slot = sid & 0xFFFFFF, sid >> 24; e = int(eids[r // bm])
    W = dequant(w2_q[e], w2_s[e])  # [H, I]
    got = p1[tok * TOPK + slot].float()
    wr = float(sw[r])
    refs = wr * W.T  # [I, H]
    cs = (refs @ got) / (refs.norm(dim=1) * got.norm() + 1e-9)
    top = torch.topk(cs, 3)
    best = [(round(v, 4), int(i)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
    print(f"c={c} (row {r}, e {e}): |got| {got.abs().mean():.4f} |ref| {refs[c].abs().mean():.4f} cos(ref c) {cs[c]:.4f}; best c' {best}", flush=True)
    A = torch.stack([refs[c], refs[best[0][1]]], dim=1)
    sol = torch.linalg.lstsq(A, got.unsqueeze(1)).solution.flatten().tolist()
    print(f"   fit got = {sol[0]:.3f}*ref(c) + {sol[1]:.3f}*ref(c'={best[0][1]})", flush=True)
