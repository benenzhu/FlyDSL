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
for r in [0, 5, 17, 40, 77, 100, 128 + 3, 256 + 9]:
    sid = int(sorted_ids[r]); tok, slot = sid & 0xFFFFFF, sid >> 24; e = int(eids[r // bm]); c = r % INTER
    if tok >= m: continue
    W = dequant(w2_q[e], w2_s[e])
    got = p1[tok * TOPK + slot].float(); ref = float(sw[r]) * W[:, c]
    bad = ((got - ref).abs() > 0.02 * ref.abs().max()).view(HIDDEN // 256, 2, 2, 4, 16).any(dim=-1)  # ntile, hf, wave_j, j
    badt = bad.reshape(HIDDEN // 256, 16).any(dim=1)
    print(f"row {r} (tile_i {r // 128}, r%128={r % 128}, c={c}, e={e}): bad n-tiles {torch.nonzero(badt).flatten().tolist()}")
    for nt in range(3):
        print(f"   ntile {nt}: " + " ".join(f"hf{hf}w{wj}:" + "".join("X" if bad[nt, hf, wj, j] else "." for j in range(4)) for hf in range(2) for wj in range(2)))
