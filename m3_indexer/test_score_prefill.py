"""FlyDSL prefill scorer vs the vllm Triton scorer: scores (valid causal columns)
and the top-k indices the Triton top-k derives from them.

  python test_score_prefill.py            # small shapes
  python test_score_prefill.py --big      # 4 x 8192 rows at 500K..800K
"""
import argparse, random, sys

import torch

sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops  # noqa: E402

import_ops("index_bf16")
from index_bf16.host import index_score_prefill  # noqa: E402
from vllm.models.minimax_m3.amd.ops.index_topk import (  # noqa: E402
    minimax_m3_index_score, minimax_m3_index_topk)

BLK = 128; D = 128; TOPK = 16


def case(q_lens, ctxs, seed):
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    seq_lens = [c + q for c, q in zip(ctxs, q_lens)]
    nblocks = [-(-s // BLK) for s in seq_lens]
    pool = sum(nblocks) + 8
    cache = (torch.randn(pool, BLK, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    bt = torch.zeros(len(q_lens), max(nblocks), dtype=torch.int32, device="cuda")
    perm = torch.randperm(pool, device="cuda", generator=g).to(torch.int32)
    o = 0
    for i, n in enumerate(nblocks):
        bt[i, :n] = perm[o:o + n]; o += n
    total_q = sum(q_lens)
    q = (torch.randn(total_q, 1, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32, device="cuda")
    t = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
    args = (q, cache, bt, cu, t(seq_lens), t(ctxs), max(q_lens), max(seq_lens), 1)
    ref = minimax_m3_index_score(*args)
    out = index_score_prefill(*args)
    torch.cuda.synchronize()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    # valid causal columns of row r of request b: blk < (prefix + r_local + 128) // 128
    worst = 0.0; bad = 0
    for b, (ql, c) in enumerate(zip(q_lens, ctxs)):
        r0 = int(cu[b])
        for r in range(ql):
            vb = (c + r + BLK) // BLK
            a = out[0, r0 + r, :vb]; e = ref[0, r0 + r, :vb]
            d = (a - e).abs()
            fin = torch.isfinite(e)
            if not torch.equal(torch.isfinite(a), fin):
                bad += 1; continue
            m = d[fin].max().item() if fin.any() else 0.0
            worst = max(worst, m)
            if m > 2e-3:
                bad += 1
    tk_ref = minimax_m3_index_topk(ref, cu, t(ctxs), max(q_lens), TOPK, 0, 1)
    tk_out = minimax_m3_index_topk(out, cu, t(ctxs), max(q_lens), TOPK, 0, 1)
    mism = (tk_ref != tk_out).any(dim=-1)[0].nonzero().flatten().tolist()
    # a mismatch whose differing blocks have (near-)equal reference scores is a tie
    # flipped by fp32 summation order, not an error
    real = 0
    for r in mism:
        a, e = set(tk_out[0, r].tolist()), set(tk_ref[0, r].tolist())
        diff = [x for x in (a ^ e) if x >= 0]
        if len(diff) == 0:
            continue  # same block set, different order (equal scores)
        sc = ref[0, r, diff]
        if (sc.max() - sc.min()).item() > 1e-5:
            real += 1
    return worst, bad, (real, len(mism)), total_q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--big", action="store_true"); a = ap.parse_args()
    rng = random.Random(0)
    if a.big:
        cases = [([rng.randint(6554, 8192) for _ in range(4)], [rng.randint(500000, 800000) for _ in range(4)])]
    else:
        cases = [
            ([7], [0]),                       # tiny, no prefix
            ([300], [1000]),                  # one request, partial tiles
            ([512, 513, 100], [4000, 130000, 129]),  # several requests, ragged
            ([2048], [140000]),               # 4 tiles x 64 segments
            ([1, 1029], [70000, 3]),
        ]
    ok = True
    for q_lens, ctxs in cases:
        worst, bad, tkm, tq = case(q_lens, ctxs, seed=len(q_lens))
        status = "OK" if (bad == 0 and tkm[0] == 0) else "FAIL"
        ok &= status == "OK"
        print(f"{status} q={q_lens} ctx={ctxs}: max|d|={worst:.2e} bad_rows={bad} topk_rows: {tkm[0]} real mismatches, {tkm[1]} tie/order-only, of {tq}")
    sys.exit(0 if ok else 1)


main()
