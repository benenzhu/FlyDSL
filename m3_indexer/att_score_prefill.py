"""Eager launches of the FlyDSL prefill index scorer for rocprofv3 ATT capture:
``--iters`` calls with different inputs, no graph, no reference.

    M3_IDX_SCORER=w4 FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i att.yaml -- \\
        python att_score_prefill.py --q 8192 --ctx 650000 --iters 3
"""
import argparse, sys

import torch

sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops  # noqa: E402

import_ops("index_bf16")
from index_bf16.host import index_score_prefill, index_topk_prefill  # noqa: E402

BLK = 128; D = 128


def build(q_lens, ctxs, seed):
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
    q = (torch.randn(sum(q_lens), 1, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32, device="cuda")
    t = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
    return (q, cache, bt, cu, t(seq_lens), t(ctxs), max(q_lens), max(seq_lens), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8192)
    ap.add_argument("--ctx", type=int, default=650000)
    ap.add_argument("--reqs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--topk", action="store_true", help="also run the prefill top-k on the scores")
    a = ap.parse_args()
    for i in range(a.iters):
        args = build([a.q] * a.reqs, [a.ctx] * a.reqs, 100 + i)
        score = index_score_prefill(*args)
        if a.topk:
            q, cache, bt, cu, seq_lens, prefix, max_q, max_seq, _ = args
            index_topk_prefill(score, cu, prefix, max_q, 16, 0, 1)
        torch.cuda.synchronize()
        del args
    print("done", flush=True)


if __name__ == "__main__":
    main()
