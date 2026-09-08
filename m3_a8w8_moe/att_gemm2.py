# SPDX-License-Identifier: Apache-2.0
"""Eager launches of the a8w8 prefill kernels for rocprofv3 ATT capture (no HIP graph:
the tracer needs plain dispatches). Runs stage 1 once, then gemm2 ``--iters`` times.

    FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i att.yaml -- \\
        python3 -m m3_a8w8_moe.att_gemm2 --tokens 4096 --iters 3
"""
import argparse

import torch

from m3_a8w8_moe.test_prefill_gemm2 import (
    HIDDEN, INTER, NUM_EXPERTS, TOPK, make_weights, routing, stage1,
)

from moe_a4w4_prefill import _run_compiled, block_m_for  # noqa: E402
from moe_a8w8_prefill import GEMM2_N_SPLIT  # noqa: E402
from moe_a8w8_prefill.gemm2 import compile_moe_gemm2, gemm2_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--n-split", type=int, default=GEMM2_N_SPLIT)
    p.add_argument("--gemm1-only", action="store_true", help="only stage 1 (sort/quant/gemm1), --iters times")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    w13, w2, w13_s, w2_s = shuffled
    m = a.tokens
    torch.manual_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
    ids, w = routing(m, dev)
    bm = block_m_for(m)
    if a.gemm1_only:
        for _ in range(a.iters):
            stage1(x, shuffled, ids, w)
        torch.cuda.synchronize()
        return
    bufs, a_q, a_s, h_q, h_s, nmb = stage1(x, shuffled, ids, w)
    torch.cuda.synchronize()
    nmb2 = (nmb * bm) // 128
    launch = compile_moe_gemm2(H=HIDDEN, I=INTER, E=NUM_EXPERTS, topk=TOPK, n_split=a.n_split, sort_block_m=bm)
    partial = torch.empty((m * TOPK, HIDDEN), dtype=torch.bfloat16, device=dev)
    dummy_sc = torch.empty((16,), dtype=torch.uint8, device=dev)
    grid2 = gemm2_grid(nmb2, a.n_split)
    for _ in range(a.iters):
        _run_compiled(launch, h_q.view(-1), w2.view(torch.uint8).view(-1), partial.view(-1), h_s,
                      w2_s.view(torch.uint8).view(-1), dummy_sc, bufs.sorted_ids, bufs.sorted_expert_ids,
                      bufs.sorted_weights, bufs.num_valid_ids, m, nmb2, grid2, torch.cuda.current_stream())
        torch.cuda.synchronize()
    print("done", flush=True)


if __name__ == "__main__":
    main()
