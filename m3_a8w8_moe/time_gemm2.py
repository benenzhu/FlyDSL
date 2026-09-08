# SPDX-License-Identifier: Apache-2.0
"""Time moe_a8w8_prefill.gemm2 alone (HIP graph, 4 replays) for compile-time variants.

    python3 -m m3_a8w8_moe.time_gemm2 --tokens 4096 32768 --n-split 2 4 --rotate 1 0
"""
import argparse

import torch

from m3_a8w8_moe.ref_mxfp8 import time_graph
from m3_a8w8_moe.test_prefill_gemm2 import HIDDEN, INTER, NUM_EXPERTS, TOPK, make_weights, routing, stage1

from moe_a4w4_prefill import _run_compiled, block_m_for  # noqa: E402
from moe_a8w8_prefill.gemm2 import compile_moe_gemm2, gemm2_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 32768])
    p.add_argument("--n-split", type=int, nargs="+", default=[2])
    p.add_argument("--rotate", type=int, nargs="+", default=[1])
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    w13, w2, w13_s, w2_s = shuffled
    launches = {}
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        bm = block_m_for(m)
        bufs, a_q, a_s, h_q, h_s, nmb = stage1(x, shuffled, ids, w)
        torch.cuda.synchronize()
        nmb2 = (nmb * bm) // 128
        for ns in a.n_split:
            for rot in a.rotate:
                key = (bm, ns, rot)
                if key not in launches:
                    launches[key] = compile_moe_gemm2(H=HIDDEN, I=INTER, E=NUM_EXPERTS, topk=TOPK, n_split=ns,
                                                      sort_block_m=bm, rotate=bool(rot))
                launch = launches[key]
                partial = torch.empty((m * TOPK, HIDDEN), dtype=torch.bfloat16, device=dev)
                grid2 = gemm2_grid(nmb2, ns)

                def fn():
                    _run_compiled(launch, h_q.view(-1), w2.view(torch.uint8).view(-1), partial.view(-1), h_s,
                                  w2_s.view(torch.uint8).view(-1), bufs.sorted_ids, bufs.sorted_expert_ids,
                                  bufs.sorted_weights, bufs.num_valid_ids, m, nmb2, grid2,
                                  torch.cuda.current_stream())

                t, _ = time_graph(lambda *args: fn(), [()] * 4)
                print(f"gemm2 alone M={m} n_split={ns} rotate={rot}: {t:.1f} us", flush=True)


if __name__ == "__main__":
    main()
