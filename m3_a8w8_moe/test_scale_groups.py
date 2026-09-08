# SPDX-License-Identifier: Apache-2.0
"""Do the fp8 kernels apply the right e8m0 to every 32-K block? Random N(0, 0.02)
weights give every block the same scale, which hides a wrong block-to-lane pairing.
Here alternate 32-K blocks of W13 / W2 are scaled x1 / x16 (and rows too), so a
kernel that applies one scale to values from two different blocks is off by 16x on
half of its products: compare aiter's a8w8 chain and our a16w8 chain against the
float reference.

    python3 -m m3_a8w8_moe.test_scale_groups
"""
import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import (
    HIDDEN, INTER, NUM_EXPERTS, SWIGLU_ALPHA, SWIGLU_LIMIT, aiter_mxfp8_moe, cos, float_reference,
    quantize_mxfp8, routing, shuffle_like_vllm,
)

import_ops("moe_a16w4_decode")
import_ops("moe_a8w8_decode")
from moe_a8w8_decode import a16w8_decode_moe  # noqa: E402


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    w13 = torch.randn((NUM_EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=dev) * 0.02
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device=dev) * 0.02
    # a random magnitude (x1 / x4 / x16 / x64) per (row, 32-K block): any kernel that
    # pairs a value with the scale of a different block is off by up to 64x there
    g = torch.Generator(device=dev).manual_seed(1)
    f13 = 4.0 ** torch.randint(0, 4, (NUM_EXPERTS, 2 * INTER, HIDDEN // 32), generator=g, device=dev)
    f2 = 4.0 ** torch.randint(0, 4, (NUM_EXPERTS, HIDDEN, INTER // 32), generator=g, device=dev)
    w13 = (w13.float() * f13.repeat_interleave(32, dim=2)).to(torch.bfloat16)
    w2 = (w2.float() * f2.repeat_interleave(32, dim=2)).to(torch.bfloat16)
    w13_q, w13_s = quantize_mxfp8(w13)
    w2_q, w2_s = quantize_mxfp8(w2)
    print("distinct e8m0 in w13 row 0:", sorted(set(w13_s[0, 0].tolist()))[:6])
    raw = (w13_q, w13_s, w2_q, w2_s)
    shuffled = shuffle_like_vllm(w13_q, w2_q, w13_s, w2_s)
    for m in (4, 32):
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        ref = float_reference(x, raw, ids, w)
        w13k, w2k, w13sk, w2sk = shuffled
        ours = a16w8_decode_moe(x, w13k, w13sk, w2k, w2sk, w, ids, hidden_size=HIDDEN, intermediate_size=INTER,
                                num_experts=NUM_EXPERTS, swiglu_alpha=SWIGLU_ALPHA, swiglu_limit=SWIGLU_LIMIT)
        ai = aiter_mxfp8_moe(x, shuffled, w, ids)
        torch.cuda.synchronize()
        print(f"M={m}: ours vs float cos {cos(ours, ref):.5f} | aiter vs float cos {cos(ai, ref):.5f} | ours vs aiter cos {cos(ours, ai):.5f}", flush=True)


if __name__ == "__main__":
    main()
