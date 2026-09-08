# SPDX-License-Identifier: Apache-2.0
"""Whole-chain check and timing of moe_a8w8_decode (a16w8) against the float
reference and the production aiter a8w8 call, 100 different inputs per graph.

    python3 -m m3_a8w8_moe.test_chain --tokens 1 4 16 32 64 128 256 [--no-time] [--no-aiter]
"""
import argparse

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import (
    HIDDEN, INTER, NUM_EXPERTS, SWIGLU_ALPHA, SWIGLU_LIMIT, aiter_mxfp8_moe, cos, float_reference,
    make_weights, routing, time_graph,
)

import_ops("moe_a16w4_decode")
import_ops("moe_a8w8_decode")
from moe_a8w8_decode import a16w8_decode_moe  # noqa: E402


def ours(x, shuffled, w, ids):
    w13, w2, w13_s, w2_s = shuffled
    return a16w8_decode_moe(x, w13, w13_s, w2, w2_s, w, ids, hidden_size=HIDDEN, intermediate_size=INTER,
                            num_experts=NUM_EXPERTS, swiglu_alpha=SWIGLU_ALPHA, swiglu_limit=SWIGLU_LIMIT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 256])
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--no-aiter", action="store_true")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        out = ours(x, shuffled, w, ids)
        torch.cuda.synchronize()
        ref = float_reference(x, raw, ids, w)
        line = f"M={m}: ours vs float cos {cos(out, ref):.6f} max|d| {(out.float() - ref).abs().max().item():.4f} (ref max {ref.abs().max().item():.3f})"
        if not a.no_aiter:
            ra = aiter_mxfp8_moe(x, shuffled, w, ids)
            torch.cuda.synchronize()
            line += f" | aiter vs float cos {cos(ra, ref):.5f} | ours vs aiter cos {cos(out, ra):.5f}"
        print(line, flush=True)
        if a.no_time:
            continue
        inputs = []
        for i in range(100):
            torch.manual_seed(1000 + i)
            xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids_i, w_i = routing(m, dev)
            inputs.append((xi, shuffled, w_i, ids_i))
        med, meds = time_graph(ours, inputs)
        line = f"[chain] M={m}: ours {med:.1f} us (rounds {', '.join(f'{v:.1f}' for v in meds)})"
        if not a.no_aiter:
            med_a, meds_a = time_graph(aiter_mxfp8_moe, inputs)
            line += f" | aiter a8w8 {med_a:.1f} us -> {med_a / med:.2f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
