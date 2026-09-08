# SPDX-License-Identifier: Apache-2.0
"""Full a8w8 prefill chain (moe_a8w8_prefill.a8w8_prefill_moe) vs the production aiter
call and a float reference on sampled tokens; HIP-graph timing of ours vs aiter over
different inputs.

    python3 -m m3_a8w8_moe.test_prefill_chain --tokens 4096 8192 16384 32768
"""
import argparse

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import (
    HIDDEN, INTER, NUM_EXPERTS, aiter_mxfp8_moe, cos, float_reference, make_weights, routing,
    time_graph,
)

import_ops("moe_a16w4_decode")
import_ops("moe_a4w4_prefill")
import_ops("moe_a8w8_prefill")
from moe_a8w8_prefill import a8w8_prefill_moe  # noqa: E402


OUT_MODE = "bf16"


def ours(x, shuffled, w, ids, out_mode=None):
    w13, w2, w13_s, w2_s = shuffled
    return a8w8_prefill_moe(x, w13, w13_s, w2, w2_s, w, ids, hidden_size=HIDDEN,
                            intermediate_size=INTER, num_experts=NUM_EXPERTS,
                            out_mode=OUT_MODE if out_mode is None else out_mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[4096])
    p.add_argument("--check-tokens", type=int, default=64)
    p.add_argument("--copies", type=int, default=4)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--no-aiter", action="store_true")
    p.add_argument("--out-mode", default="bf16", choices=["bf16", "atomic", "fp8"],
                   help="gemm2 output mode: bf16 partials + aiter reduce, bf16 atomics, fp8 partials + reduce_fp8")
    a = p.parse_args()
    global OUT_MODE
    OUT_MODE = a.out_mode
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        out = ours(x, shuffled, w, ids)
        out2 = ours(x, shuffled, w, ids)
        torch.cuda.synchronize()
        if not torch.equal(out, out2):
            bad = (out != out2).any(dim=1)
            tag = "expected for atomics" if a.out_mode == "atomic" else "!!"
            print(f"  {tag} run-to-run mismatch: {int(bad.sum())}/{m} tokens differ, cos(out, out2) {cos(out, out2):.5f}",
                  flush=True)
        ref_aiter = aiter_mxfp8_moe(x, shuffled, w, ids)
        torch.cuda.synchronize()
        gen = torch.Generator().manual_seed(m)
        toks = torch.randperm(m, generator=gen)[: a.check_tokens].tolist()
        ref = float_reference(x[toks], raw, ids[toks], w[toks])
        c_ref = cos(out[toks], ref)
        c_ait = cos(out[toks], ref_aiter[toks])
        c_ait_ref = cos(ref_aiter[toks], ref)
        worst = min(cos(out[t], ref_aiter[t]) for t in toks)
        print(f"M={m}: ours vs float ref cos {c_ref:.5f} | ours vs aiter {c_ait:.5f} (worst token {worst:.5f}) "
              f"| aiter vs float ref {c_ait_ref:.5f} | max|ours-ref| {(out[toks].float() - ref).abs().max().item():.4f} "
              f"(ref max {ref.abs().max().item():.3f})", flush=True)
        if a.no_time:
            continue
        inputs = []
        for i in range(a.copies):
            torch.manual_seed(1000 + i)
            xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids_i, w_i = routing(m, dev)
            inputs.append((xi, shuffled, w_i, ids_i))
        t_ours, _ = time_graph(ours, inputs)
        line = f"[a8w8 prefill chain {a.out_mode}] M={m}: ours {t_ours:.1f} us"
        if not a.no_aiter:
            t_ait, _ = time_graph(aiter_mxfp8_moe, inputs)
            line += f" | aiter {t_ait:.1f} us | {t_ait / t_ours:.2f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
