# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MXFP8 MoE reference: the exact aiter call vLLM's ``AiterMxfp8Experts``
makes (``aiter.fused_moe`` per_1x32, fp8 weights, gate/up INTERLEAVE, swiglu-OAI with
limit 7, activations quantized to MXFP8 inside aiter), on weights quantized and
shuffled the way vLLM does at load time (``shuffle_mxfp8_moe_weights``).

Also a float reference on the dequantized experts and a HIP-graph timing loop
(100 different inputs per graph), so the lab numbers can be checked against
aiter's tuned table (``minimax_m3_mxfp8_tuned_fmoe.csv``, TP4 rows).

    python3 -m m3_a8w8_moe.ref_mxfp8 --tokens 1 8 32 128 256 [--check]
"""
import argparse
import time

import torch

# MiniMax-M3 at TP4: hidden 6144, intermediate 3072/4, 128 routed + 1 fused shared.
HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0


def quantize_mxfp8(w: torch.Tensor):
    """bf16 ``[E, N, K]`` -> (fp8 e4m3 ``[E, N, K]``, e8m0 uint8 ``[E, N, K/32]``), the
    checkpoint format (``weight`` + ``weight_scale_inv``)."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    e, n, k = w.shape
    q, s = per_1x32_mx_quant_hip(
        w.reshape(e * n, k), quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    return q.view(e, n, k), s.view(torch.uint8).view(e, n, k // 32)


def shuffle_like_vllm(w13_q, w2_q, w13_s, w2_s):
    """What ``convert_to_fp8_moe_kernel_format(AITER_MXFP8)`` stores on the layer."""
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.shuffle_mxfp8_moe_weights(w13_q, w2_q, w13_s, w2_s)


def make_weights(device, seed=0):
    torch.manual_seed(seed)
    w13 = torch.randn((NUM_EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=device) * 0.02
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device=device) * 0.02
    w13_q, w13_s = quantize_mxfp8(w13)
    w2_q, w2_s = quantize_mxfp8(w2)
    raw = (w13_q, w13_s, w2_q, w2_s)
    return raw, shuffle_like_vllm(w13_q, w2_q, w13_s, w2_s)


def routing(m: int, device):
    """4 distinct routed experts + the shared expert per token (aiter's fused
    shared expert: routed weights renormalized x2, shared weight 1)."""
    routed = torch.stack([torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)])
    shared = torch.full((m, 1), NUM_ROUTED, device=device)
    topk_ids = torch.cat([routed, shared], dim=1).to(torch.int32)
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    topk_weights = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return topk_ids.contiguous(), topk_weights.to(torch.float32).contiguous()


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp8 ``[N, K]`` x e8m0 ``[N, K/32]`` -> f32 ``[N, K]``."""
    n, k = q.shape
    sc = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
    return (q.float().view(n, k // 32, 32) * sc.unsqueeze(-1)).view(n, k)


def float_reference(x, raw, topk_ids, topk_weights):
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = xf[t] @ dequant(w13_q[e], w13_s[e]).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = (g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)).to(torch.bfloat16).float()
            out[t] += float(topk_weights[t, j]) * (a @ dequant(w2_q[e], w2_s[e]).T)
    return out


def aiter_mxfp8_moe(x, shuffled, topk_weights, topk_ids):
    """The production call (``AiterMxfp8Experts.apply`` -> ``rocm_aiter_ops.fused_moe``)."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, w13_s, w2_s = shuffled
    return fused_moe(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=w13_s,
        w2_scale=w2_s,
        a1_scale=None,
        a2_scale=None,
        gate_mode=GateMode.INTERLEAVE.value,
        swiglu_limit=SWIGLU_LIMIT,
    )


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def time_graph(fn, inputs, rounds=3):
    """Median per-call time of ``fn`` over ``len(inputs)`` different inputs replayed
    from one HIP graph."""
    for inp in inputs[:3]:
        fn(*inp)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for inp in inputs[:2]:
            fn(*inp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for inp in inputs:
                fn(*inp)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / len(inputs) * 1e6)
        meds.append(sorted(ts)[len(ts) // 2])
    return sorted(meds)[len(meds) // 2], meds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    p.add_argument("--copies", type=int, default=100)
    p.add_argument("--check", action="store_true", help="cosine vs the float reference")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    print("w13", tuple(shuffled[0].shape), shuffled[0].dtype, "w13_scale", tuple(shuffled[2].shape),
          "w2", tuple(shuffled[1].shape), "w2_scale", tuple(shuffled[3].shape), flush=True)
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        out = aiter_mxfp8_moe(x, shuffled, w, ids)
        torch.cuda.synchronize()
        if a.check:
            ref = float_reference(x, raw, ids, w)
            print(f"M={m}: aiter vs float ref cos {cos(out, ref):.5f}, max|d| {(out.float() - ref).abs().max().item():.4f} (ref max {ref.abs().max().item():.3f})", flush=True)
        inputs = []
        for i in range(a.copies):
            torch.manual_seed(1000 + i)
            xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids_i, w_i = routing(m, dev)
            inputs.append((xi, shuffled, w_i, ids_i))
        med, meds = time_graph(aiter_mxfp8_moe, inputs)
        print(f"[aiter mxfp8 a8w8] M={m}: per call median {med:.1f} us (rounds {', '.join(f'{v:.1f}' for v in meds)})", flush=True)


if __name__ == "__main__":
    main()
