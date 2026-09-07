#!/usr/bin/env python3
"""MiniMax-M3 (TP4) decode MoE at M=4 on the FlyDSL a16w4 3-kernel chain.

Runs inside the production vLLM image (flydsl 0.2.4 + its aiter). Chain per call:
  aiter moe_sorting (sort + zero output)  ->  gemm1 (gate/up + swigluoai, bf16 out)
  ->  gemm2 (down, routing-weighted atomic add)
Inputs are generated exactly like m3-compare/scripts/bench_moe_m4.py (same seed -> same
weights and routing), so numbers are comparable with the CK-tile a16w4 baseline
(43.9 us graph replay: sort 5.7 + fill 4.1 + gemm1 19.5 + swiglu 4.3 + gemm2 10.0).

  PYTHONPATH=/flydsl python3 /flydsl/m3_a16w4_moe/bench_m3.py --tile-m 16 --g1-tile-n 64 \
      --g1-tile-k 128 --k-wave 4
"""
import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=4)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tile-m", type=int, default=16)
p.add_argument("--g1-tile-n", type=int, default=128)
p.add_argument("--g1-tile-k", type=int, default=256)
p.add_argument("--k-wave", type=int, default=1)
p.add_argument("--g1-b-nt", type=int, default=0)
p.add_argument("--g1-xcd", type=int, default=0)
p.add_argument("--g1-wpe", type=int, default=0, help="waves_per_eu attr (0 = unset)")
p.add_argument("--g1-a-direct", type=int, default=0, help="1: A straight global->VGPR (no LDS/barrier)")
p.add_argument("--g1-pf", type=int, default=1, help="K tiles in flight ahead of compute (a_direct only)")
p.add_argument("--g1-a4", type=int, default=0, help="gemm1: <=4-row blocks stage A through LDS, one load per lane per 128 K")
p.add_argument("--g1-ss", type=int, default=0, help="1: share W-scale dwords across tiles of one 256-K group")
p.add_argument("--g1-impl", choices=["base", "persist", "persist-lookahead", "persist-fused-reduce", "persist-interleave"], default="base")
p.add_argument("--g1-ctas", type=int, default=512, help="persistent gemm1 CTA count")
p.add_argument("--g2-tile-n", type=int, default=256)
p.add_argument("--g2-tile-k", type=int, default=256)
p.add_argument("--g2-b-nt", type=int, default=0)
p.add_argument("--g2-xcd", type=int, default=1)
p.add_argument("--g2-wpe", type=int, default=0)
p.add_argument("--g2-a-direct", type=int, default=0)
p.add_argument("--g2-pf", type=int, default=1)
p.add_argument("--g2-ksplit", type=int, default=1, help="gemm2 split-K across CTAs (atomics sum the partials)")
p.add_argument("--g2-pad-mask", type=int, default=0, help="gemm2: OOB-mask padding rows in the A loads")
p.add_argument("--g2-ss", type=int, default=0, help="gemm2: share scale dwords across 128-K halves / 16-col halves")
p.add_argument("--g2-hoist", type=int, default=-1, help="gemm2 prologue hoist: -1 = follow a_direct, 0/1 force")
p.add_argument("--w-layout", default="standard", choices=["standard", "guinterleave"])
p.add_argument("--sort", default="aiter", choices=["aiter", "mxfp4", "pairs", "decode", "decode-wave"],
               help="aiter: opus moe_sorting (production); mxfp4: aiter#3832 single-CTA sort + zero-init (BM=16)")
p.add_argument("--reps", type=int, default=10)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--no-check", action="store_true")
p.add_argument("--no-shared", action="store_true", help="timing experiment: all 5 slots routed (no M-row shared expert)")
p.add_argument("--loop", type=int, default=0, help="run N eager iterations and exit (for rocprofv3)")
p.add_argument("--graph-copies", type=int, default=100,
               help="calls captured per graph, each with its OWN x / routing (different experts -> weights come "
                    "from HBM like real decode, and the per-graph launch cost is amortised as in vLLM's model graph)")
p.add_argument("--stages", type=int, default=3, choices=[1, 2, 3], help="1: sort only, 2: sort+gemm1, 3: full chain")
args = p.parse_args()

import torch  # noqa: E402
import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting, _adaptive_moe_sort  # noqa: E402
from m3_a16w4_moe.sort_decode import moe_sort_decode  # noqa: E402
if args.sort == "decode-wave":
    from m3_a16w4_moe.sort_decode_wave import moe_sort_decode
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m3_a16w4_moe.host import a16w4_gemm1, a16w4_gemm2  # noqa: E402
if args.g1_impl != "base":
    from functools import partial
    from m3_a16w4_moe.host_persist import a16w4_gemm1_persist
    a16w4_gemm1 = partial(a16w4_gemm1_persist, n_ctas=args.g1_ctas, kernel_variant=args.g1_impl)

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K = args.tokens, args.hidden, args.inter, args.experts, args.topk
BM = args.tile_m
ALPHA, LIMIT = 1.702, 7.0  # MiniMax-M3 swigluoai

# ---- weights: random bf16 -> mxfp4 per-1x32 (identical to bench_moe_m4.py) ----
w1 = (torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=dev) * 0.02)
w2 = (torch.randn((E, H, I), dtype=torch.bfloat16, device=dev) * 0.02)
w1_q, w1_s = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(E, 2 * I, H // 2)
w2_q = w2_q.view(E, H, I // 2)

if args.w_layout == "standard":
    # N-major GGUU preshuffle + e8m0 scale shuffle (what the a4w4 FlyDSL paths consume)
    w1_k = shuffle_weight(w1_q, layout=(16, 16)).view(torch.uint8).contiguous()
    w1_sk = fp4_utils.e8m0_shuffle(w1_s.view(-1, H // 32)).view(torch.uint8).contiguous()
else:
    # production layout: vLLM AITER_MXFP4_BF16 passes gate_up=True (GUGU interleave) for w13
    w1_k = shuffle_weight_a16w4(w1_q, 16, True).view(torch.uint8).contiguous()
    w1_sk = shuffle_scale_a16w4(w1_s.view(-1, H // 32), E, True).view(torch.uint8).contiguous()
w2_k = shuffle_weight(w2_q, layout=(16, 16)).view(torch.uint8).contiguous()
w2_sk = fp4_utils.e8m0_shuffle(w2_s.view(-1, I // 32)).view(torch.uint8).contiguous()

# ---- routing like production: 4 distinct routed experts + the shared expert (id E-1) ----
# One input set per captured call: different tokens AND different routing per call, so a
# graph replay touches ~copies x 5 experts (913 MB of W for 129 experts) instead of re-reading
# one 35 MB expert set out of the 256 MB infinity cache.
def make_input():
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
    if args.no_shared:
        # timing experiment: 5 distinct routed experts, no expert with M rows
        topk_ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
    else:
        routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
        topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
    w_r = torch.rand((M, K - 1), device=dev)
    w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0  # renormalised, routed_scaling_factor 2.0
    topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32)
    return x, topk_ids, topk_w


inputs = [make_input() for _ in range(max(1, args.graph_copies))]
x, topk_ids, topk_w = inputs[0]  # call 0 == the old single-input bench (same seed stream)

# sorted rows upper bound (aiter pads every expert to a BM multiple)
max_sorted = M * K + E * BM - K
inter_sorted = torch.empty((max_sorted, I), dtype=torch.bfloat16, device=dev)
last_sort = None

g1_kw = dict(
    tile_m=BM, tile_n=args.g1_tile_n, tile_k=args.g1_tile_k, k_wave=args.k_wave,
    b_nt=args.g1_b_nt, xcd_swizzle=args.g1_xcd, waves_per_eu=args.g1_wpe or None,
    act="swigluoai", alpha=ALPHA, swiglu_limit=LIMIT, w_layout=args.w_layout, a_direct=bool(args.g1_a_direct),
    prefetch=args.g1_pf, scale_share=bool(args.g1_ss), a_rows4=bool(args.g1_a4),
)
g2_kw = dict(
    tile_m=BM, tile_n=args.g2_tile_n, tile_k=args.g2_tile_k,
    b_nt=args.g2_b_nt, xcd_swizzle=args.g2_xcd, waves_per_eu=args.g2_wpe or None,
    a_direct=bool(args.g2_a_direct), prefetch=args.g2_pf,
    ksplit=args.g2_ksplit, pad_mask=bool(args.g2_pad_mask),
    hoist=None if args.g2_hoist < 0 else bool(args.g2_hoist), scale_share=bool(args.g2_ss),
)


def run(inp=None):
    global last_sort
    x, topk_ids, topk_w = inputs[0] if inp is None else inp
    if args.sort == "pairs":
        # no sort kernel: gemm1/gemm2 derive expert + rows from the routing pairs
        # (n_tokens <= BM) and gemm1 zeroes `out` for gemm2's atomics.
        if args.stages < 2:
            return torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
        out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
        a16w4_gemm1(
            x_bf16=x, w1_u8=w1_k, w1_scale_u8=w1_sk, inter_sorted_bf16=inter_sorted,
            n_tokens=M, NE=E, D_HIDDEN=H, D_INTER=I, topk=K, pairs=True, topk_ids=topk_ids, zero_out=out, **g1_kw,
        )
        if args.stages < 3:
            return out
        a16w4_gemm2(
            inter_sorted_bf16=inter_sorted, w2_u8=w2_k, w2_scale_u8=w2_sk, out_bf16=out,
            n_tokens=M, NE=E, D_HIDDEN=H, D_INTER=I, pairs=True, topk=K, topk_ids=topk_ids, topk_weights=topk_w, **g2_kw,
        )
        return out
    if args.sort in ("decode", "decode-wave"):
        # our one-kernel sort + zero (sort_decode.py): block 0 sorts, the other blocks zero `out`
        sorted_ids, sorted_w, sorted_eids, num_valid, out = moe_sort_decode(topk_ids, topk_w, E, H, BM)
    elif args.sort == "mxfp4":
        # aiter#3832 moe_sort_quant with kSkipQuant: block 0 sorts (LDS counters), the other
        # CTAs zero `out`; same output contract as moe_sorting (token | slot<<24, pad = M).
        sorted_ids, sorted_w, sorted_eids, num_valid, out = _adaptive_moe_sort(
            topk_ids, topk_w, E, K, BM, H, atomic=True
        )
    else:
        sorted_ids, sorted_w, sorted_eids, num_valid, out = moe_sorting(
            topk_ids, topk_w, E, H, torch.bfloat16, block_size=BM
        )
    last_sort = (sorted_ids, num_valid)
    if args.stages < 2:
        return out
    a16w4_gemm1(
        x_bf16=x, w1_u8=w1_k, w1_scale_u8=w1_sk, sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid, sorted_token_ids=sorted_ids, inter_sorted_bf16=inter_sorted,
        n_tokens=M, NE=E, D_HIDDEN=H, D_INTER=I, topk=K, **g1_kw,
    )
    if args.stages < 3:
        return inter_sorted
    a16w4_gemm2(
        inter_sorted_bf16=inter_sorted, w2_u8=w2_k, w2_scale_u8=w2_sk, sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid, sorted_token_ids=sorted_ids, sorted_weights=sorted_w, out_bf16=out,
        n_tokens=M, NE=E, D_HIDDEN=H, D_INTER=I, **g2_kw,
    )
    return out


# ---- reference on the dequantised weights (bf16 math on the selected experts) ----
def deq(q, s, n_cols):
    v = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(q.shape[0], -1)
    sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(q.shape[0], -1)
    return (v.view(q.shape[0], -1, 32) * sc.unsqueeze(-1)).view(q.shape[0], n_cols)


def reference(inp=None):
    x, topk_ids, topk_w = inputs[0] if inp is None else inp
    shape = (M, K, I) if args.stages == 2 else (M, H)
    out = torch.zeros(shape, dtype=torch.float32, device=dev)
    xf = x.float()
    for t in range(M):
        for j in range(K):
            e = int(topk_ids[t, j])
            w1e = deq(w1_q[e], w1_s.view(E, 2 * I, -1)[e], H)  # (2I, H)
            h = xf[t] @ w1e.T
            g, u = h[:I], h[I:]
            g = g.clamp(max=LIMIT)
            u = u.clamp(-LIMIT, LIMIT)
            a = g * torch.sigmoid(ALPHA * g) * (u + 1.0)
            if args.stages == 2:
                out[t, j] = a.to(torch.bfloat16).float()
                continue
            w2e = deq(w2_q[e], w2_s.view(E, H, -1)[e], I)      # (H, I)
            # kernel rounds the stage-1 intermediate to bf16 before gemm2
            out[t] += float(topk_w[t, j]) * (a.to(torch.bfloat16).float() @ w2e.T)
    return out


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def snapshot(out, sort_state=None):
    """Canonical gemm1 output: ignore unwritten padding, preserve every routed row.

    Called outside the timed graph. Scatter by token/slot so the check does not
    depend on an implementation's ordering of rows within each expert.
    """
    if args.stages != 2:
        return out.clone()
    if args.sort == "pairs":
        raise ValueError("stage-2 validation currently requires a sorted chain")
    ids, num_valid = last_sort if sort_state is None else sort_state
    ids = ids[:int(num_valid.flatten()[0].item())]
    tokens, slots = ids & 0x00FFFFFF, ids >> 24
    valid = tokens < M
    dest = (tokens[valid].long() * K + slots[valid].long())
    assert dest.numel() == M * K and dest.unique().numel() == M * K, "routing coverage"
    result = torch.empty((M * K, I), dtype=torch.bfloat16, device=dev)
    result[dest] = inter_sorted[:ids.numel()][valid]
    return result.view(M, K, I)


def check_ref(out, ref, label):
    score = cos(out, ref)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"[a16w4-flydsl] {label}: cos {score:.8f} max|d| {err:.4f}", flush=True)
    assert torch.isfinite(out).all() and score >= 0.9999, f"{label}: numerical check failed"


tag = (f"g1 bm{BM} tn{args.g1_tile_n} tk{args.g1_tile_k} kw{args.k_wave} nt{args.g1_b_nt} xcd{args.g1_xcd} ad{args.g1_a_direct} pf{args.g1_pf} ss{args.g1_ss} a4{args.g1_a4}"
       f" | g2 tn{args.g2_tile_n} tk{args.g2_tile_k} nt{args.g2_b_nt} xcd{args.g2_xcd} ad{args.g2_a_direct} pf{args.g2_pf} ks{args.g2_ksplit} pm{args.g2_pad_mask} ho{args.g2_hoist} ss{args.g2_ss} | {args.w_layout} sort={args.sort}")
t0 = time.time()
tag += f" g1_impl={args.g1_impl} ctas={args.g1_ctas} stages={args.stages}"
out = run()
torch.cuda.synchronize()
print(f"[a16w4-flydsl] first call (JIT) {time.time() - t0:.1f}s  {tag}", flush=True)

if args.loop:
    for _ in range(args.loop):
        run()
    torch.cuda.synchronize()
    sys.exit(0)

if args.stages == 1:
    args.no_check = True
if not args.no_check:
    out = snapshot(out)
    ref = reference()
    check_ref(out, ref, "swigluoai reference")
    for trial in range(2):
        again = snapshot(run())
        assert torch.equal(out.view(torch.int16), again.view(torch.int16)), f"eager trial {trial + 2}: not bitwise deterministic"
    print("[a16w4-flydsl] eager same input x3: bitwise identical", flush=True)

# ---- HIP-graph replay timing (how vLLM runs it) ----
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        run()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    outs_g = [run(inp) for inp in inputs]
g.replay()
torch.cuda.synchronize()
if not args.no_check:
    # Gemm1 scratch is shared by all calls; after replay it contains the LAST
    # input. Snapshot outside capture so validation adds no work to timing.
    graph_sort = last_sort
    graph_out = snapshot(outs_g[-1], graph_sort)
    graph_ref = reference(inputs[-1]) if len(inputs) > 1 else ref
    check_ref(graph_out, graph_ref, "graph last-input reference")
    for trial in range(2):
        g.replay()
        again = snapshot(outs_g[-1], graph_sort)
        assert torch.equal(graph_out.view(torch.int16), again.view(torch.int16)), f"graph trial {trial + 2}: not bitwise deterministic"
    print("[a16w4-flydsl] graph same input x3: bitwise identical", flush=True)
meds = []
for r in range(args.rounds):
    for _ in range(20):
        g.replay()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(args.reps):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    meds.append(st.elapsed_time(en) * 1000.0 / args.reps / args.graph_copies)
meds.sort()
print(f"[a16w4-flydsl] M={M} graph replay per call: median {meds[len(meds) // 2]:.2f} us "
      f"(min {meds[0]:.2f}, max {meds[-1]:.2f}) over {args.rounds} rounds x {args.reps}"
      f"{f' x {args.graph_copies} copies/graph' if args.graph_copies > 1 else ''}  {tag}")
