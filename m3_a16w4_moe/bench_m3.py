#!/usr/bin/env python3
"""MiniMax-M3 (TP4) decode MoE (M <= 256) on the FlyDSL a16w4 chain, vs the aiter sort.

Runs inside the production vLLM image. The kernels are the vLLM ones
(``vllm/models/minimax_m3/amd/ops/moe_a16w4_decode``, imported through
``vllm_ops.py`` from the worktree on /dev/shm). Chain per call:
  sort (aiter moe_sorting / vLLM sort_decode / lab sort_decode_wave / sort-free pairs)
  ->  gemm1 (gate/up + swigluaoi, bf16 out)  ->  gemm2 (down, routing-weighted atomic add)
Inputs are generated exactly like m3-compare/scripts/bench_moe_m4.py (same seed -> same
weights and routing); every captured call has its own input (weights come from HBM like real
decode). The tiles are the production ones, fixed inside the kernels (gemm1.py / gemm2.py).

  PYTHONPATH=/flydsl python3 /flydsl/m3_a16w4_moe/bench_m3.py --tokens 32 --sort decode
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=4)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tile-m", type=int, default=16, help="sort block / gemm m-block rows (the kernels are built for 16)")
# gemm1 / gemm2 tiles are fixed in the vLLM kernels (gemm1.py LARGE_M_TOKENS, gemm2.py KSPLIT_SMALL_M_TOKENS)
p.add_argument("--w-layout", default="standard", choices=["standard", "guinterleave"])
p.add_argument("--sort", default="aiter", choices=["aiter", "mxfp4", "pairs", "decode", "decode-wave"],
               help="aiter: opus moe_sorting; mxfp4: aiter#3832 single-CTA sort + zero-init; decode: the vLLM "
                    "sort_decode kernel (production); decode-wave: lab sort_decode_wave.py; pairs: sort-free "
                    "(n_tokens <= 16, production for M <= 16)")
p.add_argument("--reps", type=int, default=10)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--no-check", action="store_true")
p.add_argument("--check-determinism", action="store_true", help="optional bitwise diagnostic; cosine is the default acceptance check")
p.add_argument("--no-shared", action="store_true", help="timing experiment: all 5 slots routed (no M-row shared expert)")
p.add_argument("--loop", type=int, default=0, help="run N eager iterations and exit (for rocprofv3)")
p.add_argument("--graph-copies", type=int, default=100,
               help="calls captured per graph, each with its OWN x / routing (different experts -> weights come "
                    "from HBM like real decode, and the per-graph launch cost is amortised as in vLLM's model graph)")
p.add_argument("--stages", type=int, default=3, choices=[1, 2, 3], help="1: sort only, 2: sort+gemm1, 3: full chain")
args = p.parse_args()

import torch  # noqa: E402
import aiter  # noqa: E402,F401
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting, _adaptive_moe_sort  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402
from m3_a16w4_moe.vllm_ops import import_ops  # noqa: E402

dec = import_ops("moe_a16w4_decode")
from moe_a16w4_decode.host import a16w4_gemm1, a16w4_gemm2  # noqa: E402
from moe_a16w4_decode.sort_decode import moe_sort_decode  # noqa: E402
if args.sort == "decode-wave":
    from m3_a16w4_moe.sort_decode_wave import moe_sort_decode  # noqa: F811

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

assert BM == 16, "the vLLM decode kernels are built for 16-row m-blocks"
g1_kw = dict(alpha=ALPHA, swiglu_limit=LIMIT, w_layout=args.w_layout)
g2_kw = {}


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


large_m, small_m = M > dec.gemm1.LARGE_M_TOKENS, M <= dec.gemm2.KSPLIT_SMALL_M_TOKENS
tag = (f"g1 {'tn64 kw1 wpe3' if large_m else 'tn32 kw2'} kb2 pf3 nt2 | g2 tn256 tk256 nt2 ks{3 if small_m else 1} "
       f"| {args.w_layout} sort={args.sort} stages={args.stages}")
t0 = time.time()
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
    for trial in range(2 if args.check_determinism else 0):
        again = snapshot(run())
        assert torch.equal(out.view(torch.int16), again.view(torch.int16)), f"eager trial {trial + 2}: not bitwise deterministic"
    if args.check_determinism:
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
    for trial in range(2 if args.check_determinism else 0):
        g.replay()
        again = snapshot(outs_g[-1], graph_sort)
        assert torch.equal(graph_out.view(torch.int16), again.view(torch.int16)), f"graph trial {trial + 2}: not bitwise deterministic"
    if args.check_determinism:
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
