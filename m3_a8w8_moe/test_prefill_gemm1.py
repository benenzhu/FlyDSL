# SPDX-License-Identifier: Apache-2.0
"""Stage-1 check of moe_a8w8_prefill.gemm1 (a8w8): a4w4 sort + aiter fp8 quant +
tile map + gemm1, the fp8 intermediate compared per sorted row with a float
reference built from the dequantized fp8 activations and experts. Also times the
stage (and gemm1 alone) in a HIP graph over different inputs.

    python3 -m m3_a8w8_moe.test_prefill_gemm1 --tokens 4096 8192 [--rows 64]
"""
import argparse
import time

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import (
    HIDDEN, INTER, NUM_EXPERTS, SWIGLU_ALPHA, SWIGLU_LIMIT, cos, dequant, make_weights, routing,
)

import_ops("moe_a16w4_decode")
import_ops("moe_a4w4_prefill")
import_ops("moe_a8w8_prefill")
from moe_a8w8_prefill import a8w8_prefill_stage1  # noqa: E402


def stage1(x, shuffled, ids, w):
    w13, w2, w13_s, w2_s = shuffled
    return a8w8_prefill_stage1(x, w13, w13_s, w, ids, hidden_size=HIDDEN,
                               intermediate_size=INTER, num_experts=NUM_EXPERTS)


def prefix_only(x, shuffled, ids, w):
    """the stage without gemm1 (same calls as a8w8_prefill_stage1), for timing"""
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort
    from moe_a4w4_prefill import _get_sort, _get_tile_map, _run_compiled, block_m_for
    from moe_a4w4_prefill.sort import SortBuffers
    from moe_a4w4_prefill.tile_map import tile_map_grid

    n_tokens = x.shape[0]
    bm = block_m_for(n_tokens)
    bufs = SortBuffers.allocate(n_tokens, NUM_EXPERTS, 5, bm, x.device)
    _get_sort(NUM_EXPERTS, 5, bm)(*bufs.launch_args(ids, w, n_tokens))
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(x, bufs.sorted_ids, bufs.num_valid_ids, token_num=n_tokens,
                                               topk=5, block_size=bm, quant_dtype=dtypes.fp8)
    num_m_blocks = bufs.max_sorted // bm
    grid1 = tile_map_grid(num_m_blocks, INTER)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=x.device)
    _run_compiled(_get_tile_map(INTER, bm), bufs.sorted_expert_ids, bufs.num_valid_ids, tile_map, grid1,
                  torch.cuda.current_stream())
    return a_q, a_s, tile_map


def dequant_rows(q_u8, s_u8, cols):
    """fp8 rows [n, cols] (uint8 view) x e8m0 [n, cols/32] -> f32."""
    q = q_u8.view(torch.float8_e4m3fn).float()
    sc = torch.ldexp(torch.ones_like(s_u8, dtype=torch.float32), s_u8.to(torch.int32) - 127)
    n = q.shape[0]
    return (q.view(n, cols // 32, 32) * sc.view(n, cols // 32, 1)).view(n, cols)


def unshuffle_scale_rows(s_flat, rows, cols_grp, row_list):
    """e8m0 [pad32(rows), cols_grp] in the moe-sort shuffled layout -> per-row [len, cols_grp]
    (block = 32 rows x 8 groups = 256 B: byte kl*64 + nl*4 + kp*2 + n2 -> row 16*n2 + nl,
    group 4*kp + kl; aiter mx_scale_shuffle_idx)."""
    out = torch.empty((len(row_list), cols_grp), dtype=torch.uint8, device=s_flat.device)
    blocks_per_row32 = cols_grp // 8
    for i, r in enumerate(row_list):
        r32, rin = r // 32, r % 32
        n2, nl = rin // 16, rin % 16
        for g in range(cols_grp):
            blk = r32 * blocks_per_row32 + g // 8
            gin = g % 8  # lane klane's dword, byte kp*2 + n2 = K block 128*kp + 32*klane
            kl, kp = gin % 4, gin // 4
            out[i, g] = s_flat[blk * 256 + kl * 64 + nl * 4 + kp * 2 + n2]
    return out


def check(x, raw, shuffled, ids, w, n_rows, seed):
    w13_q, w13_s, _, _ = raw
    bufs, a_q, a_s, h_q, h_s, nmb = stage1(x, shuffled, ids, w)
    torch.cuda.synchronize()
    n_valid = int(bufs.num_valid_ids[0].item())
    sorted_ids = bufs.sorted_ids[:n_valid].cpu()
    eids = bufs.sorted_expert_ids.cpu()
    bm = bufs.block_m
    valid_rows = [r for r in range(n_valid) if int(sorted_ids[r]) & 0xFFFFFF < x.shape[0]]
    gen = torch.Generator().manual_seed(seed)
    pick = [valid_rows[i] for i in torch.randperm(len(valid_rows), generator=gen)[:n_rows].tolist()]
    # activations as the kernel sees them: fp8 per token (a_q) x e8m0 per sorted row (a_s)
    toks = [int(sorted_ids[r]) & 0xFFFFFF for r in pick]
    a_scale_rows = unshuffle_scale_rows(a_s.view(torch.uint8).view(-1), nmb * bm, HIDDEN // 32, pick)
    xa = dequant_rows(a_q.view(torch.uint8)[toks], a_scale_rows, HIDDEN)
    xcos = min(cos(xa[i], x[toks[i]].float()) for i in range(len(pick)))
    print(f"  activation dequant check (a_q x unshuffled a_s vs x): worst cos {xcos:.5f}", flush=True)
    h_scale_rows = unshuffle_scale_rows(h_s, nmb * bm, INTER // 32, pick)
    got = dequant_rows(h_q[pick], h_scale_rows, INTER)
    worst = 1.0
    n_exp_match = n_exp = n_q_match = n_q_1ulp = n_q = 0
    for i, r in enumerate(pick):
        e = int(eids[r // bm])
        hh = xa[i] @ dequant(w13_q[e], w13_s[e]).T
        g = hh[:INTER].clamp(max=SWIGLU_LIMIT)
        u = hh[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        ref = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
        c = cos(got[i], ref)
        # fp8 e4m3 per-element error is <= 2^-4 relative: cos ~0.9995
        if c < worst:
            worst = c
            worst_row = (r, e, (got[i] - ref).abs().max().item(), ref.abs().max().item())
        # the kernel's rounding rule: e8m0 = ceil_pow2(amax/448), then RNE to e4m3
        amax = ref.view(INTER // 32, 32).abs().amax(dim=1)
        exp = torch.ceil(torch.log2(amax / 448.0)).clamp(min=-127)
        exp = torch.where(amax > 0, exp, torch.full_like(exp, -127))
        n_exp += exp.numel()
        n_exp_match += int((exp.to(torch.int32) + 127 == h_scale_rows[i].to(torch.int32)).sum())
        qref = (ref.view(INTER // 32, 32) / torch.exp2(exp).view(-1, 1)).view(INTER)
        qref = qref.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
        qgot = h_q[pick[i]].to(torch.int32)
        same_exp = (exp.to(torch.int32) + 127 == h_scale_rows[i].to(torch.int32)).repeat_interleave(32)
        d = (qref - qgot).abs()[same_exp]
        n_q += d.numel()
        n_q_match += int((d == 0).sum())
        n_q_1ulp += int((d <= 1).sum())
    print(f"  e8m0 match {n_exp_match}/{n_exp}; fp8 bits (groups with matching e8m0): "
          f"exact {n_q_match / max(n_q, 1):.4f}, within 1 ulp {n_q_1ulp / max(n_q, 1):.5f}", flush=True)
    return worst, worst_row, n_valid


def time_graph(fn, inputs, rounds=3):
    for inp in inputs[:2]:
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
    return sorted(meds)[len(meds) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[4096])
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--copies", type=int, default=4)
    p.add_argument("--no-time", action="store_true")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        worst, wr, n_valid = check(x, raw, shuffled, ids, w, a.rows, m)
        print(f"M={m}: {n_valid} sorted rows, worst row cos {worst:.5f} (row {wr[0]} expert {wr[1]} "
              f"max|d| {wr[2]:.4f} ref max {wr[3]:.3f})", flush=True)
        if a.no_time:
            continue
        inputs = []
        for i in range(a.copies):
            torch.manual_seed(1000 + i)
            xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids_i, w_i = routing(m, dev)
            inputs.append((xi, shuffled, ids_i, w_i))
        t = time_graph(stage1, inputs)
        t0 = time_graph(prefix_only, inputs)
        print(f"[a8w8 prefill stage1] M={m}: sort+quant+tilemap+gemm1 {t:.1f} us, "
              f"sort+quant+tilemap {t0:.1f} us -> gemm1 ~{t - t0:.1f} us", flush=True)


if __name__ == "__main__":
    main()
