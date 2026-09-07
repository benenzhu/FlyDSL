# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Memory-only replay of gemm1's load stream: same tile map (expert-major,
n-major inside an expert, contiguous chunk per XCD), same per-lane LDS-DMA
loads (W13 gate + up slab of the block's expert, optionally the gathered A
rows), same 4-wave CTA with one CTA per CU (96 KB LDS), no MFMA. Answers
"how long does the memory system need for gemm1's traffic" for a given
prefetch depth and W13 layout:

  --layout aiter   the production shuffle_weight(16,16) layout: a K-step of a
                   256-row slab is 16 pieces of 2 KB, 48 KB apart
  --layout kmajor  hypothetical K-step-major slab: the same 32 KB contiguous
  --depth N        batches in flight (gemm1 = 2)
  --with-a         add the gathered A rows (128 B per row per K-step)

Timing: HIP graph of ``--copies`` routings, median of ``--rounds`` x ``--reps``."""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("FLYDSL_ROOT", "/flydsl"))
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import const_expr, range_constexpr  # noqa: E402
from flydsl.expr import rocdl as _rocdl  # noqa: E402
from flydsl._mlir import ir as _ir  # noqa: E402
from flydsl._mlir.dialects import arith as _arith  # noqa: E402
from flydsl.expr.typing import T as _T  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops  # noqa: E402
from m3_a16w4_moe.vllm_ops import import_ops  # noqa: E402

import_ops("moe_a4w4_prefill")
from moe_a4w4_prefill.gemm1 import G2SLoaderAsm, _Buf, _divmod_nonneg, wait_barrier  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=4096)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--bm", type=int, default=128)
p.add_argument("--layout", choices=["aiter", "kmajor"], default="aiter")
p.add_argument("--depth", type=int, default=2)
p.add_argument("--with-a", action="store_true")
p.add_argument("--path", choices=["dma", "vgpr"], default="dma", help="LDS-DMA (gemm1) or global->VGPR loads")
p.add_argument("--cpol", default="", help="cache policy word for the loads (e.g. nt)")
p.add_argument("--copies", type=int, default=8)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K, BM = args.tokens, args.hidden, args.inter, args.experts, args.topk, args.bm
K_BYTES = H // 2
K_ITERS = H // 256
NB_N = I // 128
_N_WAVES = 4
N_TILES_A = BM // 2 // 2 // 16  # gemm1: LDS_BLOCK_M // 2 // 16 (=2 at BM128)
N_TILES_B = 4
A_K_STEP = 128
B_K_STEP = 2048 if args.layout == "aiter" else 256 * 128
DEPTH = args.depth
P = 2 * N_TILES_B + (2 * N_TILES_A if args.with_a else 0)


def build_tile_map(sorted_eids, num_valid, num_m_blocks):
    nb = num_m_blocks
    m_idx = torch.arange(nb, device=dev)
    valid_m = (m_idx * BM) < num_valid[0]
    e_m = torch.where(valid_m, sorted_eids[:nb].long(), torch.full_like(m_idx, E))
    hist = torch.zeros(E + 1, dtype=torch.long, device=dev).scatter_add_(0, e_m, torch.ones_like(e_m))
    starts = torch.cumsum(hist, 0) - hist
    rank = m_idx - starts[e_m]
    cnt = hist[e_m]
    grid = (nb * NB_N + 7) // 8 * 8
    n = torch.arange(NB_N, device=dev)
    idx = (NB_N * starts[e_m])[:, None] + n[None, :] * cnt[:, None] + rank[:, None]
    idx = torch.where(valid_m[:, None], idx, torch.full_like(idx, grid))
    val = (m_idx[:, None] << 3) | n[None, :]
    tm = torch.full((grid + 1,), -1, dtype=torch.long, device=dev)
    tm.scatter_(0, idx.reshape(-1), val.reshape(-1))
    tm[grid] = valid_m.sum() * NB_N
    return tm.to(torch.int32).contiguous(), grid


def compile_stream():
    LDS_BYTES = 96 * 1024  # gemm1's tile ring -> 1 CTA per CU
    LDS_BLOCK_M = BM // 2

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Int8, LDS_BYTES, 16]

    @flyc.kernel
    def kernel_stream(
        W13: fx.Tensor,
        A: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        tile_map_t: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(Shared).peek()
        base_ptr = lds.buf.ptr
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        ids_rsrc = _buffer_ops.create_buffer_resource(sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4))
        eid_rsrc = _buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4)
        intra_xcd, xcd = _divmod_nonneg(fx.block_idx.x, 8)
        remapped = xcd * (grid_size // 8) + intra_xcd
        tm_rsrc = _buffer_ops.create_buffer_resource(tile_map_t, max_size=False, num_records_bytes=grid_size * 4)
        entry = fx.Int32(_buffer_ops.buffer_load(tm_rsrc, remapped, vec_width=1, dtype=fx.Int32))
        tile_i = entry >> 3
        tile_j = entry & 7
        block_valid = entry >= 0
        expert = fx.Int32(_buffer_ops.buffer_load(eid_rsrc, tile_i, vec_width=1, dtype=fx.Int32))
        m_base = tile_i * BM
        if block_valid:
            # ---- B offsets: gemm1's _b_offsets (aiter layout) or K-major ----
            offs_b = []
            for rnd in range_constexpr(N_TILES_B):
                row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                col = (lane_id // 8) * 16
                if const_expr(args.layout == "aiter"):
                    offs_b.append(
                        (row // 16) * (K_BYTES * 16) + (row % 16) * 16 + (col // 64) * 1024 + ((col % 64) // 16) * 256 + (col % 16)
                    )
                else:
                    offs_b.append(row * 128 + col)
            b_row0 = expert * (2 * I) + tile_j * 128
            if const_expr(args.layout == "aiter"):
                B0 = b_row0 * K_BYTES
                B1 = B0 + I * K_BYTES
            else:
                # K-major slab: [expert][n_tile][k][256 rows x 128 B]; up half = rows 128..255
                B0 = expert * (2 * I * K_BYTES) + tile_j * (256 * K_BYTES)
                B1 = B0 + 128 * 128
            b_rsrc = _buffer_ops.create_buffer_resource(W13, max_size=False, num_records_bytes=E * (2 * I) * K_BYTES)
            b_g2s = G2SLoaderAsm(b_rsrc, offs_b, N_TILES_B, wave_id, cpol=args.cpol)
            b_g2s.set_wave_base(base_ptr)
            b_dst0 = _Buf(base_ptr, 0)
            b_dst1 = _Buf(base_ptr, 16 * 1024)
            if const_expr(args.with_a):
                def _a_offs(half):
                    offs = []
                    for rnd in range_constexpr(N_TILES_A):
                        row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                        col = (lane_id % 8) * 16
                        sid = fx.Int32(_buffer_ops.buffer_load(ids_rsrc, m_base + half * LDS_BLOCK_M + row, vec_width=1, dtype=fx.Int32))
                        tok = sid & fx.Int32(0x00FFFFFF)
                        offs.append(tok * fx.Int32(K_BYTES) + col)
                    return offs

                a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=n_tokens * K_BYTES)
                a0_g2s = G2SLoaderAsm(a_rsrc, _a_offs(0), N_TILES_A, wave_id, cpol=args.cpol)
                a1_g2s = G2SLoaderAsm(a_rsrc, _a_offs(1), N_TILES_A, wave_id, cpol=args.cpol)
                a0_g2s.set_wave_base(base_ptr)
                a1_g2s.set_wave_base(base_ptr)
                a_dst0 = _Buf(base_ptr, 32 * 1024)
                a_dst1 = _Buf(base_ptr, 40 * 1024)

            if const_expr(args.path == "dma"):
                def _batch(k):
                    kb = k * fx.Int32(B_K_STEP)
                    ka = k * fx.Int32(A_K_STEP)
                    if const_expr(args.with_a):
                        a0_g2s.load(a_dst0, ka)
                    b_g2s.load(b_dst0, fx.Int32(B0) + kb)
                    b_g2s.load(b_dst1, fx.Int32(B1) + kb)
                    if const_expr(args.with_a):
                        a1_g2s.load(a_dst1, ka)

                for d in range_constexpr(DEPTH):
                    _batch(fx.Int32(d))
                for k, st in range(0, K_ITERS, init=[fx.as_ir_value(fx.Int32(0))]):
                    k_i = fx.Int32(k)
                    # batch k landed; DEPTH-1 newer batches may fly
                    wait_barrier((DEPTH - 1) * P)
                    nxt = k_i + fx.Int32(DEPTH)
                    nxt = fx.arith.select(nxt < fx.Int32(K_ITERS), nxt, fx.Int32(K_ITERS - 1))
                    _batch(nxt)
                    st = yield [st[0]]
                wait_barrier(0)
            else:
                # global -> VGPR: DEPTH register sets of P dwordx4 each, consumed by XOR
                v4 = _ir.VectorType.get([4], _T.i32)
                zero4 = _arith.ConstantOp(v4, _ir.DenseElementsAttr.get_splat(v4, _ir.IntegerAttr.get(_T.i32, 0))).result
                if const_expr(args.with_a):
                    a_offs_all = _a_offs(0) + _a_offs(1)
                    a_rsrc_v = _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=n_tokens * K_BYTES)
                b_offs_all = [fx.Int32(B0) + o for o in offs_b] + [fx.Int32(B1) + o for o in offs_b]

                def _batch_v(k):
                    kb = fx.as_ir_value(k * fx.Int32(B_K_STEP))
                    ka = fx.as_ir_value(k * fx.Int32(A_K_STEP))
                    vals = []
                    if const_expr(args.with_a):
                        vals += [
                            _buffer_ops.buffer_load(a_rsrc_v, o // fx.Int32(4), vec_width=4, dtype=fx.Int32, soffset_bytes=ka)
                            for o in a_offs_all
                        ]
                    vals += [
                        _buffer_ops.buffer_load(b_rsrc, o // fx.Int32(4), vec_width=4, dtype=fx.Int32, soffset_bytes=kb)
                        for o in b_offs_all
                    ]
                    return [fx.as_ir_value(v) for v in vals]

                sets = []
                for d in range_constexpr(DEPTH):
                    sets += _batch_v(fx.Int32(d))
                for k, st in range(0, K_ITERS, init=[zero4] + sets):
                    acc = st[0]
                    regs = list(st[1:])
                    k_i = fx.Int32(k)
                    _rocdl.s_waitcnt(vmcnt=(DEPTH - 1) * P, lgkmcnt=0)
                    _rocdl.s_barrier()
                    # consume set (k % DEPTH) = the oldest = the first P regs after rotation
                    for v in regs[:P]:
                        acc = _arith.XOrIOp(acc, v).result
                    nxt = k_i + fx.Int32(DEPTH)
                    nxt = fx.arith.select(nxt < fx.Int32(K_ITERS), nxt, fx.Int32(K_ITERS - 1))
                    new = _batch_v(nxt)
                    st = yield [acc] + regs[P:] + new
                _rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                acc = st[0]
                for v in st[1:]:
                    acc = _arith.XOrIOp(acc, v).result
                out_rsrc = _buffer_ops.create_buffer_resource(tile_map_t, max_size=False, num_records_bytes=grid_size * 4)
                # keep the loads alive: a store nobody reads, only from lane 0 of a block that never exists
                if fx.block_idx.x == fx.Int32(-1):
                    _buffer_ops.buffer_store(acc, out_rsrc, fx.Int32(0), offset_is_bytes=True)

    @flyc.jit
    def launch(
        W13: fx.Tensor,
        A: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        tile_map_t: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_stream(W13, A, sorted_ids, sorted_expert_ids, tile_map_t, n_tokens, num_m_blocks, grid_size).launch(
            grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


w13 = torch.randint(0, 255, (E * 2 * I * K_BYTES,), dtype=torch.uint8, device=dev)
a_q = torch.randint(0, 255, (M * K_BYTES,), dtype=torch.uint8, device=dev)


class Case:
    def __init__(self):
        routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
        topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
        topk_w = torch.ones((M, K), dtype=torch.float32, device=dev)
        sorted_ids, _sw, sorted_eids, num_valid, _buf = moe_sorting(topk_ids, topk_w, E, H, torch.bfloat16, block_size=BM)
        self.sorted_ids = sorted_ids
        self.sorted_eids = sorted_eids
        self.num_valid = num_valid
        self.num_m_blocks = int(sorted_eids.shape[0])
        self.tile_map, self.grid = build_tile_map(sorted_eids, num_valid, self.num_m_blocks)
        self.nv = int(num_valid[0])

    def args(self):
        return (w13, a_q, self.sorted_ids, self.sorted_eids, self.tile_map, M, self.num_m_blocks, self.grid, torch.cuda.current_stream())


t0 = time.time()
cases = [Case() for _ in range(args.copies)]
launch = compile_stream()
fn = flyc.compile(launch, *cases[0].args())
for c in cases:
    fn(*c.args())
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(args.reps):
        for c in cases:
            fn(*c.args())
torch.cuda.synchronize()
meds = []
for _ in range(args.rounds):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    meds.append(s.elapsed_time(e) * 1e3 / (args.reps * args.copies))
meds.sort()
us = meds[len(meds) // 2]
blocks = sum(c.nv // BM for c in cases) / len(cases)
w_bytes = blocks * NB_N * 2 * 128 * K_BYTES  # every block streams its 256-row slab once (L2 dedups)
a_bytes = blocks * NB_N * BM * K_BYTES if args.with_a else 0
print(
    f"[wstream] tokens={M} BM={BM} layout={args.layout} path={args.path} cpol={args.cpol!r} depth={DEPTH} with_a={args.with_a}: "
    f"median {us:.1f} us (range {meds[0]:.1f}..{meds[-1]:.1f}); issued W {w_bytes / 1e6:.0f} MB + A {a_bytes / 1e6:.0f} MB "
    f"-> {(w_bytes + a_bytes) / us / 1e6:.2f} TB/s issued (setup {time.time() - t0:.1f}s)",
    flush=True,
)
