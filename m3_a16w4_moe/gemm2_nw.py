# SPDX-License-Identifier: Apache-2.0
"""Decode down projection (bf16 sorted intermediate x mxfp4 W2, routing-weighted bf16 atomic add):
A tile staged once in LDS, NB n-blocks per workgroup on one continuous W ring.

``gemm2.py`` runs one workgroup per (16-row m-block x TILE_N columns) over the whole K
(768 = 6 K-tiles of 128): 96 KB of W2 per workgroup, the 24 KB A tile re-fetched by every
n-block through LDS-DMA with a vmcnt(0) drain per K-tile, so a workgroup is half prologue
and tail (62-65 us for 322 MB at M=256 = 5.2 TB/s; 4 workgroups per CU hide part of it).

Here a workgroup owns ``NB`` consecutive n-blocks of one m-block: the A tile is loaded once
(6 dwordx4 per lane, each instruction 4 rows x 256 B = 8 full lines) into LDS (row stride
K*2+16 B, conflict-free 16 B reads), then the NB*6 K-tiles run as one flat W ring
``prefetch`` deep with no drains, and each n-block ends with gemm2.py's LDS-transposed
routing-weighted bf16 atomic epilogue (``_atomic_bf16_epilog``; its LDS region is separate
from the A tile so the next n-block's ring keeps running under the atomics).
W ring / scale / dequant / MFMA as in ``gemm1_nw.py`` (read that first).
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .gemm1_nw import _sconst
from .gemm2 import _atomic_bf16_epilog
from .host import _run_compiled
from .utils import _e8m0_byte_to_f32, _gep1, _global_base_ptr1, _global_i32_at, _raw, buffer_ops as bop, s_waitcnt_lgkm0

# timing-only diagnostics (wrong results): FAKE_W = every block reads expert 0, NO_A = no A loads / LDS reads
_G2_FAKE_W = int(os.environ.get("M3_NW_G2_FAKE_W", "0"))
_G2_NO_A = int(os.environ.get("M3_NW_G2_NO_A", "0"))
_G2_NO_EPI = int(os.environ.get("M3_NW_G2_NO_EPI", "0"))
_G2_SCHED = int(os.environ.get("M3_NW_G2_SCHED", "1"))


def compile_gemm2_nw(*, NE, N_OUT, D_INTER, BM=16, TILE_N=256, NB=4, prefetch=3, b_cache_mod=2, waves_per_eu=None):
    K = D_INTER
    assert BM == 16 and TILE_N in (128, 256) and N_OUT % (TILE_N * NB) == 0 and K % 256 == 0
    KT = K // 128
    NT = NB * KT                       # K-tiles per workgroup (flat ring)
    assert 1 <= prefetch < NT
    NPW = TILE_N // 4
    NI = NPW // 16
    NNB = N_OUT // TILE_N              # n-blocks per m-block
    NGRP = NNB // NB                   # n-groups (workgroups) per m-block
    ROWB = K * 2                       # bytes of one A row
    RS = ROWB + 16
    A_LDS = BM * RS
    EPI_LDS = BM * TILE_N * 4
    LDS_BYTES = A_LDS + EPI_LDS
    W_BYTES = NE * N_OUT * (K // 2)
    SC_K1 = (((K + 255) // 256) * 256) // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    NLD = (BM * ROWB) // (256 * 16)    # dwordx4 per lane for the A tile (= KT)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    _wpe = f"_w{waves_per_eu}" if waves_per_eu else ""

    @flyc.kernel(name=f"gemm2_a16w4_nw_ne{NE}_h{N_OUT}_i{K}_bm{BM}_tn{TILE_N}_nb{NB}_pf{prefetch}_bcm{b_cache_mod}{_wpe}",
                 known_block_size=[256, 1, 1])
    def kernel(
        arg_a: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
        arg_stids: fx.Int64, arg_sweights: fx.Int64, i32_M: fx.Int32, arg_out: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        cumsum0 = fx.Int32(_global_i32_at(arg_cumsum, fx.Int32(0)))
        mb, ng = pid // fx.Int32(NGRP), pid % fx.Int32(NGRP)
        mbase = mb * fx.Int32(BM)
        if mbase < cumsum0:
            e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(_global_i32_at(arg_eids, mb)))))
            if const_expr(_G2_FAKE_W):
                e = fx.Int32(0)
            # epilogue routing (token | slot<<24, weight) per row mr*8 + tx//32, issued up front
            stids_base = _global_base_ptr1(arg_stids)
            sw_base = _global_base_ptr1(arg_sweights)
            m_lane = tx // fx.Int32(32)
            pre_packed = [llvm.load(T.i32, _gep1(stids_base, (mbase + fx.Int32(mr * 8) + m_lane) * fx.Int32(4)), invariant=True)
                          for mr in range_constexpr(BM // 8)]
            pre_weight = [llvm.load(T.f32, _gep1(sw_base, (mbase + fx.Int32(mr * 8) + m_lane) * fx.Int32(4)), invariant=True)
                          for mr in range_constexpr(BM // 8)]
            ar = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_a)), num_records_bytes=_raw(fx.Int64(cumsum0) * fx.Int64(ROWB)))
            wr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_bq)), num_records_bytes=min(W_BYTES, 0xFFFFFFFF))
            sr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_bscale)), num_records_bytes=min(SW_BYTES, 0xFFFFFFFF))
            lds_i32 = fx.Int32(fx.ptrtoint(smem))
            lds = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(lds_i32))

            # A tile (sorted rows are contiguous): load j covers K-tile j, lane -> row tx//16, chunk tx%16
            ld_row = tx // fx.Int32(16)
            ld_gdw = ((mbase + ld_row) * fx.Int32(ROWB) + l16 * fx.Int32(16)) // fx.Int32(4)
            ld_lbyte = ld_row * fx.Int32(RS) + l16 * fx.Int32(16)
            if const_expr(not _G2_NO_A):
                abuf = [bop.buffer_load(ar, ld_gdw + fx.Int32(j * 64), vec_width=4, dtype=fx.Int32) for j in range_constexpr(NLD)]

            # ---- W addressing: per column tile a vector address; n-block / K position in an SGPR + imm ----
            col0 = ng * fx.Int32(NB * TILE_N) + wave * fx.Int32(NPW)     # first column of this wave in n-block 0
            nblk0 = [(e * fx.Int32(N_OUT) + col0 + fx.Int32(ni * 16)) // fx.Int32(16) for ni in range_constexpr(NI)]
            mni0 = [(e * fx.Int32(N_OUT) + col0 + fx.Int32(ni * 16)) // fx.Int32(32) for ni in range_constexpr(NI)]
            npk = [((col0 + fx.Int32(ni * 16)) // fx.Int32(16)) % fx.Int32(2) for ni in range_constexpr(NI)]
            wvo = [lane * fx.Int32(4) + nblk0[ni] * fx.Int32(KT * 256) for ni in range_constexpr(NI)]
            svo = [lane + mni0[ni] * fx.Int32(SC_STRIDE_N0) for ni in range_constexpr(NI)]

            def load_b_tile(t, prev):
                i, kt = t // KT, t % KT
                so = _sconst(i * (TILE_N // 16) * KT * 1024 + (kt // 4) * 4096)
                bb = [bop.buffer_load(wr, wvo[ni] + fx.Int32((kt % 4) * 256), vec_width=4, dtype=fx.Int32,
                                      cache_modifier=b_cache_mod, soffset_bytes=so) for ni in range_constexpr(NI)]
                if const_expr(kt % 2 == 0):
                    sso = _sconst((i * (TILE_N // 32) * SC_STRIDE_N0 + (kt // 2) * 64) * 4)
                    sc = [bop.buffer_load(sr, svo[ni], vec_width=1, dtype=fx.Int32, soffset_bytes=sso) for ni in range_constexpr(NI)]
                else:
                    sc = prev[1]
                return bb, sc

            rd_base = l16 * fx.Int32(RS) + q16 * fx.Int32(64)

            def read_a_tile(kt):
                out = []
                for ku in range_constexpr(4):
                    if const_expr(_G2_NO_A):
                        out.append(fx.Vector.filled(4, ku + 1, fx.Int32).bitcast(fx.BFloat16))
                        continue
                    ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(rd_base + fx.Int32(kt * 256 + ku * 16)), elem_type=T.i8)
                    out.append(fx.Vector(llvm.load(T.vec(4, T.i32), ptr, alignment=16)).bitcast(fx.BFloat16))
                return out

            vec2_bf16 = ir.Type.parse("vector<2xbf16>")

            def upconvert(raw4, ku, scale_f32):
                i32_val = _raw(fx.Int32(raw4[ku]))
                s_raw = _raw(scale_f32)
                i32s = []
                for sel in range_constexpr(4):
                    p = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
                    i32s.append(fx.Int32(fx.Vector(p).bitcast(fx.Int32)[0]))
                return fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32).bitcast(fx.BFloat16)

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            acc_layout = fx.make_layout(4, 1)
            acc = [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(NI)]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for ni in range_constexpr(NI):
                acc[ni].store(zero4)

            def _frag(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def compute_tile(bb, sc, aa, kt):
                a_t = [_frag(aa[ku]) for ku in range_constexpr(4)]
                for ni in range_constexpr(NI):
                    s = _e8m0_byte_to_f32(fx.Int32(sc[ni]), fx.Int32((kt % 2) * 2) + npk[ni])
                    raw4 = fx.Vector(bb[ni])
                    for ku in range_constexpr(4):
                        fx.gemm(mma_atom, acc[ni], a_t[ku], _frag(upconvert(raw4, ku, s)), acc[ni])

            # ---- W ring first (HBM), then the A tile into LDS (its wait does not touch the ring) ----
            ring = []
            for t in range_constexpr(prefetch):
                ring.append(load_b_tile(t, ring[-1] if ring else None))
            if const_expr(not _G2_NO_A):
                for j in range_constexpr(NLD):
                    ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(ld_lbyte + fx.Int32(j * 256)), elem_type=T.i8)
                    llvm.StoreOp(fx.as_ir_value(abuf[j]), ptr, alignment=16)
            s_waitcnt_lgkm0()
            gpu.barrier()
            for t in range_constexpr(NT):
                i, kt = t // KT, t % KT
                if const_expr(t + prefetch < NT):
                    ring.append(load_b_tile(t + prefetch, ring[-1]))
                bb, sc = ring.pop(0)
                aa = read_a_tile(kt)
                compute_tile(bb, sc, aa, kt)
                if const_expr(_G2_SCHED):
                    rocdl.sched_barrier(0)
                if const_expr(kt == KT - 1):
                    # n-block i done: routing-weighted bf16 atomic scatter through the epilogue LDS region
                    # (barrier: the previous n-block's readers are done with it), then restart the accumulators
                    if const_expr(not _G2_NO_EPI):
                        gpu.barrier()
                        _atomic_bf16_epilog(
                            lds_i32 + fx.Int32(A_LDS), [[fx.memref_load_vec(acc[ni]) for ni in range_constexpr(NI)]],
                            arg_out, arg_stids, arg_sweights, mbase, ng * fx.Int32(NB) + fx.Int32(i), wave, lane, i32_M,
                            BM, N_OUT, TILE_N, pre_packed=pre_packed, pre_weight=pre_weight,
                        )
                    if const_expr(i + 1 < NB):
                        for ni in range_constexpr(NI):
                            acc[ni].store(zero4)

    @flyc.jit
    def launch(
        arg_a: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
        arg_stids: fx.Int64, arg_sweights: fx.Int64, i32_M: fx.Int32, i32_grid: fx.Int32, arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_a, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids, arg_sweights, i32_M, arg_out,
            **({"value_attrs": {"rocdl.waves_per_eu": waves_per_eu}} if waves_per_eu else {}),
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


@functools.cache
def get_gemm2_nw(**kw):
    return compile_gemm2_nw(**kw)


def a16w4_gemm2_nw(
    *, inter_sorted_bf16, w2_u8, w2_scale_u8, out_bf16, n_tokens, NE, D_HIDDEN, D_INTER, tile_m, tile_n, tile_k,
    sorted_expert_ids, num_valid_ids, sorted_token_ids, sorted_weights, nb=4, prefetch=3, b_nt=2, waves_per_eu=None,
    stream=None, **_unused,
):
    """Drop-in for ``host.a16w4_gemm2`` on the sorted path (tile_m 16); ``nb`` n-blocks per workgroup."""
    assert tile_m == 16
    launch = get_gemm2_nw(NE=NE, N_OUT=D_HIDDEN, D_INTER=D_INTER, BM=tile_m, TILE_N=tile_n, NB=nb,
                          prefetch=prefetch, b_cache_mod=b_nt, waves_per_eu=waves_per_eu)
    grid = int(sorted_expert_ids.numel()) * (D_HIDDEN // tile_n // nb)
    _run_compiled(
        launch, inter_sorted_bf16.data_ptr(), w2_u8.data_ptr(), w2_scale_u8.data_ptr(), sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(), sorted_token_ids.data_ptr(), sorted_weights.data_ptr(), int(n_tokens), int(grid),
        out_bf16.data_ptr(), torch.cuda.current_stream() if stream is None else stream,
    )
    return out_bf16
