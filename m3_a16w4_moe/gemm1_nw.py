# SPDX-License-Identifier: Apache-2.0
"""Decode gate/up (bf16 A x mxfp4 W1, swigluoai) with 4 N-waves sharing one A batch through LDS.

The production decode gemm1 (``gemm1.py``: tile-m 16 / tile-n 32 / k_wave 4 / a_direct) gives
each of the 4 waves a quarter of K, so every wave gathers its own 16-row A fragments straight
into VGPRs: 4 dwordx4 per lane per 128-K tile, each instruction touching 16 different rows
(16 L1 requests for 1 KB). Timing diagnostics (M3_D1_NO_A=1) put that gather at 17 of 124 us
at M=256 and 6 of 108 at M=128; the kernel streams W at 5.4 TB/s at M=256 where M=64 reaches 6.6.

Here the 4 waves split N instead (TILE_N/4 columns of gate and of up each) and share the A
rows: the workgroup loads one batch of ``KB`` 128-K tiles (16 rows x KB*256 B; one dwordx4 per
lane per K-tile, each instruction 4 rows x 256 B = 8 full 128 B lines) into LDS and every wave
ds_reads its MFMA A fragments from there. Two LDS slots (row stride KB*256+16 B, conflict-free
16 B reads), one barrier per batch; the batch loads go out before the same iteration's W loads
so the in-order vmcnt wait for that W tile also covers them and the W ring keeps its depth.
No K reduce, no reduce barriers; per K-tile per wave: 2*NI W loads (1 KB each, contiguous in
the preshuffled layout), 4 ds_read_b128, 8*NI MFMA 16x16x32 bf16. W ring, scale sharing
(one packed dword per 256 K), dequant and the swigluoai epilogue follow ``gemm1.py``.

TILE_N 64/128/256 -> 12/6/3 n-blocks per expert (A re-read 12/6/3x instead of 24x). At M=256
that is 1872/936/468 workgroups: coarser tiles amortise the prologue but balance worse over
the 256 CUs, so sweep it per M.
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

from .gemm1 import _swigluoai_f32
from .host import _run_compiled
from .utils import _e8m0_byte_to_f32, _global_i32_at, _raw, buffer_ops as bop, s_waitcnt_lgkm0

_NW_STORE_CPOL = int(os.environ.get("M3_D1_STORE_CPOL", "0"), 0)
# timing-only diagnostics (wrong results): FAKE_A = every block reads rows 0..15 (A always L2-hot),
# FAKE_W = every block reads expert 0, NO_A = no A loads / LDS traffic at all.
_NW_FAKE_A = int(os.environ.get("M3_NW_FAKE_A", "0"))
_NW_FAKE_W = int(os.environ.get("M3_NW_FAKE_W", "0"))
_NW_NO_A = int(os.environ.get("M3_NW_NO_A", "0"))
# LDS row padding in bytes (16 = conflict-free 16 B reads; knob for the bank sweep)
_NW_PAD = int(os.environ.get("M3_NW_PAD", "16"))
# 1 (default): a scheduling barrier after every K-tile's MFMAs. Without it the machine scheduler hoists the
# fp4->bf16 conversions of the next 2-3 tiles above the current tile's MFMAs (runs of 45-62 v_cvt in the
# ISA), keeps their 4x larger bf16 results live and spills: 256 VGPR + 160 AGPR, 1 wave/SIMD, 111 vmcnt(0)
# drains, 280 us at M=256 (2.3x slower than gemm1.py).
_NW_SCHED = int(os.environ.get("M3_NW_SCHED", "1"))


def _sconst(v):
    """An i32 constant in an SGPR behind an asm the backend cannot see through. Passed as a buffer load's
    soffset it stays scalar; a plain constant (or uniform sum) there gets folded into the vector address
    instead, one VGPR per 4 K-tiles per column tile (66 address VGPRs for 240 loads, spills to AGPR)."""
    return fx.Int32(llvm.inline_asm(T.i32, [_raw(fx.Int32(v))], "s_mov_b32 $0, $1", "=s,s", has_side_effects=True))


def compile_gemm1_nw(*, D_HIDDEN, D_INTER, NE, TOPK, BM=16, TILE_N=128, prefetch=3, k_batch=4,
                     b_cache_mod=2, w_layout="standard", waves_per_eu=None):
    K, INTER = D_HIDDEN, D_INTER
    N_OUT = 2 * INTER
    assert BM == 16 and TILE_N in (64, 128, 256) and INTER % TILE_N == 0 and K % 256 == 0
    assert w_layout in ("standard", "guinterleave")
    KT = K // 128                      # 128-K tiles: 1 KB of W per 16 columns, 256 B of A per row
    KB = k_batch
    assert KT % KB == 0 and KB % 2 == 0 and 1 <= prefetch < KT
    NPW = TILE_N // 4                  # columns per wave (gate and up each)
    NI = NPW // 16
    NNB = INTER // TILE_N
    ROWB = KB * 256                    # A bytes per row per batch
    RS = ROWB + _NW_PAD                # padded LDS row stride
    SLOT = BM * RS
    LDS_BYTES = 2 * SLOT
    # W (mxfp4) preshuffle layout (aiter make_preshuffle_b_layout, N-major, fp4 bytes):
    # (N_OUT/16, K/128, klane 4, nlane 16, kpack 16 B): one 16-col x 128-K block = 1 KB contiguous,
    # lane (klane, n) holds the 32 K of column n at klane -> voffset = lane * 16 B.
    W_BYTES = NE * N_OUT * (K // 2)
    # scale preshuffle (make_preshuffle_scale_layout, e8m0 u8): (N_OUT/32, K_pad/256, klane 4, nlane 16)
    # dwords; one dword = 4 bytes = 2 K-tiles (128 K each) x 2 N-halves (16 cols each).
    SC_K1 = (((K + 255) // 256) * 256) // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    _wl = "" if w_layout == "standard" else "_gu"
    _wpe = f"_w{waves_per_eu}" if waves_per_eu else ""

    @flyc.kernel(name=f"gemm1_a16w4_nw{_wl}_h{K}_i{INTER}_ne{NE}_bm{BM}_tn{TILE_N}_pf{prefetch}_kb{KB}_bcm{b_cache_mod}{_wpe}",
                 known_block_size=[256, 1, 1])
    def kernel(
        arg_x: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64, arg_eids: fx.Int64,
        arg_cumsum: fx.Int64, arg_mind: fx.Int64, i32_ntok: fx.Int32,
        f32_alpha: fx.Float32, f32_limit: fx.Float32, arg_out: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        cumsum0 = fx.Int32(_global_i32_at(arg_cumsum, fx.Int32(0)))
        mb, nb = pid // fx.Int32(NNB), pid % fx.Int32(NNB)
        mbase = mb * fx.Int32(BM)
        if mbase < cumsum0:
            e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(_global_i32_at(arg_eids, mb)))))
            if const_expr(_NW_FAKE_W):
                e = fx.Int32(0)
            # A batch staging: wave w, load j: lane -> row w*4 + lane//16, 16 B chunk j*16 + lane%16 of
            # the batch's ROWB bytes. Padding rows carry a token >= ntok and read as 0 through the
            # OOB-clamped resource (sized to the real [ntok, K] bf16 buffer).
            ld_row = wave * fx.Int32(4) + q16
            ld_tok = fx.Int32(_global_i32_at(arg_mind, mbase + ld_row)) & fx.Int32(0xFFFFFF)
            if const_expr(_NW_FAKE_A):
                ld_tok = ld_row
            # epilogue rows: lane (q16, l16) holds rows q16*4 + ii of column l16
            ep_tok = [fx.Int32(_global_i32_at(arg_mind, mbase + q16 * fx.Int32(4) + fx.Int32(ii))) & fx.Int32(0xFFFFFF)
                      for ii in range_constexpr(4)]
            xr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_x)), num_records_bytes=_raw(fx.Int64(i32_ntok) * fx.Int64(K * 2)))
            wr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_bq)), num_records_bytes=min(W_BYTES, 0xFFFFFFFF))
            sr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_bscale)), num_records_bytes=min(SW_BYTES, 0xFFFFFFFF))
            outr = bop.create_buffer_resource_from_addr(_raw(fx.Int64(arg_out)), num_records_bytes=_raw(fx.Int64(cumsum0) * fx.Int64(INTER * 2)))
            lds = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int32(fx.ptrtoint(smem))))

            ld_gdw = (ld_tok * fx.Int32(K * 2) + l16 * fx.Int32(16)) // fx.Int32(4)
            ld_lbyte = ld_row * fx.Int32(RS) + l16 * fx.Int32(16)

            def load_a_batch(b):
                so = _sconst(b * ROWB)   # batch base in an SGPR; j*256 rides in the immediate offset field
                return [bop.buffer_load(xr, ld_gdw + fx.Int32(j * 64), vec_width=4, dtype=fx.Int32, soffset_bytes=so)
                        for j in range_constexpr(KB)]

            def stage_a_batch(regs, slot):
                if const_expr(not _NW_NO_A):
                    for j in range_constexpr(KB):
                        ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(ld_lbyte + fx.Int32(slot * SLOT + j * 256)), elem_type=T.i8)
                        llvm.StoreOp(fx.as_ir_value(regs[j]), ptr, alignment=16)
                s_waitcnt_lgkm0()
                gpu.barrier()

            # MFMA A fragment (K-step ku of tile kt): row l16, K = klane*32 + ku*8 .. +8 -> bytes q16*64 + ku*16
            rd_base = l16 * fx.Int32(RS) + q16 * fx.Int32(64)

            def read_a_tile(kt):
                slot, j = (kt // KB) % 2, kt % KB
                out = []
                for ku in range_constexpr(4):
                    if const_expr(_NW_NO_A):
                        out.append(fx.Vector.filled(4, ku + 1, fx.Int32).bitcast(fx.BFloat16))
                        continue
                    ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(rd_base + fx.Int32(slot * SLOT + j * 256 + ku * 16)), elem_type=T.i8)
                    out.append(fx.Vector(llvm.load(T.vec(4, T.i32), ptr, alignment=16)).bitcast(fx.BFloat16))
                return out

            # ---- W addressing (gu 0 = gate, 1 = up), per wave NI 16-col tiles -------------------
            nbase = nb * fx.Int32(TILE_N) + wave * fx.Int32(NPW)
            if const_expr(w_layout == "guinterleave"):
                # GUGU: gate/up 16-row blocks interleaved; scale packs gate (byte 0/2) and up (1/3)
                n0 = [(nbase + fx.Int32(ni * 16)) // fx.Int32(16) for ni in range_constexpr(NI)]
                nblk = [[e * fx.Int32(N_OUT // 16) + n0[ni] * fx.Int32(2) + fx.Int32(gu) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                mni = [[e * fx.Int32(N_OUT // 32) + n0[ni] for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                npk = [[fx.Int32(gu) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
            else:
                ng = [[e * fx.Int32(N_OUT) + nbase + fx.Int32(ni * 16 + gu * INTER) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                nblk = [[ng[gu][ni] // fx.Int32(16) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                mni = [[ng[gu][ni] // fx.Int32(32) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                npk = [[(ng[gu][ni] // fx.Int32(16)) % fx.Int32(2) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
            # per column tile one vector address (lane*16 B + the tile's block base); the K position is
            # (kt//4)*4096 in an SGPR (_sconst) + (kt%4)*1024 in the immediate offset field
            wvo = [[lane * fx.Int32(4) + nblk[gu][ni] * fx.Int32(KT * 256) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
            svo = [[lane + mni[gu][ni] * fx.Int32(SC_STRIDE_N0) for ni in range_constexpr(NI)] for gu in range_constexpr(2)]

            def load_b_tile(kt, prev):
                so = _sconst((kt // 4) * 4096)
                bb = [[bop.buffer_load(wr, wvo[gu][ni] + fx.Int32((kt % 4) * 256), vec_width=4, dtype=fx.Int32,
                                       cache_modifier=b_cache_mod, soffset_bytes=so)
                       for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                if const_expr(kt % 2 == 0):
                    sso = _sconst((kt // 8) * 1024)   # scale dword per 2 K-tiles: (kt//2)*256 B = SGPR + imm
                    sc = [[bop.buffer_load(sr, svo[gu][ni] + fx.Int32(((kt // 2) % 4) * 64), vec_width=1, dtype=fx.Int32, soffset_bytes=sso)
                           for ni in range_constexpr(NI)] for gu in range_constexpr(2)]
                else:
                    sc = prev[1]
                return bb, sc

            vec2_bf16 = ir.Type.parse("vector<2xbf16>")

            def upconvert(raw4, ku, scale_f32):
                # raw4[ku] holds 8 fp4 for K-step ku -> 4x cvt (v2bf16, sel 0..3) -> v8bf16
                i32_val = _raw(fx.Int32(raw4[ku]))
                s_raw = _raw(scale_f32)
                i32s = []
                for sel in range_constexpr(4):
                    p = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
                    i32s.append(fx.Int32(fx.Vector(p).bitcast(fx.Int32)[0]))
                return fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32).bitcast(fx.BFloat16)

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            acc_layout = fx.make_layout(4, 1)
            acc = [[fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(NI)] for _ in range(2)]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for gu in range_constexpr(2):
                for ni in range_constexpr(NI):
                    acc[gu][ni].store(zero4)

            def _frag(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def compute_tile(bb, sc, aa, kt):
                a_t = [_frag(aa[ku]) for ku in range_constexpr(4)]
                for gu in range_constexpr(2):
                    for ni in range_constexpr(NI):
                        s = _e8m0_byte_to_f32(fx.Int32(sc[gu][ni]), fx.Int32((kt % 2) * 2) + npk[gu][ni])
                        raw4 = fx.Vector(bb[gu][ni])
                        for ku in range_constexpr(4):
                            fx.gemm(mma_atom, acc[gu][ni], a_t[ku], _frag(upconvert(raw4, ku, s)), acc[gu][ni])

            # ---- pipeline: batch 0 -> LDS, W ring, then per tile: (batch loads) W load, ds_read, MFMA ----
            abuf = None if const_expr(_NW_NO_A) else load_a_batch(0)
            ring = []
            for t in range_constexpr(prefetch):
                ring.append(load_b_tile(t, ring[-1] if ring else None))
            stage_a_batch(abuf, 0)
            abuf = None
            for kt in range_constexpr(KT):
                if const_expr(kt % KB == 0 and kt + KB < KT and not _NW_NO_A):
                    abuf = load_a_batch(kt // KB + 1)      # before this iteration's W loads
                if const_expr(kt + prefetch < KT):
                    ring.append(load_b_tile(kt + prefetch, ring[-1]))
                bb, sc = ring.pop(0)
                aa = read_a_tile(kt)
                compute_tile(bb, sc, aa, kt)
                if const_expr(_NW_SCHED):
                    rocdl.sched_barrier(0)
                if const_expr(kt % KB == KB - 1 and kt + 1 < KT):
                    stage_a_batch(abuf, (kt // KB + 1) % 2)   # this batch's reads are done (MFMAs above)
                    abuf = None

            # ---- epilogue: swigluoai(gate, up) -> bf16 [sorted_row, inter], padding rows masked ----
            neg_limit = -fx.Float32(f32_limit)
            alpha = fx.Float32(f32_alpha)
            for ii in range_constexpr(4):
                sorted_row = mbase + q16 * fx.Int32(4) + fx.Int32(ii)
                valid = ep_tok[ii] < i32_ntok
                for ni in range_constexpr(NI):
                    g = fx.Float32(fx.Vector(fx.memref_load_vec(acc[0][ni]))[ii])
                    u = fx.Float32(fx.Vector(fx.memref_load_vec(acc[1][ni]))[ii])
                    y = _swigluoai_f32(g, u, alpha, neg_limit)
                    yb = y.to(fx.BFloat16)
                    out_idx = sorted_row * fx.Int32(INTER) + nbase + fx.Int32(ni * 16) + l16
                    bop.buffer_store(yb, outr, _raw(out_idx), mask=valid, cache_modifier=_NW_STORE_CPOL)

    @flyc.jit
    def launch(
        arg_x: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64, arg_eids: fx.Int64,
        arg_cumsum: fx.Int64, arg_mind: fx.Int64, i32_ntok: fx.Int32, i32_grid: fx.Int32,
        f32_alpha: fx.Float32, f32_limit: fx.Float32, arg_out: fx.Int64, stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_x, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_mind, i32_ntok, f32_alpha, f32_limit, arg_out,
            **({"value_attrs": {"rocdl.waves_per_eu": waves_per_eu}} if waves_per_eu else {}),
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


@functools.cache
def get_gemm1_nw(**kw):
    return compile_gemm1_nw(**kw)


def a16w4_gemm1_nw(
    *, x_bf16, w1_u8, w1_scale_u8, inter_sorted_bf16, n_tokens, NE, D_HIDDEN, D_INTER, topk,
    tile_m, tile_n, tile_k, sorted_expert_ids, num_valid_ids, sorted_token_ids,
    prefetch=3, k_batch=4, b_nt=2, w_layout="standard", waves_per_eu=None,
    alpha=1.702, swiglu_limit=7.0, act="swigluoai", stream=None, **_unused,
):
    """Drop-in for ``host.a16w4_gemm1`` on the sorted path (tile_m 16, tile_k 128)."""
    assert tile_m == 16 and tile_k == 128 and act == "swigluoai"
    launch = get_gemm1_nw(
        D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, NE=NE, TOPK=topk, BM=tile_m, TILE_N=tile_n,
        prefetch=prefetch, k_batch=k_batch, b_cache_mod=b_nt, w_layout=w_layout, waves_per_eu=waves_per_eu,
    )
    grid = int(sorted_expert_ids.numel()) * (D_INTER // tile_n)
    _run_compiled(
        launch, x_bf16.data_ptr(), w1_u8.data_ptr(), w1_scale_u8.data_ptr(), sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(), sorted_token_ids.data_ptr(), int(n_tokens), int(grid), float(alpha),
        float(swiglu_limit), inter_sorted_bf16.data_ptr(), torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16
