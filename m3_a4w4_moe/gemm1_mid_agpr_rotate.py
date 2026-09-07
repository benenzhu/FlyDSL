# SPDX-License-Identifier: Apache-2.0
"""TN256 AGPR gate/up with a different K starting quarter per N wave.

Input/output layouts match gemm1.py. Unlike its 2x2-wave LDS pipeline, every
wave owns 32 gate columns and their up columns; A/W flow directly to VGPRs.
FP4 MFMA operands are swapped to put one output row in lanes L,L^16,L^32,L^48,
so the existing per-32-column quantization epilogue can be reused.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from m3_a16w4_moe.gemm1_agpr_ring import _pin_accumulators

from .gemm1 import (
    _buffer_ops as bop, _swiglu_oai, _quant_prep_fp4, _fmax,
    _e8m0_roundup_fp4, _as_f32, _cvt_pk_fp4, _permlane16_swap,
    Mfma16x16x128Fp4, _asm_void,
)


def compile_moe_gemm1_mid(*, H, I, E, BLOCK_M=32, prefetch=3):
    assert BLOCK_M in (32, 64) and H % 256 == 0 and I % 128 == 0
    BM, BN, KT = BLOCK_M, 256, H // 128
    assert H % 1024 == 0, "quarter rotations must preserve K256 scale pairing"
    assert I % BN == 0
    NI = BN // 4 // 16
    MR = BM // 16
    assert 1 <= prefetch < KT

    @flyc.kernel(name=f"gemm1_a4w4_mid_agpr_rotate_h{H}_i{I}_e{E}_bm{BM}_pf{prefetch}", known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32, asbytes: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        nr = bop.create_buffer_resource(nvalid, max_size=False, num_records_bytes=4)
        valid_rows = fx.Int32(bop.buffer_load(nr, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        mb, nb = pid // fx.Int32(I // BN), pid % fx.Int32(I // BN)
        if mb * fx.Int32(BM) < valid_rows:
            er = bop.create_buffer_resource(eids, max_size=False, num_records_bytes=nblocks * 4)
            expert = fx.Int32(bop.buffer_load(er, mb, vec_width=1, dtype=fx.Int32, is_scalar=True))
            ir = bop.create_buffer_resource(ids, max_size=False, num_records_bytes=nblocks * BM * 4)
            ar = bop.create_buffer_resource(A, max_size=False, num_records_bytes=ntok * (H // 2))
            wr = bop.create_buffer_resource(W, max_size=False, num_records_bytes=E * 2 * I * H // 2)
            asr = bop.create_buffer_resource(AS, max_size=False, num_records_bytes=asbytes)
            wsr = bop.create_buffer_resource(WS, max_size=False, num_records_bytes=E * 2 * I * H // 32)
            orsrc = bop.create_buffer_resource(O, max_size=False, num_records_bytes=nblocks * BM * (I // 2))
            osr = bop.create_buffer_resource(OS, max_size=False, num_records_bytes=nblocks * BM * (I // 32))
            mbase = mb * fx.Int32(BM)
            nbase = nb * fx.Int32(BN) + wave * fx.Int32(NI * 16)
            tokens = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Int32)) & fx.Int32(0xFFFFFF) for mi in range_constexpr(MR)]

            def rotated_k(kt):
                k = wave * fx.Int32(KT // 4) + fx.Int32(kt)
                return (k < fx.Int32(KT)).select(k, k - fx.Int32(KT))

            def load_b_tile(kt, prev):
                kp = rotated_k(kt)
                bb = []
                for gu in range_constexpr(2):
                    slab = []
                    for ni in range_constexpr(NI):
                        nblk = expert * fx.Int32(2 * I // 16) + nbase // fx.Int32(16) + fx.Int32(gu * I // 16 + ni)
                        off = q16 * fx.Int32(256) + l16 * fx.Int32(16)
                        slab.append(bop.buffer_load(wr, off // fx.Int32(4), vec_width=4, dtype=fx.Int32,
                                                    soffset_bytes=nblk * fx.Int32(H * 8) + kp * fx.Int32(1024)))
                    bb.append(slab)
                if const_expr(kt % 2 == 0):
                    sb = [[bop.buffer_load(wsr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                         soffset_bytes=((expert * fx.Int32(2 * I // 32) + nbase // fx.Int32(32) + fx.Int32(gu * I // 32 + np)) * fx.Int32(H // 256 * 64) + kp // fx.Int32(2) * fx.Int32(64)) * fx.Int32(4)) for np in range_constexpr(NI // 2)] for gu in range_constexpr(2)]
                else:
                    sb = prev[1]
                return bb, sb

            def load_a_tile(kt, prev):
                kp = rotated_k(kt)
                aa = [bop.buffer_load(ar, (tokens[mi] * fx.Int32(H // 2) + q16 * fx.Int32(16)) // fx.Int32(4), vec_width=4, dtype=fx.Int32, soffset_bytes=kp * fx.Int32(64)) for mi in range_constexpr(MR)]
                if const_expr(kt % 2 == 0):
                    sa = [bop.buffer_load(asr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                         soffset_bytes=((mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(H // 256 * 64) + kp // fx.Int32(2) * fx.Int32(64)) * fx.Int32(4)) for mp in range_constexpr(MR // 2)]
                else:
                    sa = prev[1]
                return aa, sa

            mfma = Mfma16x16x128Fp4(MR, NI)
            acc = [[[None for gu in range(2)] for ni in range(NI)] for mi in range(MR)]
            ring = []
            for kt in range_constexpr(prefetch):
                ring.append(load_b_tile(kt, ring[-1] if ring else None))
            aring = [load_a_tile(0, None)]
            for kt in range_constexpr(KT):
                if const_expr(kt + prefetch < KT):
                    ring.append(load_b_tile(kt + prefetch, ring[-1]))
                if const_expr(kt + 1 < KT):
                    aring.append(load_a_tile(kt + 1, aring[-1]))
                bb, sb = ring.pop(0)
                aa, sa = aring.pop(0)
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(NI):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = mfma._mfma_agpr(
                                aa[mi], bb[gu][ni], acc[mi][ni][gu], sa[mi // 2], sb[gu][ni // 2],
                                kt % 2, mi % 2, ni % 2,
                            )

            # accumulator fence with a data dependency (see gemm1_mid_alds.py): a free-floating s_nop lets the
            # scheduler hoist the v_accvgpr_read copies right behind the last inline-asm MFMA.
            _nm, _nn = len(acc), len(acc[0])
            flat = _pin_accumulators([acc[mi][ni][gu] for mi in range_constexpr(_nm) for ni in range_constexpr(_nn) for gu in range_constexpr(2)])
            for mi in range_constexpr(_nm):
                for ni in range_constexpr(_nn):
                    for gu in range_constexpr(2):
                        acc[mi][ni][gu] = fx.Vector(flat[(mi * _nn + ni) * 2 + gu])
            for np in range_constexpr(NI // 2):
                exps = []
                for mi in range_constexpr(MR):
                    h = [_swiglu_oai(fx.Float32(fx.Vector(acc[mi][2 * np + ni][0])[ii]), fx.Float32(fx.Vector(acc[mi][2 * np + ni][1])[ii])) for ni in range_constexpr(2) for ii in range_constexpr(4)]
                    h, amax = _quant_prep_fp4(h)
                    amax = _fmax(amax, amax.shuffle_xor(16, 64))
                    amax = _fmax(amax, amax.shuffle_xor(32, 64))
                    exponent = _e8m0_roundup_fp4(amax)
                    exps.append(exponent)
                    scale = _as_f32(exponent << 23)
                    pa = _cvt_pk_fp4(fx.Int32(0), h[0], h[1], scale, 0)
                    pa = _cvt_pk_fp4(pa, h[2], h[3], scale, 1)
                    pb = _cvt_pk_fp4(fx.Int32(0), h[4], h[5], scale, 0)
                    pb = _cvt_pk_fp4(pb, h[6], h[7], scale, 1)
                    pa, pb = _permlane16_swap(pa, pb)
                    packed = pa | (pb << 16)
                    row = mbase + fx.Int32(mi * 16) + l16
                    col = nbase + fx.Int32(np * 32) + (q16 % fx.Int32(2)) * fx.Int32(16) + (q16 // fx.Int32(2)) * fx.Int32(8)
                    bop.buffer_store(packed, orsrc, row * fx.Int32(I // 2) + col // fx.Int32(2), offset_is_bytes=True)
                colgrp = nbase // fx.Int32(32) + fx.Int32(np)
                for mp in range_constexpr(MR // 2):
                    blk = (mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(I // 256) + colgrp // fx.Int32(8)
                    off = blk * fx.Int32(256) + (colgrp % fx.Int32(4)) * fx.Int32(64) + l16 * fx.Int32(4) + ((colgrp % fx.Int32(8)) // fx.Int32(4)) * fx.Int32(2)
                    pair = fx.Int32(exps[2 * mp] | (exps[2 * mp + 1] << 8)).to(fx.Int16)
                    bop.buffer_store(pair, osr, off, offset_is_bytes=True, mask=q16 == fx.Int32(0))

    @flyc.jit
    def launch(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32, asbytes: fx.Int32,
        tile_map: fx.Tensor, grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(A, W, O, AS, WS, OS, ids, eids, nvalid, ntok, nblocks, asbytes).launch(
            grid=(nblocks * (I // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
