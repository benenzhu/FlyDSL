# SPDX-License-Identifier: Apache-2.0
"""Middle-batch down projection with per-32-column MXFP8 partials.

BM32/64 and TN256, four N waves. Output/scales match reduce_fp8.py.
Routing weights are applied by that reduction, not before fp8 quantization.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import buffer_ops as bop
from .gemm2 import _cvt_pk_fp8, _e8m0_fp8, _fabs, _maxf
from .gemm1 import _as_f32


def compile_moe_gemm2_mid(*, H, I, E, topk=5, BLOCK_M=32, TILE_N=256, prefetch=3):
    BM, BN, MR = BLOCK_M, TILE_N, BLOCK_M // 16
    NI, KT = BN // 4 // 16, I // 128
    assert BM in (32, 64) and BN in (128, 256) and H % BN == 0 and I % 256 == 0
    assert 1 <= prefetch < KT

    @flyc.kernel(name=f"gemm2_a4w4_mid_fp8_h{H}_i{I}_e{E}_bm{BM}_tn{BN}_pf{prefetch}", known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor, weights: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        nr = bop.create_buffer_resource(nvalid, max_size=False, num_records_bytes=4)
        valid_rows = fx.Int32(bop.buffer_load(nr, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        mb, nb = pid // fx.Int32(H // BN), pid % fx.Int32(H // BN)
        if mb * fx.Int32(BM) < valid_rows:
            er = bop.create_buffer_resource(eids, max_size=False, num_records_bytes=nblocks * 4)
            expert = fx.Int32(bop.buffer_load(er, mb, vec_width=1, dtype=fx.Int32, is_scalar=True))
            ir = bop.create_buffer_resource(ids, max_size=False, num_records_bytes=nblocks * BM * 4)
            rr = bop.create_buffer_resource(weights, max_size=False, num_records_bytes=nblocks * BM * 4)
            ar = bop.create_buffer_resource(A, max_size=False, num_records_bytes=nblocks * BM * (I // 2))
            wr = bop.create_buffer_resource(W, max_size=False, num_records_bytes=E * H * I // 2)
            asr = bop.create_buffer_resource(AS, max_size=False, num_records_bytes=nblocks * BM * (I // 32))
            wsr = bop.create_buffer_resource(WS, max_size=False, num_records_bytes=E * H * I // 32)
            orsrc = bop.create_buffer_resource(O, max_size=False, num_records_bytes=ntok * (topk * H))
            osr = bop.create_buffer_resource(OS, max_size=False, num_records_bytes=ntok * (topk * H // 32))
            mbase = mb * fx.Int32(BM)
            nbase = nb * fx.Int32(BN) + wave * fx.Int32(NI * 16)
            fused = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Int32)) for mi in range_constexpr(MR)]
            rw = [fx.Float32(bop.buffer_load(rr, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Float32)) for mi in range_constexpr(MR)]

            def load_tile(kt, prev):
                aa = [bop.buffer_load(ar, ((mbase + fx.Int32(mi * 16) + l16) * fx.Int32(I // 2) + fx.Int32(kt * 64) + q16 * fx.Int32(16)) // fx.Int32(4), vec_width=4, dtype=fx.Int32) for mi in range_constexpr(MR)]
                bb = []
                for ni in range_constexpr(NI):
                    nblk = expert * fx.Int32(H // 16) + nbase // fx.Int32(16) + fx.Int32(ni)
                    off = nblk * fx.Int32(I * 8) + fx.Int32(kt * 1024) + q16 * fx.Int32(256) + l16 * fx.Int32(16)
                    bb.append(bop.buffer_load(wr, off // fx.Int32(4), vec_width=4, dtype=fx.Int32))
                if const_expr(kt % 2 == 0):
                    sa = [bop.buffer_load(asr, (mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(I // 256 * 64) + fx.Int32(kt // 2 * 64) + q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32) for mp in range_constexpr(MR // 2)]
                    sb = [bop.buffer_load(wsr, (expert * fx.Int32(H // 32) + nbase // fx.Int32(32) + fx.Int32(np)) * fx.Int32(I // 256 * 64) + fx.Int32(kt // 2 * 64) + q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32) for np in range_constexpr(NI // 2)]
                else:
                    sa, sb = prev[2], prev[3]
                return aa, bb, sa, sb

            acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for ni in range(NI)] for mi in range(MR)]
            ring = []
            for kt in range_constexpr(prefetch):
                ring.append(load_tile(kt, ring[-1] if ring else None))
            for kt in range_constexpr(KT):
                if const_expr(kt + prefetch < KT):
                    ring.append(load_tile(kt + prefetch, ring[-1]))
                aa, bb, sa, sb = ring.pop(0)
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(NI):
                        acc[mi][ni] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32), [
                                fx.as_ir_value(bb[ni]), fx.as_ir_value(aa[mi]), fx.as_ir_value(acc[mi][ni]),
                                4, 4, ni % 2 + 2 * (kt % 2), fx.as_ir_value(sb[ni // 2]),
                                mi % 2 + 2 * (kt % 2), fx.as_ir_value(sa[mi // 2]),
                            ],
                        ))
            for mi in range_constexpr(MR):
                token = fused[mi] & fx.Int32(0xFFFFFF)
                slot = fused[mi].shrui(fx.Int32(24))
                outrow = token * fx.Int32(topk) + slot
                for np in range_constexpr(NI // 2):
                    vals = [fx.Float32(acc[mi][2 * np + ni][ii]) for ni in range_constexpr(2) for ii in range_constexpr(4)]
                    amax = _fabs(vals[0])
                    for ii in range_constexpr(1, 8):
                        amax = _maxf(amax, _fabs(vals[ii]))
                    amax = _maxf(amax, amax.shuffle_xor(16, 64))
                    amax = _maxf(amax, amax.shuffle_xor(32, 64))
                    exponent = _e8m0_fp8(amax)
                    scale = _as_f32(exponent << 23)
                    for ni in range_constexpr(2):
                        v = vals[ni * 4:ni * 4 + 4]
                        packed = _cvt_pk_fp8(fx.Int32(0), v[0], v[1], scale, False)
                        packed = _cvt_pk_fp8(packed, v[2], v[3], scale, True)
                        col = nbase + fx.Int32((2 * np + ni) * 16) + q16 * fx.Int32(4)
                        bop.buffer_store(packed, orsrc, outrow * fx.Int32(H) + col, offset_is_bytes=True, mask=token < ntok)
                    sc_off = outrow * fx.Int32(H // 32) + nbase // fx.Int32(32) + fx.Int32(np)
                    bop.buffer_store(fx.Int32(exponent).to(fx.Uint8), osr, sc_off, offset_is_bytes=True,
                                     mask=(token < ntok) & (q16 == fx.Int32(0)))

    @flyc.jit
    def launch(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        weights: fx.Tensor, nvalid: fx.Tensor, ntok: fx.Int32,
        nblocks: fx.Int32, grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(A, W, O, AS, WS, OS, ids, eids, weights, nvalid, ntok, nblocks).launch(
            grid=(nblocks * (H // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
