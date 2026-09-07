# SPDX-License-Identifier: Apache-2.0
"""Middle-batch down projection with weighted bf16 atomic accumulation.

Uses the existing decode packed-bf16 atomic epilogue. Sort must zero [M,H].
There is no partial buffer, fp8 quantization or separate topk reduction.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import buffer_ops as bop
from m3_a16w4_moe.gemm2 import _atomic_bf16_epilog


# experiment knob: cache policy bits for the W loads (2 = nt, as the decode kernel uses). Default 0.
_MID_W_CPOL = int(os.environ.get("M3_MID_W_CPOL", "0"), 0)


def compile_moe_gemm2_mid(*, H, I, E, topk=5, BLOCK_M=32, TILE_N=256, prefetch=3):
    BM, BN, MR = BLOCK_M, TILE_N, BLOCK_M // 16
    NI, KT = BN // 4 // 16, I // 128
    assert BM in (32, 64) and BN in (128, 256) and H % BN == 0 and I % 256 == 0
    assert 1 <= prefetch < KT

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, BM * BN * 4, 16]

    @flyc.kernel(name=f"gemm2_a4w4_mid_atomic_h{H}_i{I}_e{E}_bm{BM}_tn{BN}_pf{prefetch}", known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor, weights: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
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
            orsrc = bop.create_buffer_resource(O, max_size=False, num_records_bytes=ntok * (topk * H * 2))
            mbase = mb * fx.Int32(BM)
            nbase = nb * fx.Int32(BN) + wave * fx.Int32(NI * 16)
            fused = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Int32)) for mi in range_constexpr(MR)]
            rw = [fx.Float32(bop.buffer_load(rr, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Float32)) for mi in range_constexpr(MR)]
            m_lane = tx // fx.Int32(32)
            ep_ids = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mr * 8) + m_lane, vec_width=1, dtype=fx.Int32)) for mr in range_constexpr(BM // 8)]
            ep_weights = [fx.Float32(bop.buffer_load(rr, mbase + fx.Int32(mr * 8) + m_lane, vec_width=1, dtype=fx.Float32)) for mr in range_constexpr(BM // 8)]

            def load_tile(kt, prev):
                aa = [bop.buffer_load(ar, ((mbase + fx.Int32(mi * 16) + l16) * fx.Int32(I // 2) + fx.Int32(kt * 64) + q16 * fx.Int32(16)) // fx.Int32(4), vec_width=4, dtype=fx.Int32) for mi in range_constexpr(MR)]
                bb = []
                for ni in range_constexpr(NI):
                    nblk = expert * fx.Int32(H // 16) + nbase // fx.Int32(16) + fx.Int32(ni)
                    off = nblk * fx.Int32(I * 8) + fx.Int32(kt * 1024) + q16 * fx.Int32(256) + l16 * fx.Int32(16)
                    bb.append(bop.buffer_load(wr, off // fx.Int32(4), vec_width=4, dtype=fx.Int32, cache_modifier=_MID_W_CPOL))
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
                                fx.as_ir_value(aa[mi]), fx.as_ir_value(bb[ni]), fx.as_ir_value(acc[mi][ni]),
                                4, 4, mi % 2 + 2 * (kt % 2), fx.as_ir_value(sa[mi // 2]),
                                ni % 2 + 2 * (kt % 2), fx.as_ir_value(sb[ni // 2]),
                            ],
                        ))
            _atomic_bf16_epilog(
                fx.Int32(fx.ptrtoint(smem)), acc,
                fx.Int64(fx.ptrtoint(fx.get_iter(O))),
                fx.Int64(fx.ptrtoint(fx.get_iter(ids))),
                fx.Int64(fx.ptrtoint(fx.get_iter(weights))),
                mbase, nb, wave, lane, ntok, BM, H, BN,
                pre_packed=ep_ids, pre_weight=ep_weights,
            )

    @flyc.jit
    def launch(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        weights: fx.Tensor, nvalid: fx.Tensor, ntok: fx.Int32,
        nblocks: fx.Int32, grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(A, W, O, AS, WS, ids, eids, weights, nvalid, ntok, nblocks).launch(
            grid=(nblocks * (H // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
