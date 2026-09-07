# SPDX-License-Identifier: Apache-2.0
"""Middle-batch down projection, weighted bf16 atomic accumulation, A tile staged in LDS.

Same tiles, W ring, MFMA and epilogue as ``gemm2_mid_atomic.py`` (read that first).
What changes is how the fp4 intermediate rows reach the MFMAs: there every wave
gathers the whole [BM, I] A tile K-tile by K-tile (16-row x 64 B gathers, 16 L1
requests per KB, 4x per workgroup); diagnostics put those gathers at 34 of 108 us
at 1024 tokens / BM64 (M3_MID_G2_NO_A=1). Here the tile, which is contiguous in the
sorted intermediate (BM rows x I/2 bytes), is copied once per workgroup with plain
1 KB wave loads into LDS (row stride I/2 + 16 B: conflict-free 16 B reads) before
the W ring starts, and every wave ds_reads its MFMA A fragments from there. The
staging region aliases the epilogue's accumulator region; one barrier separates
the last A read from the first epilogue write.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import buffer_ops as bop
from m3_a16w4_moe.gemm2 import _atomic_bf16_epilog
from m3_a16w4_moe.utils import s_waitcnt_lgkm0

_MID_W_CPOL = int(os.environ.get("M3_MID_W_CPOL", "2"), 0)
# pair mode: workgroups (mb, nb) and (mb+1, nb) get block ids p and p+8, i.e. the same XCD in the same
# dispatch round, so when the two m-blocks belong to one expert the second one finds W in that
# XCD's L2 (a BM-row sort then costs one W read per expert up to 2*BM rows). Default 0.
_MID_PAIR = int(os.environ.get("M3_MID_PAIR", "0"))
_G2_FAKE_W = int(os.environ.get("M3_MID_G2_FAKE_W", "0"))


def compile_moe_gemm2_mid(*, H, I, E, topk=5, BLOCK_M=32, TILE_N=256, prefetch=3):
    BM, BN, MR = BLOCK_M, TILE_N, BLOCK_M // 16
    NI, KT = BN // 4 // 16, I // 128
    assert BM in (32, 64) and BN in (128, 256) and H % BN == 0 and I % 256 == 0
    assert 1 <= prefetch < KT
    ROWB = I // 2                  # bytes of one A row
    RS = ROWB + 16                 # padded LDS row stride
    NCH = BM * ROWB // 16          # 16 B chunks in the tile
    assert NCH % 256 == 0
    NLD = NCH // 256               # 1 KB loads per wave
    A_LDS = BM * RS
    EPI_LDS = BM * BN * 4
    LDS_BYTES = max(A_LDS, EPI_LDS)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=f"gemm2_a4w4_mid_atomic_alds_h{H}_i{I}_e{E}_bm{BM}_tn{BN}_pf{prefetch}", known_block_size=[256, 1, 1])
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
        if const_expr(_MID_PAIR):
            g, r = pid // fx.Int32(16), pid % fx.Int32(16)
            item = g * fx.Int32(8) + r % fx.Int32(8)
            mb, nb = (item // fx.Int32(H // BN)) * fx.Int32(2) + r // fx.Int32(8), item % fx.Int32(H // BN)
        else:
            mb, nb = pid // fx.Int32(H // BN), pid % fx.Int32(H // BN)
        if mb * fx.Int32(BM) < valid_rows:
            er = bop.create_buffer_resource(eids, max_size=False, num_records_bytes=nblocks * 4)
            expert = fx.Int32(bop.buffer_load(er, mb, vec_width=1, dtype=fx.Int32, is_scalar=True))
            if const_expr(_G2_FAKE_W):
                expert = fx.Int32(0)
            ir = bop.create_buffer_resource(ids, max_size=False, num_records_bytes=nblocks * BM * 4)
            rr = bop.create_buffer_resource(weights, max_size=False, num_records_bytes=nblocks * BM * 4)
            ar = bop.create_buffer_resource(A, max_size=False, num_records_bytes=nblocks * BM * (I // 2))
            wr = bop.create_buffer_resource(W, max_size=False, num_records_bytes=E * H * I // 2)
            asr = bop.create_buffer_resource(AS, max_size=False, num_records_bytes=nblocks * BM * (I // 32))
            wsr = bop.create_buffer_resource(WS, max_size=False, num_records_bytes=E * H * I // 32)
            mbase = mb * fx.Int32(BM)
            nbase = nb * fx.Int32(BN) + wave * fx.Int32(NI * 16)
            lds = llvm.inttoptr(_ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int32(fx.ptrtoint(smem))))

            # A tile: BM x ROWB contiguous bytes at row mbase; wave w copies chunks (w*NLD + j)*64 + lane.
            abuf = []
            a_lbyte = []
            for j in range_constexpr(NLD):
                c = (wave * fx.Int32(NLD) + fx.Int32(j)) * fx.Int32(64) + lane
                abuf.append(bop.buffer_load(ar, (mbase * fx.Int32(ROWB)) // fx.Int32(4) + c * fx.Int32(4), vec_width=4, dtype=fx.Int32))
                a_lbyte.append((c // fx.Int32(ROWB // 16)) * fx.Int32(RS) + (c % fx.Int32(ROWB // 16)) * fx.Int32(16))
            m_lane = tx // fx.Int32(32)
            ep_ids = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mr * 8) + m_lane, vec_width=1, dtype=fx.Int32)) for mr in range_constexpr(BM // 8)]
            ep_weights = [fx.Float32(bop.buffer_load(rr, mbase + fx.Int32(mr * 8) + m_lane, vec_width=1, dtype=fx.Float32)) for mr in range_constexpr(BM // 8)]

            def load_tile(kt, prev):
                bb = []
                for ni in range_constexpr(NI):
                    nblk = expert * fx.Int32(H // 16) + nbase // fx.Int32(16) + fx.Int32(ni)
                    off = nblk * fx.Int32(I * 8) + fx.Int32(kt * 1024) + q16 * fx.Int32(256) + l16 * fx.Int32(16)
                    bb.append(bop.buffer_load(wr, off // fx.Int32(4), vec_width=4, dtype=fx.Int32, cache_modifier=_MID_W_CPOL))
                if const_expr(kt % 2 == 0):
                    sa = [bop.buffer_load(asr, (mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(I // 256 * 64) + fx.Int32(kt // 2 * 64) + q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32) for mp in range_constexpr(MR // 2)]
                    sb = [bop.buffer_load(wsr, (expert * fx.Int32(H // 32) + nbase // fx.Int32(32) + fx.Int32(np)) * fx.Int32(I // 256 * 64) + fx.Int32(kt // 2 * 64) + q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32) for np in range_constexpr(NI // 2)]
                else:
                    sa, sb = prev[1], prev[2]
                return bb, sa, sb

            rd_base = [(fx.Int32(mi * 16) + l16) * fx.Int32(RS) + q16 * fx.Int32(16) for mi in range_constexpr(MR)]

            def read_a_tile(kt):
                out = []
                for mi in range_constexpr(MR):
                    ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(rd_base[mi] + fx.Int32(kt * 64)), elem_type=T.i8)
                    out.append(fx.Vector(llvm.load(T.vec(4, T.i32), ptr, alignment=16)))
                return out

            acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for ni in range(NI)] for mi in range(MR)]
            ring = []
            for kt in range_constexpr(prefetch):
                ring.append(load_tile(kt, ring[-1] if ring else None))
            # stage the A tile (its loads were issued before the W ring: waiting for them leaves W in flight)
            for j in range_constexpr(NLD):
                ptr = bop.get_element_ptr(lds, byte_offset=fx.as_ir_value(a_lbyte[j]), elem_type=T.i8)
                llvm.StoreOp(fx.as_ir_value(abuf[j]), ptr, alignment=16)
            s_waitcnt_lgkm0()
            gpu.barrier()
            for kt in range_constexpr(KT):
                if const_expr(kt + prefetch < KT):
                    ring.append(load_tile(kt + prefetch, ring[-1]))
                bb, sa, sb = ring.pop(0)
                aa = read_a_tile(kt)
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(NI):
                        acc[mi][ni] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32), [
                                fx.as_ir_value(aa[mi]), fx.as_ir_value(bb[ni]), fx.as_ir_value(acc[mi][ni]),
                                4, 4, mi % 2 + 2 * (kt % 2), fx.as_ir_value(sa[mi // 2]),
                                ni % 2 + 2 * (kt % 2), fx.as_ir_value(sb[ni // 2]),
                            ],
                        ))
            s_waitcnt_lgkm0()
            gpu.barrier()  # every wave is done reading A before the epilogue reuses the LDS
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
            grid=(((nblocks + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2 * (H // BN)) if _MID_PAIR else nblocks * (H // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
