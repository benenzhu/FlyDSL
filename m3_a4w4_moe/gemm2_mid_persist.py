# SPDX-License-Identifier: Apache-2.0
"""Four-wave middle-batch down projection with A retained across N tiles.

Each CTA covers one sorted M block and one hidden-dimension chunk. All six
K128 activation fragments stay in registers, while the weight prefetch ring
continues across N-tile boundaries. C is explicitly in AGPR. Output is the
same deterministic MXFP8 partial/scales layout consumed by reduce_fp8.py.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import buffer_ops as bop
from .gemm1 import Mfma16x16x128Fp4, _asm_void, _as_f32
from .gemm2 import _cvt_pk_fp8, _e8m0_fp8, _fabs, _maxf


def compile_moe_gemm2_mid_persist(*, H, I, E, topk=5, BLOCK_M=32, n_split=4, prefetch=3):
    BM, BN, MR, NI, KT = BLOCK_M, 256, BLOCK_M // 16, 4, I // 128
    assert BM in (32, 64) and I == 768 and H % (BN * n_split) == 0
    assert 1 <= prefetch < KT
    NT = H // BN // n_split

    @flyc.kernel(name=f"gemm2_a4w4_mid_persist_h{H}_i{I}_bm{BM}_ns{n_split}_pf{prefetch}", known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor, WS: fx.Tensor,
               OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor, nvalid: fx.Tensor,
               ntok: fx.Int32, nblocks: fx.Int32):
        tx, pid = fx.Int32(gpu.thread_id("x")), fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        mb, chunk = pid // fx.Int32(n_split), pid % fx.Int32(n_split)
        nr = bop.create_buffer_resource(nvalid, max_size=False, num_records_bytes=4)
        rows = fx.Int32(bop.buffer_load(nr, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        if mb * fx.Int32(BM) < rows:
            er = bop.create_buffer_resource(eids, max_size=False, num_records_bytes=nblocks * 4)
            expert = fx.Int32(bop.buffer_load(er, mb, vec_width=1, dtype=fx.Int32, is_scalar=True))
            ir = bop.create_buffer_resource(ids, max_size=False, num_records_bytes=nblocks * BM * 4)
            ar = bop.create_buffer_resource(A, max_size=False, num_records_bytes=nblocks * BM * (I // 2))
            wr = bop.create_buffer_resource(W, max_size=False, num_records_bytes=E * H * I // 2)
            asr = bop.create_buffer_resource(AS, max_size=False, num_records_bytes=nblocks * BM * (I // 32))
            wsr = bop.create_buffer_resource(WS, max_size=False, num_records_bytes=E * H * I // 32)
            outr = bop.create_buffer_resource(O, max_size=False, num_records_bytes=ntok * topk * H)
            osr = bop.create_buffer_resource(OS, max_size=False, num_records_bytes=ntok * topk * (H // 32))
            mbase = mb * fx.Int32(BM)
            nbase0 = chunk * fx.Int32(NT * BN) + wave * fx.Int32(NI * 16)
            fused = [fx.Int32(bop.buffer_load(ir, mbase + fx.Int32(mi * 16) + l16, vec_width=1, dtype=fx.Int32)) for mi in range_constexpr(MR)]
            a = [[bop.buffer_load(ar, ((mbase + fx.Int32(mi * 16) + l16) * fx.Int32(I // 2) + q16 * fx.Int32(16)) // fx.Int32(4), vec_width=4, dtype=fx.Int32,
                                 soffset_bytes=fx.Int32(kt * 64)) for mi in range_constexpr(MR)] for kt in range_constexpr(KT)]
            sa = [[bop.buffer_load(asr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                  soffset_bytes=((mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(I // 256 * 64) + fx.Int32(kg * 64)) * fx.Int32(4))
                   for mp in range_constexpr(MR // 2)] for kg in range_constexpr(KT // 2)]

            def load_b(step, prev):
                nt, kt = step // KT, step % KT
                nbase = nbase0 + fx.Int32(nt * BN)
                bb = [bop.buffer_load(wr, (q16 * fx.Int32(256) + l16 * fx.Int32(16)) // fx.Int32(4), vec_width=4, dtype=fx.Int32,
                                      soffset_bytes=(expert * fx.Int32(H // 16) + nbase // fx.Int32(16) + fx.Int32(ni)) * fx.Int32(I * 8) + fx.Int32(kt * 1024))
                      for ni in range_constexpr(NI)]
                if const_expr(kt % 2 == 0):
                    sb = [bop.buffer_load(wsr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                          soffset_bytes=((expert * fx.Int32(H // 32) + nbase // fx.Int32(32) + fx.Int32(np)) * fx.Int32(I // 256 * 64) + fx.Int32(kt // 2 * 64)) * fx.Int32(4))
                          for np in range_constexpr(NI // 2)]
                else:
                    sb = prev[1]
                return bb, sb

            def store_tile(acc, nt):
                # The first accumulator read is older than seven MFMAs; the
                # explicit tail delay also protects the final accumulator.
                _asm_void([], "s_nop 15\ns_nop 15", "")
                nbase = nbase0 + fx.Int32(nt * BN)
                for mi in range_constexpr(MR):
                    token = fused[mi] & fx.Int32(0xFFFFFF)
                    slot = fused[mi].shrui(fx.Int32(24))
                    row = token * fx.Int32(topk) + slot
                    for np in range_constexpr(NI // 2):
                        v = [fx.Float32(fx.Vector(acc[mi][2 * np + ni])[ii]) for ni in range_constexpr(2) for ii in range_constexpr(4)]
                        amax = _fabs(v[0])
                        for ii in range_constexpr(1, 8):
                            amax = _maxf(amax, _fabs(v[ii]))
                        amax = _maxf(amax, amax.shuffle_xor(16, 64))
                        amax = _maxf(amax, amax.shuffle_xor(32, 64))
                        e8 = _e8m0_fp8(amax)
                        sf = _as_f32(e8 << 23)
                        for ni in range_constexpr(2):
                            z = v[ni * 4:ni * 4 + 4]
                            packed = _cvt_pk_fp8(fx.Int32(0), z[0], z[1], sf, False)
                            packed = _cvt_pk_fp8(packed, z[2], z[3], sf, True)
                            col = nbase + fx.Int32((np * 2 + ni) * 16) + q16 * fx.Int32(4)
                            bop.buffer_store(packed, outr, row * fx.Int32(H) + col, offset_is_bytes=True, mask=token < ntok)
                        sc = row * fx.Int32(H // 32) + nbase // fx.Int32(32) + fx.Int32(np)
                        bop.buffer_store(fx.Int32(e8).to(fx.Uint8), osr, sc, offset_is_bytes=True, mask=(token < ntok) & (q16 == fx.Int32(0)))

            mfma = Mfma16x16x128Fp4(MR, NI)
            ring = []
            for step in range_constexpr(prefetch):
                ring.append(load_b(step, ring[-1] if ring else None))
            for nt in range_constexpr(NT):
                acc = [[None for ni in range(NI)] for mi in range(MR)]
                for kt in range_constexpr(KT):
                    step = nt * KT + kt
                    if const_expr(step + prefetch < NT * KT):
                        ring.append(load_b(step + prefetch, ring[-1]))
                    bb, sb = ring.pop(0)
                    for mi in range_constexpr(MR):
                        for ni in range_constexpr(NI):
                            acc[mi][ni] = mfma._mfma_agpr(a[kt][mi], bb[ni], acc[mi][ni], sa[kt // 2][mi // 2], sb[ni // 2], kt % 2, mi % 2, ni % 2)
                store_tile(acc, nt)

    @flyc.jit
    def launch(A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor, WS: fx.Tensor,
               OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor, weights: fx.Tensor,
               nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel(A, W, O, AS, WS, OS, ids, eids, nvalid, ntok, nblocks).launch(
            grid=(nblocks * n_split, 1, 1), block=(256, 1, 1), stream=stream)
    return launch
