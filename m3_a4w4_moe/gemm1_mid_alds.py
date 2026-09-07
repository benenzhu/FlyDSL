# SPDX-License-Identifier: Apache-2.0
"""TN256 AGPR gate/up with the A tile staged through LDS once per workgroup.

Same tiles, W ring, MFMA and epilogue as ``gemm1_mid_agpr_wide_splitring.py``
(read that first). What changes is only how the fp4 A rows reach the MFMAs:

* there every wave gathers its own copy of the whole A tile every K-tile
  (2 x 16-row gathers of 64 B per wave per K-tile, 16 L1 requests each), so the
  4 N-waves of a workgroup issue 4x the A requests the tile needs. Timing
  diagnostics (M3_MID_NO_A=1) put those gathers at 9.5 us of 120 at 512 tokens
  and 20 us of 140 at 1024 (BM64), while the A *bytes* do not matter
  (M3_MID_FAKE_A=1 changes nothing): the CU's load-request rate is the cost;
* here the workgroup loads each A row once per batch of ``KB`` K-tiles
  (KB x 64 B = 256 B per row, full 128 B lines, 4 rows per dwordx4 instruction),
  parks the batch in LDS (row stride padded to 272 B: conflict-free 16 B reads)
  and every wave ds_reads its MFMA A fragments from there. Two LDS slots, one
  ``s_barrier`` per batch; the batch loads are issued right before the W tile
  loads of the same iteration, so the in-order vmcnt wait for that W tile also
  covers them and the W ring keeps its depth.

Timing-only knobs from the base file are kept (FAKE_W, W_CPOL); the W loads
default to nt here (measured +4-6% on every M <= 2048 chain, no W reuse).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from m3_a16w4_moe.vllm_ops import import_ops
from .lab_utils import _pin_accumulators, s_waitcnt_lgkm0

import_ops("moe_a4w4_prefill")
from moe_a4w4_prefill.gemm1 import (  # noqa: E402
    buffer_ops, _swiglu_oai, _quant_prep_fp4, _fmax,
    _e8m0_roundup_fp4, _as_f32, _cvt_pk_fp4, _permlane16_swap,
    Mfma16x16x128Fp4, _asm_void,
)

_MID_FAKE_W = int(os.environ.get("M3_MID_FAKE_W", "0"))
_MID_W_CPOL = int(os.environ.get("M3_MID_W_CPOL", "2"), 0)
# pair mode: workgroups (mb, nb) and (mb+1, nb) get block ids p and p+8, i.e. the same XCD in the same
# dispatch round, so when the two m-blocks belong to one expert the second one finds W in that
# XCD's L2 (a BM-row sort then costs one W read per expert up to 2*BM rows). Default 0.
_MID_PAIR = int(os.environ.get("M3_MID_PAIR", "0"))
# probes used while hunting the prefetch-2/BM64 mismatch (kept for reference, default 0): 1 extra barrier
# before the LDS write; 3 vmcnt(0) before the write; 6 batch loads after the W loads; 7 keep the previous
# A fragments live; 8 s_nop x64 after each MFMA block; 9 pipelined ds_read; 10 every wave writes the whole
# batch; 11/12 vmcnt(0)(+lgkmcnt(0)) before each MFMA block. None of them mattered: the cause was the
# accumulator read fence (see the epilogue).
_ALDS_BAR = int(os.environ.get("M3_ALDS_BAR", "0"))


_G1_TN = int(os.environ.get("M3_MID_G1_TN", "0"))   # 0 = the caller's TILE_N


def compile_moe_gemm1_mid(*, H, I, E, BLOCK_M=32, prefetch=3, k_batch=4, TILE_N=256):
    if _G1_TN:
        TILE_N = _G1_TN
    assert BLOCK_M in (32, 64, 128) and H % 256 == 0 and I % 128 == 0 and TILE_N in (128, 256, 384)
    BM, BN, KT, KB = BLOCK_M, TILE_N, H // 128, k_batch
    assert I % BN == 0 and KT % KB == 0 and KB % 2 == 0
    NI = BN // 4 // 16
    MR = BM // 16
    assert 1 <= prefetch < KT
    ROWB = KB * 64                 # bytes of one row per batch
    CH = ROWB // 16                # 16 B chunks per row per batch
    ROWS_PER_LD = 64 // CH         # rows covered by one dwordx4 wave load
    assert ROWS_PER_LD >= 1 and 64 % CH == 0
    RPW = BM // 4                  # rows loaded by each of the 4 waves
    NLD = RPW // ROWS_PER_LD       # loads per wave per batch
    assert RPW % ROWS_PER_LD == 0
    RS = ROWB + 16                 # padded LDS row stride (bank spread for 16 B reads)
    SLOT = BM * RS
    LDS_BYTES = 2 * SLOT

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=f"gemm1_a4w4_mid_alds_h{H}_i{I}_e{E}_bm{BM}_tn{BN}_pf{prefetch}_kb{KB}", known_block_size=[256, 1, 1])
    def kernel(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32, asbytes: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        nr = buffer_ops.create_buffer_resource(nvalid, max_size=False, num_records_bytes=4)
        valid_rows = fx.Int32(buffer_ops.buffer_load(nr, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        if const_expr(_MID_PAIR):
            g, r = pid // fx.Int32(16), pid % fx.Int32(16)
            item = g * fx.Int32(8) + r % fx.Int32(8)
            mb, nb = (item // fx.Int32(I // BN)) * fx.Int32(2) + r // fx.Int32(8), item % fx.Int32(I // BN)
        else:
            mb, nb = pid // fx.Int32(I // BN), pid % fx.Int32(I // BN)
        if mb * fx.Int32(BM) < valid_rows:
            er = buffer_ops.create_buffer_resource(eids, max_size=False, num_records_bytes=nblocks * 4)
            expert = fx.Int32(buffer_ops.buffer_load(er, mb, vec_width=1, dtype=fx.Int32, is_scalar=True))
            if const_expr(_MID_FAKE_W):
                expert = fx.Int32(0)
            ir = buffer_ops.create_buffer_resource(ids, max_size=False, num_records_bytes=nblocks * BM * 4)
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=ntok * (H // 2))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=E * 2 * I * H // 2)
            asr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=asbytes)
            wsr = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * 2 * I * H // 32)
            orsrc = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=nblocks * BM * (I // 2))
            osr = buffer_ops.create_buffer_resource(OS, max_size=False, num_records_bytes=nblocks * BM * (I // 32))
            mbase = mb * fx.Int32(BM)
            nbase = nb * fx.Int32(BN) + wave * fx.Int32(NI * 16)
            lds = llvm.inttoptr(_ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int32(fx.ptrtoint(smem))))

            # A batch staging: this wave's rows are wave*RPW + j*ROWS_PER_LD + lane//CH (j < NLD),
            # 16 B chunk lane%CH of the batch's ROWB bytes. Padding rows carry a token >= ntok and
            # read as 0 through the OOB-clamped resource.
            if const_expr(_ALDS_BAR == 10):
                NLD_ = BM // ROWS_PER_LD
                ld_row = [fx.Int32(j * ROWS_PER_LD) + lane // fx.Int32(CH) for j in range_constexpr(NLD_)]
            else:
                NLD_ = NLD
                ld_row = [wave * fx.Int32(RPW) + fx.Int32(j * ROWS_PER_LD) + lane // fx.Int32(CH) for j in range_constexpr(NLD)]
            ld_chunk = (lane % fx.Int32(CH)) * fx.Int32(16)
            ld_tok = [fx.Int32(buffer_ops.buffer_load(ir, mbase + ld_row[j], vec_width=1, dtype=fx.Int32)) & fx.Int32(0xFFFFFF) for j in range_constexpr(NLD_)]
            ld_gbyte = [(ld_tok[j] * fx.Int32(H // 2) + ld_chunk) // fx.Int32(4) for j in range_constexpr(NLD_)]
            ld_lbyte = [ld_row[j] * fx.Int32(RS) + ld_chunk for j in range_constexpr(NLD_)]

            def load_a_batch(b):
                return [buffer_ops.buffer_load(ar, ld_gbyte[j], vec_width=4, dtype=fx.Int32, soffset_bytes=fx.Int32(b * ROWB)) for j in range_constexpr(NLD_)]

            def _bar():
                if const_expr(_ALDS_BAR == 2):
                    _asm_void([], "s_waitcnt lgkmcnt(0)\ns_barrier", "", "~{memory}")
                else:
                    s_waitcnt_lgkm0()
                    gpu.barrier()

            def stage_a_batch(regs, slot):
                if const_expr(_ALDS_BAR == 1):
                    _bar()
                if const_expr(_ALDS_BAR == 3):
                    rocdl.s_waitcnt(0x0F70)   # vmcnt(0): probe whether the batch loads are really complete here
                for j in range_constexpr(NLD_):
                    ptr = buffer_ops.get_element_ptr(lds, byte_offset=fx.as_ir_value(fx.Int32(slot * SLOT) + ld_lbyte[j]), elem_type=T.i8)
                    llvm.StoreOp(fx.as_ir_value(regs[j]), ptr, alignment=16)
                _bar()

            rd_base = [fx.Int32(mi * 16) * fx.Int32(RS) + l16 * fx.Int32(RS) + q16 * fx.Int32(16) for mi in range_constexpr(MR)]

            def read_a_tile(kt):
                slot, j = (kt // KB) % 2, kt % KB
                out = []
                for mi in range_constexpr(MR):
                    ptr = buffer_ops.get_element_ptr(lds, byte_offset=fx.as_ir_value(rd_base[mi] + fx.Int32(slot * SLOT + j * 64)), elem_type=T.i8)
                    out.append(fx.Vector(llvm.load(T.vec(4, T.i32), ptr, alignment=16)))
                return out

            def load_b_tile(kt, prev):
                bb = []
                for gu in range_constexpr(2):
                    slab = []
                    for ni in range_constexpr(NI):
                        nblk = expert * fx.Int32(2 * I // 16) + nbase // fx.Int32(16) + fx.Int32(gu * I // 16 + ni)
                        off = q16 * fx.Int32(256) + l16 * fx.Int32(16)
                        slab.append(buffer_ops.buffer_load(wr, off // fx.Int32(4), vec_width=4, dtype=fx.Int32, cache_modifier=_MID_W_CPOL,
                                                    soffset_bytes=nblk * fx.Int32(H * 8) + fx.Int32(kt * 1024)))
                    bb.append(slab)
                if const_expr(kt % 2 == 0):
                    sb = [[buffer_ops.buffer_load(wsr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                         soffset_bytes=((expert * fx.Int32(2 * I // 32) + nbase // fx.Int32(32) + fx.Int32(gu * I // 32 + np)) * fx.Int32(H // 256 * 64) + fx.Int32(kt // 2 * 64)) * fx.Int32(4)) for np in range_constexpr(NI // 2)] for gu in range_constexpr(2)]
                else:
                    sb = prev[1]
                return bb, sb

            def load_a_scale(kt, prev):
                if const_expr(kt % 2 == 0):
                    return [buffer_ops.buffer_load(asr, q16 * fx.Int32(16) + l16, vec_width=1, dtype=fx.Int32,
                                            soffset_bytes=((mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(H // 256 * 64) + fx.Int32(kt // 2 * 64)) * fx.Int32(4)) for mp in range_constexpr(MR // 2)]
                return prev

            mfma = Mfma16x16x128Fp4(MR, NI)
            acc = [[[None for gu in range(2)] for ni in range(NI)] for mi in range(MR)]
            # prologue: batch 0 -> LDS, then the W ring; the batch-0 wait does not touch the W loads.
            abuf = load_a_batch(0)
            ring = []
            for kt in range_constexpr(prefetch):
                ring.append(load_b_tile(kt, ring[-1] if ring else None))
            sring = [load_a_scale(0, None)]
            stage_a_batch(abuf, 0)
            abuf = None
            aa_prev = None
            aa_next = None
            for kt in range_constexpr(KT):
                if const_expr(kt % KB == 0 and kt + KB < KT and _ALDS_BAR != 6):
                    abuf = load_a_batch(kt // KB + 1)      # before this iteration's W loads
                if const_expr(kt + prefetch < KT):
                    ring.append(load_b_tile(kt + prefetch, ring[-1]))
                if const_expr(kt % KB == 0 and kt + KB < KT and _ALDS_BAR == 6):
                    abuf = load_a_batch(kt // KB + 1)      # probe: after the W loads instead
                if const_expr(kt + 1 < KT):
                    sring.append(load_a_scale(kt + 1, sring[-1]))
                bb, sb = ring.pop(0)
                sa = sring.pop(0)
                if const_expr(_ALDS_BAR == 9):
                    aa = aa_next if aa_next is not None else read_a_tile(kt)
                    aa_next = read_a_tile(kt + 1) if (kt % KB != KB - 1 and kt + 1 < KT) else None
                else:
                    aa = read_a_tile(kt)
                if const_expr(_ALDS_BAR == 11):
                    rocdl.s_waitcnt(0x0070)   # probe: vmcnt(0) lgkmcnt(0) before every MFMA block
                if const_expr(_ALDS_BAR == 12):
                    rocdl.s_waitcnt(0x0F70)   # probe: vmcnt(0) only
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(NI):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = mfma._mfma_agpr(
                                aa[mi], bb[gu][ni], acc[mi][ni][gu], sa[mi // 2], sb[gu][ni // 2],
                                kt % 2, mi % 2, ni % 2,
                            )
                if const_expr(_ALDS_BAR == 7):
                    if aa_prev is not None:
                        _asm_void([fx.as_ir_value(v) for v in aa_prev], "", ",".join(["v"] * MR))
                    aa_prev = aa
                if const_expr(_ALDS_BAR == 8):
                    _asm_void([], "s_nop 15\ns_nop 15\ns_nop 15\ns_nop 15", "")
                if const_expr(kt % KB == KB - 1 and kt + 1 < KT):
                    stage_a_batch(abuf, (kt // KB + 1) % 2)   # this batch's reads are done (MFMAs above)
                    abuf = None

            # Root cause of the prefetch-2/BM64 mismatch (2026-09-07): the epilogue's v_accvgpr_read copies
            # depend only on the asm MFMA outputs, so the scheduler hoisted them above a dependency-free
            # "s_nop" fence, right behind the last MFMAs (the compiler cannot see the XDL-write ->
            # accvgpr-read hazard inside inline asm). Route every accumulator through an asm that outputs
            # it again (tied "=a"/"0") with the wait states inside: reads now depend on the fence.
            flat = _pin_accumulators([acc[mi][ni][gu] for mi in range_constexpr(MR) for ni in range_constexpr(NI) for gu in range_constexpr(2)])
            for mi in range_constexpr(MR):
                for ni in range_constexpr(NI):
                    for gu in range_constexpr(2):
                        acc[mi][ni][gu] = fx.Vector(flat[(mi * NI + ni) * 2 + gu])
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
                    buffer_ops.buffer_store(packed, orsrc, row * fx.Int32(I // 2) + col // fx.Int32(2), offset_is_bytes=True)
                colgrp = nbase // fx.Int32(32) + fx.Int32(np)
                for mp in range_constexpr(MR // 2):
                    blk = (mbase // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(I // 256) + colgrp // fx.Int32(8)
                    off = blk * fx.Int32(256) + (colgrp % fx.Int32(4)) * fx.Int32(64) + l16 * fx.Int32(4) + ((colgrp % fx.Int32(8)) // fx.Int32(4)) * fx.Int32(2)
                    pair = fx.Int32(exps[2 * mp] | (exps[2 * mp + 1] << 8)).to(fx.Int16)
                    buffer_ops.buffer_store(pair, osr, off, offset_is_bytes=True, mask=q16 == fx.Int32(0))

    @flyc.jit
    def launch(
        A: fx.Tensor, W: fx.Tensor, O: fx.Tensor, AS: fx.Tensor,
        WS: fx.Tensor, OS: fx.Tensor, ids: fx.Tensor, eids: fx.Tensor,
        nvalid: fx.Tensor, ntok: fx.Int32, nblocks: fx.Int32, asbytes: fx.Int32,
        tile_map: fx.Tensor, grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(A, W, O, AS, WS, OS, ids, eids, nvalid, ntok, nblocks, asbytes).launch(
            grid=(((nblocks + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2 * (I // BN)) if _MID_PAIR else nblocks * (I // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
