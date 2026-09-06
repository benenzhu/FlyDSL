# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MiniMax-M3 prefill MoE stage 2 (a4w4, down projection) with an MXFP8
token-major output.

    y[tok, slot, :] = h[row, :] @ W2[e]^T          row = sorted row of (tok, slot)
    out[tok*topk + slot, :], out_scale[..] = mxfp8_quant(y)   (e4m3, per 32 cols)

Why a new kernel and not gemm1's structure again: K = I = 768 is only 3
K-steps, so per 128 x 256 output tile the production kernel spends most of
its time on the prologue / epilogue, and its bf16 sorted output (1 GB at
16384 tokens) is re-read by the reduce. Here

  * one CTA owns an m-tile of 128 sorted rows and sweeps ``N_TILES`` n-tiles
    of 256 columns (``n_split`` CTAs share the 6144 columns). The A tile
    (128 x 384 B fp4 + its scales) is DMA'd to LDS once, read into
    registers once and pinned in AGPRs for the whole sweep: the K loop
    only streams W2 (32 KB per step) through a 2-set LDS ping-pong. That
    keeps the LDS read traffic at 64 KB per 1024-cycle step (B only,
    2 waves share each half) instead of 96 KB.
  * the (n-tile, k-step) sequence is one flat depth-2 pipeline: the loads
    for the first two steps of n-tile t+1 are in flight while the last step
    of n-tile t computes and its epilogue stores.
  * epilogue: per-32-col amax across the 4 lanes of a row, E8M0 scale
    ``floor(log2 amax) - 7`` (the block's max lands in [128, 256): never
    saturates e4m3), ``v_cvt_scalef32_pk_fp8_f32``, permlane16 swap so a
    lane owns 8 consecutive fp8 -> one dwordx2 store; the 4 scale bytes of
    a row's 128 columns of the wave -> one dword store. Rows are written
    token-major (``(tok*topk + slot) * H``) straight from ``sorted_ids``;
    padded rows (tok == n_tokens) fall outside the buffer resource and are
    dropped, so the reduce reads ``topk`` contiguous rows per token and no
    reverse map is needed.
  * block -> work: consecutive m-tiles (same expert, same 2.36 MB W2 slab)
    go to the same XCD; the per-XCD share is computed from the device-side
    valid row count so the fully padded tail tiles cost nothing.

Layouts (bytes):
  A        [num_m_blocks*BM, I/2]          sorted rows, fp4 (gemm1 OUT_Q)
  A_scale  [num_m_blocks*BM, I/32]         sorted rows, e8m0-shuffled (gemm1 OUT_sc)
  W2       [E, H, I/2]                     aiter shuffle_weight(layout=(16,16))
  W2_sc    [E*H, I/32]                     aiter e8m0_shuffle
  OUT      [n_tokens*topk, H]              fp8 e4m3 (OCP), token-major
  OUT_sc   [n_tokens*topk, H/32]           e8m0, row-major
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops

from m3_a4w4_moe.gemm1 import (
    G2SLoaderAsm,
    Mfma16x16x128Fp4,
    S2RLoaderFp4,
    _Buf,
    _FP4_PACK,
    _N_WAVES,
    _as_f32,
    _asm_void,
    _bits,
    _divmod_nonneg,
    _fmax,
    _g2s_thunks,
    _intrin_f32,
    _lds_ptr_t,
    _min,
    _permlane16_swap,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _uniform_i32,
    wait_barrier,
)

_NUM_XCDS = 8
# timing knock-out experiments (never for real use)
import os as _os

_NO_EPI = _os.environ.get("M3_G2_NO_EPI", "0") == "1"  # skip the epilogue entirely (no stores)


class _MfmaAgprA(Mfma16x16x128Fp4):
    """Same MFMA, but the A-matrix fragment is read from AGPRs (the A tile is
    pinned there for the whole n-sweep; gfx950 accepts AGPR src A/B)."""

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        a_op, b_op = b_op, a_op  # feeds (B, A): lane L holds C[row L%16, cols 4*(L//16)..]
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        src2 = "$0" if acc is not None else "0"
        asm = f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 {opsel} {opsel_hi} cbsz:4 blgp:4"
        ops = [fx.as_ir_value(a_op), fx.as_ir_value(b_op), fx.as_ir_value(sa_v), fx.as_ir_value(sb_v)]
        cons = "=a,v,a,v,v"  # $2 = the A-matrix fragment -> AGPR
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


def _lds_load_i32(addr_i32):
    ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(addr_i32))
    return fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, ptr, alignment=4).result)


def _i1(v: bool):
    t = _ir.IntegerType.get_signless(1)
    return _arith.ConstantOp(t, _ir.IntegerAttr.get(t, 1 if v else 0)).result


def _cvt_pk_fp8(old, a, b, scale_f32, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32: (a, b) / scale -> 2 x e4m3 into the low (hi=False)
    or high 16 bits of ``old`` (the intrinsic works on <2 x i16>; i32 in/out here)."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    old_v = _llvm.bitcast(v2i16, fx.as_ir_value(old))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [old_v, fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(scale_f32), _i1(hi)],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(_T.i32, res))


def _e8m0_fp8(amax):
    """floor(log2 amax) - 7, floored at 0: amax / 2^(e-127) lands in [128, 256),
    inside e4m3 (max 448) with one bit of headroom, never saturates."""
    e = (_bits(amax) >> 23) - fx.Int32(7)
    return fx.arith.select(e > fx.Int32(0), e, fx.Int32(0))


def _fabs(v):
    return _intrin_f32("llvm.fabs.f32", [v])


def _maxf(a, b):
    """v_max_f32 (arith.maxnumf), not the select-based gemm1 helper"""
    return fx.Float32(fx.arith.maxnumf(fx.as_ir_value(a), fx.as_ir_value(b)))


def _permlane32_swap(d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane32_swap(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(_llvm.extractvalue(_T.i32, res, [1]))


def _xlane_max4(x):
    """max over the 4 lanes {L, L^16, L^32, L^48} (one row's 4 column groups), in every lane.
    permlane32_swap(x, x) yields [lo, lo] / [hi, hi]; permlane16_swap(y, y) yields
    [r0, r0, r2, r2] / [r1, r1, r3, r3]; a max after each gives the butterfly."""
    xi = _bits(x)
    a, b = _permlane32_swap(xi, xi)
    m = _maxf(_as_f32(a), _as_f32(b))
    mi = _bits(m)
    a, b = _permlane16_swap(mi, mi)
    return _maxf(_as_f32(a), _as_f32(b))


def _pin_vec4(v):
    """Route an accumulator through a side-effecting no-op asm with a VGPR operand.
    The AGPR -> VGPR copies feeding an inline-asm operand are glued to the asm, so they
    are emitted where the asm sits in the (ordered) asm stream instead of being hoisted
    by the SelectionDAG scheduler to right behind the MFMA that produced the value
    (which the compiler cannot see as an MFMA -> no hazard wait states)."""
    ty = Vec.make_type(4, fx.Float32)
    return _llvm.inline_asm(ty, [fx.as_ir_value(v)], "; pin $0", "=v,0", has_side_effects=True)


def _v2i32(a, b):
    ty = _ir.VectorType.get([2], _T.i32)
    return _vector.FromElementsOp(ty, [fx.as_ir_value(a), fx.as_ir_value(b)]).result


class _BScaleGather:
    """Per pipeline step the 8 e8m0 blocks (256 B = 32 W2 rows x 8 K-groups) of
    the n-tile's 256 rows for one K-step, 2 ``buffer_load_dword ... lds`` per
    wave (wave w fetches row-groups 2w, 2w+1 into slot bytes [2w*256, +512))."""

    def __init__(self, rsrc, lane_id, wave_id, region_base_i32):
        self.rsrc = fx.as_ir_value(rsrc)
        self.voff = [fx.as_ir_value((wave_id * 2 + q) * fx.Int32(768) + lane_id * fx.Int32(4)) for q in range(2)]
        wave_u = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(wave_id)))
        self.base_s = _uniform_i32(region_base_i32 + wave_u * fx.Int32(512))

    def gather(self, slot_byte_off, soff_bytes):
        m0 = fx.as_ir_value(fx.Int32(self.base_s) + slot_byte_off)
        soff = _uniform_i32(soff_bytes)
        asm = (
            "s_mov_b32 m0, $0\n"
            "buffer_load_dword $1, $2, $3 offen lds\n"
            "s_add_u32 m0, 256, m0\n"
            "buffer_load_dword $4, $2, $3 offen lds"
        )
        _asm_void([m0, self.voff[0], self.rsrc, soff, self.voff[1]], asm, "s,v,s,s,v", "~{scc}")


def compile_moe_gemm2(
    *,
    H: int,
    I: int,
    E: int,
    topk: int,
    n_split: int = 2,
):
    """fp4 grouped down-projection with mxfp8 token-major output. Sorted inputs
    must come from a block_m = 128 sort (a 256 sort works too: the expert of
    m-tile t is ``sorted_expert_ids[t // 2]`` -- pass ``eid_shift=1`` on the
    launcher; not exposed yet)."""
    BM = 128
    BN = 256
    K = I
    K_BYTES = K // 2
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    assert K % BLOCK_K == 0
    K_ITERS = K // BLOCK_K  # 3
    N_TILES_ALL = H // BN  # 24
    assert N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    assert NT % 2 == 0, "the n loop is unrolled by 2 (static LDS ping-pong)"
    LDS_BLOCK_M = BM // 2  # 64 rows per A half
    LDS_BLOCK_N = BN // 2  # 128 W2 rows per B half
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 2
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    SC_COLS = K // 32  # 24 e8m0 per A row
    SC_BLOCKS_PER_G = SC_COLS // 8  # 3 blocks of 256 B per 32-row group
    OUT_SC_COLS = H // 32

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BUFS = K_ITERS * 2 * a_lds_size  # 48 KB, whole A tile
    B_SETS = 2
    B_BUFS = B_SETS * 2 * b_lds_size  # 64 KB
    LDS_TILES_BYTES = A_BUFS + B_BUFS
    A_SC_BYTES = 4096  # 12 blocks used (4 groups x 3 K-blocks), 4 waves x 1 KB DMA
    B_SC_SLOT = 2048  # 8 blocks
    B_SC_SLOTS = 4
    SC_LDS_BYTES = A_SC_BYTES + B_SC_SLOTS * B_SC_SLOT
    B_SC_OFF = A_SC_BYTES

    NB = N_TILES_B
    NG = 2  # gather loads per wave per step
    P_STEP = 2 * NB + NG  # loads per lane per step
    SEG2 = NB + NG
    ST = 4 * 4 + 4  # epilogue stores per lane: 16 dwordx2 + 4 scale dwords
    assert P_STEP + ST <= 63

    B_TILE_BYTES = BN * K_BYTES  # W2 bytes per n-tile
    B_K_STEP = 2 * 1024  # preshuffled: 128 B of K = 2 x (16 rows x 64 B)

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SC_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr
        _sc_ptr = lds.scale_lds.ptr

        def a_buf(kb, half):
            return _Buf(_base_ptr, (kb * 2 + half) * a_lds_size)

        def b_buf(s, half):
            return _Buf(_base_ptr, A_BUFS + (s * 2 + half) * b_lds_size)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # ---- work item: (m-tile, n-chunk); consecutive m-tiles share an XCD ----
        nv_rsrc = _buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=4)
        num_valid = fx.Int32(_buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        intra, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        work = xcd * per_xcd + intra
        block_valid = (intra < per_xcd) & (work < n_work)
        work_safe = fx.arith.select(block_valid, work, fx.Int32(0))
        tile_i, chunk = _divmod_nonneg(work_safe, n_split)
        m_base = tile_i * BM
        eid_rsrc = _buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4)
        expert = fx.Int32(_buffer_ops.buffer_load(eid_rsrc, tile_i, vec_width=1, dtype=fx.Int32, is_scalar=True))
        chunk_n0 = chunk * NT  # first n-tile (global index) of this CTA

        if block_valid:
            ids_rsrc = _buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES))
            as_rsrc = _buffer_ops.create_buffer_resource(
                A_scale, max_size=False, num_records_bytes=num_m_blocks * (BM * SC_COLS)
            )
            b_rsrc = _buffer_ops.create_buffer_resource(W2, max_size=False, num_records_bytes=E * H * K_BYTES)
            bs_rsrc = _buffer_ops.create_buffer_resource(W2_scale, max_size=False, num_records_bytes=E * H * SC_COLS)

            # ---- A: contiguous sorted rows, swizzled 128-B LDS rows ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    offs.append((m_base + half * LDS_BLOCK_M + row) * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            # ---- B: LDS half hb, LDS row r -> W2 row (r//64)*128 + hb*64 + r%64 of the
            #      n-tile, so wave_j's 64 rows of both halves are 128 consecutive columns ----
            def _b_offsets(hb):
                offs = []
                for rnd in range_constexpr(NB):
                    r = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    w2row = (r // 64) * 128 + hb * 64 + (r % 64)
                    offs.append(
                        (w2row // 16) * (K_BYTES * 16)
                        + (w2row % 16) * 16
                        + (col // 64) * 1024
                        + ((col % 64) // 16) * 256
                        + (col % 16)
                    )
                return offs

            a0_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(0), N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(1), N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(0), NB, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(1), NB, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            # A scales: 4 x 1 KB = the 3 KB of this m-tile's 4 row groups (+1 KB spill, harmless)
            as_g2s = G2SLoaderAsm(
                as_rsrc, [(m_base // 32) * fx.Int32(SC_BLOCKS_PER_G * 256) + wave_id * 1024 + lane_id * 16], 1, wave_id
            )
            as_g2s.set_wave_base(_sc_ptr)
            sc_base_i32 = fx.Int32(fx.ptrtoint(_sc_ptr))
            bsg = _BScaleGather(bs_rsrc, lane_id, wave_id, sc_base_i32 + fx.Int32(B_SC_OFF))

            a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
            b_s2r = S2RLoaderFp4(wave_j, NB)
            mfma = _MfmaAgprA(N_TILES_A, NB)

            expert_rows = expert * fx.Int32(H)
            b_base_bytes = expert_rows * fx.Int32(K_BYTES)
            bs_base_bytes = expert_rows * fx.Int32(SC_COLS)

            def _b_soff(nt, kb):
                """W2 byte offset of (n-tile nt [CTA-local], K-step kb)"""
                return b_base_bytes + (chunk_n0 + nt) * fx.Int32(B_TILE_BYTES) + fx.Int32(kb * B_K_STEP)

            def _g_soff(nt, kb):
                return bs_base_bytes + (chunk_n0 + nt) * fx.Int32(BN * SC_COLS) + fx.Int32(kb * 256)

            def _slot_off(nt, kb):
                """scale slot of flat step 3*nt + kb"""
                s = (nt * fx.Int32(K_ITERS) + fx.Int32(kb)) & fx.Int32(B_SC_SLOTS - 1)
                return s * fx.Int32(B_SC_SLOT)

            def _gather(nt, kb):
                bsg.gather(_slot_off(nt, kb), _g_soff(nt, kb))

            # ---- prologue: A (12), A scales (1), then steps 0/1 of n-tile 0 ----
            for kb in range_constexpr(K_ITERS):
                a0_g2s.load(a_buf(kb, 0), fx.Int32(kb * BLOCK_K_BYTES))
                a1_g2s.load(a_buf(kb, 1), fx.Int32(kb * BLOCK_K_BYTES))
            as_g2s.load(_Buf(_sc_ptr, 0), fx.Int32(0))
            n0 = fx.Int32(0)
            _gather(n0, 0)
            b0_g2s.load(b_buf(0, 0), _b_soff(n0, 0))
            b1_g2s.load(b_buf(0, 1), _b_soff(n0, 0))
            _gather(n0, 1)
            b0_g2s.load(b_buf(1, 0), _b_soff(n0, 1))
            b1_g2s.load(b_buf(1, 1), _b_soff(n0, 1))
            _gather(n0, 2)

            # A + A scales landed (steps 0/1 of B may fly)
            wait_barrier(NG + 2 * NB + NG + 2 * NB + NG)
            aF = [[a_s2r.load(a_buf(kb, h)) for h in range(2)] for kb in range(K_ITERS)]
            lane_sc = (lane_id // 4) * 16 + (lane_id % 4) * 4
            saA = [
                [
                    _lds_load_i32(sc_base_i32 + ((h * 2 + wave_i) * SC_BLOCKS_PER_G + kb) * fx.Int32(256) + lane_sc)
                    for kb in range(K_ITERS)
                ]
                for h in range(2)
            ]

            def _read_bsc_thunks(nt, kb, holder):
                base = sc_base_i32 + fx.Int32(B_SC_OFF) + _slot_off(nt, kb) + lane_sc

                def _rd(idx, hb, sub):
                    gi = wave_j * 4 + hb * 2 + sub
                    holder[idx] = _lds_load_i32(base + fx.Int32(gi * 256))

                return [lambda: _rd(0, 0, 0), lambda: _rd(1, 0, 1), lambda: _rd(2, 1, 0), lambda: _rd(3, 1, 1)]

            # step (0,0) B + its scales landed
            wait_barrier(NG + 2 * NB + NG)
            b0f = b_s2r.load(b_buf(0, 0), preshuffled=True)
            b1f = b_s2r.load(b_buf(0, 1), preshuffled=True)
            _sc0 = [None] * 4
            for t in _read_bsc_thunks(n0, 0, _sc0):
                t()

            # ---- output rows (token-major) of this lane's 4 row tiles ----
            out_rsrc = _buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=n_tokens * (topk * H))
            osc_rsrc = _buffer_ops.create_buffer_resource(
                OUT_scale, max_size=False, num_records_bytes=n_tokens * (topk * OUT_SC_COLS)
            )
            out_off = [[None, None], [None, None]]
            sc_off = [[None, None], [None, None]]
            for h in range_constexpr(2):
                for ti in range_constexpr(N_TILES_A):
                    srow = m_base + h * LDS_BLOCK_M + wave_i * (N_TILES_A * 16) + ti * 16 + r16
                    sid = fx.Int32(_buffer_ops.buffer_load(ids_rsrc, srow, vec_width=1, dtype=fx.Int32))
                    tok = sid & fx.Int32(0x00FFFFFF)  # padded rows: tok == n_tokens -> OOB -> dropped
                    slot = (sid >> 24) & fx.Int32(0xFF)
                    orow = tok * fx.Int32(topk) + slot
                    out_off[h][ti] = orow * fx.Int32(H)
                    sc_off[h][ti] = orow * fx.Int32(OUT_SC_COLS)
            g_is0 = g4 == fx.Int32(0)

            # ---- epilogue pieces, interleaved into the MFMA shadow ----
            def _epi_group(cq, h, hb, ti, p, nt, mask):
                """one 32-col group of one 16-row tile: amax over the row's 4 lanes, e8m0,
                8 fp8, dwordx2 store. Returns the e8m0 (i32)."""
                cv = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p)]))
                cw = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p + 1)]))
                v = [fx.Float32(cv[k]) for k in range_constexpr(4)] + [fx.Float32(cw[k]) for k in range_constexpr(4)]
                amax = _fabs(v[0])
                for k in range_constexpr(1, 8):
                    amax = _maxf(amax, _fabs(v[k]))
                amax = _xlane_max4(amax)
                e8 = _e8m0_fp8(amax)
                sf = _as_f32(e8 << 23)
                da = _cvt_pk_fp8(fx.Int32(0), v[0], v[1], sf, False)
                da = _cvt_pk_fp8(da, v[2], v[3], sf, True)
                db = _cvt_pk_fp8(fx.Int32(0), v[4], v[5], sf, False)
                db = _cvt_pk_fp8(db, v[6], v[7], sf, True)
                da, db = _permlane16_swap(da, db)
                # lane group g now holds tile (2p + g%2), cols (g//2)*8 .. +8
                col = (chunk_n0 + nt) * fx.Int32(BN) + wave_j * 128 + hb * 64 + (2 * p + (g4 % 2)) * 16 + (g4 // 2) * 8
                _buffer_ops.buffer_store(_v2i32(da, db), out_rsrc, out_off[h][ti] + col, mask=mask, offset_is_bytes=True)
                return e8

            def _epi_scale(h, ti, e4, nt, mask):
                """the row's 4 scale bytes of this wave's 128 columns -> one dword (lanes g==0)"""
                packed = e4[0] | (e4[1] << 8) | (e4[2] << 16) | (e4[3] << 24)
                sc_col = (chunk_n0 + nt) * fx.Int32(BN // 32) + wave_j * 4
                m = g_is0 if mask is None else (g_is0 & mask)
                _buffer_ops.buffer_store(packed, osc_rsrc, sc_off[h][ti] + sc_col, mask=fx.as_ir_value(m), offset_is_bytes=True)

            def _group_thunks(cq, h, hb, nt, hold, mask=None):
                ts = []
                if const_expr(_NO_EPI):
                    return ts
                for ti in range_constexpr(N_TILES_A):
                    for p in range_constexpr(NB // 2):

                        def t(ti=ti, p=p):
                            hold[(hb, ti, p)] = _epi_group(cq, h, hb, ti, p, nt, mask)

                        ts.append(t)
                return ts

            def _scale_thunks(h, nt, hold, mask=None):
                ts = []
                if const_expr(_NO_EPI):
                    return ts
                for ti in range_constexpr(N_TILES_A):

                    def t(ti=ti):
                        _epi_scale(h, ti, [hold[(0, ti, 0)], hold[(0, ti, 1)], hold[(1, ti, 0)], hold[(1, ti, 1)]], nt, mask)

                    ts.append(t)
                return ts

            def _wb2(first, c_first, c_other):
                """``first`` is a Python bool (static) or a wave-uniform DSL bool"""
                if const_expr(isinstance(first, bool)):
                    wait_barrier(c_first if first else c_other)
                elif const_expr(c_first == c_other):
                    wait_barrier(c_first)
                else:
                    if first:
                        wait_barrier(c_first)
                    else:
                        wait_barrier(c_other)

            # vmcnt allowances (loads AND stores retire in issue order). Per step the loads are
            # b0 (NB) in phase 1, b1 (NB) + gather (NG) in phase 3; the last K-step of a tile
            # also stores c00 (4) in phase 2, c01 (4) + h0 scales (2) in phase 3, c10 (4) in
            # phase 4, and the next tile's first step stores c11 (4) + h1 scales (2) in phase 1.
            ST_A, ST_B, ST_C = 4, 4 + 2, 4  # phase 2 / 3 / 4 stores of a last step
            ST_P = 4 + 2  # pending stores in phase 1 of a first step
            TOP0 = P_STEP + ST_A + ST_B + ST_C  # 24
            SEG0 = NG + ST_B + ST_C + NB + ST_P  # 22
            TOP1 = ST_B + ST_C + P_STEP + ST_P  # 26
            SEG1 = NG + NB  # 6
            TOP2 = P_STEP  # 10
            SEG2L = NG + NB + ST_A  # 10
            TOP0_F, SEG0_F, TOP1_F = P_STEP, NG + NB + ST_P, P_STEP + ST_P  # first tile (prologue order)
            for c in (TOP0, SEG0, TOP1, SEG1, TOP2, SEG2L, TOP0_F, SEG0_F, TOP1_F):
                assert c <= 63, c

            def _one_step(nt, nt_next, n_par, kb, first, b0f, b1f, sc, accs, pend):
                """flat step s = (nt, kb); B(s) in set (n_par+kb)%2; issues B(s+2), scales(s+3).
                ``pend`` (kb == 0 only): thunks of the previous tile's c11 epilogue for phase 1.
                Returns (b0n, b1n, sc_next, accs, hold) with hold = e8m0 of this step's groups
                (kb == 2 only)."""
                s_par = (n_par + kb) % 2
                bc0, bc1 = b_buf(s_par, 0), b_buf(s_par, 1)
                bn0, bn1 = b_buf(1 - s_par, 0), b_buf(1 - s_par, 1)
                kb2 = (kb + 2) % K_ITERS
                nt2 = nt if (kb + 2) < K_ITERS else nt_next
                kb1 = (kb + 1) % K_ITERS
                nt1 = nt if (kb + 1) < K_ITERS else nt_next
                sbC0, sbC1 = sc
                c00, c01, c10, c11 = accs
                a0f, a1f = aF[kb][0], aF[kb][1]
                zero = kb == 0
                last = kb == K_ITERS - 1
                hold = {}

                _scn = [None] * 4
                rd_scn = _read_bsc_thunks(nt1, kb1, _scn)
                b_off2 = _b_soff(nt2, kb2)

                # top: everything issued at step s-2 landed (B(s), scales(s+1))
                if kb == 0:
                    _wb2(first, TOP0_F, TOP0)
                elif kb == 1:
                    _wb2(first, TOP1_F, TOP1)
                else:
                    wait_barrier(TOP2)
                il = _g2s_thunks(b0_g2s, bc0, b_off2, NB) + rd_scn[:2]
                c00 = mfma.call(a0f, b0f, c00, [saA[0][kb]], sbC0, interleave=il, zero_acc=zero,
                                late=(pend if kb == 0 else None))
                c01 = mfma.call(a0f, b1f, c01, [saA[0][kb]], sbC1, interleave=rd_scn[2:], zero_acc=zero,
                                late=(_group_thunks(c00, 0, 0, nt, hold) if last else None))

                # SEG2: B(s+1) landed (issued at s-1)
                if kb == 0:
                    _wb2(first, SEG0_F, SEG0)
                elif kb == 1:
                    wait_barrier(SEG1)
                else:
                    wait_barrier(SEG2L)
                _b0n = [None] * NB
                _b1n = [None] * NB
                # the gather sets m0 too: it must follow the whole m0-relative DMA sequence
                il = _g2s_thunks(b1_g2s, bc1, b_off2, NB) + [lambda: _gather(nt_next, kb)]
                late = (_group_thunks(c01, 0, 1, nt, hold) + _scale_thunks(0, nt, hold)) if last else None
                c10 = mfma.call(a1f, b0f, c10, [saA[1][kb]], sbC0, interleave=il, zero_acc=zero, late=late)
                il = _s2r_thunks(b_s2r, bn0, _b0n, NB, True) + _s2r_thunks(b_s2r, bn1, _b1n, NB, True)
                late = _group_thunks(c10, 1, 0, nt, hold) if last else None
                c11 = mfma.call(a1f, b1f, c11, [saA[1][kb]], sbC1, interleave=il, zero_acc=zero, late=late)
                return _b0n, _b1n, (_scn[:2], _scn[2:]), (c00, c01, c10, c11), hold

            _R = fx.as_ir_value

            def _flat_b(frag):
                return [_R(t[0]) for t in frag] + [_R(t[1]) for t in frag]

            def _unflat_b(flat):
                return [[flat[i], flat[NB + i]] for i in range(NB)]

            def _flat_state(b0f, b1f, sc, c11, e10):
                return (
                    _flat_b(b0f) + _flat_b(b1f) + [_R(v) for v in sc[0]] + [_R(v) for v in sc[1]]
                    + [_R(v) for v in c11] + [_R(v) for v in e10]
                )

            def _unflat_state(st):
                o = 0
                b0f = _unflat_b(st[o : o + 2 * NB])
                o += 2 * NB
                b1f = _unflat_b(st[o : o + 2 * NB])
                o += 2 * NB
                sc = (list(st[o : o + 2]), list(st[o + 2 : o + 4]))
                o += 4
                c11 = list(st[o : o + N_ACCUMS])
                o += N_ACCUMS
                e10 = [fx.Int32(v) for v in st[o : o + 4]]
                return b0f, b1f, sc, c11, e10

            def _pending_thunks(c11_prev, e10_prev, nt_prev, mask):
                """previous tile's c11 groups + its h=1 scale dwords"""
                hold = {(0, ti, p): e10_prev[2 * ti + p] for ti in range(N_TILES_A) for p in range(NB // 2)}
                return _group_thunks(c11_prev, 1, 1, nt_prev, hold, mask) + _scale_thunks(1, nt_prev, hold, mask)

            def _n_tile(nt, nt_next, n_par, first, b0f, b1f, sc, pend):
                accs = tuple([None] * N_ACCUMS for _ in range(4))
                hold = {}
                for kb in range_constexpr(K_ITERS):
                    b0f, b1f, sc, accs, h = _one_step(nt, nt_next, n_par, kb, first, b0f, b1f, sc, accs, pend)
                    hold.update(h)
                e10 = [hold.get((0, ti, p), fx.Int32(0)) for ti in range(N_TILES_A) for p in range(NB // 2)]
                return b0f, b1f, sc, accs[3], e10

            zero_v4 = _arith.ConstantOp(
                mfma.res_ty, _ir.DenseElementsAttr.get_splat(mfma.res_ty, _ir.FloatAttr.get(_T.f32, 0.0))
            ).result
            init_state = _flat_state(
                b0f, b1f, (_sc0[:2], _sc0[2:]), [zero_v4] * N_ACCUMS, [fx.Int32(0)] * 4
            )
            for np_, state in range(0, NT // 2, init=init_state):
                b0f, b1f, sc, c11p, e10p = _unflat_state(state)
                np_i = fx.Int32(np_)
                first = np_i == fx.Int32(0)
                nt_e = np_i * fx.Int32(2)
                nt_o = nt_e + fx.Int32(1)
                nt_o_next = _min(nt_e + fx.Int32(2), fx.Int32(NT - 1))  # last pair: redundant reloads
                pend = _pending_thunks(c11p, e10p, nt_e - fx.Int32(1), np_i > fx.Int32(0))
                b0f, b1f, sc, c11e, e10e = _n_tile(nt_e, nt_o, 0, first, b0f, b1f, sc, pend)
                pend = _pending_thunks(c11e, e10e, nt_e, None)
                b0f, b1f, sc, c11o, e10o = _n_tile(nt_o, nt_o_next, 1, False, b0f, b1f, sc, pend)
                state = yield _flat_state(b0f, b1f, sc, c11o, e10o)

            # the last tile's c11 epilogue has no next step to hide in
            b0f, b1f, sc, c11p, e10p = _unflat_state(state)
            for t in _pending_thunks(c11p, e10p, fx.Int32(NT - 1), None):
                t()

            # never retire with LDS DMA in flight (the CU reuses the LDS)
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            grid_size,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


def gemm2_grid(num_m_blocks: int, n_split: int) -> int:
    """blocks to launch: every (m-tile, chunk) of the allocation, rounded to the 8 XCDs
    (the kernel drops the fully padded tail tiles at runtime)."""
    return (num_m_blocks * n_split + _NUM_XCDS - 1) // _NUM_XCDS * _NUM_XCDS
