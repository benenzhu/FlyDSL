# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MiniMax-M3 prefill MoE stage 1 (a4w4): persistent version of ``gemm1.py``.

Same 2x2-wave / 8-LDS-buffer / depth-2 K pipeline as gemm1.py, but one CTA per
CU walks through several (m-tile, n-tile) blocks and the K pipeline never
drains between them: the two tail K-steps of block t already issue the first
two K-steps of block t+1 (the expert switch, i.e. the HBM fetch of the next
gate/up slab, happens while block t's tail MFMAs and epilogue run), and the
epilogue of block t overlaps the DMA of block t+1. In gemm1.py every block
pays that latency in its prologue.

Block order: the host ``tile_map`` (expert by expert, n-slab-major inside an
expert) split in 8 contiguous chunks, one per XCD; CTA b (on XCD ``b % 8``)
takes entries ``b // 8, b // 8 + P8, ...`` of its chunk (``P8`` = CTAs per XCD).
The table carries the number of valid entries as its last element so every
CTA computes an exact trip count (no wasted blocks, no branches in the loop).

The next block's routing (table entry, expert, sorted_ids of its rows) is
fetched with *scalar* buffer loads (SMEM, counted by lgkmcnt) so they do not
disturb the hand-counted ``vmcnt`` schedule of the inline-asm LDS DMA. The
epilogue's ``S`` buffer stores of block t are younger than the loads for
block t+1 (issued at the tail), so the first waits of block t+1 allow ``S``
extra outstanding ops; from K-step 2 on the counts are the steady-state ones.

Layouts, epilogue, numerics: identical to gemm1.py (bit-exact same output).

Result (2026-09-06, us per call, graph of 2 inputs): 4096 tok BM128 223.9,
8192 BM128 326.7, 16384 BM256 534.6 -- no gain over gemm1.py (224 / 320 / 523):
the block prologue latency was not what the non-persistent kernel loses; the
time goes to the LDS-DMA / vector-memory path inside the K loop (see NIGHTLOG).
Kept as the validated cross-block pipeline template.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops  # the copy shipped in the vLLM image

from m3_a4w4_moe.gemm1 import (
    _FP4_PACK,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_LDS_BYTES,
    _SCALE_QUARTER_BYTES,
    _SCALE_SLOT_BYTES,
    _SCALE_SLOTS,
    G2SLoaderAsm,
    Mfma16x16x128Fp4,
    S2RLoaderFp4,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _asm_void,
    _Buf,
    _cvt_pk_fp4,
    _divmod_nonneg,
    _e8m0_roundup_fp4,
    _as_f32,
    _f32,
    _flat_frag,
    _fmax,
    _g2s_thunks,
    _intrin_f32,
    _min,
    _permlane16_swap,
    _riffle,
    _s2r_thunks,
    _quant_prep_fp4,
    _swiglu_oai,
    _swizzled_col,
    _unflat_frag,
    _uniform_i32,
    wait_barrier,
)

_NUM_XCDS = 8
# M3_PERSIST_DBG=1: every CTA stores a progress marker (block index * 100 + point) to the
# ``dbg_ptr`` buffer; point it at pinned host memory and the markers stay readable while
# the GPU is stuck (this is how the SCC-clobber hang was located).
_DBG_PROGRESS = __import__("os").environ.get("M3_PERSIST_DBG", "0") == "1"


def _max(a, b):
    return fx.arith.select(a > b, a, b)


class _ScaleGatherCtx(ScaleGatherMoE):
    """ScaleGatherMoE with the per-block part of the context (the 32-row group
    ``G`` of this wave's first scale block) passed explicitly, so one object
    serves the current and the next block."""

    def set_wave_const(self):
        wid = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(self.wave_id)))
        self._wave_base_s = fx.as_ir_value(self._lds_base + wid * fx.Int32(_SCALE_QUARTER_BYTES))
        self._is_a = wid < fx.Int32(2)
        self._q = wid % fx.Int32(2)
        self._HS = _uniform_i32(
            fx.arith.select(self._is_a, fx.Int32(self._a_half_groups), fx.Int32(self._b_half_groups))
        )
        self._rsrc = fx.arith.select(self._is_a, self.a_rsrc, self.b_rsrc)
        self._soff0 = _uniform_i32(fx.Int32(0))

    def ctx(self, a_base_row, b_base_row):
        """Uniform i32: first 32-row group this wave gathers for the block."""
        g_a = a_base_row // fx.Int32(32) + self._q * fx.Int32(self._a_wave_groups)
        g_b = b_base_row // fx.Int32(32) + self._q * fx.Int32(self._b_wave_groups)
        return _uniform_i32(fx.arith.select(self._is_a, g_a, g_b))

    def gather_g(self, kstep, slot, G):
        grp = fx.Int32(G) + (self._blk // 2) * fx.Int32(self._HS) + (self._blk % 2)
        i32_off = grp * fx.Int32(self.row_i32) + fx.Int32(kstep) * fx.Int32(64) + self._in16 * fx.Int32(4)
        voff = fx.as_ir_value(i32_off * fx.Int32(4))
        addr = fx.Int32(self._wave_base_s) + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        _asm_void([fx.as_ir_value(addr), voff, self._rsrc, self._soff0], asm, "s,v,s,s", "~{m0}")


def compile_moe_gemm1_persist(*, H: int, I: int, E: int, BLOCK_M: int = 256, n_cta: int = 256):
    """Persistent grouped fp4 gemm1. ``BLOCK_M`` = moe_sorting block size (128 or
    256); ``n_cta`` CTAs (multiple of 8; 256 = one per CU on MI355X, the LDS
    footprint allows exactly one CTA per CU anyway)."""
    K = H
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    BLOCK_N = 256
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16
    N_TILES_B = LDS_BLOCK_N // 2 // 16
    N_BLOCKS_N = I // LDS_BLOCK_N
    NA, NB = N_TILES_A, N_TILES_B

    assert BLOCK_M in (128, 256)
    assert K % BLOCK_K == 0 and I % LDS_BLOCK_N == 0 and (2 * I) % 256 == 0
    assert n_cta % _NUM_XCDS == 0
    P8 = n_cta // _NUM_XCDS
    K_ITERS = K // BLOCK_K
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert K_ITERS >= 4 and (K_ITERS - 4) % UNROLL == 0, K_ITERS
    # block boundary must keep the LDS ping-pong parity and the scale slot ring aligned
    assert K_ITERS % 2 == 0 and K_ITERS % _SCALE_SLOTS == 0
    N_ACCUMS = N_TILES_A * N_TILES_B
    K_BYTES = K // 2

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES
    A_BUFS = 4 * a_lds_size
    LDS_TILES_BYTES = A_BUFS + 4 * b_lds_size

    A_WAVE_GROUPS = N_TILES_A // 2
    A_HALF_GROUPS = LDS_BLOCK_M // 32
    B_WAVE_GROUPS = N_TILES_B // 2
    B_HALF_GROUPS = I // 32
    I_BYTES = I // 2
    SCALE_COLS_OUT = I // 32
    OUT_SC_BLOCKS_PER_ROW32 = SCALE_COLS_OUT // 8

    # vmcnt bookkeeping (loads complete in issue order). Per K-step, issue order:
    # a0 (NA), b0 (NB), [SEG2 wait], b1 (NB), scale gather (1), a1 (NA).
    P_STEP = 2 * NA + 2 * NB + 1
    SEG2 = 2 * NA + NB + 1
    # block top: outstanding = [g0][step0: a0 b0 b1 g1 a1][step1: a0 b0 b1 g2 a1](+S stores)
    TOP_A = 3 * NA + 4 * NB + 2  # a_cur0 (+g0) landed
    TOP_B = 3 * NA + 2 * NB + 2  # b_cur0, b_cur1 landed
    # epilogue buffer stores per lane: 2 quadrant pairs x (NB/2 col groups) x (NA dwords + NA/2 scale i16)
    S_STORES = 2 * (N_TILES_B // 2) * (N_TILES_A + N_TILES_A // 2)
    assert TOP_A + S_STORES <= 63, (TOP_A, S_STORES)

    A_K_STEP = BLOCK_K_BYTES
    B_K_STEP = 2 * 1024
    N_CTX = 4 + 2 * NA  # tile_i, tile_j, m_base, b_row0, gather offsets (2 halves)

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, _SCALE_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm1_persist(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
        dbg_ptr: fx.Pointer,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        if const_expr(_DBG_PROGRESS):
            from m3_a4w4_moe.lowlevel import ptr_buffer_resource as _pbr

            dbg_rsrc = _pbr(dbg_ptr, n_cta * 4)

        def _mark(code, it_v=None):
            if const_expr(_DBG_PROGRESS):
                v = fx.Int32(code) if it_v is None else it_v * fx.Int32(100) + fx.Int32(code)
                _buffer_ops.buffer_store(v, dbg_rsrc, fx.Int32(fx.block_idx.x), cache_modifier=4)
        _base_ptr = lds.all_lds.ptr

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        b_cur0 = _Buf(_base_ptr, A_BUFS + 0 * b_lds_size)
        b_cur1 = _Buf(_base_ptr, A_BUFS + 1 * b_lds_size)
        b_next0 = _Buf(_base_ptr, A_BUFS + 2 * b_lds_size)
        b_next1 = _Buf(_base_ptr, A_BUFS + 3 * b_lds_size)
        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        wave_u = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(wave_id)))
        lane_g8 = lane_id // 8

        ids_rsrc = _buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * 4)
        )
        eid_rsrc = _buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4)
        tm_rsrc = _buffer_ops.create_buffer_resource(tile_map_t, max_size=False, num_records_bytes=(grid_size + 1) * 4)

        def _sload(rsrc, elem_off):
            return fx.Int32(_buffer_ops.buffer_load(rsrc, elem_off, vec_width=1, dtype=fx.Int32, is_scalar=True))

        def _sload4(rsrc, elem_off):
            v = Vec(_buffer_ops.buffer_load(rsrc, elem_off, vec_width=4, dtype=fx.Int32, is_scalar=True))
            return [fx.Int32(v[i]) for i in range_constexpr(4)]

        # ---- this CTA's share of the block table ----
        n_valid = _sload(tm_rsrc, grid_size)
        intra, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        chunk = grid_size // _NUM_XCDS
        chunk_base = xcd * chunk
        v_in_chunk = _min(_max(n_valid - chunk_base, fx.Int32(0)), chunk)
        rem = v_in_chunk - intra
        n_iters = fx.arith.select(rem > fx.Int32(0), (rem + fx.Int32(P8 - 1)) // fx.Int32(P8), fx.Int32(0))
        base_entry = chunk_base + intra

        # ---- next-block routing, in three stages so no load is waited on right away ----
        def _ctx_stage1(it_i32):
            """table entry (invalid / past the end -> block 0, harmless loads)"""
            e_idx = _min(base_entry + it_i32 * fx.Int32(P8), grid_size - fx.Int32(1))  # never read past the table
            entry = _sload(tm_rsrc, e_idx)
            return fx.arith.select(entry >= fx.Int32(0), entry, fx.Int32(0))

        def _ctx_stage2(entry):
            """expert + the 8 consecutive sorted_ids each (wave, round) needs, as SGPRs"""
            tile_i = entry >> 3
            expert = _sload(eid_rsrc, tile_i)
            m_base = tile_i * BLOCK_M
            sids = []
            for half in range_constexpr(2):
                for rnd in range_constexpr(N_TILES_A):
                    base_row = m_base + fx.Int32(half * LDS_BLOCK_M + rnd * 32) + wave_u * fx.Int32(8)
                    sids.append(_sload4(ids_rsrc, base_row) + _sload4(ids_rsrc, base_row + fx.Int32(4)))
            return tile_i, expert, sids

        def _ctx_stage3(entry, tile_i, expert, sids):
            tile_j = entry & 7
            m_base = tile_i * BLOCK_M
            b_row0 = expert * (2 * I) + tile_j * LDS_BLOCK_N
            offs = []
            n = 0
            for half in range_constexpr(2):
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_g8 + wave_id * 8 + rnd * 32
                    col = (lane_id % 8) * 16
                    vals = sids[n]
                    n += 1
                    sid = vals[0]
                    for i in range_constexpr(1, 8):
                        sid = fx.arith.select(lane_g8 == fx.Int32(i), vals[i], sid)
                    tok = sid & fx.Int32(0x00FFFFFF)  # padding rows: token == n_tokens -> OOB -> zeros
                    offs.append(tok * fx.Int32(K_BYTES) + _swizzled_col(row, col))
            return [tile_i, tile_j, m_base, b_row0] + offs

        def _ctx_all(it_i32):
            entry = _ctx_stage1(it_i32)
            tile_i, expert, sids = _ctx_stage2(entry)
            return _ctx_stage3(entry, tile_i, expert, sids)

        # ---- B (static per-lane part of the preshuffled offsets) ----
        def _b_offsets():
            offs = []
            for rnd in range_constexpr(N_TILES_B):
                row = lane_id % 8 + wave_id * 8 + rnd * 32
                col = (lane_id // 8) * 16
                offs.append(
                    (row // 16) * (K_BYTES * 16)
                    + (row % 16) * 16
                    + (col // 64) * 1024
                    + ((col % 64) // 16) * 256
                    + (col % 16)
                )
            return offs

        gl_off_b = _b_offsets()
        a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=n_tokens * K_BYTES)
        b_rsrc = _buffer_ops.create_buffer_resource(W13, max_size=False, num_records_bytes=E * (2 * I) * K_BYTES)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B)
        mfma = Mfma16x16x128Fp4(N_TILES_A, N_TILES_B)

        _scale_base_ptr = lds.scale_lds.ptr
        sg = _ScaleGatherCtx(
            A_scale,
            W13_scale,
            K,
            lane_id,
            wave_id,
            _scale_base_ptr,
            a_scale_bytes,
            E * 2 * I * (K // 32),
            A_WAVE_GROUPS,
            A_HALF_GROUPS,
            B_WAVE_GROUPS,
            B_HALF_GROUPS,
        )
        sg.set_wave_const()
        a_scale_ld = ScaleLoaderLDS(N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION)
        b_scale_ld = ScaleLoaderLDS(N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION)

        out_rsrc = _buffer_ops.create_buffer_resource(
            OUT_Q, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * I_BYTES)
        )
        osc_rsrc = _buffer_ops.create_buffer_resource(
            OUT_scale, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * SCALE_COLS_OUT)
        )

        def _loaders(ctx):
            a0_g2s = G2SLoaderAsm(a_rsrc, ctx[4 : 4 + NA], N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, ctx[4 + NA : 4 + 2 * NA], N_TILES_A, wave_id)
            a0_g2s.set_wave_base(_base_ptr)
            a1_g2s.set_wave_base(_base_ptr)
            B0 = ctx[3] * K_BYTES
            B1 = B0 + I * K_BYTES
            return a0_g2s, a1_g2s, B0, B1

        def _read_scale_thunks_slot(slot, holder):
            def _r(dst, ld, half):
                holder[dst] = ld.read_half(slot, half)

            return [
                lambda: _r(0, a_scale_ld, 0),
                lambda: _r(1, a_scale_ld, 1),
                lambda: _r(2, b_scale_ld, 0),
                lambda: _r(3, b_scale_ld, 1),
            ]

        def _one_step(a0f, b0f, b1f_in, sc, accs, bufs, ld, wb_main, wb_seg2, zero_acc=False, read_next=True):
            """One K-step: MFMAs on (a0f, b0f, b1f) while loading K-step ``ld``
            (the loaders / offsets / gather / scale slot chosen by the caller)."""
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            a0_g2s, a1_g2s, a_off, b0_off, b1_off, gather_thunk, rd_slot = ld

            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            _scn = [None, None, None, None]
            _rd_scn = _read_scale_thunks_slot(rd_slot, _scn) if read_next else []

            wb_main()
            il = (
                _riffle(_g2s_thunks(a0_g2s, ac0, a_off, N_TILES_A), _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False))
                + _rd_scn[:2]
            )
            c00f = mfma.call(a0f, b0f, c00f, saR0, sbC0, interleave=il, zero_acc=zero_acc)
            il = _riffle(_g2s_thunks(b_g2s, bc0, b0_off, N_TILES_B), _rd_scn[2:])
            c01f = mfma.call(a0f, b1f_in, c01f, saR0, sbC1, interleave=il, zero_acc=zero_acc)
            a1f = _a1

            wb_seg2()
            nxt_a = _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False) if read_next else []
            il = _riffle(_g2s_thunks(b_g2s, bc1, b1_off, N_TILES_B), nxt_a) + [gather_thunk]
            c10f = mfma.call(a1f, b0f, c10f, saR1, sbC0, interleave=il, zero_acc=zero_acc)
            nxt_b = (
                _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True) + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True)
                if read_next
                else []
            )
            il = _riffle(_g2s_thunks(a1_g2s, ac1, a_off, N_TILES_A), nxt_b)
            c11f = mfma.call(a1f, b1f_in, c11f, saR1, sbC1, interleave=il, zero_acc=zero_acc)

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return _a0n, _b0n, _b1n, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        def _swap_bufs(bufs):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

        n_a = 2 * N_TILES_A
        n_b = 2 * N_TILES_B
        n_ga = N_TILES_A // _FP4_PACK
        n_gb = N_TILES_B // _FP4_PACK
        n_sc = 2 * n_ga + 2 * n_gb
        _R = fx.as_ir_value

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return [_R(v) for v in saR0] + [_R(v) for v in saR1] + [_R(v) for v in sbC0] + [_R(v) for v in sbC1]

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga])
            o += n_ga
            saR1 = list(flat[o : o + n_ga])
            o += n_ga
            sbC0 = list(flat[o : o + n_gb])
            o += n_gb
            sbC1 = list(flat[o : o + n_gb])
            return (saR0, saR1, sbC0, sbC1)

        def _flat_all(a0f, b0f, b1f, sc, accs):
            return (
                _flat_frag(a0f)
                + _flat_frag(b0f)
                + _flat_frag(b1f)
                + _flat_sc(sc)
                + [_R(x) for x in accs[0]]
                + [_R(x) for x in accs[1]]
                + [_R(x) for x in accs[2]]
                + [_R(x) for x in accs[3]]
            )

        def _unflat_all(state):
            off = 0
            a0f = _unflat_frag(state[off : off + n_a], N_TILES_A)
            off += n_a
            b0f = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            b1f = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            accs = []
            for q in range_constexpr(4):
                accs.append(list(state[off : off + N_ACCUMS]))
                off += N_ACCUMS
            return a0f, b0f, b1f, sc, tuple(accs)

        g = lane_id // 16
        r16 = lane_id % 16

        def _epilogue(c_gate, c_up, base_row, col_base, colgrp_base):
            """One quadrant pair: rows base_row + ti*16 + r16, this wave's 64 gate cols."""
            for p in range_constexpr(N_TILES_B // 2):
                colgrp = colgrp_base + p
                sc_in_block = (colgrp % 4) * 64 + r16 * 4 + ((colgrp % 8) // 4) * 2
                e8m0_of_ti = []
                for ti in range_constexpr(N_TILES_A):
                    gv = Vec(c_gate[mfma.idx(ti, 2 * p)])
                    gw = Vec(c_gate[mfma.idx(ti, 2 * p + 1)])
                    uv = Vec(c_up[mfma.idx(ti, 2 * p)])
                    uw = Vec(c_up[mfma.idx(ti, 2 * p + 1)])
                    h = [_swiglu_oai(_f32(gv[v]), _f32(uv[v])) for v in range_constexpr(4)] + [
                        _swiglu_oai(_f32(gw[v]), _f32(uw[v])) for v in range(4)
                    ]
                    h, amax = _quant_prep_fp4(h)
                    amax = _fmax(amax, amax.shuffle_xor(16, 64))
                    amax = _fmax(amax, amax.shuffle_xor(32, 64))
                    e8m0 = _e8m0_roundup_fp4(amax)
                    e8m0_of_ti.append(e8m0)
                    scale_f = _as_f32(e8m0 << 23)
                    pa = _cvt_pk_fp4(fx.Int32(0), h[0], h[1], scale_f, 0)
                    pa = _cvt_pk_fp4(pa, h[2], h[3], scale_f, 1)
                    pb = _cvt_pk_fp4(fx.Int32(0), h[4], h[5], scale_f, 0)
                    pb = _cvt_pk_fp4(pb, h[6], h[7], scale_f, 1)
                    pa, pb = _permlane16_swap(pa, pb)
                    dword = pa | (pb << 16)
                    row = base_row + ti * 16 + r16
                    col = col_base + (2 * p + (g % 2)) * 16 + (g // 2) * 8
                    _buffer_ops.buffer_store(dword, out_rsrc, row * I_BYTES + col // 2, offset_is_bytes=True)
                for tp in range_constexpr(N_TILES_A // 2):
                    row32 = (base_row // 32) + tp
                    blk = row32 * OUT_SC_BLOCKS_PER_ROW32 + colgrp // 8
                    pair = e8m0_of_ti[2 * tp] | (e8m0_of_ti[2 * tp + 1] << 8)
                    pair16 = _arith.TruncIOp(_T.i16, fx.as_ir_value(pair)).result
                    _buffer_ops.buffer_store(pair16, osc_rsrc, blk * 256 + sc_in_block, offset_is_bytes=True)

        # ---- block 0: routing + pipeline prologue in the tail-step issue order:
        #      g0, [a0 b0 b1 g1 a1] (K-step 0), [a0 b0 b1 g2 a1] (K-step 1) ----
        ctx0 = _ctx_all(fx.Int32(0))
        a0_g2s0, a1_g2s0, B0_0, B1_0 = _loaders(ctx0)
        G0 = sg.ctx(ctx0[2], ctx0[3])
        sg.gather_g(0, 0, G0)
        a0_g2s0.load(a_cur0, fx.Int32(0))
        b_g2s.load(b_cur0, B0_0)
        b_g2s.load(b_cur1, B1_0)
        sg.gather_g(1, 1, G0)
        a1_g2s0.load(a_cur1, fx.Int32(0))
        a0_g2s0.load(a_next0, fx.Int32(A_K_STEP))
        b_g2s.load(b_next0, B0_0 + B_K_STEP)
        b_g2s.load(b_next1, B1_0 + B_K_STEP)
        sg.gather_g(2, 2, G0)
        a1_g2s0.load(a_next1, fx.Int32(A_K_STEP))

        _mark(1)
        ctx0_flat = [_R(v) for v in ctx0]
        for it, ostate in range(0, n_iters, init=ctx0_flat):
            ctx = [fx.Int32(v) for v in ostate]
            it_i32 = fx.Int32(it)
            _mark(2, it_i32)
            first = it_i32 == fx.Int32(0)
            tile_j = ctx[1]
            m_base = ctx[2]
            b_row0 = ctx[3]
            a0_g2s, a1_g2s, B0_gl, B1_gl = _loaders(ctx)
            G_cur = sg.ctx(m_base, b_row0)

            def _wb2(c_first, c_other):
                if const_expr(c_first == c_other):
                    wait_barrier(c_first)
                else:
                    if first:
                        wait_barrier(c_first)
                    else:
                        wait_barrier(c_other)

            # ---- block top: K-steps 0/1 are in LDS (issued by the previous tail) ----
            _wb2(TOP_A, TOP_A + S_STORES)
            a0_frag = a_s2r.load(a_cur0)
            _wb2(TOP_B, TOP_B + S_STORES)
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            sc0_saR0, sc0_saR1 = a_scale_ld.read(fx.Int32(0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(fx.Int32(0))
            sc = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)
            _mark(3, it_i32)

            # next block's routing, stage 1 (SMEM; used one K-step later)
            entry_n = _ctx_stage1(it_i32 + fx.Int32(1))

            def _ld_cur(kc):
                """loads for K-step kc+2 of this block (kc static)"""
                k2 = kc + 2
                return (
                    a0_g2s,
                    a1_g2s,
                    fx.Int32(k2 * A_K_STEP),
                    B0_gl + k2 * B_K_STEP,
                    B1_gl + k2 * B_K_STEP,
                    (lambda: sg.gather_g(kc + 3, (kc + 3) % _SCALE_SLOTS, G_cur)),
                    fx.Int32((kc + 1) % _SCALE_SLOTS),
                )

            _accs0 = ([None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS)
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                a0_frag, b0_frag, b1_frag, sc, _accs0, bufs0, _ld_cur(0),
                lambda: _wb2(P_STEP, P_STEP + S_STORES), lambda: _wb2(SEG2, SEG2 + S_STORES), zero_acc=True,
            )
            _mark(4, it_i32)
            # next block's routing, stage 2 (depends on the entry)
            tile_i_n, expert_n, sids_n = _ctx_stage2(entry_n)
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                a0f, b0f, b1f, sc, accs, _swap_bufs(bufs0), _ld_cur(1),
                lambda: _wb2(P_STEP, P_STEP + S_STORES), lambda: _wb2(SEG2, SEG2 + S_STORES),
            )
            # next block's routing, stage 3: per-lane gather offsets + scale context
            ctx_n = _ctx_stage3(entry_n, tile_i_n, expert_n, sids_n)
            a0_g2s_n, a1_g2s_n, B0_n, B1_n = _loaders(ctx_n)
            G_next = sg.ctx(ctx_n[2], ctx_n[3])

            _mark(5, it_i32)
            init_state = _flat_all(a0f, b0f, b1f, sc, accs)
            for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):
                a0f, b0f, b1f, sc, accs = _unflat_all(state)
                bufs = bufs0
                for u in range_constexpr(UNROLL):
                    kc_i = fx.Int32(kk + u)
                    k2 = kc_i + fx.Int32(2)
                    gk = kc_i + fx.Int32(3)
                    use_next = gk >= fx.Int32(K_ITERS)  # last 3 gathers of a block fetch the next block's k = 0, 1, 2
                    k_g = fx.arith.select(use_next, gk - fx.Int32(K_ITERS), gk)
                    G_g = _uniform_i32(fx.arith.select(use_next, fx.Int32(G_next), fx.Int32(G_cur)))
                    ld = (
                        a0_g2s,
                        a1_g2s,
                        k2 * fx.Int32(A_K_STEP),
                        B0_gl + k2 * fx.Int32(B_K_STEP),
                        B1_gl + k2 * fx.Int32(B_K_STEP),
                        (lambda k_g=k_g, G_g=G_g: sg.gather_g(k_g, k_g % fx.Int32(_SCALE_SLOTS), G_g)),
                        (kc_i + fx.Int32(1)) % fx.Int32(_SCALE_SLOTS),
                    )
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        a0f, b0f, b1f, sc, accs, bufs, ld,
                        lambda: wait_barrier(P_STEP), lambda: wait_barrier(SEG2),
                    )
                    _mark(fx.Int32(50) + kc_i, it_i32)
                state = yield _flat_all(a0f, b0f, b1f, sc, accs)

            a0f, b0f, b1f, sc, accs = _unflat_all(state)
            _mark(6, it_i32)

            # ---- tail K-steps K_ITERS-2 / K_ITERS-1: load the NEXT block's K-steps 0 / 1 ----
            def _ld_next(k):
                return (
                    a0_g2s_n,
                    a1_g2s_n,
                    fx.Int32(k * A_K_STEP),
                    B0_n + k * B_K_STEP,
                    B1_n + k * B_K_STEP,
                    (lambda: sg.gather_g(k + 1, k + 1, G_next)),
                    fx.Int32((K_ITERS - 1 + k) % _SCALE_SLOTS),
                )

            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                a0f, b0f, b1f, sc, accs, bufs0, _ld_next(0),
                lambda: wait_barrier(P_STEP), lambda: wait_barrier(SEG2),
            )
            _, _, _, _, accs, _ = _one_step(
                a0f, b0f, b1f, sc, accs, bufs, _ld_next(1),
                lambda: wait_barrier(P_STEP), lambda: wait_barrier(SEG2), read_next=False,
            )
            c00_frag, c01_frag, c10_frag, c11_frag = accs
            _mark(7, it_i32)

            # ---- epilogue of this block (its stores overlap the next block's DMA) ----
            col_base = tile_j * LDS_BLOCK_N + wave_j * (N_TILES_B * 16)
            colgrp_base = col_base // 32
            row_r0 = m_base + wave_i * (N_TILES_A * 16)
            row_r1 = row_r0 + LDS_BLOCK_M
            _epilogue(c00_frag, c01_frag, row_r0, col_base, colgrp_base)
            _epilogue(c10_frag, c11_frag, row_r1, col_base, colgrp_base)

            _mark(8, it_i32)
            ostate = yield [_R(v) for v in ctx_n]

        # never leave LDS DMA in flight when the CTA retires (the LDS is reused)
        wait_barrier(0)
        _mark(9)

    @flyc.jit
    def launch_gemm1_persist(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
        stream: fx.Stream,
        dbg_ptr: fx.Pointer,
    ):
        kernel_gemm1_persist(
            A,
            W13,
            OUT_Q,
            A_scale,
            W13_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            tile_map_t,
            grid_size,
            dbg_ptr,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(n_cta, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1_persist
