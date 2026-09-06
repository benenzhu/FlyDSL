"""MiniMax-M3 prefill MoE down-projection: persistent-CTA variant of ``gemm2.py``.

Same tiles, pipeline, epilogue and output modes as ``gemm2.py`` (read that first).
What changes: one CTA per CU walks a list of work items (m-tile, n-chunk) instead of
one item per CTA. ``gemm2.py`` pays ~20k cycles per CTA before its first MFMA (kernel
args -> num_valid -> expert ids, then 33 LDS-DMA issues that throttle after the first
dozen, then the row-id loads): with one CTA per CU that is ~18% of the CTA's life as
dead MFMA time. Here the W2 pipeline simply continues across the item boundary (the
last n-tile of item j prefetches n-tile 0 of item j+1 like any other next tile), and
everything the next item needs besides W2 streams in during the current item's last
n-tile and the new item's first one. The constraint is the per-wave VMEM issue credit
(~16 instructions in flight; a step already issues 10 loads + 8 stores): the first
version reloaded A straight into AGPRs (24 dwordx4 per lane, both wave_j halves fetch
the same rows) plus 6 scale and 20 row-id loads inside the last tile, and the ATT
showed the wave stalling ~1-3k cycles behind each of them -- the boundary cost exactly
the prologue it had removed. Now 21 VMEM instructions per boundary, spread out:

* A K-slices 0 and 1 go through a 16 KB LDS buffer by LDS-DMA (4 instructions per
  slice, no redundancy): slice 0 issued at the top of the last tile's step 0 and read
  into the (dead) old slice-0 AGPRs at the top of step 2; slice 1 issued in step 2's
  phase 3 (after the reads freed the buffer) and read at the top of the new item's
  step 1. Each read sits behind a wait_barrier whose vmcnt target is younger than the
  DMA, so in-order retirement covers it and every wave's part has landed.
* K-slice 2 is loaded straight into its (dead after step 2) AGPRs: 3 fragments after
  step 2, 3 in the new item's step 0 phase 1, 2 in its phase 3.
* the A scales (3 KB) by one LDS-DMA per wave with slice 0; read at slice 0's read
  point (K-slices 0, 1) and at the top of the new item's step 0 (slice 2).
* the row ids: each lane's flush rows are made consecutive (lane group g stores rows
  g*N_ST .. g*N_ST+N_ST-1) so they come as dwordx4 (2 bf16 / 1 fp8 per row half) plus
  the routing weights (bf16) or the scale-row ids (fp8): 8 / 4 instructions in the
  item tail, tied to the registers of the derived offsets they replace, pinned behind
  the first wait of the new item's third K-step.
* the expert id and the rotation gate of item j+2 are scalar loads issued at the start
  of item j and consumed at once (a scalar load left in flight makes LLVM turn every
  LDS wait into lgkmcnt(0)).

Item 0 takes the same tile-0 code: its prologue leaves the item in the same state a
boundary does (slice 1 and the scales in LDS, slice 2 fragments 3..7 not yet loaded).

The last pair of n-tiles is peeled out of the pair loop so that the reloads are
static code; the first n-tile after a boundary and the last n-tile of an item use
their own vmcnt allowances, derived by the same issue-order simulation as the steady
state (``_derive_vmcnt``). The output of every item is flushed by the item's own tail
(the previous kernel deferred the last row half into the next tile; across an item
boundary that would need a second set of row offsets, which the bf16 mode has no
VGPRs for).

Work list: items are ordered m-tile-major (n-chunk fastest) like the non-persistent
kernel; XCD x owns items [x*per_xcd, (x+1)*per_xcd) and its 32 CTAs take them
round-robin (item = base + slot + 32*j), so the 32 items in flight on an XCD are
consecutive m-tiles of the same expert, which is what the rotated n-tile sweep and
the L2 sharing rely on.

Launch: ``launch_gemm2_persist(A, W2, OUT, A_scale, W2_scale, OUT_scale, sorted_ids,
sorted_expert_ids, sorted_weights, num_valid_ids, n_tokens, num_m_blocks, n_ctas,
stream)`` -- the argument list of ``gemm2.py`` with the grid size replaced by the
number of persistent CTAs (256 = one per CU; must be a multiple of 8).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops

from m3_a4w4_moe.gemm1 import (
    G2SLoaderAsm,
    S2RLoaderFp4,
    _Buf,
    _N_WAVES,
    _as_f32,
    _asm_void,
    _bits,
    _divmod_nonneg,
    _g2s_thunks,
    _min,
    _permlane16_swap,
    _s2r_thunks,
    _swizzled_col,
    _uniform_i32,
    wait_barrier,
)
from m3_a4w4_moe.gemm2 import (
    _BScaleGather,
    _MfmaAgprAB,
    _NUM_XCDS,
    _bf16x2,
    _cvt_pk_fp8,
    _e8m0_fp8,
    _fabs,
    _lds_load_i32,
    _lds_load_vec,
    _lds_store_i16,
    _lds_store_vec,
    _maxf_nn,
    _pin_vec4,
    _undef_i32,
    _v2i32,
    _wait_pin,
    _xlane_max4_pair,
)


def _ld_dword(rsrc, voff_bytes, soff, old=None):
    """buffer_load_dword tracked by our own vmcnt accounting (LLVM sees no load). With
    ``old`` the destination is tied to the value it replaces: same register, ordered
    after that value's readers. Results go through ``_wait_pin`` before use."""
    if old is None:
        return fx.Int32(
            _llvm.inline_asm(
                _T.i32,
                [fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
                "buffer_load_dword $0, $1, $2, $3 offen",
                "=v,v,s,s",
                has_side_effects=True,
            )
        )
    return fx.Int32(
        _llvm.inline_asm(
            _T.i32,
            [fx.as_ir_value(old), fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
            "buffer_load_dword $0, $2, $3, $4 offen",
            "=v,0,v,s,s",
            has_side_effects=True,
        )
    )


def _pin_sgprs(vals):
    """Consume wave-uniform values right here (tied SGPR operands of a no-op asm). The
    scalar loads behind them are otherwise sunk to their first real use in the *next*
    item and stay in flight across the loop back-edge, and with a scalar load pending
    LLVM turns every LDS wait in the loop into lgkmcnt(0) (scalar loads can return out
    of order), which drains the interleaved ds_reads at each of them."""
    n = len(vals)
    ty = _ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * n) + ")>")
    cons = ",".join(["=s"] * n + [str(i) for i in range(n)])
    res = _llvm.inline_asm(ty, [fx.as_ir_value(v) for v in vals], "; pin sgprs", cons, has_side_effects=True)
    return [fx.Int32(_llvm.extractvalue(_T.i32, res, [i])) for i in range(n)]


def _ld_a_frag(rsrc, voff_bytes, soff, old=None):
    """buffer_load_dwordx4 straight into AGPRs (the MFMA operand class); ``old`` ties the
    destination to the fragment being replaced."""
    ty = _ir.VectorType.get([4], _T.i32)
    if old is None:
        return _llvm.inline_asm(
            ty,
            [fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
            "buffer_load_dwordx4 $0, $1, $2, $3 offen",
            "=a,v,s,s",
            has_side_effects=True,
        )
    return _llvm.inline_asm(
        ty,
        [fx.as_ir_value(old), fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
        "buffer_load_dwordx4 $0, $2, $3, $4 offen",
        "=a,0,v,s,s",
        has_side_effects=True,
    )


def _ld_vec4(rsrc, voff_bytes, soff, old=None):
    """buffer_load_dwordx4 into VGPRs, our own vmcnt accounting; ``old`` ties the
    destination (a vector<4xi32>) to the value it replaces."""
    ty = _ir.VectorType.get([4], _T.i32)
    if old is None:
        return _llvm.inline_asm(
            ty,
            [fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
            "buffer_load_dwordx4 $0, $1, $2, $3 offen",
            "=v,v,s,s",
            has_side_effects=True,
        )
    return _llvm.inline_asm(
        ty,
        [fx.as_ir_value(old), fx.as_ir_value(voff_bytes), fx.as_ir_value(rsrc), _uniform_i32(soff)],
        "buffer_load_dwordx4 $0, $2, $3, $4 offen",
        "=v,0,v,s,s",
        has_side_effects=True,
    )


def _wait_pin_any(vals, vmcnt):
    """``_wait_pin`` for a mix of i32 and vector<4xi32> values (raw IR values)."""
    vals = [fx.as_ir_value(v) for v in vals]
    tys = [v.type for v in vals]
    ty = _ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    cons = ",".join(["=v"] * len(vals) + [str(i) for i in range(len(vals))])
    res = _llvm.inline_asm(ty, vals, f"s_waitcnt vmcnt({vmcnt})", cons, has_side_effects=True)
    return [_llvm.extractvalue(tys[i], res, [i]) for i in range(len(vals))]


def _vec4_elems(v):
    return [fx.Int32(_llvm.extractelement(fx.as_ir_value(v), fx.as_ir_value(fx.Int32(i)))) for i in range(4)]


def _make_vec4(vals):
    ty = _ir.VectorType.get([4], _T.i32)
    v = _llvm.mlir_undef(ty)
    for i, x in enumerate(vals):
        v = _llvm.insertelement(v, fx.as_ir_value(x), fx.as_ir_value(fx.Int32(i)))
    return v


def _derive_vmcnt(NB, NG, ST, NID, NA_DIR, NA_DMA, NT, K_ITERS=3):
    """vmcnt allowances from the per-wave VMEM issue order (loads and stores retire in
    order). Prologue: g(0) g(1) g(2) | A slice 0 direct (NA_DIR) | A slice 2 fragments
    0..2 | A slice 1 DMA (NA_DMA) | scales DMA (1) | B(0) B(1) B(2). Step s = (tile, kb):
    top wait | P1 b0(s+3) x NB | seg2 wait | P3 b1(s+3) x NB, g(s+3) x NG | P4 ST stores
    for kb 0 and 2 (kb 0 of an item's tile 0 flushes the previous item's last rows).
    Boundary (last tile L of an item, tile 0 of the next): (L,0) top: slice-0 DMA
    (NA_DMA) + scales DMA (1); (L,2) P3 before b1: slice-1 DMA; (L,2) after P4: slice-2
    fragments 0..2; (0,0) P1 before b0: fragments 3..5; P3 before b1: fragments 6..7;
    (0,0) after P4's stores: the item's row ids (NID). Consumers: slice 0 + scales read
    at (L,2) top, slice 1 at (0,1) top, slice 2 needed at (0,2), ids pinned at (0,2) top
    ('PIN'). Contexts: F = first
    tile of the CTA, N = first tile of a later item, L = last tile of an item, S =
    steady. 'WA' / 'WB0' = prologue waits after the A loads / after B(0). Allowances
    are capped at 63 (stricter is still correct)."""
    seq = []

    def g(s):
        return [("g", s, i) for i in range(NG)]

    def B(s):
        return [("b0", s, i) for i in range(NB)] + [("b1", s, i) for i in range(NB)]

    n_frag = NA_DIR // 3  # fragments per K-slice
    seq += g(0) + g(1) + g(2)
    seq += [("A0", 0, i) for i in range(n_frag)] + [("A2", 0, i) for i in range(3)]
    seq += [("ADMA1", 0, i) for i in range(NA_DMA)] + [("SDMA", 0, 0)]
    seq += B(0) + B(1) + B(2)
    seq += [("W", "WA", 0), ("W", "WB0", 0)]
    n_items = 3
    for j in range(n_items):
        for t in range(NT):
            for kb in range(K_ITERS):
                s = (j * NT + t) * K_ITERS + kb
                ctx = "F" if (j == 0 and t == 0) else "N" if t == 0 else "L" if t == NT - 1 else "S"
                seq.append(("W", ("top", ctx, kb), s))
                if t == 0 and kb == 2:
                    seq.append(("W", "PIN", j))
                if t == 0 and kb == 1:
                    seq.append(("W", "RD1", j))  # slice-1 read: needs ADMA1(j)
                if t == 0 and kb == 2:
                    seq.append(("W", "USE2", j))  # slice 2 complete: needs A2(j) all
                if t == NT - 1 and kb == 0:
                    seq += [("ADMA0", j + 1, i) for i in range(NA_DMA)] + [("SDMA", j + 1, 0)]
                if t == NT - 1 and kb == 2:
                    seq.append(("W", "RD0", j + 1))  # slice-0 + scales read: needs ADMA0/SDMA(j+1)
                if t == 0 and kb == 0:
                    seq += [("A2", j, i) for i in range(3, 6)]
                seq += [("b0", s + 3, i) for i in range(NB)]
                seq.append(("W", ("seg2", ctx, kb), s))
                if t == NT - 1 and kb == 2:
                    seq += [("ADMA1", j + 1, i) for i in range(NA_DMA)]
                if t == 0 and kb == 0:
                    seq += [("A2", j, i) for i in range(6, 8)]
                seq += [("b1", s + 3, i) for i in range(NB)] + g(s + 3)
                if kb in (0, 2):
                    seq += [("st", s, i) for i in range(ST)]
                if t == 0 and kb == 0:
                    seq += [("id", j, i) for i in range(NID)]  # after the previous item's last flush
                if t == NT - 1 and kb == 2:
                    seq += [("A2", j + 1, i) for i in range(3)]

    def idx_last(pred, before):
        c = [i for i, y in enumerate(seq[:before]) if y[0] != "W" and pred(y)]
        return max(c) if c else None

    res = {}
    for idx, x in enumerate(seq):
        if x[0] != "W":
            continue
        kind = x[1]
        if kind == "WA":
            tpos = idx_last(lambda y: y[0] in ("A0", "A2", "ADMA1", "SDMA"), idx)
        elif kind == "WB0":
            tpos = idx_last(lambda y: y[0] == "b1" and y[1] == 0, idx)
        elif kind == "PIN":
            tpos = idx_last(lambda y: y[0] == "id" and y[1] == x[2], idx)
        elif kind in ("RD0", "RD1", "USE2"):
            j = x[2]
            if kind == "RD0":
                tpos = idx_last(lambda y: y[0] in ("ADMA0", "SDMA") and y[1] == j, idx)
            elif kind == "RD1":
                tpos = idx_last(lambda y: y[0] == "ADMA1" and y[1] == j, idx)
            else:
                tpos = idx_last(lambda y: y[0] == "A2" and y[1] == j, idx)
            if tpos is None or (kind == "RD1" and j == 0):
                continue  # item 0's slice 1 was covered by the prologue's WA wait
            # these readers rely on the wait_barrier just before them: its target must be
            # younger than the producer (in-order retirement then covers the producer)
            wpos = max(i for i in range(idx) if seq[i][0] == "W" and isinstance(seq[i][1], tuple))
            s_ = seq[wpos][2]
            want = ("g", s_ + 1, NG - 1) if seq[wpos][1][0] == "top" else ("b1", s_ + 1, NB - 1)
            assert seq.index(want) > tpos, (kind, j, "producer not covered by the preceding wait")
            continue
        else:
            s_ = x[2]
            want = ("g", s_ + 1, NG - 1) if kind[0] == "top" else ("b1", s_ + 1, NB - 1)
            tpos = seq.index(want)
            assert tpos < idx, (x, want)
        n = sum(1 for y in seq[tpos + 1 : idx] if y[0] != "W")
        res.setdefault(kind, set()).add(n)
    out = {}
    for k, v in res.items():
        assert len(v) == 1, (k, v)
        out[k] = min(v.pop(), 63)
    return out


def compile_moe_gemm2_persist(
    *,
    H: int,
    I: int,
    E: int,
    topk: int,
    n_split: int = 2,
    out_dtype: str = "bf16",
):
    assert out_dtype in ("bf16", "fp8"), out_dtype
    FP8 = out_dtype == "fp8"
    BM = 128
    BN = 256
    K = I
    K_BYTES = K // 2
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    assert K % BLOCK_K == 0
    K_ITERS = K // BLOCK_K  # 3
    assert K_ITERS == 3, "the flat pipeline uses LDS set = k-step (3 sets)"
    N_TILES_ALL = H // BN  # 24
    assert N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per item
    assert NT % 2 == 0 and NT >= 4, "the n loop is unrolled by 2 with the last pair peeled"
    LDS_BLOCK_M = BM // 2  # 64 rows per A half
    LDS_BLOCK_N = BN // 2  # 128 W2 rows per B half
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 2
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    SC_COLS = K // 32  # 24 e8m0 per A row
    SC_BLOCKS_PER_G = SC_COLS // 8  # 3 blocks of 256 B per 32-row group
    OUT_ELEM = 1 if FP8 else 2
    OUT_ROW_BYTES = H * OUT_ELEM
    OUT_SC_COLS = H // 32
    WAVE_COLS = BN // 2  # 128 output columns per wave per n-tile

    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    B_SET_BYTES = 2 * b_lds_size  # 32 KB
    B_SET_OFF = [0, B_SET_BYTES, 2 * B_SET_BYTES]
    LDS_TILES_BYTES = 3 * B_SET_BYTES  # 96 KB
    B_SC_SLOT = 2048  # 8 blocks
    B_SC_SLOTS = 4
    SC_LDS_BYTES = B_SC_SLOTS * B_SC_SLOT  # 8 KB
    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB: one A half of one K-slice
    A_LDS_BYTES = 2 * a_lds_size  # 16 KB: one K-slice of the next item
    AS_LDS_BYTES = 4 * 1024  # 3 KB of A scales (+1 KB spill)
    JUNK_LDS_BYTES = _N_WAVES * 1024  # landing zone of the L2 touch-prefetch DMAs
    STG_ROW = WAVE_COLS * OUT_ELEM  # 128 (fp8) / 256 (bf16) B per staged row
    CH = STG_ROW // 16
    STG_DATA = 32 * STG_ROW
    STG_SC = 32 * 4 if FP8 else 0
    STG_WAVE = (STG_DATA + STG_SC + 255) // 256 * 256
    STG_LDS_BYTES = _N_WAVES * STG_WAVE
    ROWS_PER_ST = 64 // CH
    N_ST = 32 // ROWS_PER_ST
    assert LDS_TILES_BYTES + SC_LDS_BYTES + A_LDS_BYTES + AS_LDS_BYTES + JUNK_LDS_BYTES + STG_LDS_BYTES <= 160 * 1024
    assert N_ST % 4 == 0, "row ids come as dwordx4 per 4 consecutive flush rows"

    NB = N_TILES_B
    NG = 2
    P_STEP = 2 * NB + NG
    ST = N_ST + (1 if FP8 else 0)
    N_IDV = N_ST // 4  # dwordx4 id loads per row half
    NSA = 2 * K_ITERS  # A scales per lane (loop-carried)
    NID = 2 * N_IDV + (2 if FP8 else 2 * N_TILES_A)  # id vec4s + (fp8: scale-row ids | bf16: weights)
    NA = K_ITERS * 2 * N_TILES_A * 2  # A fragments per lane (24)
    NA_DMA = 2 * N_TILES_A  # LDS-DMA instructions per K-slice (2 halves x 2)
    VM = _derive_vmcnt(NB, NG, ST, NID, NA, NA_DMA, NT, K_ITERS)
    # the steady state must agree with gemm2.py's hand-derived constants
    assert VM[("top", "S", 0)] == P_STEP + ST and VM[("seg2", "S", 0)] == NG + P_STEP + ST + NB, VM
    assert VM[("top", "S", 1)] == ST + P_STEP + ST and VM[("top", "S", 2)] == ST + P_STEP, VM

    B_TILE_BYTES = BN * K_BYTES
    B_K_STEP = 2 * 1024

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SC_LDS_BYTES, 16]
        a_lds: fx.Array[fx.Int8, A_LDS_BYTES, 16]
        as_lds: fx.Array[fx.Int8, AS_LDS_BYTES, 16]
        junk_lds: fx.Array[fx.Int8, JUNK_LDS_BYTES, 16]
        stage_lds: fx.Array[fx.Int8, STG_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm2_persist(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        n_ctas: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr
        _sc_ptr = lds.scale_lds.ptr
        _a_ptr = lds.a_lds.ptr
        _as_ptr = lds.as_lds.ptr
        _junk_ptr = lds.junk_lds.ptr
        _stg_ptr = lds.stage_lds.ptr

        def b_buf(s, half):
            return _Buf(_base_ptr, B_SET_OFF[s] + half * b_lds_size)

        def a_buf(half):
            return _Buf(_a_ptr, half * a_lds_size)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # ---- work list ----
        nv_rsrc = _buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=4)
        num_valid = fx.Int32(_buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True))
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        slot, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        ctas_per_xcd = n_ctas // fx.Int32(_NUM_XCDS)
        xcd_lo = xcd * per_xcd
        xcd_n = _min(per_xcd, n_work - xcd_lo)  # items of this XCD (may be <= 0 on the tail)
        n_items = fx.arith.select(
            slot < xcd_n, (xcd_n - slot + ctas_per_xcd - fx.Int32(1)) // ctas_per_xcd, fx.Int32(0)
        )
        eid_rsrc = _buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4)

        def _sload(rsrc, idx):
            return fx.Int32(_buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=fx.Int32, is_scalar=True))

        ROT_STRIDE, ROT_GATE = 2, 3  # see gemm2.py

        def _desc(j):
            """item descriptor (wave-uniform): tile_i, chunk_n0, expert, nt_rot"""
            valid = j < n_items
            work = fx.arith.select(valid, xcd_lo + slot + j * ctas_per_xcd, fx.Int32(0))
            tile_i, chunk = _divmod_nonneg(work, n_split)
            expert = _sload(eid_rsrc, tile_i)
            _d = fx.Int32(ROT_GATE)
            _lo_ok = tile_i >= _d
            _hi_ok = tile_i + _d < num_m_blocks
            _e_lo = _sload(eid_rsrc, fx.arith.select(_lo_ok, tile_i - _d, fx.Int32(0)))
            _e_hi = _sload(eid_rsrc, fx.arith.select(_hi_ok, tile_i + _d, fx.Int32(0)))
            rot_on = (_lo_ok & (_e_lo == expert)) | (_hi_ok & (_e_hi == expert))
            nt_rot = fx.arith.select(rot_on, _divmod_nonneg(tile_i * fx.Int32(ROT_STRIDE), NT)[1], fx.Int32(0))
            return [tile_i, chunk * fx.Int32(NT), expert, nt_rot]

        _D_TILE, _D_N0, _D_EXP, _D_ROT = 0, 1, 2, 3

        # experiment: stagger CTA starts by slot phase (M3_G2P_STAGGER=<phases>,<sleeps>)
        _STG = os.environ.get("M3_G2P_STAGGER", "0,0").split(",")
        STAGGER_PHASES, STAGGER_SLEEPS = int(_STG[0]), int(_STG[1])
        # experiment: skip the next item's A / scale / id loads (timing only, output wrong)
        _NOA = os.environ.get("M3_G2P_NOALOAD", "0")
        NOALOAD = _NOA == "1"  # skip A / scales / ids
        NOALOAD_A = _NOA in ("1", "2")  # skip A / scales (ids still loaded)
        NOSTORE = os.environ.get("M3_G2P_NOSTORE", "0") == "1"  # timing only: no output stores
        TOUCH = os.environ.get("M3_G2P_TOUCH", "0") == "1"  # L2 touch-prefetch of the next item's A / scales / ids
        if const_expr(STAGGER_PHASES > 1):
            n_sleep = _divmod_nonneg(slot, STAGGER_PHASES)[1] * fx.Int32(STAGGER_SLEEPS)
            for _ in range(0, n_sleep):
                _asm_void([], "s_sleep 127", "")

        if n_items > fx.Int32(0):
            ids_rsrc = _buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            sw_rsrc = _buffer_ops.create_buffer_resource(
                sorted_weights, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES))
            as_rsrc = _buffer_ops.create_buffer_resource(
                A_scale, max_size=False, num_records_bytes=num_m_blocks * (BM * SC_COLS)
            )
            b_rsrc = _buffer_ops.create_buffer_resource(W2, max_size=False, num_records_bytes=E * H * K_BYTES)
            bs_rsrc = _buffer_ops.create_buffer_resource(W2_scale, max_size=False, num_records_bytes=E * H * SC_COLS)
            out_rsrc = _buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=n_tokens * (topk * OUT_ROW_BYTES)
            )
            osc_rsrc = _buffer_ops.create_buffer_resource(
                OUT_scale, max_size=False, num_records_bytes=n_tokens * (topk * OUT_SC_COLS)
            )

            # ---- B: LDS half hb, LDS row r -> W2 row (r//64)*128 + hb*64 + r%64 ----
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

            # ---- A: LDS-DMA of one K-slice (swizzled 128-B rows, as gemm2.py's prologue) ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    offs.append((half * LDS_BLOCK_M + row) * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            a0_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(0), N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(1), N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(0), NB, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(1), NB, wave_id)
            for ld in (b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            for ld in (a0_g2s, a1_g2s):
                ld.set_wave_base(_a_ptr)
            # A scales: 4 x 1 KB = the m-tile's 3 KB (+1 KB spill, harmless)
            as_g2s = G2SLoaderAsm(as_rsrc, [wave_id * 1024 + lane_id * 16], 1, wave_id)
            as_g2s.set_wave_base(_as_ptr)
            as_base_i32 = fx.Int32(fx.ptrtoint(_as_ptr))
            # L2 touch-prefetch: lane L pulls one 128-B line; wave w covers lines
            # w*128 .. w*128+127 of the A tile (2 instructions), every wave the scales / ids
            _touch_a = [
                G2SLoaderAsm(a_rsrc, [lane_id * 128 + wave_id * 16384 + i * 8192], 1, wave_id) for i in range(2)
            ]
            _touch_sa = G2SLoaderAsm(as_rsrc, [lane_id * 128], 1, wave_id)
            _touch_ids = G2SLoaderAsm(ids_rsrc, [lane_id * 128], 1, wave_id)
            _touch_w = G2SLoaderAsm(sw_rsrc, [lane_id * 128], 1, wave_id)
            for ld in _touch_a + [_touch_sa, _touch_ids, _touch_w]:
                ld.set_wave_base(_junk_ptr)
            _junk = _Buf(_junk_ptr, 0)

            def _touch_thunks(d):
                """4-5 junk DMAs that bring item ``d``'s A tile, scales and row ids into L2"""
                a_soff = d[_D_TILE] * fx.Int32(BM * K_BYTES)
                sa_soff = (d[_D_TILE] * fx.Int32(BM // 32)) * fx.Int32(SC_BLOCKS_PER_G * 256)
                id_soff = d[_D_TILE] * fx.Int32(BM * 4)
                ts = [lambda i=i: _touch_a[i].load_one(_junk, a_soff, 0) for i in range(2)]
                ts.append(lambda: _touch_sa.load_one(_junk, sa_soff, 0))
                ts.append(lambda: _touch_ids.load_one(_junk, id_soff, 0))
                ts += [lambda: _touch_w.load_one(_junk, id_soff, 0)] if not FP8 else []
                return ts
            sc_base_i32 = fx.Int32(fx.ptrtoint(_sc_ptr))
            bsg = _BScaleGather(bs_rsrc, lane_id, wave_id, sc_base_i32)
            a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
            b_s2r = S2RLoaderFp4(wave_j, NB)
            mfma = _MfmaAgprAB(N_TILES_A, NB)

            def _pn(d, nt):
                x = nt + d[_D_ROT]
                return fx.arith.select(x >= fx.Int32(NT), x - fx.Int32(NT), x)

            def _b_soff(d, nt, kb):
                return d[_D_EXP] * fx.Int32(H * K_BYTES) + (d[_D_N0] + _pn(d, nt)) * fx.Int32(B_TILE_BYTES) + fx.Int32(
                    kb * B_K_STEP
                )

            def _g_soff(d, nt, kb):
                return d[_D_EXP] * fx.Int32(H * SC_COLS) + (d[_D_N0] + _pn(d, nt)) * fx.Int32(BN * SC_COLS) + fx.Int32(
                    kb * 256
                )

            def _slot_off(nt, kb):
                s = (nt * fx.Int32(K_ITERS) + fx.Int32(kb)) & fx.Int32(B_SC_SLOTS - 1)
                return s * fx.Int32(B_SC_SLOT)

            def _gather(d, nt, kb):
                bsg.gather(_slot_off(nt, kb), _g_soff(d, nt, kb))

            def _load_b(d, nt, kb):
                b0_g2s.load(b_buf(kb, 0), _b_soff(d, nt, kb))
                b1_g2s.load(b_buf(kb, 1), _b_soff(d, nt, kb))

            # ---- A K-slice kb of item d: LDS-DMA thunks (4) / read from LDS / direct ----
            def _a_soff_dma(d, kb):
                return d[_D_TILE] * fx.Int32(BM * K_BYTES) + fx.Int32(kb * BLOCK_K_BYTES)

            def _a_dma_thunks(d, kb):
                soff = _a_soff_dma(d, kb)
                return _g2s_thunks(a0_g2s, a_buf(0), soff, N_TILES_A) + _g2s_thunks(a1_g2s, a_buf(1), soff, N_TILES_A)

            def _read_a_slice():
                """the DMA'd K-slice -> [h][ti][ksub] fragments (every wave's part landed)"""
                return [a_s2r.load(a_buf(h)) for h in range(2)]

            # direct global -> AGPR fragments: lane L holds row L%16 of its 16-row tile,
            # 16 B at K byte (L//16)*16 of each 64-B (128 fp4) chunk
            _a_voff = [
                (fx.Int32(h * LDS_BLOCK_M) + wave_i * fx.Int32(32) + r16) * fx.Int32(K_BYTES) + g4 * fx.Int32(16)
                for h in range_constexpr(2)
            ]
            _FRAGS = [(h, ti, ksub) for h in range(2) for ti in range(N_TILES_A) for ksub in range(2)]

            def _a_soff(d, kb, ti, ksub):
                return d[_D_TILE] * fx.Int32(BM * K_BYTES) + fx.Int32(ti * 16 * K_BYTES + kb * BLOCK_K_BYTES + ksub * 64)

            def _load_a_frags(d, kb, slice_, which):
                """direct loads of fragments ``which`` (indices into _FRAGS) of K-slice kb
                into ``slice_`` ([h][ti][ksub], mutated; None entries = first load)"""
                for f in which:
                    h, ti, ksub = _FRAGS[f]
                    slice_[h][ti][ksub] = _ld_a_frag(a_rsrc, _a_voff[h], _a_soff(d, kb, ti, ksub), slice_[h][ti][ksub])

            def _empty_slice():
                return [[[None for _ in range(2)] for _ in range(N_TILES_A)] for _ in range(2)]

            # A scales: LDS-DMA of the m-tile's 3 KB; block ((h*2 + wave_i)*3 + kb), 4 B per lane
            lane_sc = (lane_id // 4) * 16 + (lane_id % 4) * 4

            def _sa_dma_thunk(d):
                return [lambda: as_g2s.load(_Buf(_as_ptr, 0), (d[_D_TILE] * fx.Int32(BM // 32)) * fx.Int32(SC_BLOCKS_PER_G * 256))]

            def _read_sa(kb):
                return [
                    _lds_load_i32(as_base_i32 + fx.Int32(((h * 2) * SC_BLOCKS_PER_G + kb) * 256) + wave_i * fx.Int32(SC_BLOCKS_PER_G * 256) + lane_sc)
                    for h in range_constexpr(2)
                ]

            # row ids of the rows this lane touches. Flush: lane group g = lane//CH stores
            # staged rows g*N_ST .. g*N_ST+N_ST-1 (store k: rows {g*N_ST + k}), so the ids
            # of a row half come as N_ST/4 dwordx4; plus the scale-row ids (fp8, lane%32)
            # or the routing weights of the accumulator rows ti*16 + r16 (bf16).
            def _wave_row(h, r):
                return h * LDS_BLOCK_M + wave_i * 32 + r

            _id_vec_rows = [_wave_row(h, (lane_id // CH) * N_ST + 4 * q) for h in range(2) for q in range(N_IDV)]
            _id_one_rows = [_wave_row(h, lane_id % 32) for h in range(2)] if FP8 else []
            _w_rows = [] if FP8 else [_wave_row(h, ti * 16 + r16) for h in range(2) for ti in range(N_TILES_A)]

            def _load_ids(d, old):
                """[vec4 x 2*N_IDV] + [i32 ...]; ``old`` = the values being replaced"""
                soff = d[_D_TILE] * fx.Int32(BM * 4)
                o = old if old is not None else [None] * NID
                vals = [_ld_vec4(ids_rsrc, r * fx.Int32(4), soff, o[i]) for i, r in enumerate(_id_vec_rows)]
                n = len(vals)
                vals += [_ld_dword(ids_rsrc, r * fx.Int32(4), soff, o[n + i]) for i, r in enumerate(_id_one_rows)]
                n = len(vals)
                vals += [_ld_dword(sw_rsrc, r * fx.Int32(4), soff, o[n + i]) for i, r in enumerate(_w_rows)]
                return vals

            def _orow(sid):
                tok = sid & fx.Int32(0x00FFFFFF)  # padded rows: tok == n_tokens -> OOB -> dropped
                slot_ = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot_

            def _bind_rows(raw):
                """raw row ids (+ weights) of the item -> the values the epilogue uses:
                data-flush byte offsets [h][k], fp8 scale-flush offsets [h], bf16 routing
                weights [h][ti]. The raw values die here, so the derived ones take over
                their registers; the next item's ids are reloaded on top of these."""
                cur["off"] = [[None] * N_ST for _ in range(2)]
                for h in range_constexpr(2):
                    for q in range_constexpr(N_IDV):
                        for e, sid in enumerate(_vec4_elems(raw[h * N_IDV + q])):
                            cur["off"][h][4 * q + e] = _orow(sid) * fx.Int32(OUT_ROW_BYTES)
                n = 2 * N_IDV
                cur["sc_off"] = [_orow(raw[n + h]) * fx.Int32(OUT_SC_COLS) for h in range(2)] if FP8 else None
                cur["wbits"] = (
                    [[raw[n + h * N_TILES_A + ti] for ti in range(N_TILES_A)] for h in range(2)] if not FP8 else None
                )

            def _bound_rows_flat():
                """the registers the next item's raw ids are loaded into (order of _load_ids);
                also the loop-carried form of the epilogue values"""
                vals = [_make_vec4(cur["off"][h][4 * q : 4 * q + 4]) for h in range(2) for q in range(N_IDV)]
                if FP8:
                    vals += [cur["sc_off"][h] for h in range(2)]
                else:
                    vals += [cur["wbits"][h][ti] for h in range(2) for ti in range(N_TILES_A)]
                return vals

            def _set_rows(vals):
                """inverse of _bound_rows_flat (loop-carried values -> cur)"""
                cur["off"] = [
                    [v for q in range(N_IDV) for v in _vec4_elems(vals[h * N_IDV + q])] for h in range(2)
                ]
                n = 2 * N_IDV
                cur["sc_off"] = [fx.Int32(vals[n + h]) for h in range(2)] if FP8 else None
                cur["wbits"] = (
                    [[fx.Int32(vals[n + h * N_TILES_A + ti]) for ti in range(N_TILES_A)] for h in range(2)]
                    if not FP8
                    else None
                )

            # ---- epilogue (as gemm2.py; the offsets come from ``cur`` filled per item) ----
            stg_base = fx.Int32(fx.ptrtoint(_stg_ptr)) + wave_id * fx.Int32(STG_WAVE)
            stg_sc_base = stg_base + fx.Int32(STG_DATA)
            cur = {}  # per-item epilogue values, see _bind_rows

            def _stg_addr(row, chunk, half):
                return stg_base + row * fx.Int32(STG_ROW) + ((chunk ^ (row % CH)) * 16 + half * 8)

            def _stage_bf16(cq, h, hb, ti, tj):
                cv = Vec(_pin_vec4(cq[mfma.idx(ti, tj)]))
                w = _as_f32(cur["wbits"][h][ti])
                v = [fx.Float32(cv[k]) * w for k in range_constexpr(4)]
                row = ti * 16 + r16
                chunk = hb * 8 + tj * 2 + g4 // 2
                _lds_store_vec(_v2i32(_bf16x2(v[0], v[1]), _bf16x2(v[2], v[3])), _stg_addr(row, chunk, g4 % 2), 2)

            def _stage_fp8_pair(cq, hb, ti):
                v = []
                for p in range_constexpr(2):
                    cv = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p)]))
                    cw = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p + 1)]))
                    v.append(
                        [fx.Float32(cv[k]) for k in range_constexpr(4)] + [fx.Float32(cw[k]) for k in range_constexpr(4)]
                    )
                am = []
                for p in range_constexpr(2):
                    a = _maxf_nn(_fabs(v[p][0]), _fabs(v[p][1]))
                    for k in range_constexpr(2, 8):
                        a = _maxf_nn(a, _fabs(v[p][k]))
                    am.append(a)
                amax = _xlane_max4_pair(am[0], am[1])
                row = ti * 16 + r16
                e8s = []
                for p in range_constexpr(2):
                    e8 = _e8m0_fp8(amax[p])
                    sf = _as_f32(e8 << 23)
                    da = _cvt_pk_fp8(_undef_i32(), v[p][0], v[p][1], sf, False)
                    da = _cvt_pk_fp8(da, v[p][2], v[p][3], sf, True)
                    db = _cvt_pk_fp8(_undef_i32(), v[p][4], v[p][5], sf, False)
                    db = _cvt_pk_fp8(db, v[p][6], v[p][7], sf, True)
                    da, db = _permlane16_swap(da, db)
                    chunk = hb * 4 + 2 * p + g4 % 2
                    _lds_store_vec(_v2i32(da, db), _stg_addr(row, chunk, g4 // 2), 2)
                    e8s.append(e8)
                _lds_store_i16(e8s[0] | (e8s[1] << 8), stg_sc_base + row * fx.Int32(4) + hb * 2)

            def _stage_thunks(cq, h, hb):
                ts = []
                for ti in range_constexpr(N_TILES_A):
                    if const_expr(FP8):
                        ts.append(lambda ti=ti: _stage_fp8_pair(cq, hb, ti))
                    else:
                        for tj in range_constexpr(NB):
                            ts.append(lambda ti=ti, tj=tj: _stage_bf16(cq, h, hb, ti, tj))
                return ts

            def _flush_thunks(d, h, nt, mask):
                col_wave = ((d[_D_N0] + _pn(d, nt)) * fx.Int32(BN) + wave_j * WAVE_COLS) * OUT_ELEM
                ts = []
                for k in range_constexpr(N_ST):

                    def _st(k=k):
                        row = (lane_id // CH) * N_ST + k
                        chunk = lane_id % CH
                        data = _lds_load_vec(_stg_addr(row, chunk, 0), 4)
                        if const_expr(not NOSTORE):
                            _buffer_ops.buffer_store(
                                data, out_rsrc, cur["off"][h][k] + col_wave + chunk * 16, mask=mask, offset_is_bytes=True
                            )

                    ts.append(_st)
                if const_expr(FP8):

                    def _st_sc():
                        scv = _lds_load_i32(stg_sc_base + (lane_id % 32) * 4)
                        sc_col = (d[_D_N0] + _pn(d, nt)) * fx.Int32(BN // 32) + wave_j * 4
                        if const_expr(not NOSTORE):
                            _buffer_ops.buffer_store(scv, osc_rsrc, cur["sc_off"][h] + sc_col, mask=mask, offset_is_bytes=True)

                    ts.append(_st_sc)
                return ts

            def _read_bsc_thunks(nt, kb, holder):
                base = sc_base_i32 + _slot_off(nt, kb) + lane_sc

                def _rd(idx, hb, sub):
                    gi = wave_j * 4 + hb * 2 + sub
                    holder[idx] = _lds_load_i32(base + fx.Int32(gi * 256))

                return [lambda: _rd(0, 0, 0), lambda: _rd(1, 0, 1), lambda: _rd(2, 1, 0), lambda: _rd(3, 1, 1)]

            # ---- waits: context F / N / L static or dynamic ----
            def _wait(kind, kb, ctx):
                if const_expr(isinstance(ctx, str)):
                    wait_barrier(VM[(kind, ctx, kb)])
                else:
                    # tile 0 of an item: ctx = wave-uniform bool "first item of the CTA"
                    cF, cN = VM[(kind, "F", kb)], VM[(kind, "N", kb)]
                    if const_expr(cF == cN):
                        wait_barrier(cF)
                    else:
                        if ctx:
                            wait_barrier(cF)
                        else:
                            wait_barrier(cN)

            def _one_step(d, d_nt, nt, nt_next, kb, ctx, b0f, b1f, sc, accs, epi, st, hooks=None):
                """flat step (nt, kb) of item ``d``; issues B(nt_next, kb) of item ``d_nt``.
                ``st``: mutable item state {"aF", "saA", "ids_raw"}. ``hooks``: {"top":
                [fn], "p1": [thunk], "p3": [thunk], "p4": [fn]} -- boundary work: "top"
                runs right after the top wait_barrier, "p1"/"p3" thunks go in front of
                that phase's DMA thunks (m0 sequences stay contiguous), "p4" after the step."""
                hooks = hooks or {}
                kb1 = (kb + 1) % K_ITERS
                nt1 = nt if (kb + 1) < K_ITERS else nt_next
                bn0, bn1 = b_buf(kb1, 0), b_buf(kb1, 1)
                sbC0, sbC1 = sc
                c00, c01, c10, c11 = accs
                zero = kb == 0

                def _late(ph):
                    return epi[ph](accs) if ph in epi else None

                _scn = [None] * 4
                rd_scn = _read_bsc_thunks(nt1, kb1, _scn)
                b_off3 = _b_soff(d_nt, nt_next, kb)

                _wait("top", kb, ctx)
                for fn in hooks.get("top", []):
                    fn()
                a0f, a1f = st["aF"][kb][0], st["aF"][kb][1]
                il = list(hooks.get("p1", [])) + _g2s_thunks(b0_g2s, b_buf(kb, 0), b_off3, NB) + rd_scn[:2]
                mfma.call(a0f, b0f, c00, [st["saA"][0][kb]], sbC0, interleave=il, zero_acc=zero, late=_late(1))
                mfma.call(a0f, b1f, c01, [st["saA"][0][kb]], sbC1, interleave=rd_scn[2:], zero_acc=zero, late=_late(2))

                _wait("seg2", kb, ctx)
                _b0n = [None] * NB
                _b1n = [None] * NB
                il = (
                    list(hooks.get("p3", []))
                    + _g2s_thunks(b1_g2s, b_buf(kb, 1), b_off3, NB)
                    + [lambda: _gather(d_nt, nt_next, kb)]
                    + list(hooks.get("p3post", []))
                )
                mfma.call(a1f, b0f, c10, [st["saA"][1][kb]], sbC0, interleave=il, zero_acc=zero, late=_late(3))
                il = _s2r_thunks(b_s2r, bn0, _b0n, NB, True) + _s2r_thunks(b_s2r, bn1, _b1n, NB, True)
                mfma.call(a1f, b1f, c11, [st["saA"][1][kb]], sbC1, interleave=il, zero_acc=zero, late=_late(4))
                for fn in hooks.get("p4", []):
                    fn()
                return _b0n, _b1n, (_scn[:2], _scn[2:])

            _R = fx.as_ir_value

            def _flat_b(frag):
                return [_R(t[0]) for t in frag] + [_R(t[1]) for t in frag]

            def _unflat_b(flat):
                return [[flat[i], flat[NB + i]] for i in range(NB)]

            def _flat_state(b0f, b1f, sc, c11):
                return _flat_b(b0f) + _flat_b(b1f) + [_R(v) for v in sc[0]] + [_R(v) for v in sc[1]] + [_R(v) for v in c11]

            def _unflat_state(s_):
                o = 0
                b0f = _unflat_b(s_[o : o + 2 * NB])
                o += 2 * NB
                b1f = _unflat_b(s_[o : o + 2 * NB])
                o += 2 * NB
                sc = (list(s_[o : o + 2]), list(s_[o + 2 : o + 4]))
                o += 4
                c11 = list(s_[o : o + N_ACCUMS])
                return b0f, b1f, sc, c11

            def _n_tile(d, d_nt, nt, nt_next, ctx, b0f, b1f, sc, c11_prev, nt_prev, st, hooks_by_kb=None, d_prev=None, mask_prev=None):
                """as gemm2.py: the previous tile's c11 is staged in kb 0 P1 and its rows
                flushed in kb 0 P4 (``d_prev`` / ``mask_prev`` when that tile belonged to
                the previous item -- or to nothing, for the CTA's first tile)"""
                accs = tuple([None] * N_ACCUMS for _ in range(4))
                hooks_by_kb = hooks_by_kb or {}
                dp = d_prev if d_prev is not None else d
                epi_by_kb = {
                    0: {
                        1: lambda a: _stage_thunks(c11_prev, 1, 1),
                        4: lambda a: _flush_thunks(dp, 1, nt_prev, mask_prev),
                    },
                    1: {},
                    2: {
                        2: lambda a: _stage_thunks(a[0], 0, 0),
                        3: lambda a: _stage_thunks(a[1], 0, 1),
                        4: lambda a: _flush_thunks(d, 0, nt, None) + _stage_thunks(a[2], 1, 0),
                    },
                }
                for kb in range_constexpr(K_ITERS):
                    b0f, b1f, sc = _one_step(
                        d, d_nt, nt, nt_next, kb, ctx, b0f, b1f, sc, accs, epi_by_kb[kb], st, hooks_by_kb.get(kb)
                    )
                return b0f, b1f, sc, accs[3]

            # ---- item state flattening (A frags, A scales, raw row ids, descriptors) ----
            def _flat_a(aF):
                return [_R(aF[kb][h][ti][ks]) for kb in range(K_ITERS) for h in range(2) for ti in range(N_TILES_A) for ks in range(2)]

            def _unflat_a(flat):
                it = iter(flat)
                return [[[[next(it) for ks in range(2)] for ti in range(N_TILES_A)] for h in range(2)] for kb in range(K_ITERS)]

            def _flat_sa(saA):
                return [_R(saA[h][kb]) for h in range(2) for kb in range(K_ITERS)]

            def _unflat_ids(flat):
                return list(flat)  # raw IR values (vec4 / i32), used only by _wait_pin_any / _load_ids

            def _unflat_sa(flat):
                return [[flat[h * K_ITERS + kb] for kb in range(K_ITERS)] for h in range(2)]

            zero_v4 = _arith.ConstantOp(
                mfma.res_ty, _ir.DenseElementsAttr.get_splat(mfma.res_ty, _ir.FloatAttr.get(_T.f32, 0.0))
            ).result

            # ---- prologue (item 0): leaves the same state a boundary does ----
            d0 = _desc(fx.Int32(0))
            d1 = _pin_sgprs(_desc(fx.Int32(1)))
            n0 = fx.Int32(0)
            for kb in range_constexpr(K_ITERS):
                _gather(d0, n0, kb)
            aF0 = [_empty_slice() for _ in range(K_ITERS)]
            _load_a_frags(d0, 0, aF0[0], list(range(len(_FRAGS))))  # slice 0: direct (all 8)
            _load_a_frags(d0, 2, aF0[2], [0, 1, 2])  # slice 2: fragments 0..2 (3..7 in tile 0)
            for t in _a_dma_thunks(d0, 1) + _sa_dma_thunk(d0):  # slice 1 + scales into LDS
                t()
            for kb in range_constexpr(K_ITERS):
                _load_b(d0, n0, kb)
            wait_barrier(VM["WA"])  # A loads, slice-1 / scale DMAs landed (B(0..2) may fly)
            # the epilogue values (row offsets / weights) an item's tile 0 replaces with the
            # next ids: undefined before the first item (its tile-0 flush is masked)
            cur["off"] = [[_undef_i32() for _ in range(N_ST)] for _ in range(2)]
            cur["sc_off"] = [_undef_i32() for _ in range(2)] if FP8 else None
            cur["wbits"] = [[_undef_i32() for _ in range(N_TILES_A)] for _ in range(2)] if not FP8 else None
            saA0 = [[None] * K_ITERS for _ in range(2)]
            for kb in range_constexpr(2):
                r = _read_sa(kb)
                saA0[0][kb], saA0[1][kb] = r[0], r[1]
            wait_barrier(VM["WB0"])  # B(0) + its scales landed
            b0f = b_s2r.load(b_buf(0, 0), preshuffled=True)
            b1f = b_s2r.load(b_buf(0, 1), preshuffled=True)
            _sc0 = [None] * 4
            for t in _read_bsc_thunks(n0, 0, _sc0):
                t()
            # slice 1 and sa[.][2] are read in tile 0 (kb 1 top / kb 0 top) like every item;
            # their register slots start undefined
            und_v4 = _llvm.mlir_undef(_ir.VectorType.get([4], _T.i32))
            for h in range_constexpr(2):
                for ti in range_constexpr(N_TILES_A):
                    for ks in range_constexpr(2):
                        aF0[1][h][ti][ks] = und_v4
                        if aF0[2][h][ti][ks] is None:
                            aF0[2][h][ti][ks] = und_v4
                saA0[h][2] = _undef_i32()

            item_init = (
                [_R(v) for v in d0]
                + [_R(v) for v in d1]
                + _flat_a(aF0)
                + _flat_sa(saA0)
                + _flat_state(b0f, b1f, (_sc0[:2], _sc0[2:]), [zero_v4] * N_ACCUMS)
                + [_R(v) for v in d0]  # "previous item" of item 0 (its flush is masked)
                + [_R(v) for v in _bound_rows_flat()]
            )
            N_D = 4
            for j, ist in range(0, n_items, init=item_init):
                o = 0
                d = [fx.Int32(v) for v in ist[o : o + N_D]]
                o += N_D
                dn = [fx.Int32(v) for v in ist[o : o + N_D]]
                o += N_D
                st = {"aF": _unflat_a(ist[o : o + NA])}
                o += NA
                st["saA"] = _unflat_sa(ist[o : o + NSA])
                o += NSA
                b0f, b1f, sc, c11_prev = _unflat_state(ist[o : o + 4 * NB + 4 + N_ACCUMS])
                o += 4 * NB + 4 + N_ACCUMS
                d_prev = [fx.Int32(v) for v in ist[o : o + N_D]]
                o += N_D
                _set_rows(ist[o : o + NID])
                o += NID
                j_i = fx.Int32(j)
                first_item = j_i == fx.Int32(0)
                mask_prev = fx.as_ir_value(j_i > fx.Int32(0))  # item 0: nothing to flush
                # the descriptor of item j+2, used by the next iteration's last tile; its
                # scalar loads hide behind this item's tiles and tail
                dnn = _pin_sgprs(_desc(j_i + fx.Int32(2)))

                # tile 0 (peeled): finishes the item's own A / scales, pins the row ids
                def _rd_sa2():
                    r = _read_sa(2)
                    st["saA"][0][2], st["saA"][1][2] = r[0], r[1]

                def _rd_slice1():
                    st["aF"][1] = _read_a_slice()

                def _pin():
                    _bind_rows(_wait_pin_any(st["ids_raw"], VM["PIN"]))

                def _ld_ids():
                    # right after the previous item's last rows went out: their offsets die
                    # here and this item's raw ids take their registers
                    st["ids_raw"] = _load_ids(d, _bound_rows_flat()) if not NOALOAD else _bound_rows_flat()

                hooks0 = {
                    0: {
                        "top": [_rd_sa2],
                        "p1": [lambda f=f: _load_a_frags(d, 2, st["aF"][2], [f]) for f in range(3, 6)],
                        "p3": [lambda f=f: _load_a_frags(d, 2, st["aF"][2], [f]) for f in range(6, 8)],
                        "p4": [_ld_ids],
                    },
                    1: {"top": [_rd_slice1]},
                    2: {"top": [_pin]},
                }
                if const_expr(NOALOAD_A):
                    hooks0 = {0: {"top": [_rd_sa2], "p4": [_ld_ids]}, 1: {"top": [_rd_slice1]}, 2: {"top": [_pin]}}
                b0f, b1f, sc, c11p = _n_tile(
                    d, d, fx.Int32(0), fx.Int32(1), first_item, b0f, b1f, sc, c11_prev, fx.Int32(NT - 1), st, hooks0,
                    d_prev=d_prev, mask_prev=mask_prev,
                )
                inner_init = _flat_state(b0f, b1f, sc, c11p)
                touch = _touch_thunks(dn)
                for np_, state in range(0, NT // 2 - 1, init=inner_init):
                    b0f, b1f, sc, c11p = _unflat_state(state)
                    np_i = fx.Int32(np_)
                    nt_e = np_i * fx.Int32(2) + fx.Int32(1)
                    nt_o = nt_e + fx.Int32(1)
                    is_last_pair = np_i == fx.Int32(NT // 2 - 2)

                    def _guarded(t):
                        def _run():
                            if is_last_pair:
                                t()

                        return _run

                    hooks_e = {kb: {"p3post": [_guarded(touch[kb])]} for kb in range(K_ITERS)} if TOUCH else None
                    hooks_o = (
                        {kb: {"p3post": [_guarded(touch[K_ITERS + kb])]} for kb in range(K_ITERS) if K_ITERS + kb < len(touch)}
                        if TOUCH
                        else None
                    )
                    b0f, b1f, sc, c11e = _n_tile(d, d, nt_e, nt_o, "S", b0f, b1f, sc, c11p, nt_e - fx.Int32(1), st, hooks_e)
                    b0f, b1f, sc, c11o = _n_tile(d, d, nt_o, nt_o + fx.Int32(1), "S", b0f, b1f, sc, c11e, nt_e, st, hooks_o)
                    state = yield _flat_state(b0f, b1f, sc, c11o)
                # last tile (peeled): streams the next item's A / scales in, prefetches its
                # n-tile 0, then the tail flushes what this tile leaves behind
                b0f, b1f, sc, c11p = _unflat_state(state)
                nt_l = fx.Int32(NT - 1)

                def _rd_slice0_sa01():
                    st["aF"][0] = _read_a_slice()
                    for kb in range_constexpr(2):
                        r = _read_sa(kb)
                        st["saA"][0][kb], st["saA"][1][kb] = r[0], r[1]

                hooksL = {
                    0: {"top": _a_dma_thunks(dn, 0) + _sa_dma_thunk(dn)},
                    2: {
                        "top": [_rd_slice0_sa01],
                        "p3": _a_dma_thunks(dn, 1),
                        "p4": [lambda: _load_a_frags(dn, 2, st["aF"][2], [0, 1, 2])],
                    },
                }
                if const_expr(NOALOAD_A):
                    hooksL = {2: {"top": [_rd_slice0_sa01]}}
                b0f, b1f, sc, c11o = _n_tile(d, dn, nt_l, fx.Int32(0), "L", b0f, b1f, sc, c11p, nt_l - fx.Int32(1), st, hooksL)
                # the last tile's c11 / row half 1 go out in the next item's tile 0 (kb 0)
                ist = yield (
                    [_R(v) for v in dn]
                    + [_R(v) for v in dnn]
                    + _flat_a(st["aF"])
                    + _flat_sa(st["saA"])
                    + _flat_state(b0f, b1f, sc, c11o)
                    + [_R(v) for v in d]
                    + [_R(v) for v in _bound_rows_flat()]
                )

            # the CTA's last rows: nothing left to hide them behind
            o = 2 * N_D + NA + NSA
            _, _, _, c11_last = _unflat_state(ist[o : o + 4 * NB + 4 + N_ACCUMS])
            o += 4 * NB + 4 + N_ACCUMS
            d_last = [fx.Int32(v) for v in ist[o : o + N_D]]
            o += N_D
            _set_rows(ist[o : o + NID])
            for t in _stage_thunks(c11_last, 1, 1) + _flush_thunks(d_last, 1, fx.Int32(NT - 1), None):
                t()
            # never retire with DMA in flight (the CU reuses the LDS)
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2_persist(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        n_ctas: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2_persist(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            n_ctas,
        ).launch(grid=(n_ctas, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2_persist


def gemm2_persist_grid(n_ctas: int = 256) -> int:
    assert n_ctas % _NUM_XCDS == 0
    return n_ctas
