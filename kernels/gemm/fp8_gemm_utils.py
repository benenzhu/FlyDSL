# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

# ceildiv is the canonical cdiv from the shared layer; re-exported here for the
# gemm kernels that historically imported it from this module.
from kernels.common.utils import cdiv as ceildiv  # noqa: F401


def divmod(a, b):
    """Integer divmod that works on DSL values (e.g. ``Int32``).

    The builtin ``divmod`` rejects DSL scalar types, so this uses the overloaded
    ``//`` / ``%`` operators to emit the corresponding ops.
    """
    return (a // b, a % b)


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` for ``b_preshuffled=True``."""
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16) + (row % 16) * 16 + (col // 64) * 1024 + ((col % 64) // 16) * 256 + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


class StoreC:
    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        # Exact byte counts from compile-time shape (BF16 C output, FP32 scales).
        # ``num_records_bytes`` is required when ``max_size=False`` -- see
        # ``make_buffer_tensor`` docstring for the silent-OOB rationale.
        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        sa_nbytes = c_rows * 4  # Float32 row-wise scale
        sb_nbytes = c_cols * 4  # Float32 col-wise scale
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=sa_nbytes)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=sb_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _load_scale_vec4(self, row):
        fx.copy(self.scale_atom_4, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _load_scale_scalar(self, col):
        fx.copy(self.scale_atom_1, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        a_scales = [
            self._load_scale_vec4(base_row + i * 16 + (self.lane_id // 16) * 4) for i in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_scale_scalar(base_col + i * 16 + self.lane_id % 16) for i in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(scaled, arith.select(col_valid, c_index, oob))


def _cvt_pk_bf16(a, b):
    """Pack two f32 into 2xbf16 (i32), as ``arith.truncf <2xf32> -> <2xbf16>``.

    Selects to the same single ``v_cvt_pk_bf16_f32`` on gfx950 as
    ``rocdl.cvt_pk_bf16_f32``, but that helper is inline asm (ROCDL has no op
    for this instruction), which plants an ASMSTART/ASMEND wall the machine
    scheduler cannot move across -- 128 of them in the epilogue.
    """
    v2f32 = _ir.VectorType.get([2], fx.Float32.ir_type)
    vec = Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
    src = fx.as_ir_value(vec)
    if src.type != v2f32:
        src = fx.arith.bitcast(v2f32, src)
    v2bf16 = _ir.VectorType.get([2], fx.BFloat16.ir_type)
    # llvm.bitcast, not arith.bitcast: the latter requires operand and result to
    # have the same shape, and this one is <2xbf16> -> i32.
    return _llvm.BitcastOp(fx.Int32.ir_type, fx.arith.trunc_f(v2bf16, src)).result


def _permlane16_swap(d_a, d_b):
    """Exchange the row-16 halves of two VGPRs between lane groups."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = rocdl.permlane16_swap(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return _llvm.extractvalue(_T.i32, res, [0]), _llvm.extractvalue(_T.i32, res, [1])


class StoreCTransposed:
    """``StoreC`` for a kernel whose MFMAs are fed (B, A) instead of (A, B).

    Since C^T = B^T A^T that swap transposes the accumulator: lane L holds
    ``C[L%16, 4 consecutive cols]`` instead of ``C[4 consecutive rows, L%16]``.
    The 4 values per accumulator are then CONTIGUOUS in memory, so a pair of
    N-tiles can be permlane16_swap'd into 16 bytes per lane and written with one
    ``buffer_store_dwordx4`` -- vs 4 separate ``buffer_store_short`` per
    accumulator on the untransposed path.

    The two scales swap roles to match: A's (per row) becomes a per-lane scalar,
    B's (per col) becomes the 8 consecutive columns this lane ends up owning.

    Requires ``n_tiles_b`` even (tiles are consumed in pairs).
    """

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        assert n_tiles_b % 2 == 0, f"StoreCTransposed needs an even n_tiles_b, got {n_tiles_b}"
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        sa_nbytes = c_rows * 4  # Float32 row-wise scale
        sb_nbytes = c_cols * 4  # Float32 col-wise scale
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=sa_nbytes)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=sb_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)

    def _load_a_scale(self, row):
        """Transposed: A's scale is per ROW, and a lane owns ONE row -> scalar."""
        fx.copy(self.scale_atom_1, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _load_b_scale_vec4(self, col):
        """Transposed: B's scale is per COL, and a lane owns 4 consecutive cols."""
        fx.copy(self.scale_atom_4, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _store_one(self, c_frag, base_row, base_col, ti, tj, a_scale, b_scales):
        """One 16-byte store covering N-tiles ``tj`` and ``tj+1``."""
        vec_lo = Vec(c_frag[self.c_idx_fn(ti, tj)])
        vec_hi = Vec(c_frag[self.c_idx_fn(ti, tj + 1)])
        sc_lo, sc_hi = b_scales[tj], b_scales[tj + 1]
        lo = [(vec_lo[i] * (a_scale * sc_lo[i])) for i in range_constexpr(4)]
        hi = [(vec_hi[i] * (a_scale * sc_hi[i])) for i in range_constexpr(4)]
        a0 = _cvt_pk_bf16(lo[0], lo[1])
        a1 = _cvt_pk_bf16(lo[2], lo[3])
        b0 = _cvt_pk_bf16(hi[0], hi[1])
        b1 = _cvt_pk_bf16(hi[2], hi[3])

        # Swap halves between lane groups so each lane ends up with 8 consecutive
        # columns (16 bytes) instead of two disjoint 4-column runs.
        a0, b0 = _permlane16_swap(a0, b0)
        a1, b1 = _permlane16_swap(a1, b1)

        g = self.lane_id // 16
        row = base_row + ti * 16 + self.lane_id % 16
        col = base_col + (tj + g % 2) * 16 + (g // 2) * 8
        pack = Vec.from_elements([fx.Int32(a0), fx.Int32(a1), fx.Int32(b0), fx.Int32(b1)], fx.Int32).bitcast(
            fx.BFloat16
        )
        fx.memref_store_vec(pack, self.reg_bf16_8)
        c_index = row * self.c_cols + col
        fx.copy(self.out_atom_8, self.reg_bf16_8, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        # b_scales[tj] = the 4 columns tile tj contributes for this lane; the
        # permlane16_swap below re-groups them, but the scale must be applied
        # BEFORE the swap, while each value is still with its own column.
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + (self.lane_id // 16) * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            a_scale = self._load_a_scale(base_row + ti * 16 + self.lane_id % 16)
            for tj in range_constexpr(0, self.n_tiles_b, 2):
                self._store_one(c_frag, base_row, base_col, ti, tj, a_scale, b_scales)


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


class Mfma16x16x128:
    """``swap_ab`` feeds the MFMA (B, A) instead of (A, B). Since C^T = B^T A^T
    that produces the transposed accumulator ``StoreCTransposed`` expects, whose
    4 values per lane are contiguous in C and so store 16 bytes at a time."""

    def __init__(self, n_tiles_a, n_tiles_b, swap_ab=False):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.swap_ab = swap_ab

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _do_mma(self, a, b, c):
        a_frag = self._make_operand_frag(a)
        b_frag = self._make_operand_frag(b)
        c_frag = self._make_accum_frag(c)
        fx.gemm(self.atom, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load().ir_value()

    def call(self, a, b, c, *, set_prio=True):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [self._make_operand_frag(a[idx]) for idx in range_constexpr(self.n_tiles_a)]
        b_frags = [self._make_operand_frag(b[idx]) for idx in range_constexpr(self.n_tiles_b)]
        c_frags = [self._make_accum_frag(c[idx]) for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                if const_expr(self.swap_ab):
                    fx.gemm(self.atom, cf, b_frags[j], a_frags[i], cf)
                else:
                    fx.gemm(self.atom, cf, a_frags[i], b_frags[j], cf)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [c_frags[idx].load().ir_value() for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b

        if const_expr(self.swap_ab):
            return self._do_mma(b[j], a[i], c[self.idx(i, j)])
        return self._do_mma(a[i], b[j], c[self.idx(i, j)])
