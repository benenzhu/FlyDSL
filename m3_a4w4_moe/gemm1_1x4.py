# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MiniMax-M3 prefill MoE stage 1 (a4w4), aiter-style 1x4 kernel.

Port of the BM128 FlyDSL gemm1 from the Kimi mxfp4 migration
(``kimi_fp4_moe_16384_opt.py``, ``..._v36_agprasm``: A through a 2-slot LDS
pipeline, B straight from global into VGPRs two K-tiles ahead, 4 waves side by
side on N, accumulators pinned in AGPR by inline asm) to the M3 shape and the
production buffer layouts used by ``gemm1.py``:

  * routing: ``sorted_ids`` (aiter moe_sorting, ``token | slot << 24``, padded
    rows carry ``token == n_tokens`` -> out-of-range -> zeros), one expert per
    128-row m-tile from ``sorted_expert_ids``, block order from ``tile_map``
    (expert by expert, n-slab-major, one contiguous chunk per XCD).
  * W13 is the production *separated* layout: gate rows ``[0, I)``, up rows
    ``[I, 2I)`` of ``shuffle_weight(layout=(16,16))``. Waves 0/1 of a block
    load the 128 gate rows of n-tile ``by``, waves 2/3 the matching 128 up rows,
    so the tile's 256 accumulator columns are [gate | up] of the same 128
    intermediate columns and the epilogue pairs col ``c`` with ``c + 128``.
  * epilogue: swiglu-OAI (alpha 1.702, limit 7, up+1), per-32-col amax over the
    4 lanes of a quad (DPP), E8M0 = aiter's fused FlyDSL rule (nearest pow2,
    headroom 2), ``v_cvt_scalef32_pk_fp4_f32``, fp4 rows in sorted order and
    e8m0 scales in the sorted-shuffled layout (identical to ``gemm1.py``).

Layouts: see gemm1.py.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math
import re

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf, vector
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from aiter.ops.flydsl.kernels import buffer_ops  # the copy shipped in the vLLM image

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


# ---------------------------------------------------------------------------
# small helpers (copied from the Kimi branch's kernels/layout_utils.py and
# kernels/mfma_preshuffle_pipeline.py so this file is self-contained)
# ---------------------------------------------------------------------------
def _wrap(v):
    if isinstance(v, ArithValue):
        return v
    if isinstance(v, ir.Value):
        return ArithValue(v)
    return v


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _div_pow2(val, divisor):
    shift = _math.log2(divisor)
    assert shift == int(shift), f"{divisor} is not a power of 2"
    return arith.shrui(val, arith.index(int(shift)))


def _mod_pow2(val, modulus):
    return arith.andi(val, arith.index(modulus - 1))


def _parse_dim(tok):
    tok = tok.strip()
    return None if tok == "?" else int(tok)


def _parse_layout(ly):
    ly_str = str(ly.type) if hasattr(ly, "type") else str(ly)
    m = re.search(r"\(([^)]+)\):\(([^)]+)\)", ly_str)
    if not m:
        return None
    shapes = [_parse_dim(s) for s in m.group(1).split(",")]
    strides = [_parse_dim(s) for s in m.group(2).split(",")]
    return shapes, strides


def _has_dynamic_strides(strides):
    return any(s is None for s in strides)


def idx2crd(idx, layout):
    parsed = _parse_layout(layout)
    if hasattr(idx, "ir_value"):
        idx = idx.ir_value()
    if parsed is None or _has_dynamic_strides(parsed[1]):
        result = fx.idx2crd(fx.Int32(idx), layout)
        ndims = len(parsed[1]) if parsed else 1
        return [_wrap(fx.get(result, i)) for i in range(ndims)]
    if isinstance(idx, ir.Value) and not isinstance(idx.type, ir.IndexType):
        idx = arith.index_cast(T.index, idx)
    shapes, strides = parsed
    ndims = len(strides)
    ordered = sorted(
        [(i, s, sz) for i, s, sz in _builtins.zip(range(ndims), strides, shapes) if s != 0],
        key=lambda x: x[1],
        reverse=True,
    )
    coords = [None] * ndims
    remaining = idx
    for i, stride_val, size_val in ordered:
        if stride_val == 1:
            c = remaining
        elif _is_pow2(stride_val):
            c = _div_pow2(remaining, stride_val)
        else:
            c = remaining / arith.index(stride_val)
        if size_val is not None:
            if _is_pow2(size_val):
                c = _mod_pow2(c, size_val)
            else:
                c = c % arith.index(size_val)
        coords[i] = c
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


def crd2idx(crd, layout):
    if not isinstance(crd, (list, tuple)):
        crd = [crd]
    parsed = _parse_layout(layout)
    assert parsed is not None and not _has_dynamic_strides(parsed[1]), "static layouts only"
    _, strides = parsed
    result = None
    for coord_v, stride_v in _builtins.zip(crd, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * arith.index(stride_v)
        result = term if result is None else result + term
    return result if result is not None else arith.index(0)


def layout_get(int_tuple, mode):
    return int_tuple[mode]


def swizzle_xor16(row, col, k_blocks16):
    """col XOR ((row & (k_blocks16 - 1)) * 16): 16-B-granular XOR swizzle on K."""
    mask = k_blocks16 - arith.index(1)
    rem = arith.andi(row, mask)
    return col ^ (rem * 16)


def _buffer_load_vec(rsrc, idx, *, elem_type, vec_elems, elem_bytes, offset_in_bytes, cache_modifier=0):
    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4
    if offset_in_bytes:
        idx_i32 = arith.shrui(idx, arith.index(2))
    elif elem_bytes == 2:
        idx_i32 = arith.shrui(idx, arith.index(1))
    else:
        idx_i32 = idx
    i32_val = buffer_ops.buffer_load(rsrc, idx_i32, vec_width=vec_width, dtype=T.i32, cache_modifier=cache_modifier)
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


def tile_chunk_coord_i32(*, tx_i32_base, i, total_threads, layout_tile_div4, chunk_i32=4):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for the A tile DMA."""
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = fx.idx2crd(fx.Int32(tile_idx_i32), layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
    return row_local, col_local_i32


def _ptr_buffer_resource(ptr, num_records_bytes):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)


def _raw_rsrc(fat):
    return fx.as_ir_value(fx.rocdl.get_buffer_rsrc(fat))


def _extract_global_ptr(ptr):
    addr = fx.ptrtoint(ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), addr_i64)


def _global_load_i32(global_ptr, elem_offset):
    if isinstance(elem_offset, int):
        elem_offset = arith.constant(elem_offset, type=T.i32)
    raw = elem_offset.ir_value() if hasattr(elem_offset, "ir_value") else elem_offset
    if isinstance(raw.type, ir.IndexType):
        off_i64 = arith.index_cast(T.i64, raw)
    else:
        int_type = ir.IntegerType(raw.type)
        off_i64 = ArithValue(raw) if int_type.width == 64 else ArithValue(arith.ExtSIOp(T.i64, raw).result)
    byte_offset_i64 = off_i64 * arith.constant(4, type=T.i64)
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=byte_offset_i64, elem_type=T.i8)
    return ArithValue(llvm.LoadOp(T.i32, ptr, alignment=4).result)


def _dpp_xor_f32(src, offset: int, *, bound_ctrl: bool = True):
    src_i32 = src.bitcast(T.i32) if hasattr(src, "bitcast") else ArithValue(src).bitcast(T.i32)
    dpp_ctrl = {1: 0xB1, 2: 0x4E}[offset]
    out_i32 = llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.update.dpp.i32",
        [
            src_i32,
            src_i32,
            arith.constant(dpp_ctrl, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(0xF, type=T.i32),
            arith.constant(bound_ctrl, type=ir.IntegerType.get_signless(1)),
        ],
        [],
        [],
    )
    return ArithValue(out_i32).bitcast(T.f32)


def ptr_arg(t: torch.Tensor):
    """torch tensor -> fx.Pointer argument (byte pointer)."""
    v = t.view(torch.uint8) if t.element_size() == 1 and t.dtype != torch.uint8 else t
    return flyc.from_c_void_p(fx.Uint8, v.data_ptr())


# ---------------------------------------------------------------------------
def compile_moe_gemm1_1x4(*, H: int, I: int, E: int, tile_m: int = 128):
    """Returns the jit launcher. Args (all byte pointers unless noted):
    out_q, out_scale, a_quant, w13, a_scale_sorted, w13_scale,
    sorted_expert_ids (i32), sorted_ids (i32), tile_map (i32),
    n_tokens, num_m_blocks, a_scale_bytes, grid_size (ints), stream."""
    MODEL_DIM, INTER_DIM, EXPERTS = H, I, E
    tile_n = 256  # 128 gate + 128 up columns of one n-tile
    tile_k = 256
    total_threads = 256
    num_n_blocks = INTER_DIM // 128
    assert INTER_DIM % 128 == 0 and MODEL_DIM % tile_k == 0 and tile_m == 128

    _scale_pack_m = 2
    _scale_pack_n = 2
    pack_M = 2
    pack_N = 2
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves  # 64
    num_acc_n = n_per_wave // 16  # 4
    k_unroll = tile_k // 128
    m_repeat_packed = m_repeat // pack_M
    num_acc_n_packed = num_acc_n // pack_N
    num_k_tiles = MODEL_DIM // tile_k

    a_elem_vec_pack = 2
    _eff_lds_stride = tile_k // a_elem_vec_pack
    _eff_tile_k_bytes = tile_k // a_elem_vec_pack
    _single_x_bytes = tile_m * _eff_lds_stride
    _x_slots_bytes = 2 * _single_x_bytes
    _lds_scale_bytes = (tile_m // 32) * num_k_tiles * 256
    _kloop_lds_bytes = _x_slots_bytes + _lds_scale_bytes
    _lds_acc_bytes = tile_m * tile_n * 4

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_m3_gemm1_1x4")
    lds_offset = allocator._align(allocator.ptr, 16)
    lds_scale_offset = lds_offset + _x_slots_bytes
    allocator.ptr = lds_offset + max(_kloop_lds_bytes, _lds_acc_bytes)

    K_BYTES = MODEL_DIM // 2
    w_nbytes = EXPERTS * (2 * INTER_DIM) * K_BYTES
    w_scale_nbytes = EXPERTS * (2 * INTER_DIM) * (MODEL_DIM // 32)

    # preshuffled (16,16) W13: per 16-row block, per 64-B K chunk, 1024 B
    _b_stride_nlane = 16
    _b_stride_klane = 16 * _b_stride_nlane
    _b_stride_k0 = 4 * _b_stride_klane
    _b_stride_n0 = (K_BYTES // 64) * _b_stride_k0
    _expert_b_stride = ((2 * INTER_DIM) // 16) * _b_stride_n0

    # e8m0-shuffled scales: one 32-row chunk = (K/32) bytes x 32 rows, dwords:
    k_as_per_chunk_dw = ((MODEL_DIM // 32) // 4 // 2) * 64
    k_bs_stride_k0_dw = 64
    k_bs_stride_n0_dw = k_as_per_chunk_dw
    k_bs_per_expert_dw = (((2 * INTER_DIM) // 16) // 2) * k_bs_stride_n0_dw
    k_out_as_per_chunk_dw = ((INTER_DIM // 32) // 4 // 2) * 64
    scale_chunk_bytes_py = k_as_per_chunk_dw * 4
    _n_dma16 = scale_chunk_bytes_py // 4096
    _n_dma4 = (scale_chunk_bytes_py - _n_dma16 * 4096) // 1024
    assert _n_dma16 * 4096 + _n_dma4 * 1024 == scale_chunk_bytes_py

    module_name = f"m3_gemm1_1x4_E{EXPERTS}_H{MODEL_DIM}_I{INTER_DIM}_BM{tile_m}"

    @flyc.kernel(name=module_name)
    def gemm1(
        arg_out_q: fx.Pointer,
        arg_out_scale: fx.Pointer,
        arg_a_quant: fx.Pointer,
        arg_w: fx.Pointer,
        arg_a_scale_sorted: fx.Pointer,
        arg_w_scale: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_ids: fx.Pointer,
        arg_tile_map: fx.Pointer,
        arg_n_tokens: fx.Int32,
        arg_num_m_blocks: fx.Int32,
        arg_a_scale_bytes: fx.Int32,
        arg_grid: fx.Int32,
    ):
        i32 = T.i32
        i64 = T.i64
        f32 = T.f32
        vec4_f32 = T.vec(4, f32)
        vec2_i64 = T.vec(2, i64)
        vec4_i32 = T.vec(4, i32)
        vec16_x = T.vec(16, T.i8)
        idx_t = ir.IndexType.get()

        n_tokens_i32 = ArithValue(fx.as_ir_value(arg_n_tokens))
        num_m_blocks_i32 = ArithValue(fx.as_ir_value(arg_num_m_blocks))
        a_scale_bytes_i32 = ArithValue(fx.as_ir_value(arg_a_scale_bytes))
        grid_i32 = ArithValue(fx.as_ir_value(arg_grid))
        n_tokens_idx = arith.index_cast(idx_t, n_tokens_i32)
        max_rows_idx = arith.index_cast(idx_t, num_m_blocks_i32) * arith.constant(tile_m, index=True)

        x_nbytes = n_tokens_idx * arith.constant(K_BYTES, index=True)
        out_q_nbytes = max_rows_idx * arith.constant(INTER_DIM // 2, index=True)
        out_scale_nbytes = max_rows_idx * arith.constant(INTER_DIM // 32, index=True)

        out_q_rsrc = _ptr_buffer_resource(arg_out_q, out_q_nbytes)
        out_scale_rsrc = _ptr_buffer_resource(arg_out_scale, out_scale_nbytes)
        x_rsrc = _raw_rsrc(_ptr_buffer_resource(arg_a_quant, x_nbytes))
        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)
        sx_rsrc = _raw_rsrc(_ptr_buffer_resource(arg_a_scale_sorted, arith.index_cast(idx_t, a_scale_bytes_i32)))
        sw_rsrc = _ptr_buffer_resource(arg_w_scale, w_scale_nbytes)
        # sorted_ids has max_num_tokens_padded entries (< num_m_blocks * tile_m): the last
        # tile's tail rows read past it -> buffer OOB -> 0 (they are padding rows anyway)
        ids_rsrc = _ptr_buffer_resource(arg_sorted_ids, max_rows_idx * arith.constant(4, index=True))
        eid_rsrc = _ptr_buffer_resource(arg_expert_ids, arith.index_cast(idx_t, num_m_blocks_i32) * arith.constant(4, index=True))
        tile_map_ptr = _extract_global_ptr(arg_tile_map)

        tx = gpu.thread_id("x")
        bid = gpu.block_id("x")
        # contiguous chunk of the tile map per XCD (grid % 8 == 0)
        c8_idx = arith.constant(8, index=True)
        grid_idx = arith.index_cast(idx_t, grid_i32)
        remapped = (bid % c8_idx) * (grid_idx / c8_idx) + bid / c8_idx
        entry = _global_load_i32(tile_map_ptr, remapped)
        entry_ok = arith.cmpi(CmpIPredicate.sge, entry, arith.constant(0, type=i32))
        # blocks with nothing to do (-1) read tile 0's routing (harmless) and skip the body
        entry_safe = arith.select(entry_ok, entry, arith.constant(0, type=i32))
        bx = arith.index_cast(idx_t, entry_safe >> arith.constant(3, type=i32))  # m-tile
        by = arith.index_cast(idx_t, entry_safe & arith.constant(7, type=i32))  # n-tile
        bx_m = bx * arith.constant(tile_m, index=True)

        expert_i32 = ArithValue(buffer_ops.buffer_load(eid_rsrc, bx, vec_width=1, dtype=i32))
        expert_idx = arith.index_cast(idx_t, expert_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(EXPERTS, type=i32))
        do_gemm = arith.andi(entry_ok, exp_valid)

        base_ptr = allocator.get_base()
        lds_x0 = SmemPtr(base_ptr, lds_offset, T.i8, shape=(_single_x_bytes,)).get()
        lds_x1 = SmemPtr(base_ptr, lds_offset + _single_x_bytes, T.i8, shape=(_single_x_bytes,)).get()
        lds_x_slots = [lds_x0, lds_x1]
        lds_scale_i8 = SmemPtr(base_ptr, lds_scale_offset, T.i8, shape=(_lds_scale_bytes,)).get()
        lds_scale_i32 = SmemPtr(base_ptr, lds_scale_offset, T.i32, shape=(_lds_scale_bytes // 4,)).get()
        lds_acc = SmemPtr(base_ptr, lds_offset, T.f32, shape=(tile_m * tile_n,)).get()

        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = idx2crd(tx, layout_tx_wave_lane)
        wave_id = layout_get(coord_wl, 0)
        lane_id = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * arith.constant(16, index=True)

        # This wave's first W13 row: waves 0/1 -> gate rows by*128 + w*64,
        # waves 2/3 -> up rows I + by*128 + (w-2)*64 (production separated layout).
        c2_idx = arith.constant(2, index=True)
        is_gate = arith.cmpi(CmpIPredicate.ult, wave_id, c2_idx)
        gate_row0 = by * arith.constant(128, index=True) + wave_id * arith.constant(64, index=True)
        up_row0 = (
            arith.constant(INTER_DIM, index=True)
            + by * arith.constant(128, index=True)
            + (wave_id - c2_idx) * arith.constant(64, index=True)
        )
        wave_n_base = arith.select(is_gate, gate_row0, up_row0)
        expert_b_base = expert_idx * arith.constant(_expert_b_stride, index=True)

        n_intra_list = [None] * num_acc_n
        n_blk_list = [None] * num_acc_n
        for ni in range_constexpr(num_acc_n):
            global_n = wave_n_base + arith.constant(ni * 16, index=True) + lane_mod_16
            n_blk_list[ni] = _div_pow2(global_n, 16)
            n_intra_list[ni] = global_n % arith.constant(16, index=True)

        bytes_per_thread_x = tile_m * tile_k // a_elem_vec_pack // total_threads
        x_load_bytes = 16
        num_x_loads = bytes_per_thread_x // x_load_bytes
        chunk_i32 = x_load_bytes // 4
        tile_k_dwords = tile_k // (4 * a_elem_vec_pack)
        layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
        c_chunk_i32 = arith.constant(chunk_i32, index=True)
        tx_i32_base = tx * c_chunk_i32
        c_k_div4 = arith.constant(K_BYTES // 4, index=True)

        x_row_base_div4 = []
        x_col_local_i32 = []
        x_row_local = []
        for i in range_constexpr(num_x_loads):
            row_local, col_local_i32 = tile_chunk_coord_i32(
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_x_tile_div4,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(row_local)
            x_col_local_i32.append(col_local_i32)
            sorted_row_i = bx_m + row_local
            packed = ArithValue(buffer_ops.buffer_load(ids_rsrc, sorted_row_i, vec_width=1, dtype=i32))
            actual_row = packed & arith.constant(0x00FFFFFF, type=i32)
            row_ok = arith.cmpi(CmpIPredicate.ult, actual_row, n_tokens_i32)
            # padded rows read one past the A buffer: MUBUF OOB returns zeros
            row_safe = arith.select(row_ok, actual_row, n_tokens_i32)
            row_idx = arith.index_cast(idx_t, row_safe)
            x_row_base_div4.append(row_idx * c_k_div4)

        def load_b_packs_k64(k_tile: int, ku: int, ni: int):
            k0 = arith.constant(k_tile * k_unroll + ku, index=True)
            idx_pack = (
                expert_b_base
                + n_blk_list[ni] * arith.constant(_b_stride_n0, index=True)
                + k0 * arith.constant(_b_stride_k0, index=True)
                + lane_div_16 * arith.constant(_b_stride_klane, index=True)
                + n_intra_list[ni] * arith.constant(_b_stride_nlane, index=True)
            )
            b16 = _buffer_load_vec(
                w_rsrc, idx_pack, elem_type=T.i8, vec_elems=16, elem_bytes=1, offset_in_bytes=True, cache_modifier=0
            )
            b_i64x2 = vector.bitcast(vec2_i64, b16)
            return (
                vector.extract(b_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(b_i64x2, static_position=[1], dynamic_position=[]),
            )

        def load_b_tile(k_tile: int):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(k_tile, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        def make_empty_b_tile():
            b_packs0 = []
            b_packs1 = []
            for _ku in range_constexpr(k_unroll):
                b_packs0.append([None] * num_acc_n)
                b_packs1.append([None] * num_acc_n)
            return b_packs0, b_packs1

        def finish_b_tile(b_packs0, b_packs1):
            return [(b_packs0[ku], b_packs1[ku]) for ku in range(k_unroll)]

        lane_word = lane_div_16 * arith.constant(16, index=True) + lane_mod_16

        def load_a_scale_tile(k_tile: int):
            a_scale_tile = []
            for mi in range_constexpr(m_repeat_packed):
                scale_idx = (
                    arith.constant(mi * k_as_per_chunk_dw, index=True)
                    + arith.constant(k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = vector.extract(
                    vector.load(T.vec(1, i32), lds_scale_i32, [scale_idx]),
                    static_position=[0],
                    dynamic_position=[],
                )
                a_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return a_scale_tile

        # W scale 32-row chunk index of this wave's first row (multiple of 32)
        mni_base = wave_n_base / arith.constant(32, index=True)

        def load_b_scale_tile(k_tile: int):
            b_scale_tile = []
            for ni in range_constexpr(num_acc_n_packed):
                scale_idx = (
                    expert_idx * arith.constant(k_bs_per_expert_dw, index=True)
                    + (mni_base + arith.constant(ni, index=True)) * arith.constant(k_bs_stride_n0_dw, index=True)
                    + arith.constant(k_tile * k_bs_stride_k0_dw, index=True)
                    + lane_word
                )
                s = buffer_ops.buffer_load(sw_rsrc, scale_idx, vec_width=1, dtype=i32)
                b_scale_tile.append(vector.from_elements(T.vec(1, i32), [s]))
            return b_scale_tile

        def lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base):
            col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
            loaded_a16 = vector.load(vec16_x, lds_x_tile, [idx_a16])
            a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
            return (
                vector.extract(a_i64x2, static_position=[0], dynamic_position=[]),
                vector.extract(a_i64x2, static_position=[1], dynamic_position=[]),
            )

        def pack_i64x2_to_i32x4(x0, x1):
            v2 = vector.from_elements(vec2_i64, [x0, x1])
            return vector.bitcast(vec4_i32, v2)

        def _raw(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        def mfma_scale_agpr_init(a128_v4, b128_v4, op_sel_a, scale_a, op_sel_b, scale_b):
            alo, ahi = op_sel_a & 1, (op_sel_a >> 1) & 1
            blo, bhi = op_sel_b & 1, (op_sel_b >> 1) & 1
            asm = (
                "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 "
                f"op_sel:[{alo},{blo},0] op_sel_hi:[{ahi},{bhi},0] cbsz:4 blgp:4"
            )
            return llvm.inline_asm(
                vec4_f32,
                [_raw(a128_v4), _raw(b128_v4), _raw(scale_a), _raw(scale_b)],
                asm,
                "=a,v,v,v,v",
                has_side_effects=True,
            )

        def mfma_scale_agpr_accum(a128_v4, b128_v4, acc, op_sel_a, scale_a, op_sel_b, scale_b):
            alo, ahi = op_sel_a & 1, (op_sel_a >> 1) & 1
            blo, bhi = op_sel_b & 1, (op_sel_b >> 1) & 1
            asm = (
                "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 "
                f"op_sel:[{alo},{blo},0] op_sel_hi:[{ahi},{bhi},0] cbsz:4 blgp:4"
            )
            return llvm.inline_asm(
                vec4_f32,
                [_raw(a128_v4), _raw(b128_v4), _raw(scale_a), _raw(scale_b), _raw(acc)],
                asm,
                "=a,v,v,v,v,0",
                has_side_effects=True,
            )

        def _read_a_pairs(lds_x_tile):
            a_pairs_v4 = []
            for k_idx in range_constexpr(k_unroll):
                col_base = col_offset_base + arith.constant(k_idx * 128 // a_elem_vec_pack, index=True)
                for mi_idx in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(lds_x_tile, curr_row_a_lds, col_base)
                    a_pairs_v4.append(pack_i64x2_to_i32x4(a0, a1))
            return a_pairs_v4

        def _mfma_jmajor(acc_list, a_pairs_v4, b_tile_in, a_scale, b_scale, init_zero, after_ni=None):
            for ni in range_constexpr(num_acc_n_packed):
                b_scale_val = vector.extract(b_scale[ni], static_position=[0], dynamic_position=[])
                for inxdl in range_constexpr(pack_N):
                    ni_idx = ni * pack_N + inxdl
                    for mi in range_constexpr(m_repeat_packed):
                        a_scale_val = vector.extract(a_scale[mi], static_position=[0], dynamic_position=[])
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi * pack_M + imxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            for k_idx in range_constexpr(k_unroll):
                                b_packs0, b_packs1 = b_tile_in[k_idx]
                                op_sel_a = k_idx * _scale_pack_m + imxdl
                                op_sel_b = k_idx * _scale_pack_n + inxdl
                                b128_v4 = pack_i64x2_to_i32x4(b_packs0[ni_idx], b_packs1[ni_idx])
                                a128_v4 = a_pairs_v4[k_idx * m_repeat + mi_idx]
                                if const_expr(init_zero and k_idx == 0):
                                    acc_list[acc_idx] = mfma_scale_agpr_init(
                                        a128_v4, b128_v4, op_sel_a, a_scale_val, op_sel_b, b_scale_val
                                    )
                                else:
                                    acc_list[acc_idx] = mfma_scale_agpr_accum(
                                        a128_v4, b128_v4, acc_list[acc_idx], op_sel_a, a_scale_val, op_sel_b, b_scale_val
                                    )
                    if after_ni is not None:
                        after_ni(ni_idx)
            return acc_list

        def compute_tile_jmajor(acc_in, lds_x_tile, b_tile_in, a_scale, b_scale, init_zero=False):
            acc_list = list(acc_in)
            a_pairs_v4 = _read_a_pairs(lds_x_tile)
            return _mfma_jmajor(acc_list, a_pairs_v4, b_tile_in, a_scale, b_scale, init_zero)

        def compute_tile_jmajor_prefetch_a_b(
            acc_in, lds_x_tile, b_tile_in, a_scale, b_scale, next_a_tile, next_a_slot, next_b_tile, init_zero=False
        ):
            acc_list = list(acc_in)
            next_b_packs0, next_b_packs1 = make_empty_b_tile()
            a_pairs_v4 = _read_a_pairs(lds_x_tile)
            dma_x_tile_to_lds(next_a_tile, lds_x_slots[next_a_slot])

            def _issue_next_b(ni_idx):
                rocdl.sched_barrier(0)
                for k_idx in range_constexpr(k_unroll):
                    nb0, nb1 = load_b_packs_k64(next_b_tile, k_idx, ni_idx)
                    next_b_packs0[k_idx][ni_idx] = nb0
                    next_b_packs1[k_idx][ni_idx] = nb1
                rocdl.sched_barrier(0)

            acc_list = _mfma_jmajor(acc_list, a_pairs_v4, b_tile_in, a_scale, b_scale, init_zero, after_ni=_issue_next_b)
            b_scale_next = load_b_scale_tile(next_b_tile)
            return acc_list, finish_b_tile(next_b_packs0, next_b_packs1), b_scale_next

        _dma_bytes = 16
        _wave_size = 64

        def dma_x_tile_to_lds(k_tile: int, lds_x_tile):
            c4_idx = arith.index(4)
            base_k_div4 = arith.constant(k_tile * tile_k // a_elem_vec_pack // 4, index=True)
            lds_ptr_i64 = None
            for i in range_constexpr(num_x_loads):
                row_local_i = x_row_local[i]
                col_local_i32_i = x_col_local_i32[i]
                col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                row_k_dw = x_row_base_div4[i] + base_k_div4
                global_byte_idx = row_k_dw * c4_idx + col_local_sw
                global_offset = arith.index_cast(i32, global_byte_idx)
                if const_expr(i == 0):
                    lds_addr = memref.extract_aligned_pointer_as_index(lds_x_tile) + wave_id * arith.constant(
                        _wave_size * _dma_bytes, index=True
                    )
                    lds_ptr_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr))
                else:
                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=i64)
                lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)
                rocdl.raw_ptr_buffer_load_lds(
                    x_rsrc,
                    lds_ptr,
                    arith.constant(_dma_bytes, type=i32),
                    global_offset,
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                    arith.constant(0, type=i32),
                )

        def dma_a_scales_to_lds():
            """The m-tile's A scales for all K (tile_m/32 chunks x K/32 bytes x 32 rows) -> LDS once."""
            c4_idx = arith.index(4)
            c16_idx = arith.index(16)
            scale_chunk_bytes = arith.constant(scale_chunk_bytes_py, index=True)
            chunk_base = bx_m / arith.constant(32, index=True)
            lds_scale_base = memref.extract_aligned_pointer_as_index(lds_scale_i8)
            tx_bytes16 = tx * c16_idx
            tx_bytes4 = tx * c4_idx
            for sub in range_constexpr(tile_m // 32):
                sub_idx = arith.constant(sub, index=True)
                global_sub_base = (chunk_base + sub_idx) * scale_chunk_bytes
                lds_sub_base = sub_idx * scale_chunk_bytes
                for p in range_constexpr(_n_dma16):
                    byte_off = arith.constant(p * 4096, index=True)
                    lds_addr = lds_scale_base + lds_sub_base + byte_off + wave_id * arith.constant(1024, index=True)
                    lds_ptr_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr))
                    lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)
                    global_offset = arith.index_cast(i32, global_sub_base + byte_off + tx_bytes16)
                    rocdl.raw_ptr_buffer_load_lds(
                        sx_rsrc,
                        lds_ptr,
                        arith.constant(16, type=i32),
                        global_offset,
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                    )
                for tail in range_constexpr(_n_dma4):
                    byte_off = arith.constant(_n_dma16 * 4096 + tail * 1024, index=True)
                    lds_addr_tail = lds_scale_base + lds_sub_base + byte_off + wave_id * arith.constant(256, index=True)
                    lds_ptr_tail_i64 = rocdl.readfirstlane(i64, arith.index_cast(i64, lds_addr_tail))
                    lds_ptr_tail = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_tail_i64)
                    global_offset_tail = arith.index_cast(i32, global_sub_base + byte_off + tx_bytes4)
                    rocdl.raw_ptr_buffer_load_lds(
                        sx_rsrc,
                        lds_ptr_tail,
                        arith.constant(4, type=i32),
                        global_offset_tail,
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                        arith.constant(0, type=i32),
                    )

        def _then_body():
            acc = [arith.constant_vector(0.0, vec4_f32)] * (m_repeat * num_acc_n)
            dma_a_scales_to_lds()
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

            dma_x_tile_to_lds(0, lds_x_slots[0])
            dma_x_tile_to_lds(1, lds_x_slots[1])
            b_tiles = [load_b_tile(0), load_b_tile(1)]
            b_scales = [load_b_scale_tile(0), load_b_scale_tile(1)]
            # B1 + its scales (8 + 2 loads) may still fly; A0/A1 must be in LDS.
            rocdl.s_waitcnt(vmcnt=20)
            gpu.barrier()
            a_scale = load_a_scale_tile(0)

            for k_tile in range_constexpr(num_k_tiles):
                curr_slot = k_tile % 2
                if const_expr(k_tile + 2 < num_k_tiles):
                    acc, b_tile_next, b_scale_next = compute_tile_jmajor_prefetch_a_b(
                        acc,
                        lds_x_slots[curr_slot],
                        b_tiles[curr_slot],
                        a_scale,
                        b_scales[curr_slot],
                        k_tile + 2,
                        curr_slot,
                        k_tile + 2,
                        init_zero=(k_tile == 0),
                    )
                else:
                    acc = compute_tile_jmajor(
                        acc, lds_x_slots[curr_slot], b_tiles[curr_slot], a_scale, b_scales[curr_slot], init_zero=(k_tile == 0)
                    )
                if const_expr(k_tile + 1 < num_k_tiles):
                    # keep the newest future tile's 4 A + 10 B/B-scale loads outstanding
                    if const_expr(k_tile + 2 < num_k_tiles):
                        rocdl.s_waitcnt(vmcnt=14)
                    else:
                        rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    a_scale = load_a_scale_tile(k_tile + 1)
                    if const_expr(k_tile + 2 < num_k_tiles):
                        b_tiles[curr_slot] = b_tile_next
                        b_scales[curr_slot] = b_scale_next

            gpu.barrier()

            # ---- epilogue: accumulators -> LDS (f32), then per thread 8 cols x 8 row-groups ----
            c0_f32 = arith.constant(0.0, type=f32)
            c1_f32 = arith.constant(1.0, type=f32)
            c_lim = arith.constant(SWIGLU_LIMIT, type=f32)
            c_neg_lim = arith.constant(-SWIGLU_LIMIT, type=f32)
            c_neg_alpha_log2e = arith.constant(-SWIGLU_ALPHA * 1.4426950408889634, type=f32)
            c0_i32 = arith.constant(0, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c8_i32 = arith.constant(8, type=i32)
            c23_i32 = arith.constant(23, type=i32)
            c_half_i32 = arith.constant(0x400000, type=i32)
            c_expmask_i32 = arith.constant(0xFF800000, type=i32)
            fm_fast = arith.FastMathFlags.fast

            def _fmax_num(a, b):
                return ArithValue(arith.MaxNumFOp(arith._to_raw(a), arith._to_raw(b), fastmath=fm_fast).result)

            for mi in range_constexpr(m_repeat):
                row_base = arith.constant(mi * 16, index=True) + lane_div_16 * arith.constant(4, index=True)
                for j in range_constexpr(num_acc_n):
                    # wave w's 64 columns: gate waves -> cols [w*64, +64), up waves -> 128 + [(w-2)*64, +64)
                    col_local = wave_id * arith.constant(64, index=True) + arith.constant(j * 16, index=True) + lane_mod_16
                    for v_i in range_constexpr(4):
                        row = row_base + arith.constant(v_i, index=True)
                        val = vector.extract(acc[mi * num_acc_n + j], static_position=[v_i], dynamic_position=[])
                        vector.store(
                            vector.from_elements(T.vec(1, f32), [val]),
                            lds_acc,
                            [row * arith.constant(tile_n, index=True) + col_local],
                            alignment=4,
                        )

            gpu.barrier()

            m_lane = tx / arith.constant(16, index=True)
            n_lane = tx % arith.constant(16, index=True)
            wave_grp = n_lane / arith.constant(4, index=True)  # 32-col group 0..3
            kk = n_lane % arith.constant(4, index=True)  # 8-col chunk within it
            scales_per_mr = []

            def _swiglu_oai(g, u):
                g = arith.minimumf(g, c_lim)
                u = arith.maximumf(arith.minimumf(u, c_lim), c_neg_lim)
                t = g * c_neg_alpha_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = c1_f32 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return g * sig * (u + c1_f32)

            for mr in range_constexpr(m_repeat):
                row_local = arith.constant(mr * 16, index=True) + m_lane
                result_vals = []
                local_max = c0_f32
                for e in range_constexpr(8):
                    col_in_grp = kk * arith.constant(8, index=True) + arith.constant(e, index=True)
                    gate_col = wave_grp * arith.constant(32, index=True) + col_in_grp
                    up_col = gate_col + arith.constant(128, index=True)
                    gate_v = vector.extract(
                        vector.load(T.vec(1, f32), lds_acc, [row_local * arith.constant(tile_n, index=True) + gate_col]),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    up_v = vector.extract(
                        vector.load(T.vec(1, f32), lds_acc, [row_local * arith.constant(tile_n, index=True) + up_col]),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    res = _swiglu_oai(gate_v, up_v)
                    result_vals.append(res)
                    abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [res], [], [])
                    local_max = _fmax_num(local_max, abs_v)

                peer1 = _dpp_xor_f32(local_max, 1)
                local_max = _fmax_num(local_max, peer1)
                peer2 = _dpp_xor_f32(local_max, 2)
                local_max = _fmax_num(local_max, peer2)

                # aiter's fused FlyDSL stage-1 rule: nearest pow2 of amax, exponent - 2, floor 0
                amax_i32 = local_max.bitcast(i32)
                rounded = (amax_i32 + c_half_i32) & c_expmask_i32
                e8m0 = arith.maxsi((rounded >> c23_i32) - c2_i32, c0_i32)
                scales_per_mr.append(e8m0)
                quant_scale = (e8m0 << c23_i32).bitcast(f32)

                packed = c0_i32
                for pair in range_constexpr(4):
                    packed = ArithValue(
                        llvm.call_intrinsic(
                            i32,
                            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                            [packed, result_vals[2 * pair], result_vals[2 * pair + 1], quant_scale, arith.constant(pair, type=i32)],
                            [],
                            [],
                        )
                    )

                row_i32 = arith.index_cast(i32, bx_m + row_local)
                byte_pos = (
                    by * arith.constant(64, index=True)
                    + wave_grp * arith.constant(16, index=True)
                    + kk * arith.constant(4, index=True)
                )
                byte_pos_i32 = arith.index_cast(i32, byte_pos)
                q_word_off = row_i32 * arith.constant((INTER_DIM // 2) // 4, type=i32) + (byte_pos_i32 >> c2_i32)
                buffer_ops.buffer_store(packed, out_q_rsrc, q_word_off, cache_modifier=4)

            _if_scale = scf.IfOp(arith.cmpi(CmpIPredicate.eq, kk, arith.constant(0, index=True)))
            with ir.InsertionPoint(_if_scale.then_block):
                ku = by >> arith.constant(1, index=True)  # 8-col-group index of this n-tile
                ikxdl = by % arith.constant(2, index=True)  # q within the block
                for sub in range_constexpr(tile_m // 32):
                    chunk = bx * arith.constant(tile_m // 32, index=True) + arith.constant(sub, index=True)
                    dword_off = (
                        chunk * arith.constant(k_out_as_per_chunk_dw, index=True)
                        + ku * arith.constant(64, index=True)
                        + wave_grp * arith.constant(16, index=True)
                        + m_lane
                    )
                    lo = scales_per_mr[sub * 2]
                    hi = scales_per_mr[sub * 2 + 1]
                    pair_i32 = lo | (hi << c8_i32)
                    pair_i16 = arith.TruncIOp(T.i16, pair_i32).result
                    byte_off = arith.index_cast(
                        i32, dword_off * arith.constant(4, index=True) + ikxdl * arith.constant(2, index=True)
                    )
                    buffer_ops.buffer_store(pair_i16, out_scale_rsrc, byte_off, cache_modifier=4, offset_is_bytes=True)
                scf.YieldOp([])

        _if_valid = scf.IfOp(do_gemm)
        with ir.InsertionPoint(_if_valid.then_block):
            _then_body()
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm1(
        out_q: fx.Pointer,
        out_scale: fx.Pointer,
        a_quant: fx.Pointer,
        w: fx.Pointer,
        a_scale_sorted: fx.Pointer,
        w_scale: fx.Pointer,
        expert_ids: fx.Pointer,
        sorted_ids: fx.Pointer,
        tile_map: fx.Pointer,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm1(
            out_q,
            out_scale,
            a_quant,
            w,
            a_scale_sorted,
            w_scale,
            expert_ids,
            sorted_ids,
            tile_map,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            grid_size,
        ).launch(grid=(grid_size, 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_gemm1
