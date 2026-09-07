# SPDX-License-Identifier: Apache-2.0
"""Helpers the lab-only mid-batch kernels need that are not part of the vLLM
packages (the production helpers come from ``moe_a4w4_prefill.gemm1`` /
``moe_a16w4_decode.gemm2``)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from m3_a16w4_moe.vllm_ops import import_ops

import_ops("moe_a16w4_decode")
from moe_a16w4_decode.utils import _gep1, _gep3, _global_base_ptr1, _lds_ptr3  # noqa: E402


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def s_waitcnt_lgkm0():
    """``s_waitcnt lgkmcnt(0)`` (vmcnt/expcnt left at max); CDNA simm16 encoding."""
    return rocdl.s_waitcnt(0xC07F)


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that reads them, so the reads
    that follow cannot be hoisted above the MFMAs (see acc_hazard.py: after an
    inline-asm MFMA the compiler sees no dependency and moves a plain read above a
    dependency-free s_nop fence)."""
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>")
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(ty, [_raw(v) for v in values], "s_nop 15\ns_nop 15", constraints, has_side_effects=True)
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


# @flyc.jit is load-bearing: it AST-rewrites ``if token_id < i32_M`` into an scf.if.
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    pre_packed=None,
    pre_weight=None,
):
    """General-BM (32/64) version of the decode gemm2 atomic epilogue: accm[i][J]
    (f32[4] per lane, MFMA C layout, i = 16-row chunk) -> LDS [BM, BN] f32 -> per row:
    2 columns per lane x routing weight -> packed bf16 atomic add at out[token, col].
    The vLLM copy (``moe_a16w4_decode.gemm2._atomic_bf16_epilog``) is the BM=16
    special case; this one is only used by the lab mid-batch gemm2."""
    _kMChunks = BM // 16
    M_REPS = BM // 8
    _n_per_wave = BN // 4  # 4 waves split the BN tile
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed, weight = [], []
    if pre_packed is not None:
        packed, weight = pre_packed, pre_weight  # issued in the kernel prologue
    else:
        for mr in range_constexpr(M_REPS):
            sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
            packed.append(llvm.load(T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True))
            weight.append(llvm.load(T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True))

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, off),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
