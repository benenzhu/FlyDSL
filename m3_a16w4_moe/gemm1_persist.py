# SPDX-License-Identifier: Apache-2.0
"""Persistent decode gate/up GEMM, bf16 x MXFP4, for MiniMax-M3.

The work list is the same (sorted m-block, 32-column n-block) list as gemm1.py.
Each CTA walks it with a fixed stride. A/W/scale VGPR prefetch carries across
items: the last three K steps issue the next item's first three steps before
the slice-K reduction and swiglu epilogue. Only the first item has a cold ring.

This supports the validated decode geometry, not every format/routing mode
of the generic kernel. Ring reads use side-effecting asm and explicit vmcnt
accounting so LLVM does not drain them at the work-loop back edge.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .gemm1 import _swigluoai_f32
from .utils import (
    _e8m0_byte_to_f32, _gep3, _global_i32_at, _lds_ptr3, _raw, buffer_ops,
)


def _load(rsrc, offset, *, width=4, cache=0):
    ty = T.i32 if width == 1 else T.vec(4, T.i32)
    op = "buffer_load_dword" if width == 1 else "buffer_load_dwordx4"
    return llvm.inline_asm(
        ty, [_raw(offset), fx.as_ir_value(fx.rocdl.get_buffer_rsrc(rsrc))],
        f"{op} $0, $1, $2, 0 offen" + (" nt" if cache == 2 else ""),
        "=v,v,s", has_side_effects=True,
    )


def _wait_values(vals, count):
    vals = [_raw(v) for v in vals]
    # The scale dwords are shared by adjacent tiles. Repeating an input in
    # tied asm outputs makes register allocation COPY it before the wait,
    # reading an unfinished load. Pin each SSA value once, then alias outputs.
    unique, indices, keys = [], [], {}
    for v in vals:
        # DSL values overload equality and string formatting. The inherited
        # MLIR hash/identity comparison still identifies the underlying SSA.
        key = hash(v)
        if key not in keys:
            keys[key] = len(unique)
            unique.append(v)
        assert ir.Value.__eq__(unique[keys[key]], v)
        indices.append(keys[key])
    types = [v.type for v in unique]
    ty = types[0] if len(types) == 1 else ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in types) + ")>")
    constraints = ",".join(["=v"] * len(unique) + [str(i) for i in range(len(unique))])
    res = llvm.inline_asm(ty, unique, f"s_waitcnt vmcnt({count})", constraints, has_side_effects=True)
    pinned = [res] if len(types) == 1 else [llvm.extractvalue(t, res, [i]) for i, t in enumerate(types)]
    return [pinned[i] for i in indices]


def _pin_scalar(vals):
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join("i32" for _ in vals) + ")>")
    constraints = ",".join(["=s"] * len(vals) + [str(i) for i in range(len(vals))])
    res = llvm.inline_asm(ty, [_raw(v) for v in vals], "; pin descriptor", constraints, has_side_effects=True)
    return [fx.Int32(llvm.extractvalue(T.i32, res, [i])) for i in range(len(vals))]


def _replace_shared_scales(ring, before, ready):
    """Trace-time SSA bookkeeping, outside the kernel's AST loop rewriter."""
    for queued in ring:
        for s in (8, 9):
            if ir.Value.__eq__(_raw(queued[s]), _raw(before[s])):
                queued[s] = ready[s]


def compile_gemm1_persist(
    *, D_HIDDEN, D_INTER, NE, TOPK, BM=16, TILE_N=32, TILE_K=128,
    prefetch=3, b_cache_mod=2, w_layout="standard",
):
    assert BM == 16 and TILE_N == 32 and TILE_K == 128
    assert D_HIDDEN % 1024 == 0 and D_INTER % 128 == 0
    assert w_layout in ("standard", "guinterleave")
    assert b_cache_mod in (0, 2)
    K, I, NN = D_HIDDEN, D_INTER, D_INTER // TILE_N
    KT = K // 4 // TILE_K
    assert 1 <= prefetch < KT
    # A: four 128-bit fragments, B: gate/up x two 16-column fragments,
    # scales: two dwords shared by a pair of K tiles and by the two N halves.
    NV = 10
    LDS_BYTES = 4 * 2 * 64 * 4 * 4

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=f"gemm1_a16w4_persist_h{K}_i{I}_e{NE}_pf{prefetch}_{w_layout}_nt{b_cache_mod}", known_block_size=[256, 1, 1])
    def kernel(
        ax: fx.Int64, bw: fx.Int64, bs: fx.Int64, eids: fx.Int64,
        nvalid: fx.Int64, ids: fx.Int64, ntok: fx.Int32, nctas: fx.Int32,
        alpha: fx.Float32, limit: fx.Float32, out: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tid = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tid % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tid // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        rows = fx.Int32(_global_i32_at(nvalid, fx.Int32(0)))
        total = rows // fx.Int32(BM) * fx.Int32(NN)
        ar = buffer_ops.create_buffer_resource_from_addr(_raw(ax), num_records_bytes=_raw(fx.Int64(ntok) * fx.Int64(K * 2)))
        wr = buffer_ops.create_buffer_resource_from_addr(_raw(bw), num_records_bytes=NE * 2 * I * K // 2)
        sr = buffer_ops.create_buffer_resource_from_addr(_raw(bs), num_records_bytes=NE * 2 * I * K // 32)
        outr = buffer_ops.create_buffer_resource_from_addr(_raw(out), num_records_bytes=_raw(fx.Int64(rows) * fx.Int64(I * 2)))
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        scr = _lds_ptr3(fx.Int32(fx.ptrtoint(smem)), fx.Int32(0))

        def descriptor(item):
            # Clamp look-ahead to valid storage, even on the last item.
            safe = (item < total).select(item, total - fx.Int32(1))
            mb, nb = safe // fx.Int32(NN), safe % fx.Int32(NN)
            e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(eids, mb))))
            mb, nb, e = _pin_scalar([mb, nb, e])
            r = fx.Int32(_global_i32_at(ids, mb * fx.Int32(BM) + l16)) & fx.Int32(0xFFFFFF)
            ep = [fx.Int32(_global_i32_at(ids, mb * fx.Int32(BM) + q16 * fx.Int32(4) + fx.Int32(i))) & fx.Int32(0xFFFFFF) for i in range_constexpr(4)]
            rv = _wait_values([r] + ep, 0)
            return [mb, nb, e] + [fx.Int32(v) for v in rv]

        def tile_load(d, kt, prev):
            mb, nb, e = d[:3]
            kb = wave * fx.Int32(K // 4) + fx.Int32(kt * TILE_K)
            vals = []
            for ku in range_constexpr(4):
                off = d[3] * fx.Int32(K * 2) + kb * fx.Int32(2) + q16 * fx.Int32(64) + fx.Int32(ku * 16)
                vals.append(_load(ar, off))
            for gu in range_constexpr(2):
                for ni in range_constexpr(2):
                    if const_expr(w_layout == "standard"):
                        nblk = e * fx.Int32(2 * I // 16) + nb * fx.Int32(2) + fx.Int32(gu * I // 16 + ni)
                    else:
                        nblk = e * fx.Int32(2 * I // 16) + nb * fx.Int32(4) + fx.Int32(ni * 2 + gu)
                    off = nblk * fx.Int32(K * 8) + (kb // fx.Int32(128)) * fx.Int32(1024) + q16 * fx.Int32(256) + l16 * fx.Int32(16)
                    vals.append(_load(wr, off, cache=b_cache_mod))
            if const_expr(kt % 2 == 0):
                for s in range_constexpr(2):
                    if const_expr(w_layout == "standard"):
                        sn = e * fx.Int32(2 * I // 32) + nb + fx.Int32(s * I // 32)
                    else:
                        sn = e * fx.Int32(2 * I // 32) + nb * fx.Int32(2) + fx.Int32(s)
                    off = (sn * fx.Int32(K // 256 * 64) + kb // fx.Int32(256) * fx.Int32(64) + q16 * fx.Int32(16) + l16) * fx.Int32(4)
                    vals.append(_load(sr, off, width=1))
            else:
                vals += prev[-2:]
            return vals

        def frag(v):
            r = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            r.store(fx.Vector(v).bitcast(fx.BFloat16))
            return r

        def compute(vals, kt, acc):
            for ni in range_constexpr(2):
                for ku in range_constexpr(4):
                    a = frag(vals[ku])
                    for gu in range_constexpr(2):
                        packed = fx.Vector(vals[4 + gu * 2 + ni])[ku]
                        if const_expr(w_layout == "standard"):
                            sc = _e8m0_byte_to_f32(fx.Int32(vals[8 + gu]), fx.Int32((kt % 2) * 2 + ni))
                        else:
                            sc = _e8m0_byte_to_f32(fx.Int32(vals[8 + ni]), fx.Int32((kt % 2) * 2 + gu))
                        bv = []
                        for sel in range_constexpr(4):
                            v = rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), _raw(packed), _raw(sc), sel)
                            bv.append(fx.Vector(v).bitcast(fx.Int32)[0])
                        b = frag(_raw(fx.Vector.from_elements(bv, fx.Int32)))
                        fx.gemm(mma, acc[gu][ni], a, b, acc[gu][ni])

        def finish(d, acc):
            # Match gemm1.py's fixed slice-K summation order exactly.
            for gu in range_constexpr(2):
                gpu.barrier()
                for ni in range_constexpr(2):
                    v = fx.Vector(fx.memref_load_vec(acc[gu][ni]))
                    idx = wave * fx.Int32(512) + fx.Int32(ni * 256) + lane * fx.Int32(4)
                    for ii in range_constexpr(4):
                        llvm.StoreOp(_raw(v[ii]), _gep3(scr, (idx + fx.Int32(ii)) * fx.Int32(4)))
                gpu.barrier()
                for ni in range_constexpr(2):
                    v = fx.Vector(fx.memref_load_vec(acc[gu][ni]))
                    for peer in range_constexpr(1, 4):
                        idx = fx.Int32(peer * 512 + ni * 256) + lane * fx.Int32(4)
                        p = fx.Vector(llvm.load(T.vec(4, T.f32), _gep3(scr, idx * fx.Int32(4))))
                        v = fx.Vector.from_elements([v[ii] + p[ii] for ii in range_constexpr(4)], fx.Float32)
                    acc[gu][ni].store(v)
            for ii in range_constexpr(4):
                valid = (d[4 + ii] < ntok) & (wave == fx.Int32(0))
                row = d[0] * fx.Int32(BM) + q16 * fx.Int32(4) + fx.Int32(ii)
                for ni in range_constexpr(2):
                    g = fx.Float32(fx.Vector(fx.memref_load_vec(acc[0][ni]))[ii])
                    u = fx.Float32(fx.Vector(fx.memref_load_vec(acc[1][ni]))[ii])
                    y = _swigluoai_f32(g, u, alpha, -limit).to(fx.BFloat16)
                    idx = row * fx.Int32(I) + d[1] * fx.Int32(TILE_N) + fx.Int32(ni * 16) + l16
                    buffer_ops.buffer_store(y, outr, _raw(idx), mask=valid)

        if pid < total:
            first = descriptor(pid)
            initial = []
            for kt in range_constexpr(prefetch):
                vals = tile_load(first, kt, initial[-NV:])
                initial += vals
            # The first loop-entry PHI may copy registers as well as the back
            # edge. Its source loads must be ready before those copies execute.
            state0 = [_raw(v) for v in first] + _wait_values(initial, 0)
            count = (total - pid + nctas - fx.Int32(1)) // nctas
            for j, state in range(0, count, init=state0):
                d = [fx.Int32(v) for v in state[:8]]
                ring = [list(state[8 + t * NV:8 + (t + 1) * NV]) for t in range(prefetch)]
                nxt = descriptor(pid + (fx.Int32(j) + fx.Int32(1)) * nctas)
                acc = [[fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(2)] for _ in range(2)]
                for gu in range_constexpr(2):
                    for ni in range_constexpr(2):
                        acc[gu][ni].store(fx.Vector.filled(4, 0.0, fx.Float32))
                for kt in range_constexpr(KT):
                    future = kt + prefetch
                    vals = tile_load(d if future < KT else nxt, future % KT, ring[-1])
                    ring.append(vals)
                    # Each later tile issues 8 VMEM reads plus 2 scales on even K.
                    outstanding = sum(8 + (2 if (t % KT) % 2 == 0 else 0) for t in range(kt + 1, kt + prefetch + 1))
                    raw_cur = ring.pop(0)
                    cur = _wait_values(raw_cur, outstanding)
                    # Future tiles must refer to the READY scale values, so
                    # pinning never has to preserve an unfinished raw input.
                    _replace_shared_scales(ring, raw_cur, cur)
                    compute(cur, kt, acc)
                finish(d, acc)
                # PHI copies at the back edge cannot read pending destinations.
                # The epilogue above hides this wait for the next item's ring.
                flat = _wait_values([v for tile in ring for v in tile], 0)
                state = yield [_raw(v) for v in nxt] + flat

    @flyc.jit
    def launch(
        ax: fx.Int64, bw: fx.Int64, bs: fx.Int64, eids: fx.Int64,
        nvalid: fx.Int64, ids: fx.Int64, ntok: fx.Int32, nctas: fx.Int32,
        alpha: fx.Float32, beta_rcp: fx.Float32, linbeta: fx.Float32,
        linbeta_rcp: fx.Float32, limit: fx.Float32, out: fx.Int64,
        zero: fx.Int64, zero_dw: fx.Int32, stream: fx.Stream,
    ):
        kernel(ax, bw, bs, eids, nvalid, ids, ntok, nctas, alpha, limit, out).launch(
            grid=(nctas, 1, 1), block=(256, 1, 1), stream=stream,
        )

    return launch
