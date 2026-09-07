# SPDX-License-Identifier: Apache-2.0
"""Four-wave a16w4 gate/up with AGPR C and an in-place N-major weight ring.

One WG owns up to 256 gate columns (and up). Each wave handles one K quarter.
Unlike gemm1.py's multi-K VGPR ring, consuming column tile n immediately
replaces its packed W registers with W(k+1,n). The remaining column tiles
provide the look-ahead. This keeps large-N C in AGPR and just one W tile in
VGPR. The file is an independent experiment; no production dispatch uses it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T

from .gemm1 import _swigluoai_f32
from .gemm1_persist import _wait_values
from .utils import _raw, _global_i32_at, _e8m0_byte_to_f32, _lds_ptr3, _gep3, buffer_ops


def _load(rsrc, voff, soff, *, width=4, old=None, nt=False):
    ty = T.vec(4, T.i32) if width == 4 else T.i32
    op = "buffer_load_dwordx4" if width == 4 else "buffer_load_dword"
    rsrc = fx.as_ir_value(fx.rocdl.get_buffer_rsrc(rsrc))
    soff = rocdl.readfirstlane(T.i32, _raw(soff))
    args = [_raw(voff), rsrc, soff]
    if old is None:
        asm, constraints = f"{op} $0, $1, $2, $3 offen", "=v,v,s,s"
    else:
        args = [_raw(old)] + args
        asm, constraints = f"{op} $0, $2, $3, $4 offen", "=v,0,v,s,s"
    return llvm.inline_asm(ty, args, asm + (" nt" if nt else ""), constraints, has_side_effects=True)


def _mma(a, b, c):
    # The packed-fp4 conversion is VALU. Opaque MFMA asm needs its own
    # VALU -> MFMA operand delay; the small AGPR control verified this gap.
    args = [_raw(a), _raw(b)]
    constraints = "=a,v,v"
    tail = "0"
    if c is not None:
        args.append(_raw(c))
        constraints += ",0"
        tail = "$0"
    return llvm.inline_asm(T.vec(4, T.f32), args,
                           f"s_nop 1\nv_mfma_f32_16x16x32_bf16 $0, $1, $2, {tail}",
                           constraints, has_side_effects=True)


def _pin_accumulators(values):
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>")
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(ty, [_raw(v) for v in values], "s_nop 15\ns_nop 15", constraints, has_side_effects=True)
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


def _body(smem, ax, bw, bs, ids, out, pid, expert, valid_rows, ntok, alpha, limit, *, K, I, E, TN, b_nt):
    """Python-traced body: event dictionaries and register replacement are
    compile-time bookkeeping, not loop-carried Python containers in scf.for."""
    NI, KT = TN // 16, K // 4 // 128
    tid = fx.Int32(gpu.thread_id("x"))
    lane = tid % fx.Int32(64)
    wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tid // fx.Int32(64))))
    l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
    mb, nb = pid // fx.Int32(I // TN), pid % fx.Int32(I // TN)
    mbase, nbase = mb * fx.Int32(16), nb * fx.Int32(TN)
    kbase = wave * fx.Int32(K // 4)
    token = fx.Int32(_global_i32_at(ids, mbase + l16)) & fx.Int32(0xFFFFFF)
    ep = [fx.Int32(_global_i32_at(ids, mbase + q16 * fx.Int32(4) + fx.Int32(ii))) & fx.Int32(0xFFFFFF) for ii in range(4)]
    ready = _wait_values([token] + ep, 0)
    token, ep = fx.Int32(ready[0]), [fx.Int32(v) for v in ready[1:]]
    ar = buffer_ops.create_buffer_resource_from_addr(_raw(ax), num_records_bytes=_raw(fx.Int64(ntok) * fx.Int64(K * 2)))
    wr = buffer_ops.create_buffer_resource_from_addr(_raw(bw), num_records_bytes=E * 2 * I * K // 2)
    sr = buffer_ops.create_buffer_resource_from_addr(_raw(bs), num_records_bytes=E * 2 * I * K // 32)
    outr = buffer_ops.create_buffer_resource_from_addr(_raw(out), num_records_bytes=_raw(fx.Int64(valid_rows) * fx.Int64(I * 2)))
    woff = q16 * fx.Int32(256) + l16 * fx.Int32(16)
    soff = (q16 * fx.Int32(16) + l16) * fx.Int32(4)
    events, vals, issued = [], {}, {}

    def put(key, resource, voff, scalar, width=4, old_key=None, nt=False):
        old = vals.get(old_key) if old_key is not None else None
        vals[key] = _load(resource, voff, scalar, width=width, old=old, nt=nt)
        issued[key] = len(events)
        events.append(key)

    def a_load(kt, old_kt=None):
        for ku in range(4):
            voff = token * fx.Int32(K * 2) + q16 * fx.Int32(64) + fx.Int32(ku * 16)
            put(("a", kt, ku), ar, voff, (kbase + fx.Int32(kt * 128)) * fx.Int32(2),
                old_key=("a", old_kt, ku) if old_kt is not None else None)

    def w_load(kt, ni, old_kt=None):
        for gu in range(2):
            nblk = expert * fx.Int32(2 * I // 16) + nbase // fx.Int32(16) + fx.Int32(gu * I // 16 + ni)
            scalar = nblk * fx.Int32(K * 8) + (kbase // fx.Int32(128) + fx.Int32(kt)) * fx.Int32(1024)
            put(("b", kt, ni, gu), wr, woff, scalar,
                old_key=("b", old_kt, ni, gu) if old_kt is not None else None, nt=b_nt == 2)

    def s_load(kg, pair, old_kg=None):
        for gu in range(2):
            ng = expert * fx.Int32(2 * I // 32) + nbase // fx.Int32(32) + fx.Int32(gu * I // 32 + pair)
            scalar = (ng * fx.Int32(K // 256 * 64) + (kbase // fx.Int32(256) + fx.Int32(kg)) * fx.Int32(64)) * fx.Int32(4)
            put(("s", kg, pair, gu), sr, soff, scalar, width=1,
                old_key=("s", old_kg, pair, gu) if old_kg is not None else None)

    def pin(keys):
        # Outstanding VMEM is derived from the actual load issue order. All
        # older reads precede the newest required event; later ones may stay in flight.
        count = min(63, len(events) - 1 - max(issued[k] for k in keys))
        result = _wait_values([vals[k] for k in keys], count)
        for k, v in zip(keys, result):
            vals[k] = v  # future consumers see the ready SSA, avoiding pre-wait copies
        return result

    a_load(0)
    for ni in range(NI):
        w_load(0, ni)
        if ni % 2 == 0:
            s_load(0, ni // 2)
    acc = [[None] * NI for _ in range(2)]
    for kt in range(KT):
        for ni in range(NI):
            ak = [("a", kt, ku) for ku in range(4)]
            bk = [("b", kt, ni, gu) for gu in range(2)]
            sk = [("s", kt // 2, ni // 2, gu) for gu in range(2)]
            v = pin(ak + bk + sk)
            scales = [_e8m0_byte_to_f32(fx.Int32(v[6 + gu]), fx.Int32((kt % 2) * 2 + ni % 2)) for gu in range(2)]
            for ku in range(4):
                gb = []
                for gu in range(2):
                    packed = _raw(fx.Vector(v[4 + gu])[ku])
                    words = [fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), packed, _raw(scales[gu]), sel)).bitcast(fx.Int32)[0] for sel in range(4)]
                    gb.append(fx.Vector.from_elements(words, fx.Int32))
                for gu in range(2):
                    acc[gu][ni] = _mma(v[ku], gb[gu], acc[gu][ni])
            if kt + 1 < KT:
                w_load(kt + 1, ni, old_kt=kt)
                if ni == 0:
                    a_load(kt + 1, old_kt=kt - 1 if kt > 0 else None)
                if kt % 2 == 1 and ni % 2 == 1:
                    s_load((kt + 1) // 2, ni // 2, old_kg=kt // 2)

    flat = _pin_accumulators([v for slab in acc for v in slab])
    acc = [flat[:NI], flat[NI:]]
    scr = _lds_ptr3(fx.Int32(fx.ptrtoint(smem)), fx.Int32(0))
    stride = NI * 64 * 4
    # Both projections have their own LDS region. There is no preceding LDS
    # use, so one trailing barrier is sufficient before reading K peers.
    for gu in range(2):
        for ni in range(NI):
            idx = fx.Int32(gu * 4 * stride + ni * 256) + wave * fx.Int32(stride) + lane * fx.Int32(4)
            vv = fx.Vector(acc[gu][ni])
            for ii in range(4):
                llvm.StoreOp(_raw(vv[ii]), _gep3(scr, (idx + fx.Int32(ii)) * fx.Int32(4)))
    gpu.barrier()
    sums = [[], []]
    for gu in range(2):
        for ni in range(NI):
            s = fx.Vector(acc[gu][ni])
            for peer in range(1, 4):
                idx = fx.Int32(gu * 4 * stride + peer * stride + ni * 256) + lane * fx.Int32(4)
                p = fx.Vector(llvm.load(T.vec(4, T.f32), _gep3(scr, idx * fx.Int32(4))))
                s = fx.Vector.from_elements([s[ii] + p[ii] for ii in range(4)], fx.Float32)
            sums[gu].append(s)
    for ni in range(NI):
        for ii in range(4):
            row = mbase + q16 * fx.Int32(4) + fx.Int32(ii)
            value = _swigluoai_f32(fx.Float32(sums[0][ni][ii]), fx.Float32(sums[1][ni][ii]), alpha, -limit).to(fx.BFloat16)
            dst = row * fx.Int32(I) + nbase + fx.Int32(ni * 16) + l16
            buffer_ops.buffer_store(value, outr, _raw(dst), mask=(ep[ii] < ntok) & (wave == fx.Int32(0)))


def compile_gemm1_agpr_ring(*, D_HIDDEN, D_INTER, NE, TOPK, BM=16, TILE_N=256,
                           TILE_K=128, prefetch=1, b_cache_mod=2, w_layout="standard"):
    assert BM == 16 and TILE_K == 128 and prefetch == 1
    assert TILE_N in (32, 64, 128, 256) and D_INTER % TILE_N == 0
    assert D_HIDDEN % 1024 == 0 and w_layout == "standard"
    lds_bytes = 2 * 4 * (TILE_N // 16) * 64 * 4 * 4
    assert lds_bytes <= 160 * 1024

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=f"gemm1_a16w4_agpr_ring_h{D_HIDDEN}_i{D_INTER}_tn{TILE_N}", known_block_size=[256, 1, 1])
    def kernel(ax: fx.Int64, bw: fx.Int64, bs: fx.Int64, eids: fx.Int64,
               nvalid: fx.Int64, ids: fx.Int64, ntok: fx.Int32,
               alpha: fx.Float32, limit: fx.Float32, out: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        pid = fx.Int32(gpu.block_id("x"))
        nr = fx.Int32(_global_i32_at(nvalid, fx.Int32(0)))
        mb = pid // fx.Int32(D_INTER // TILE_N)
        if mb * fx.Int32(16) < nr:
            e = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_global_i32_at(eids, mb))))
            _body(smem, ax, bw, bs, ids, out, pid, e, nr, ntok, alpha, limit,
                  K=D_HIDDEN, I=D_INTER, E=NE, TN=TILE_N, b_nt=b_cache_mod)

    @flyc.jit
    def launch(ax: fx.Int64, bw: fx.Int64, bs: fx.Int64, eids: fx.Int64,
               nvalid: fx.Int64, ids: fx.Int64, ntok: fx.Int32, grid: fx.Int32,
               alpha: fx.Float32, beta_rcp: fx.Float32, linbeta: fx.Float32,
               linbeta_rcp: fx.Float32, limit: fx.Float32, out: fx.Int64,
               zero: fx.Int64, zero_dw: fx.Int32, stream: fx.Stream):
        kernel(ax, bw, bs, eids, nvalid, ids, ntok, alpha, limit, out).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)
    return launch
