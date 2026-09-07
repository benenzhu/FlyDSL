"""Per-CU vector-memory path rates from L2-resident data: LDS-DMA loads
(buffer_load ... lds, gemm2's W2 path), direct-to-VGPR loads, dwordx4 stores, and
gemm2's mix (16 DMA loads : 12 stores per wave). 4 waves per CTA, 1 CTA per CU;
each XCD's CTAs cycle one 2 MB region (misses L1, hits L2)."""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.environ.get("FLYDSL_ROOT", "/flydsl"))
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir as _ir  # noqa: E402
from flydsl._mlir.dialects import arith as _arith  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr import rocdl as _rocdl  # noqa: E402
from flydsl.expr.typing import T as _T  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops  # noqa: E402
from m3_a4w4_moe.gemm1 import G2SLoaderAsm, _Buf  # noqa: E402

N_LD = 16  # 16-B-per-lane instructions per wave per iteration (16 KB per wave, 64 KB per CTA)
N_ST_MIX = 12
REGION = int(os.environ.get("VMEM_REGION_MB", "2")) * 1024 * 1024  # 2 MB: L2-resident; 256 MB: HBM/MALL
N_XCD = 8


def compile_vmem(mode: str):
    assert mode in ("dma", "vgpr", "store", "mixed")

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Int8, 65536, 16]

    @flyc.kernel
    def kernel_vmem(SRC: fx.Tensor, DST: fx.Tensor, n_iter: fx.Int32):
        lds = fx.SharedAllocator().allocate(Shared).peek()
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        cta = fx.block_idx.x
        src_rsrc = _buffer_ops.create_buffer_resource(SRC, max_size=True)
        dst_rsrc = _buffer_ops.create_buffer_resource(DST, max_size=True)
        region = (cta % fx.Int32(N_XCD)) * fx.Int32(REGION)
        lane_off = [region + fx.Int32((s * 4) * 1024) + wave * fx.Int32(1024) + lane * fx.Int32(16) for s in range(N_LD)]
        v4 = _ir.VectorType.get([4], _T.i32)
        zero4 = _arith.ConstantOp(v4, _ir.DenseElementsAttr.get_splat(v4, _ir.IntegerAttr.get(_T.i32, 0))).result
        one4 = _arith.ConstantOp(v4, _ir.DenseElementsAttr.get_splat(v4, _ir.IntegerAttr.get(_T.i32, 1))).result
        g2s = G2SLoaderAsm(src_rsrc, lane_off, N_LD, wave)
        g2s.set_wave_base(lds.buf.ptr)
        dst0 = _Buf(lds.buf.ptr, 0)

        for it, st in range(0, n_iter, init=[zero4]):
            acc = st[0]
            it_i = fx.Int32(it)
            koff = (it_i % fx.Int32(REGION // 65536)) * fx.Int32(65536)
            if mode == "dma":
                g2s.load(dst0, koff)
                _rocdl.s_waitcnt(vmcnt=N_LD, lgkmcnt=0)
            elif mode == "vgpr":
                vals = [
                    _buffer_ops.buffer_load(
                        src_rsrc, lane_off[s] // fx.Int32(4), vec_width=4, dtype=fx.Int32, soffset_bytes=fx.as_ir_value(koff)
                    )
                    for s in range(N_LD)
                ]
                for v in vals:
                    acc = _arith.XOrIOp(acc, v).result
            elif mode == "store":
                for s in range_constexpr(N_LD):
                    _buffer_ops.buffer_store(one4, dst_rsrc, lane_off[s], offset_is_bytes=True, soffset_bytes=fx.as_ir_value(koff))
            else:
                g2s.load(dst0, koff)
                for s in range_constexpr(N_ST_MIX):
                    _buffer_ops.buffer_store(one4, dst_rsrc, lane_off[s], offset_is_bytes=True, soffset_bytes=fx.as_ir_value(koff))
                _rocdl.s_waitcnt(vmcnt=N_LD + N_ST_MIX, lgkmcnt=0)
            st = yield [acc]
        _rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        if mode == "vgpr":
            _buffer_ops.buffer_store(st[0], dst_rsrc, cta * fx.Int32(4096) + tid * fx.Int32(16), offset_is_bytes=True)

    @flyc.jit
    def launch(SRC: fx.Tensor, DST: fx.Tensor, n_iter: fx.Int32, n_cta: fx.Int32, stream: fx.Stream):
        kernel_vmem(SRC, DST, n_iter).launch(grid=(n_cta, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def time_us(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--modes", default="dma,vgpr,store,mixed")
    p.add_argument("--ctas", default="32,256,512")
    p.add_argument("--iters-hbm", type=int, default=0)
    a = p.parse_args()
    src = torch.randint(0, 2**31 - 1, (N_XCD * REGION // 4,), dtype=torch.int32, device="cuda")
    dst = torch.zeros(N_XCD * REGION // 4, dtype=torch.int32, device="cuda")
    for mode in a.modes.split(","):
        launch = compile_vmem(mode)
        for n_cta in (int(x) for x in a.ctas.split(",")):
            fn = lambda: launch(src, dst, a.iters, n_cta, torch.cuda.current_stream())
            t = time_us(fn, a.reps)
            ld_b = N_LD * 1024 * 4 * a.iters if mode != "store" else 0
            st_b = (N_LD if mode == "store" else N_ST_MIX if mode == "mixed" else 0) * 1024 * 4 * a.iters
            per_cta = ld_b + st_b
            cus = min(n_cta, 256)
            waves_per_cu = n_cta / 256
            total = per_cta * n_cta
            print(
                f"{mode:5s} ctas={n_cta:4d}: {t:8.1f} us  total {total / t * 1e6 / 1e12:5.2f} TB/s  "
                f"per busy CU {total / t / cus * 1e6 / 2.0e9:6.1f} B/clk@2.0GHz  (ld {ld_b >> 10} KB + st {st_b >> 10} KB per CTA)",
                flush=True,
            )
