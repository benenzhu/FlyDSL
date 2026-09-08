# SPDX-License-Identifier: Apache-2.0
"""Time the prefill final reduce alone (HIP graph over different inputs): aiter's
moe_reduction vs moe_a8w8_prefill.reduce_bf16 variants, and reduce_fp8 variants; every
variant is checked against a float reference (bf16 ones also for bit-equality with aiter).

    python3 -m m3_a8w8_moe.time_reduce --tokens 32768 --cpw 1 2 3 --nt 1 0
"""
import argparse

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import HIDDEN, time_graph

import_ops("moe_a16w4_decode")
import_ops("moe_a4w4_prefill")
import_ops("moe_a8w8_prefill")
from moe_a4w4_prefill import _run_compiled  # noqa: E402
from moe_a4w4_prefill.reduce_bf16 import compile_moe_reduce_bf16  # noqa: E402
from moe_a8w8_prefill.reduce_fp8 import compile_moe_reduce_fp8  # noqa: E402

TOPK = 5


def fp8_e4m3_to_f32(b: torch.Tensor) -> torch.Tensor:
    """OCP e4m3fn bytes -> f32 (NaN codes must be filtered by the caller)."""
    b = b.int()
    sign = 1.0 - 2.0 * (b >> 7).float()
    exp = (b >> 3) & 0xF
    mant = (b & 7).float()
    normal = torch.ldexp(1.0 + mant / 8.0, exp - 7)
    sub = torch.ldexp(mant / 8.0, torch.full_like(exp, -6))
    return sign * torch.where(exp == 0, sub, normal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[32768])
    p.add_argument("--cpw", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--nt", type=int, nargs="+", default=[1])
    p.add_argument("--nt-store", type=int, nargs="+", default=[0])
    p.add_argument("--wpc", type=int, nargs="+", default=[4], help="waves per CTA")
    p.add_argument("--calib", action="store_true", help="time torch copy_/sum over the same bytes")
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--skip-bf16", action="store_true")
    p.add_argument("--skip-fp8", action="store_true")
    a = p.parse_args()
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    dev = torch.device("cuda")
    for m in a.tokens:
        torch.manual_seed(m)
        outs = [torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev) for _ in range(a.copies)]
        if not a.skip_bf16:
            parts = [torch.randn((m * TOPK, HIDDEN), dtype=torch.bfloat16, device=dev) for _ in range(a.copies)]
            ref = parts[0].view(m, TOPK, HIDDEN).float().sum(1)
            gb = (m * TOPK * HIDDEN * 2 + m * HIDDEN * 2) * 1e-9

            def aiter_fn(part, out):
                _run_moe_reduction(part.view(m, TOPK, HIDDEN), out, m, TOPK, HIDDEN)

            aiter_fn(parts[0], outs[0])
            torch.cuda.synchronize()
            ait = outs[0].clone()
            if a.calib:
                dst = torch.empty_like(parts[0])
                t, _ = time_graph(lambda p_: dst.copy_(p_), [(p_,) for p_ in parts])
                print(f"[calib] M={m} torch copy_ 2 GB->2 GB: {t:.1f} us ({2 * m * TOPK * HIDDEN * 2e-9 / t * 1e3:.2f} TB/s r+w)", flush=True)
                t, _ = time_graph(lambda p_, o_: torch.sum(p_.view(m, TOPK, HIDDEN), dim=1, out=o_), list(zip(parts, outs)))
                print(f"[calib] M={m} torch.sum dim=1: {t:.1f} us ({gb / t * 1e3:.2f} TB/s r+w)", flush=True)
                del dst
            t, _ = time_graph(aiter_fn, list(zip(parts, outs)))
            print(f"[reduce bf16] M={m} aiter moe_reduction: {t:.1f} us ({gb / t * 1e3:.2f} TB/s r+w) "
                  f"| max|aiter-ref| {(ait.float() - ref).abs().max().item():.4f}", flush=True)
            for cpw, nt, nts, wpc in [(c, n, s_, w_) for c in a.cpw for n in a.nt for s_ in a.nt_store for w_ in a.wpc]:
                    if True:
                        launch = compile_moe_reduce_bf16(H=HIDDEN, topk=TOPK, chunks_per_wave=cpw,
                                                         nt_load=bool(nt), nt_store=bool(nts), waves_per_cta=wpc)

                        def fn(part, out, launch=launch):
                            _run_compiled(launch, part.view(-1), out.view(-1), m, torch.cuda.current_stream())

                        outs[0].zero_()
                        fn(parts[0], outs[0])
                        torch.cuda.synchronize()
                        same = torch.equal(outs[0], ait)
                        err = (outs[0].float() - ref).abs().max().item()
                        t, _ = time_graph(fn, list(zip(parts, outs)))
                        print(f"[reduce bf16] M={m} ours cpw={cpw} wpc={wpc} nt={nt} nt_store={nts}: {t:.1f} us "
                              f"({gb / t * 1e3:.2f} TB/s r+w) | == aiter {same} | max|ours-ref| {err:.4f}", flush=True)
            del parts, ref, ait
        if a.skip_fp8:
            continue
        qs = []
        for _ in range(a.copies):
            q = torch.randint(0, 256, (m * TOPK, HIDDEN), dtype=torch.uint8, device=dev)
            q[(q & 0x7F) == 0x7F] = 0  # no NaN codes
            qs.append(q)
        scs = [torch.randint(120, 134, (m * TOPK * (HIDDEN // 32),), dtype=torch.uint8, device=dev)
               for _ in range(a.copies)]
        ws = [torch.rand((m, TOPK), dtype=torch.float32, device=dev) for _ in range(a.copies)]
        scale_f = torch.ldexp(torch.ones((), device=dev), scs[0].view(m * TOPK, HIDDEN // 32).int() - 127)
        deq = fp8_e4m3_to_f32(qs[0]).view(m * TOPK, HIDDEN // 32, 32) * scale_f[..., None]
        ref8 = (deq.view(m, TOPK, HIDDEN) * ws[0][..., None]).sum(1)
        del deq, scale_f
        gb8 = (m * TOPK * HIDDEN + m * TOPK * HIDDEN // 32 + m * TOPK * 4 + m * HIDDEN * 2) * 1e-9
        for cpw, nt, nts, wpc in [(c, n, s_, w_) for c in a.cpw for n in a.nt for s_ in a.nt_store for w_ in a.wpc]:
                if True:
                    launch = compile_moe_reduce_fp8(H=HIDDEN, topk=TOPK, chunks_per_wave=cpw,
                                                    nt_load=bool(nt), nt_store=bool(nts), waves_per_cta=wpc)

                    def fn8(q, sc, w, out, launch=launch):
                        _run_compiled(launch, q.view(-1), sc, w.view(-1), out.view(-1), m, torch.cuda.current_stream())

                    outs[0].zero_()
                    fn8(qs[0], scs[0], ws[0], outs[0])
                    torch.cuda.synchronize()
                    err = (outs[0].float() - ref8).abs().max().item()
                    rel = err / ref8.abs().max().item()
                    t, _ = time_graph(fn8, list(zip(qs, scs, ws, outs)))
                    print(f"[reduce fp8] M={m} ours cpw={cpw} wpc={wpc} nt={nt} nt_store={nts}: {t:.1f} us "
                          f"({gb8 / t * 1e3:.2f} TB/s r+w) | max|ours-ref| {err:.4f} (rel {rel:.1e})", flush=True)


if __name__ == "__main__":
    main()
