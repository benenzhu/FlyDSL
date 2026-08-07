#!/usr/bin/env python3
"""FlyDSL fp4_gemm_4wave vs aiter's hand-written asm, same process, interleaved.

    python3 experiments/fp4_handoff/bench_vs_aiter.py 8192x8192x8192 16384x16384x16384

Two things keep this honest:

* ``N_SETS`` distinct input sets, rotated per iteration. At 16384^3 one set's
  A+B is 268 MB and five sets are 1.34 GB -- far past the 256 MB MALL, so by
  the time a set comes around again it has been evicted. Back-to-back timing of
  a single input set instead measures the cache-resident path, and at 8192^3
  (A+B = 67 MB) it fits in MALL entirely, which is why 8192 and 16384 were not
  measuring the same thing.
* fly and aiter are timed as separate ITERS-long blocks, but alternated PAIRS
  times over the same rotation indices, so clock and power drift are spread
  across both sides rather than favouring whoever ran first.

WARMUP=500 is not padding: at 200 the first pair came out 1.6% off the other two
(and in a different direction run to run) while aiter was already stable, so we
were reading our own warmup transient. At 500 the three pairs land within 0.04%
at 16384^3, which is tight enough to A/B a single optimization.

Env: PAIRS (default 3), ITERS (default 500), WARMUP (default 500), N_SETS (5).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import aiter  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from kernels.fp4_gemm_4wave import compile_fp4_gemm_4w  # noqa: E402
from tests.kernels.utils import fp4_utils as U  # noqa: E402

PAIRS = int(os.environ.get("PAIRS", "3"))
ITERS = int(os.environ.get("ITERS", "500"))
WARMUP = int(os.environ.get("WARMUP", "500"))
N_SETS = int(os.environ.get("N_SETS", "5"))
dev = torch.device("cuda")


def u8(t):
    return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)


def setup(M, N, K):
    """Build N_SETS input sets and return (fly_step, aiter_step), each taking an
    iteration index and issuing one GEMM on set ``i % N_SETS``."""
    fly_args, ait_args = [], []
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    # One C, reused: rotating it too would add 512 MB per set at 16384^3 without
    # changing what is measured -- C is write-only and streams past the cache
    # either way. (aiter's gemm_a4w4 has no ``out``; it allocates its own, which
    # the caching allocator serves from the same block every call.)
    c = torch.zeros(M * N, dtype=torch.bfloat16, device=dev)

    for s in range(N_SETS):
        torch.manual_seed(s)
        a = torch.randn(M, K, device=dev)
        b = torch.randn(N, K, device=dev)

        a_q, sa, _ = U.per_1x32_f4_quant(a)
        b_q, sb, _ = U.per_1x32_f4_quant(b)
        ad = u8(a_q).contiguous().view(-1)
        bd, s1, s2 = (
            u8(t).contiguous().view(-1)
            for t in (
                U.shuffle_weight_w4(b_q, 16, False, False),
                U.shuffle_scale_w4(sa, 1, False),
                U.shuffle_scale_w4(sb, 1, False),
            )
        )
        fly_args.append((ad, bd, c, s1, s2, M, N, torch.cuda.current_stream()))

        xq, xs = quant(a.to(torch.bfloat16), shuffle=True)
        wq, ws = quant(b.to(torch.bfloat16), shuffle=True)
        ait_args.append((xq, shuffle_weight(wq, layout=(16, 16)), xs, ws))
        del a, b, a_q, b_q, wq

    compiled = flyc.compile(compile_fp4_gemm_4w(K=K, MN=(M, N)), *fly_args[0])

    def fly(i):
        compiled(*fly_args[i % N_SETS])

    def ait(i):
        x, w, xs, ws = ait_args[i % N_SETS]
        aiter.gemm_a4w4(x, w, xs, ws, bpreshuffle=True)

    return fly, ait


def time_us(fn, base):
    """WARMUP + ITERS back-to-back calls under ONE event pair.

    Two deliberate omissions. No per-iteration events: those need a sync each
    time, which drains the pipe and measures single-launch latency plus event
    overhead instead of steady-state throughput. And no sync between warmup and
    the timed region: leaving the queue full means the CPU stays ahead of the
    GPU, so the timed region has no launch-gap bubble at its start. The events
    are still ordered in the stream, so what they bracket is pure GPU time.
    """
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for n in range(WARMUP):
        fn(base + n)
    st.record()
    for n in range(ITERS):
        fn(base + n)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / ITERS * 1e3


for shape in sys.argv[1:] or ["8192x8192x8192", "16384x16384x16384"]:
    M, N, K = (int(v) for v in shape.split("x"))
    flops = 2 * M * N * K
    fly, ait = setup(M, N, K)
    fly(0)  # first-call costs: module load, aiter's kernel-config lookup
    ait(0)
    torch.cuda.synchronize()

    print(f"\n=== {M}x{N}x{K} ===  ({N_SETS} input sets, warmup {WARMUP}, {PAIRS}x{ITERS} iters)")
    fly_best = ait_best = 1e9
    for p in range(PAIRS):
        f = time_us(fly, p * ITERS)
        a_ = time_us(ait, p * ITERS)
        fly_best, ait_best = min(fly_best, f), min(ait_best, a_)
        print(
            f"  pair {p}:  fly {f:8.1f} us {flops / (f / 1e6) / 1e12:6.0f} TFLOPS"
            f"   |  aiter {a_:8.1f} us {flops / (a_ / 1e6) / 1e12:6.0f} TFLOPS"
            f"   |  fly/aiter {a_ / f:.4f}x"
        )
    print(
        f"  BEST:    fly {flops / (fly_best / 1e6) / 1e12:6.0f} TFLOPS"
        f"   |  aiter {flops / (ait_best / 1e6) / 1e12:6.0f} TFLOPS"
        f"   |  fly is {(ait_best / fly_best - 1) * 100:+.2f}%"
    )
