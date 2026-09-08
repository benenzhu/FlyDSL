# SPDX-License-Identifier: Apache-2.0
"""Lab sweep of the a16w8 gemm1 tile knobs (large_m config, W prefetch depth, waves
per EU) at a few token counts; times gemm1 alone (sorted path incl. sort_decode,
inline path incl. zeroing) in a HIP graph of 100 different inputs.

    python3 -m m3_a8w8_moe.sweep_gemm1 --tokens 64 128 256
"""
import argparse
import itertools
import time

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import HIDDEN, INTER, NUM_EXPERTS, SWIGLU_ALPHA, SWIGLU_LIMIT, make_weights, routing

import_ops("moe_a16w4_decode")
import_ops("moe_a8w8_decode")
from moe_a16w4_decode.host import _run_compiled  # noqa: E402
from moe_a16w4_decode.sort_decode import max_sorted_rows, moe_sort_decode  # noqa: E402
from moe_a8w8_decode.gemm1 import compile_gemm1  # noqa: E402

TILE_M = 16
_launches = {}


def get(n_tokens, **kw):
    launch = compile_gemm1(D_HIDDEN=HIDDEN, D_INTER=INTER, NE=NUM_EXPERTS, TOPK=5, n_tokens=n_tokens,
                           inline_sort=n_tokens <= TILE_M, **kw)
    return _launches.setdefault(launch.kernel_name, launch)


def run(launch, x, w13, w13_s, out, m, ids=None, zero_out=None, s_e=None, nvl=None, s_ids=None):
    if m <= TILE_M:
        max_mb = m * 5
        args = (0, 0, ids.data_ptr(), m, max_mb * (INTER // launch.tile_n), SWIGLU_ALPHA, SWIGLU_LIMIT,
                out.data_ptr(), zero_out.data_ptr(), zero_out.numel() * 2 // 4)
    else:
        max_mb = s_e.numel()
        args = (s_e.data_ptr(), nvl.data_ptr(), s_ids.data_ptr(), m, max_mb * (INTER // launch.tile_n),
                SWIGLU_ALPHA, SWIGLU_LIMIT, out.data_ptr(), 0, 0)
    _run_compiled(launch, x.data_ptr(), w13.data_ptr(), w13_s.data_ptr(), *args, torch.cuda.current_stream())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--large", type=int, nargs="+", default=[0, 1])
    p.add_argument("--prefetch", type=int, nargs="+", default=[3, 4, 6])
    p.add_argument("--waves", type=int, nargs="+", default=[0, 2, 3, 4], help="0 = default")
    a = p.parse_args()
    dev = torch.device("cuda")
    _, shuffled = make_weights(dev)
    w13, _, w13_s, _ = shuffled
    rows = max(TILE_M * 5 * TILE_M, max_sorted_rows(256, NUM_EXPERTS, 5, TILE_M))
    out = torch.empty((rows, INTER), dtype=torch.bfloat16, device=dev)
    for m in a.tokens:
        inputs = []
        for i in range(100):
            torch.manual_seed(1000 + i)
            xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids_i, w_i = routing(m, dev)
            zero_out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            if m <= TILE_M:
                inputs.append(dict(x=xi, ids=ids_i, zero_out=zero_out))
            else:
                s_ids, s_w, s_e, nvl = moe_sort_decode(ids_i, w_i, NUM_EXPERTS, HIDDEN, TILE_M, zero_out)
                inputs.append(dict(x=xi, s_e=s_e, nvl=nvl, s_ids=s_ids))
        torch.cuda.synchronize()
        results = []
        for large, pf, wv in itertools.product(a.large, a.prefetch, a.waves):
            kw = dict(large_m=bool(large), prefetch=pf, waves_per_eu=(None if wv == 0 else wv))
            try:
                launch = get(m, **kw)
                fn = lambda inp: run(launch, w13=w13, w13_s=w13_s, out=out, m=m, **inp)
                for inp in inputs[:3]:
                    fn(inp)
                torch.cuda.synchronize()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    for inp in inputs[:2]:
                        fn(inp)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s):
                        for inp in inputs:
                            fn(inp)
                torch.cuda.synchronize()
                meds = []
                for _ in range(3):
                    ts = []
                    for _ in range(10):
                        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) / 100 * 1e6)
                    meds.append(sorted(ts)[5])
                med = sorted(meds)[1]
                results.append((med, kw, launch.kernel_name))
                print(f"M={m} large_m={large} pf={pf} waves={wv}: {med:.1f} us  ({launch.kernel_name})", flush=True)
            except Exception as e:  # compile failure of one variant
                print(f"M={m} large_m={large} pf={pf} waves={wv}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        results.sort()
        print(f"== M={m} best: {results[0][0]:.1f} us {results[0][1]}; default-config reference is large_m={'1' if m > 128 else '0'} pf=3 waves={'3' if m > 128 else '0'}", flush=True)


if __name__ == "__main__":
    main()
