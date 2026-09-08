# SPDX-License-Identifier: Apache-2.0
"""Kernel breakdown of one aiter MXFP8 MoE call (the production a8w8 path) with
torch.profiler in the lab container: which kernels, in what order, how long.

    python3 -m m3_a8w8_moe.prof_aiter --tokens 1 32 128
    python3 -m m3_a8w8_moe.prof_aiter --ours --tokens 4096 8192     # our prefill chain
"""
import argparse
import collections
import re

import torch
from torch.profiler import ProfilerActivity, profile

from m3_a8w8_moe.ref_mxfp8 import HIDDEN, aiter_mxfp8_moe, make_weights, routing


def short(n):
    return re.sub(r"\(.*", "", n)[:100]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 128])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--ours", action="store_true", help="profile moe_a8w8_prefill.a8w8_prefill_moe instead")
    p.add_argument("--ours-decode", action="store_true", help="profile moe_a8w8_decode.a16w8_decode_moe instead")
    a = p.parse_args()
    dev = torch.device("cuda")
    _, shuffled = make_weights(dev)
    fn = aiter_mxfp8_moe
    if a.ours:
        from m3_a8w8_moe.test_prefill_chain import ours

        fn = lambda x, shuffled, w, ids: ours(x, shuffled, w, ids)  # noqa: E731
    if a.ours_decode:
        from m3_a8w8_moe.test_chain import ours as ours_dec

        fn = lambda x, shuffled, w, ids: ours_dec(x, shuffled, w, ids)  # noqa: E731
    for m in a.tokens:
        inputs = []
        for i in range(a.iters + 2):
            torch.manual_seed(100 + i)
            x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            ids, w = routing(m, dev)
            inputs.append((x, w, ids))
        for x, w, ids in inputs[:2]:
            fn(x, shuffled, w, ids)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for x, w, ids in inputs[2:]:
                fn(x, shuffled, w, ids)
            torch.cuda.synchronize()
        ev = sorted(
            (e for e in prof.events() if getattr(e, "device_type", None) is not None and "cuda" in str(e.device_type).lower()),
            key=lambda e: e.time_range.start,
        )
        kern = [(e.time_range.start, e.time_range.elapsed_us(), e.name) for e in ev]
        n_per = len(kern) // a.iters
        print(f"\n=== M={m}: {len(kern)} GPU events over {a.iters} calls ({n_per} per call) ===")
        tot = collections.defaultdict(float)
        cnt = collections.Counter()
        for _, d, n in kern:
            tot[short(n)] += d / a.iters
            cnt[short(n)] += 1
        for n, d in sorted(tot.items(), key=lambda kv: -kv[1]):
            print(f"  {d:8.1f} us/call  x{cnt[n] // a.iters}  {n}")
        print(f"  ---- sum {sum(tot.values()):.1f} us/call; sequence of the last call:")
        last = kern[-n_per:] if n_per else kern
        t_prev = None
        for ts, d, n in last:
            gap = (ts - t_prev) if t_prev is not None else 0.0
            print(f"    {d:8.1f} us (+{gap:6.1f})  {short(n)}")
            t_prev = ts + d


if __name__ == "__main__":
    main()
