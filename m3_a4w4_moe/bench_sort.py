# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE routing sort at MiniMax-M3 prefill shapes: aiter ``moe_sorting`` (opus,
the production sorter) vs the FlyDSL 3-stage sorter in ``sort.py``.

    cd /flydsl && PYTHONPATH=/flydsl python m3_a4w4_moe/bench_sort.py --tokens 16384 --bm 128

Checks the 3-stage output against aiter's (same padded row count, same
expert per block, every (token, slot) pair placed once inside its expert's
rows with its weight, padding rows carry the ``n_tokens`` sentinel), then times both in a HIP graph of ``copies`` calls, each
with its own routing."""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--bm", type=int, default=128)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--copies", type=int, default=8)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-check", action="store_true")
args = p.parse_args()

import aiter  # noqa: E402,F401
from aiter.fused_moe import moe_sorting  # noqa: E402

from m3_a16w4_moe.vllm_ops import import_ops  # noqa: E402

import_ops("moe_a4w4_prefill")
from moe_a4w4_prefill.sort import SortBuffers, compile_moe_sort  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, E, K, BM = args.tokens, args.experts, args.topk, args.bm


def make_routing():
    routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
    topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32).contiguous()
    w_r = torch.rand((M, K - 1), device=dev)
    w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0
    topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32).contiguous()
    return topk_ids, topk_w


class Case:
    def __init__(self):
        self.topk_ids, self.topk_w = make_routing()
        self.bufs = SortBuffers.allocate(M, E, K, BM, dev)

    def ref(self):
        return moe_sorting(self.topk_ids, self.topk_w, E, args.hidden, torch.bfloat16, block_size=BM)

    def mine(self):
        launch(*self.bufs.launch_args(self.topk_ids, self.topk_w, M))


launch = compile_moe_sort(E=E, topk=K, block_m=BM)
cases = [Case() for _ in range(max(1, args.copies))]
t0 = time.time()
cases[0].mine()
torch.cuda.synchronize()
print(f"[sort] compile+first run {time.time() - t0:.1f}s", flush=True)


def check(c: Case):
    ids_ref, w_ref, eids_ref, nv_ref, _ = c.ref()
    c.mine()
    b = c.bufs
    torch.cuda.synchronize()
    nv_r = int(nv_ref[0].item())
    nv_m = int(b.num_valid_ids[0].item())
    ok = True
    if nv_r != nv_m:
        print(f"  padded rows differ: aiter {nv_r} vs 3stage {nv_m}")
        ok = False
    nb = min(nv_r, nv_m) // BM
    if not torch.equal(eids_ref[:nb], b.sorted_expert_ids[:nb]):
        print("  sorted_expert_ids differ")
        ok = False
    ids = b.sorted_ids[:nv_m].long()
    tok = ids & 0xFFFFFF
    slot = ids >> 24
    valid = tok < M
    pad_rows = int((~valid).sum())
    pad_ref = int(((ids_ref[:nv_r].long() & 0xFFFFFF) >= M).sum())
    if pad_rows != pad_ref:
        print(f"  padding rows: 3stage {pad_rows} vs aiter {pad_ref}")
        ok = False
    if not bool((tok[~valid] == M).all()):
        print("  padding rows do not carry the n_tokens sentinel")
        ok = False
    # every pair exactly once
    pair = (tok[valid] * K + slot[valid])
    if pair.numel() != M * K or not torch.equal(pair.sort().values, torch.arange(M * K, device=dev)):
        print("  (token, slot) pairs are not a permutation")
        ok = False
    # inside its expert's rows
    e_of_row = b.sorted_expert_ids[: nv_m // BM].long().repeat_interleave(BM)[:nv_m]
    e_pair = c.topk_ids.long()[tok[valid], slot[valid]]
    if not torch.equal(e_pair, e_of_row[valid]):
        print("  some pair sits in another expert's rows")
        ok = False
    w = b.sorted_weights[:nv_m]
    if not torch.equal(w[valid], c.topk_w[tok[valid], slot[valid]]) or not bool((w[~valid] == 0).all()):
        print("  sorted_weights differ")
        ok = False
    return ok, nv_m


if not args.no_check:
    allok = True
    for i, c in enumerate(cases):
        ok, nv = check(c)
        allok &= ok
    print(f"[sort] tokens={M} topk={K} BM={BM} E={E}: 3-stage output matches aiter moe_sorting on {len(cases)} routings: {allok}; "
          f"sorted rows {nv} (pad {nv / (M * K) - 1:+.1%})", flush=True)


def time_graph(fn):
    for c in cases:
        fn(c)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in cases:
            fn(c)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    meds = []
    for _ in range(args.rounds):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(args.reps):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        meds.append(st.elapsed_time(en) * 1000.0 / args.reps / len(cases))
    meds.sort()
    return meds[len(meds) // 2], meds[0], meds[-1]


us_ref = time_graph(lambda c: c.ref())
us_mine = time_graph(lambda c: c.mine())
print(
    f"[sort] tokens={M} BM={BM} per call: aiter moe_sorting(opus) {us_ref[0]:.1f} us ({us_ref[1]:.1f}..{us_ref[2]:.1f}); "
    f"3-stage FlyDSL {us_mine[0]:.1f} us ({us_mine[1]:.1f}..{us_mine[2]:.1f})",
    flush=True,
)
