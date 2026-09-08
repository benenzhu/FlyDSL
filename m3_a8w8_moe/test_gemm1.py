# SPDX-License-Identifier: Apache-2.0
"""Stage-1 check of moe_a8w8_decode.gemm1 (a16w8): sorted path (sort_decode from the
a16w4 package) and inline-sort path, against a float reference built from the
dequantized fp8 experts. Also times gemm1 alone in a HIP graph.

    python3 -m m3_a8w8_moe.test_gemm1 --tokens 4 32 128 256
"""
import argparse
import time

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import (
    HIDDEN, INTER, NUM_EXPERTS, SWIGLU_ALPHA, SWIGLU_LIMIT, cos, dequant, make_weights, routing,
)

import_ops("moe_a16w4_decode")
import_ops("moe_a8w8_decode")
from moe_a16w4_decode.sort_decode import max_sorted_rows, moe_sort_decode  # noqa: E402
from moe_a8w8_decode.gemm1 import BM, compile_gemm1  # noqa: E402

TILE_M = 16
_launches = {}


def get_gemm1(n_tokens, inline):
    launch = compile_gemm1(D_HIDDEN=HIDDEN, D_INTER=INTER, NE=NUM_EXPERTS, TOPK=5,
                           n_tokens=n_tokens, inline_sort=inline)
    return _launches.setdefault(launch.kernel_name, launch)


def run_compiled(exe, *args):
    import flydsl.compiler as flyc
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def gemm1(x, w13, w13_s, out, n_tokens, inline, topk_ids=None, zero_out=None,
          sorted_eids=None, num_valid=None, sorted_ids=None):
    launch = get_gemm1(n_tokens, inline)
    if inline:
        max_mb = n_tokens * 5
        eids_ptr, cumsum_ptr, mind_ptr = 0, 0, topk_ids.data_ptr()
        zero_ptr, zero_dw = zero_out.data_ptr(), zero_out.numel() * 2 // 4
    else:
        max_mb = sorted_eids.numel()
        eids_ptr, cumsum_ptr, mind_ptr = sorted_eids.data_ptr(), num_valid.data_ptr(), sorted_ids.data_ptr()
        zero_ptr, zero_dw = 0, 0
    grid = max_mb * (INTER // launch.tile_n)
    run_compiled(launch, x.data_ptr(), w13.data_ptr(), w13_s.data_ptr(), eids_ptr, cumsum_ptr, mind_ptr,
                 int(n_tokens), int(grid), float(SWIGLU_ALPHA), float(SWIGLU_LIMIT), out.data_ptr(),
                 int(zero_ptr), int(zero_dw), torch.cuda.current_stream())
    return out


def stage1_ref(x, w13_q, w13_s, e):
    h = x.float() @ dequant(w13_q[e], w13_s[e]).T
    g = h[:, :INTER].clamp(max=SWIGLU_LIMIT)
    u = h[:, INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)


def check(m, raw, shuffled, dev, time_it=True):
    w13_q, w13_s, _, _ = raw
    w13, _, w13_sk, _ = shuffled
    torch.manual_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
    ids, w = routing(m, dev)
    inline = m <= TILE_M
    rows = max(TILE_M * 5 * TILE_M, max_sorted_rows(256, NUM_EXPERTS, 5, TILE_M))
    out = torch.full((rows, INTER), float("nan"), dtype=torch.bfloat16, device=dev)
    zero_out = torch.ones((m, HIDDEN), dtype=torch.bfloat16, device=dev)
    if inline:
        gemm1(x, w13, w13_sk, out, m, True, topk_ids=ids, zero_out=zero_out)
        torch.cuda.synchronize()
        assert zero_out.abs().max().item() == 0.0, "pair-0 blocks did not zero the output"
        # rows: pair p = token*5 + slot owns m-block p only if it is the first pair of
        # its expert; row r of that block = r-th pair with that expert (pair order)
        pairs = ids.view(-1).tolist()
        first = {}
        for p, e in enumerate(pairs):
            first.setdefault(e, p)
        got, want = [], []
        for e, p0 in first.items():
            members = [p for p, ee in enumerate(pairs) if ee == e]
            toks = [p // 5 for p in members]
            want.append(stage1_ref(x[toks], w13_q, w13_s, e))
            got.append(out[p0 * TILE_M: p0 * TILE_M + len(members)].float())
    else:
        sorted_ids, sorted_w, sorted_eids, num_valid = moe_sort_decode(ids, w, NUM_EXPERTS, HIDDEN, TILE_M, zero_out)
        gemm1(x, w13, w13_sk, out, m, False, sorted_eids=sorted_eids, num_valid=num_valid, sorted_ids=sorted_ids)
        torch.cuda.synchronize()
        nv = int(num_valid[0])
        sid = sorted_ids[:nv].tolist()
        eid = sorted_eids.tolist()
        got, want = [], []
        for r in range(nv):
            tok = sid[r] & 0xFFFFFF
            if tok >= m:
                continue
            e = eid[r // TILE_M]
            got.append(out[r: r + 1].float())
            want.append(stage1_ref(x[tok: tok + 1], w13_q, w13_s, e))
    got, want = torch.cat(got), torch.cat(want)
    c = cos(got, want)
    md = (got - want).abs().max().item()
    print(f"M={m} {'inline' if inline else 'sorted'}: rows {got.shape[0]} cos {c:.6f} max|d| {md:.4f} (ref max {want.abs().max().item():.3f}) nan {int(torch.isnan(got).sum())}", flush=True)
    if not time_it:
        return
    # timing: graph of 100 different inputs (sort excluded on the sorted path is not
    # possible here; the sorted path times sort + gemm1)
    inputs = []
    for i in range(100):
        torch.manual_seed(1000 + i)
        xi = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        inputs.append((xi, *routing(m, dev)))

    def fn(xi, ids_i, w_i):
        if inline:
            gemm1(xi, w13, w13_sk, out, m, True, topk_ids=ids_i, zero_out=zero_out)
        else:
            s_ids, s_w, s_e, nvl = moe_sort_decode(ids_i, w_i, NUM_EXPERTS, HIDDEN, TILE_M, zero_out)
            gemm1(xi, w13, w13_sk, out, m, False, sorted_eids=s_e, num_valid=nvl, sorted_ids=s_ids)

    for inp in inputs[:3]:
        fn(*inp)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for inp in inputs[:2]:
            fn(*inp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for inp in inputs:
                fn(*inp)
    torch.cuda.synchronize()
    meds = []
    for _ in range(3):
        ts = []
        for _ in range(10):
            torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / 100 * 1e6)
        meds.append(sorted(ts)[5])
    print(f"[a16w8 gemm1{'' if inline else ' + sort'}] M={m}: {sorted(meds)[1]:.1f} us (rounds {', '.join(f'{v:.1f}' for v in meds)})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[4, 32, 128, 256])
    p.add_argument("--no-time", action="store_true")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    for m in a.tokens:
        check(m, raw, shuffled, dev, time_it=not a.no_time)


if __name__ == "__main__":
    main()
