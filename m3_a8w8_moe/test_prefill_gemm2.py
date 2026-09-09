# SPDX-License-Identifier: Apache-2.0
"""Stage-2 check of moe_a8w8_prefill.gemm2 alone: stage 1 (sort + fp8 quant + gemm1)
feeds gemm2, whose bf16 partials [M*topk, H] are compared per sorted row with
w * dequant(h_q[row]) @ dequant(W2[e])^T; prints the error pattern per (n-tile, 128-col
half, 64-col wave, 16-col tile) to localize layout bugs, and run-to-run equality.

    python3 -m m3_a8w8_moe.test_prefill_gemm2 --tokens 4096 --rows 16
"""
import argparse

import torch

from m3_a16w4_moe.vllm_ops import import_ops
from m3_a8w8_moe.ref_mxfp8 import HIDDEN, INTER, NUM_EXPERTS, cos, dequant, make_weights, routing
from m3_a8w8_moe.test_prefill_gemm1 import dequant_rows, stage1, unshuffle_scale_rows

import_ops("moe_a16w4_decode")
import_ops("moe_a4w4_prefill")
import_ops("moe_a8w8_prefill")
from moe_a4w4_prefill import _run_compiled, block_m_for  # noqa: E402
from moe_a8w8_prefill import _get_gemm2, gemm2_n_split_for  # noqa: E402
from moe_a8w8_prefill.gemm2 import gemm2_grid  # noqa: E402

TOPK = 5


def run_gemm2(bufs, h_q, h_s, nmb, shuffled, n_tokens, bm):
    w13, w2, w13_s, w2_s = shuffled
    partial = torch.full(((n_tokens + 1) * TOPK, HIDDEN), float("nan"), dtype=torch.bfloat16, device=h_q.device)
    nmb2 = (nmb * bm) // 128
    n_split = gemm2_n_split_for(n_tokens)
    grid2 = gemm2_grid(nmb2, n_split)
    dummy_sc = torch.empty((16,), dtype=torch.uint8, device=h_q.device)
    _run_compiled(_get_gemm2(HIDDEN, INTER, NUM_EXPERTS, TOPK, bm, n_split, "bf16"), h_q.view(-1), w2.view(torch.uint8).view(-1),
                  partial.view(-1), h_s, w2_s.view(torch.uint8).view(-1), dummy_sc, bufs.sorted_ids, bufs.sorted_expert_ids,
                  bufs.sorted_weights, bufs.num_valid_ids, n_tokens, nmb2, grid2, torch.cuda.current_stream())
    return partial[: n_tokens * TOPK]


def probe(bufs, h_q, h_s, nmb, shuffled, m, bm, w2_q, w2_s):
    p1 = run_gemm2(bufs, h_q, h_s, nmb, shuffled, m, bm)
    torch.cuda.synchronize()
    n_valid = int(bufs.num_valid_ids[0].item())
    sorted_ids = bufs.sorted_ids[:n_valid].cpu()
    eids = bufs.sorted_expert_ids.cpu()
    sw = bufs.sorted_weights[:n_valid].cpu()
    # rows r with c = r % 768: one row per K column, sampled over the first valid rows
    valid_rows = [r for r in range(n_valid) if int(sorted_ids[r]) & 0xFFFFFF < m]
    by_c = {}
    for r in valid_rows:
        by_c.setdefault(r % INTER, r)
        if len(by_c) == INTER:
            break
    errmap = torch.zeros((INTER, HIDDEN), device=p1.device)
    for c, r in sorted(by_c.items()):
        sid = int(sorted_ids[r])
        tok, slot = sid & 0xFFFFFF, sid >> 24
        e = int(eids[r // bm])
        ref = float(sw[r]) * dequant(w2_q[e], w2_s[e])[:, c]
        got = p1[tok * TOPK + slot].float()
        errmap[c] = (got - ref).abs() / (ref.abs().max() + 1e-6)
    bad = errmap > 0.02
    print(f"   probe: {int(bad.sum())} bad (c, n) of {bad.numel()}; bad K-steps (c//128): "
          f"{[int(v) for v in bad.view(6, 128, HIDDEN).any(dim=2).sum(dim=1).tolist()]} (of 128 c each)", flush=True)
    kin = bad.view(6, 8, 16, HIDDEN).any(dim=3).sum(dim=(0, 2))  # per 16-K chunk within a step
    print(f"   bad per 16-K chunk (0..7) within a step: {kin.tolist()}", flush=True)
    nn = bad.any(dim=0).view(HIDDEN // 256, 2, 2, 4, 16).any(dim=-1).sum(dim=(0,))
    print(f"   bad output tiles per (half, wave, tile16): {nn.tolist()}", flush=True)
    nt = bad.any(dim=0).view(HIDDEN // 256, 256).any(dim=1)
    print(f"   n-tiles with any bad col: {torch.nonzero(nt).flatten().tolist()}", flush=True)
    # a couple of concrete examples
    idx = torch.nonzero(bad)[:6].tolist()
    for c, n in idx:
        r = by_c[c]
        sid = int(sorted_ids[r]); tok, slot = sid & 0xFFFFFF, sid >> 24
        e = int(eids[r // bm])
        print(f"     c={c} n={n}: got {p1[tok * TOPK + slot, n].item():.5f} "
              f"ref {(float(sw[r]) * dequant(w2_q[e], w2_s[e])[n, c]).item():.5f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[4096])
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--probe", action="store_true",
                   help="replace gemm1's output by one-hot rows (col = row %% 768, scale 1): the error map "
                        "over (K, N) shows which operand rows / K-steps gemm2 misreads")
    a = p.parse_args()
    dev = torch.device("cuda")
    raw, shuffled = make_weights(dev)
    _, _, w2_q, w2_s = raw
    for m in a.tokens:
        torch.manual_seed(m)
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(m, dev)
        bm = block_m_for(m)
        bufs, a_q, a_s, h_q, h_s, nmb = stage1(x, shuffled, ids, w)
        torch.cuda.synchronize()
        if a.probe:
            rows_all = h_q.shape[0]
            h_q.zero_()
            h_q[torch.arange(rows_all, device=dev), torch.arange(rows_all, device=dev) % INTER] = 0x38  # fp8 1.0
            h_s.fill_(127)
            probe(bufs, h_q, h_s, nmb, shuffled, m, bm, w2_q, w2_s)
            continue
        p1 = run_gemm2(bufs, h_q, h_s, nmb, shuffled, m, bm)
        p2 = run_gemm2(bufs, h_q, h_s, nmb, shuffled, m, bm)
        torch.cuda.synchronize()
        nan_rows = int(torch.isnan(p1).any(dim=1).sum())
        diff = (p1 != p2) & ~(torch.isnan(p1) & torch.isnan(p2))
        print(f"M={m}: partial rows never written {nan_rows}/{m * TOPK}; run-to-run differing rows "
              f"{int(diff.any(dim=1).sum())}, cols {int(diff.any(dim=0).sum())}", flush=True)
        if diff.any():
            cols = torch.nonzero(diff.any(dim=0)).flatten().tolist()
            print(f"   differing col n-tiles: {sorted(set(c // 256 for c in cols))}, "
                  f"col%256 in {sorted(set(c % 256 for c in cols))[:16]}...", flush=True)
        n_valid = int(bufs.num_valid_ids[0].item())
        sorted_ids = bufs.sorted_ids[:n_valid].cpu()
        eids = bufs.sorted_expert_ids.cpu()
        sw = bufs.sorted_weights[:n_valid].cpu()
        valid_rows = [r for r in range(n_valid) if int(sorted_ids[r]) & 0xFFFFFF < m]
        gen = torch.Generator().manual_seed(m)
        pick = [valid_rows[i] for i in torch.randperm(len(valid_rows), generator=gen)[: a.rows].tolist()]
        h_scale_rows = unshuffle_scale_rows(h_s, nmb * bm, INTER // 32, pick)
        hd = dequant_rows(h_q[pick], h_scale_rows, INTER)
        err = torch.zeros((HIDDEN,), device=dev)
        worst = 1.0
        for i, r in enumerate(pick):
            sid = int(sorted_ids[r])
            tok, slot = sid & 0xFFFFFF, sid >> 24
            e = int(eids[r // bm])
            ref = float(sw[r]) * (hd[i] @ dequant(w2_q[e], w2_s[e]).T)
            got = p1[tok * TOPK + slot].float()
            c = cos(got, ref)
            worst = min(worst, c)
            err += (got - ref).abs() / (ref.abs().max() + 1e-6)
        err = (err / len(pick)).view(HIDDEN // 256, 2, 2, 4, 16).mean(dim=-1)  # n-tile, half, wave, tile
        print(f"   worst row cos {worst:.5f}; mean rel err per n-tile: "
              + " ".join(f"{v:.3f}" for v in err.mean(dim=(1, 2, 3)).tolist()), flush=True)
        print("   per (half, wave, tile) averaged over n-tiles:\n   "
              + "\n   ".join(f"half{h} wave{wv}: " + " ".join(f"{v:.3f}" for v in err[:, h, wv, :].mean(dim=0).tolist())
                             for h in range(2) for wv in range(2)), flush=True)


if __name__ == "__main__":
    main()
