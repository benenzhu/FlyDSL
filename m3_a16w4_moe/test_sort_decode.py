"""Check sort_decode against aiter ``moe_sorting`` (same contract) and time it alone.
Usage: python m3_a16w4_moe/test_sort_decode.py [--module sort_decode] [--tokens 4 32 256 ...]"""
import argparse, importlib, time, torch

p = argparse.ArgumentParser()
p.add_argument("--module", default="sort_decode")
p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 256])
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--block-m", type=int, default=16)
p.add_argument("--trials", type=int, default=20)
args = p.parse_args()

from aiter.fused_moe import moe_sorting  # noqa: E402
mod = importlib.import_module(f"m3_a16w4_moe.{args.module}")
dev = "cuda"
E, K, H, BM = args.experts, args.topk, args.hidden, args.block_m
torch.manual_seed(0)


def routing(M):
    routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
    ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
    w = torch.rand((M, K), device=dev, dtype=torch.float32)
    return ids, w


def canon(sorted_ids, sorted_w, eids, nvalid, M):
    """per expert: multiset of (packed id, weight) of the real rows; padding rows must be M."""
    n = int(nvalid[0])
    assert n % BM == 0
    ids = sorted_ids[:n].cpu(); w = sorted_w[:n].cpu(); e = eids[: n // BM].cpu()
    per = {}
    for b in range(n // BM):
        rows = ids[b * BM:(b + 1) * BM]; ws = w[b * BM:(b + 1) * BM]
        real = rows & 0xFFFFFF != M
        for r, ww in zip(rows[real].tolist(), ws[real].tolist()):
            per.setdefault(int(e[b]), []).append((r, round(ww, 6)))
        # padding rows carry weight 0 and sit after the real rows
        assert bool((ws[~real] == 0).all()), "padding weight != 0"
    return {k: sorted(v) for k, v in per.items()}, n


ok = True
for M in args.tokens:
    for t in range(args.trials):
        ids, w = routing(M)
        ref = moe_sorting(ids, w, E, H, torch.bfloat16, block_size=BM)
        got = mod.moe_sort_decode(ids, w, E, H, BM)
        torch.cuda.synchronize()
        cr, nr = canon(ref[0], ref[1], ref[2], ref[3], M)
        cg, ng = canon(got[0], got[1], got[2], got[3], M)
        if cr != cg or nr != ng or not bool((got[4] == 0).all()) or got[4].shape != ref[4].shape:
            ok = False
            print(f"MISMATCH M={M} trial {t}: nvalid ref {nr} got {ng}, experts ref {len(cr)} got {len(cg)}, out zero {bool((got[4] == 0).all())}")
            bad = [k for k in set(cr) | set(cg) if cr.get(k) != cg.get(k)][:5]
            print("  first differing experts:", bad, [(cr.get(k), cg.get(k)) for k in bad[:1]])
            break
    print(f"M={M}: {'ok' if ok else 'FAIL'} ({args.trials} trials vs aiter moe_sorting)", flush=True)
    if not ok:
        break

# timing: graph of 100 different routings
if ok:
    for M in args.tokens[-3:]:
        ins = [routing(M) for _ in range(100)]
        for f, name in ((lambda i, w: mod.moe_sort_decode(i, w, E, H, BM), args.module), (lambda i, w: moe_sorting(i, w, E, H, torch.bfloat16, block_size=BM), "aiter")):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for i, w in ins[:2]:
                    f(i, w)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                for i, w in ins:
                    f(i, w)
            g.replay(); torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(10):
                g.replay()
            t1.record(); torch.cuda.synchronize()
            print(f"M={M} {name:>14s}: {t0.elapsed_time(t1) * 1000 / 10 / 100:6.2f} us per call (100 routings per graph)", flush=True)
