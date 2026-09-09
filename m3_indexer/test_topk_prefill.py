"""FlyDSL prefill top-k vs the vllm Triton top-k on scores from the Triton scorer."""
import argparse, random, sys
import torch
sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops  # noqa: E402
import_ops("index_bf16")
from index_bf16.host import index_topk_prefill  # noqa: E402
from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_score, minimax_m3_index_topk  # noqa: E402
BLK = 128; D = 128; TOPK = 16


def case(q_lens, ctxs, seed, pattern="rand"):
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    seq_lens = [c + q for c, q in zip(ctxs, q_lens)]
    nblocks = [-(-s // BLK) for s in seq_lens]; pool = sum(nblocks) + 8
    cache = (torch.randn(pool, BLK, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    bt = torch.zeros(len(q_lens), max(nblocks), dtype=torch.int32, device="cuda")
    perm = torch.randperm(pool, device="cuda", generator=g).to(torch.int32); o = 0
    for i, n in enumerate(nblocks):
        bt[i, :n] = perm[o:o + n]; o += n
    total_q = sum(q_lens)
    q = (torch.randn(total_q, 1, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32, device="cuda")
    t = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
    score = minimax_m3_index_score(q, cache, bt, cu, t(seq_lens), t(ctxs), max(q_lens), max(seq_lens), 1)
    if pattern == "inc":  # monotonically increasing along the row: every block beats the threshold
        score = torch.arange(score.shape[2], device="cuda", dtype=torch.float32)[None, None, :].expand_as(score).contiguous()
    elif pattern == "ties":
        score = (score * 4).round() / 4
    ref = minimax_m3_index_topk(score, cu, t(ctxs), max(q_lens), TOPK, 0, 1)
    out = index_topk_prefill(score, cu, t(ctxs), max(q_lens), TOPK, 0, 1)
    torch.cuda.synchronize()
    mism = (ref != out).any(dim=-1)[0].nonzero().flatten().tolist()
    real = 0; order_only = 0
    for r in mism:
        a, e = set(out[0, r].tolist()), set(ref[0, r].tolist())
        diff = [x for x in (a ^ e) if x >= 0]
        if len(diff) == 0:
            order_only += 1; continue
        sc = score[0, r, diff]
        if (sc.max() - sc.min()).item() > 0:
            real += 1
    return real, order_only, len(mism), total_q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--big", action="store_true"); a = ap.parse_args()
    rng = random.Random(0)
    cases = [
        (([7], [0]), "rand"), (([40], [100]), "rand"), (([300], [1000]), "rand"),
        (([512, 513, 100], [4000, 130000, 129]), "rand"), (([2048], [140000]), "rand"),
        (([1, 1029], [70000, 3]), "rand"), (([600], [90000]), "inc"), (([600], [90000]), "ties"),
    ]
    if a.big:
        cases = [(([rng.randint(6554, 8192) for _ in range(4)], [rng.randint(500000, 800000) for _ in range(4)]), "rand")]
    ok = True
    for (q_lens, ctxs), pat in cases:
        real, order_only, n, tq = case(q_lens, ctxs, seed=len(q_lens) + len(pat), pattern=pat)
        st = "OK" if real == 0 else "FAIL"; ok &= real == 0
        print(f"{st} {pat} q={q_lens} ctx={ctxs}: {real} real mismatches, {order_only} order-only, {n} rows differ of {tq}")
    sys.exit(0 if ok else 1)


main()
