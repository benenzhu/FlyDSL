"""FlyDSL decode scorer: scores vs a torch reference (small), and the full decode
(scorer + Triton fused top-k/sparse table) vs upstream minimax_m3_index_decode."""
import argparse, random, sys
import torch
sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops  # noqa: E402
import_ops("index_bf16")
from index_bf16.host import index_decode, index_score_decode  # noqa: E402
from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_decode  # noqa: E402
from vllm.models.minimax_m3.amd.ops.sparse_pa import PAGES_PER_SPARSE_BLOCK  # noqa: E402
BLK = 128; D = 128; TOPK = 16; INIT = 0; LOCAL = 1


def make(seq_lens, qlen, seed):
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    nblocks = [-(-s // BLK) for s in seq_lens]; pool = sum(nblocks) + 8
    cache = (torch.randn(pool, BLK, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    bt = torch.zeros(len(seq_lens), max(nblocks), dtype=torch.int32, device="cuda")
    perm = torch.randperm(pool, device="cuda", generator=g).to(torch.int32); o = 0
    for i, n in enumerate(nblocks):
        bt[i, :n] = perm[o:o + n]; o += n
    M = len(seq_lens) * qlen
    q = (torch.randn(M, 1, D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    return cache, bt, q, torch.tensor(seq_lens, dtype=torch.int32, device="cuda")


def ref_scores(cache, bt, q, seq_lens, qlen):
    R = seq_lens.numel(); M = R * qlen; max_block = -(-int(seq_lens.max()) // BLK); S = -(-max_block // 16) * 16
    out = torch.full((1, M, S), float("-inf"), device="cuda")
    for r in range(R):
        L = int(seq_lens[r]); nb = -(-L // BLK)
        k = cache[bt[r, :nb].long()].float().view(nb * BLK, D)  # [tokens, D]
        for qi in range(qlen):
            pos = L - qlen + qi; kv = pos + 1
            s = (k[:kv] @ q[r * qlen + qi, 0].float())  # [kv]
            nbq = -(-kv // BLK)
            padded = torch.full((nbq * BLK,), float("-inf"), device="cuda"); padded[:kv] = s
            bs = padded.view(nbq, BLK).max(dim=1).values
            loc0 = max(0, nbq - LOCAL)
            bs[loc0:] = 1e29
            bs[:INIT] = 1e30
            out[0, r * qlen + qi, :nbq] = bs
    return out


def check_scores(seq_lens, qlen, seed):
    cache, bt, q, sl = make(seq_lens, qlen, seed)
    got = index_score_decode(q, cache, bt, sl, int(sl.max()), INIT, LOCAL, qlen); torch.cuda.synchronize()
    ref = ref_scores(cache, bt, q, sl, qlen)
    bad = 0; worst = 0.0
    for r, L in enumerate(seq_lens):
        nb = -(-L // BLK)
        for qi in range(qlen):
            a, e = got[0, r * qlen + qi, :nb], ref[0, r * qlen + qi, :nb]
            fin = torch.isfinite(e)
            if not torch.equal(torch.isfinite(a), fin): bad += 1; continue
            big = e.abs() > 1e28
            d = (a - e).abs()
            if big.any() and not torch.equal(a[big], e[big]): bad += 1; continue
            m = d[fin & ~big].max().item() if (fin & ~big).any() else 0.0
            worst = max(worst, m); bad += int(m > 2e-3)
    return worst, bad


def check_decode(seq_lens, qlen, seed):
    cache, bt, q, sl = make(seq_lens, qlen, seed)
    M = q.shape[0]
    args = (q, cache, bt, sl, int(sl.max()), TOPK, INIT, LOCAL, 1, qlen, qlen)
    def run(fn):
        out = torch.empty(1, M, TOPK, dtype=torch.int32, device="cuda")
        sbt = torch.empty(M, TOPK * PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device="cuda")
        sctx = torch.empty(M, dtype=torch.int32, device="cuda")
        cnt = torch.zeros(1, M, dtype=torch.int32, device="cuda")
        fn(*args, out=out, attention_block_table=bt, sparse_block_table_out=sbt, sparse_context_lens_out=sctx,
           block_page_stride=PAGES_PER_SPARSE_BLOCK, completion_counter=cnt)
        torch.cuda.synchronize(); return out, sbt, sctx
    o1, s1, c1 = run(minimax_m3_index_decode); o2, s2, c2 = run(index_decode)
    rows = (o1 != o2).any(dim=-1)[0].nonzero().flatten().tolist()
    order_only = sum(1 for r in rows if set(o1[0, r].tolist()) == set(o2[0, r].tolist()))
    return len(rows) - order_only, order_only, M, torch.equal(c1, c2), int((s1 != s2).any(dim=-1).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--big", action="store_true"); a = ap.parse_args()
    ok = True
    if not a.big:
        for seq_lens, qlen in (([300], 1), ([1000, 129], 4), ([3000, 2500, 777], 4), ([5000] * 9, 3), ([70000, 3], 4)):
            worst, bad = check_scores(seq_lens, qlen, seed=len(seq_lens))
            st = "OK" if bad == 0 else "FAIL"; ok &= bad == 0
            print(f"{st} scores seq={seq_lens} qlen={qlen}: max|d|={worst:.2e} bad_rows={bad}")
    rng = random.Random(1)
    cases = [([140000], 4), ([rng.randint(500000, 800000) for _ in range(2)], 4), ([rng.randint(500000, 800000) for _ in range(8)], 4)]
    if a.big:
        cases = [([rng.randint(500000, 800000) for _ in range(64)], 4)]
    for seq_lens, qlen in cases:
        real, oo, M, ctx_eq, sbt_diff = check_decode(seq_lens, qlen, seed=len(seq_lens) + 7)
        st = "OK" if (real == 0 and ctx_eq) else "FAIL"; ok &= st == "OK"
        print(f"{st} decode reqs={len(seq_lens)} qlen={qlen}: {real} real mismatches, {oo} order-only of {M}; ctx_lens equal={ctx_eq}, sparse-table rows differ={sbt_diff}")
    sys.exit(0 if ok else 1)


main()
