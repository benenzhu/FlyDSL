"""Correctness + timing of ``m3_a4w4_moe.gemm1`` (MiniMax-M3 prefill MoE stage 1).

Run inside the vLLM image (its aiter provides the production prologue:
``moe_sorting`` + ``fused_dynamic_mx_quant_moe_sort``):

    PYTHONPATH=/flydsl python /flydsl/m3_a4w4_moe/bench_gemm1.py --tokens 16384

Checks (``--check-rows``): sampled valid sorted rows against a float reference
built from the *same* fp4 activations / weights (dequantised), with the
kernel's own e8m0 rule re-applied in torch, so the fp4 payload and the scale
bytes can be compared almost bit-exactly (only accumulation-order flips).
Timing: HIP graph of ``--copies`` launches with different inputs, replayed
``--reps`` times, median per-call time.
"""

import argparse
import statistics
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--bm", type=int, default=128, help="sort / gemm1 block_m (128 or 256)")
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129, help="128 routed + 1 shared (id E-1)")
p.add_argument("--topk", type=int, default=5, help="4 routed + shared")
p.add_argument("--copies", type=int, default=8, help="inputs per graph (each call = its own routing)")
p.add_argument("--reps", type=int, default=20)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--check-rows", type=int, default=256, help="sampled sorted rows to verify (0 = skip)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-xcd", action="store_true", help="plain block order instead of the XCD-aware remap")
p.add_argument("--dump-ir", action="store_true")
p.add_argument("--fake-dense", choices=["rows", "gather"], default=None,
               help="after the real prologue, overwrite routing: 'rows' = expert 0 everywhere + contiguous token "
                    "rows (pure kernel overhead vs the dense kernel); 'gather' = expert 0 everywhere but the real "
                    "gathered rows (isolates the A gather from expert switching); disables the check")
p.add_argument("--wgm", type=int, default=4, help="m-tiles per XCD group in the block remap")
p.add_argument("--kernel", choices=["2x2", "s3", "1x4", "persist"], default="2x2",
               help="2x2 = gemm1.py (4-wave 2x2 quadrants); 1x4 = gemm1_1x4.py (Kimi v36 port, BM128 only); "
                    "persist = gemm1_persist.py (2x2, one CTA per CU, cross-block pipelined); "
                    "s3 = gemm1_s3.py (2x2, 3-stage LDS ring, DMA three K-steps ahead, BM128)")
p.add_argument("--ctas", type=int, default=256, help="persist: number of CTAs (multiple of 8)")
p.add_argument("--order", choices=["expert", "xcd"], default="expert",
               help="block order: expert = host tile_map, n-slab-major per expert (default); xcd = dense-style WGM groups")
args = p.parse_args()

import flydsl.compiler as flyc  # noqa: E402
from m3_a4w4_moe.gemm1 import SWIGLU_ALPHA, SWIGLU_LIMIT, compile_moe_gemm1  # noqa: E402
from m3_a4w4_moe.gemm1_1x4 import compile_moe_gemm1_1x4, ptr_arg  # noqa: E402
from m3_a4w4_moe.gemm1_persist import compile_moe_gemm1_persist  # noqa: E402
from m3_a4w4_moe.gemm1_s3 import compile_moe_gemm1_s3  # noqa: E402

import aiter  # noqa: E402,F401
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort, per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K, BM = args.tokens, args.hidden, args.inter, args.experts, args.topk, args.bm


def u8(t):
    return t.view(torch.uint8)


# ---- e8m0 shuffle inverse (aiter fp4_utils.e8m0_shuffle: per 32-row x 8-col 256-B block,
#      byte = s*64 + r*4 + q*2 + h with r=row%16, h=row//16, q=col//4, s=col%4) ----
def e8m0_unshuffle(flat, rows, cols):
    assert rows % 32 == 0 and cols % 8 == 0
    t = flat.reshape(rows // 32, cols // 8, 4, 16, 2, 2)  # (g, c8, s, r, q, h)
    return t.permute(0, 5, 3, 1, 4, 2).reshape(rows, cols)  # (g, h, r, c8, q, s)


_t = torch.randint(0, 255, (256, 24), dtype=torch.uint8, device=dev)
_ts = fp4_utils.e8m0_shuffle(_t.view(torch.float8_e8m0fnu)).view(torch.uint8).reshape(-1)
assert torch.equal(e8m0_unshuffle(_ts[: 256 * 24], 256, 24), _t), "e8m0_unshuffle is not the inverse"

# ---- weights: random bf16 -> mxfp4 per-1x32, production (AITER_MXFP4_MXFP4) layouts ----
w1 = torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=dev) * 0.02
w1_q, w1_s = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(E, 2 * I, H // 2)
w1_s = w1_s.view(E * 2 * I, H // 32)
w1_k = shuffle_weight(w1_q, layout=(16, 16))
w1_sk = fp4_utils.e8m0_shuffle(w1_s)
del w1


def make_input():
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
    routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
    topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
    w_r = torch.rand((M, K - 1), device=dev)
    w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0
    topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32)
    return x, topk_ids, topk_w


def prologue(x, topk_ids, topk_w):
    """Production a4w4 prologue: aiter moe_sorting(block_m) + fused per-token fp4 quant
    with the activation scales sorted into the e8m0-shuffled tile layout."""
    sorted_ids, sorted_w, sorted_eids, num_valid, _moe_buf = moe_sorting(
        topk_ids, topk_w, E, H, torch.bfloat16, block_size=BM
    )
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(x, sorted_ids, num_valid, token_num=M, topk=K, block_size=BM)
    return sorted_ids, sorted_w, sorted_eids, num_valid, a_q, a_s


def build_tile_map(sorted_eids, num_valid, num_m_blocks):
    """int32 table, one entry per launched block: m_tile << 3 | n_tile, or -1.
    Order: expert by expert (experts appear in sorted order), inside an expert
    n-slab-major (all its m-tiles for n=0, then n=1, ...). Pure device ops, no
    host sync (num_valid stays on the GPU)."""
    nb = num_m_blocks
    dev = sorted_eids.device
    NB_N = I // 128
    m_idx = torch.arange(nb, device=dev)
    valid_m = (m_idx * BM) < num_valid[0]
    e_m = torch.where(valid_m, sorted_eids[:nb].long(), torch.full_like(m_idx, E))  # dummy expert E
    hist = torch.zeros(E + 1, dtype=torch.long, device=dev).scatter_add_(0, e_m, torch.ones_like(e_m))
    starts = torch.cumsum(hist, 0) - hist
    rank = m_idx - starts[e_m]
    cnt = hist[e_m]
    grid = (nb * NB_N + 7) // 8 * 8
    n = torch.arange(NB_N, device=dev)
    idx = (NB_N * starts[e_m])[:, None] + n[None, :] * cnt[:, None] + rank[:, None]
    idx = torch.where(valid_m[:, None], idx, torch.full_like(idx, grid))  # invalid -> spare slot
    val = (m_idx[:, None] << 3) | n[None, :]
    tm = torch.full((grid + 1,), -1, dtype=torch.long, device=dev)
    tm.scatter_(0, idx.reshape(-1), val.reshape(-1))
    tm[grid] = valid_m.sum() * NB_N  # valid entries are [0, n_valid); the persistent kernel reads this
    return tm.to(torch.int32).contiguous(), grid


class Case:
    def __init__(self):
        self.x, self.topk_ids, self.topk_w = make_input()
        self.sorted_ids, self.sorted_w, self.sorted_eids, self.num_valid, self.a_q, self.a_s = prologue(
            self.x, self.topk_ids, self.topk_w
        )
        if args.fake_dense == "rows":
            n_rows = self.sorted_ids.shape[0]
            self.sorted_ids = (torch.arange(n_rows, device=dev, dtype=torch.int32) % M).contiguous()
        if args.fake_dense:
            self.sorted_eids = torch.zeros_like(self.sorted_eids)
        self.num_m_blocks = int(self.sorted_eids.shape[0])
        self.tile_map, self.grid = build_tile_map(self.sorted_eids, self.num_valid, self.num_m_blocks)
        self.dbg = torch.zeros(args.ctas, dtype=torch.int32, device=dev)  # persist: progress markers (debug)
        rows = self.num_m_blocks * BM
        self.out_q = torch.empty((rows, I // 2), dtype=torch.uint8, device=dev)
        self.out_s = torch.empty((rows * (I // 32),), dtype=torch.uint8, device=dev)

    def args(self):
        return (
            u8(self.a_q).contiguous().view(-1),
            u8(w1_k).contiguous().view(-1),
            self.out_q.view(-1),
            u8(self.a_s).contiguous().view(-1),
            u8(w1_sk).contiguous().view(-1),
            self.out_s,
            self.sorted_ids.contiguous(),
            self.sorted_eids.contiguous(),
            self.num_valid.contiguous(),
            M,
            self.num_m_blocks,
            int(u8(self.a_s).numel()),
            self.tile_map,
            self.grid,
            torch.cuda.current_stream(),
        ) + ((ptr_arg(self.dbg),) if args.kernel == "persist" else ())

    def args_1x4(self):
        def pa(t):
            assert t.is_contiguous()
            return ptr_arg(t)

        return (
            pa(self.out_q),
            pa(self.out_s),
            pa(u8(self.a_q)),
            pa(u8(w1_k)),
            pa(u8(self.a_s)),
            pa(u8(w1_sk)),
            pa(self.sorted_eids),
            pa(self.sorted_ids),
            pa(self.tile_map),
            M,
            self.num_m_blocks,
            int(u8(self.a_s).numel()),
            self.grid,
            torch.cuda.current_stream(),
        )


t0 = time.time()
cases = [Case() for _ in range(max(1, args.copies))]
torch.cuda.synchronize()
c0 = cases[0]
nv = int(c0.num_valid[0].item())
print(
    f"[gemm1] M={M} topk={K} BM={BM}: sorted rows valid {nv} / alloc {c0.num_m_blocks * BM} "
    f"(pad {nv / (M * K) - 1:+.1%}), a_s rows {u8(c0.a_s).numel() // (H // 32)}, prologue x{len(cases)} {time.time() - t0:.1f}s",
    flush=True,
)

t0 = time.time()
if args.kernel == "1x4":
    assert BM == 128 and args.order == "expert", "1x4 kernel: BM128 + expert-order tile_map only"
    launch_1x4 = compile_moe_gemm1_1x4(H=H, I=I, E=E, tile_m=BM)

    def call(case):
        launch_1x4(*case.args_1x4())
else:
    if args.kernel == "persist":
        assert args.order == "expert"
        launch = compile_moe_gemm1_persist(H=H, I=I, E=E, BLOCK_M=BM, n_cta=args.ctas)
    elif args.kernel == "s3":
        launch = compile_moe_gemm1_s3(H=H, I=I, E=E, BLOCK_M=BM, use_xcd_remap=not args.no_xcd, xcd_wgm=args.wgm,
                                      tile_map=args.order == "expert")
    else:
        launch = compile_moe_gemm1(H=H, I=I, E=E, BLOCK_M=BM, use_xcd_remap=not args.no_xcd, xcd_wgm=args.wgm,
                                   tile_map=args.order == "expert")
    fn = flyc.compile(launch, *c0.args())

    def call(case):
        fn(*case.args())
print(f"[gemm1:{args.kernel}] compile {time.time() - t0:.1f}s", flush=True)
call(c0)
torch.cuda.synchronize()
_q1, _s1 = c0.out_q.clone(), c0.out_s.clone()
call(c0)
torch.cuda.synchronize()
_nv = int(c0.num_valid[0].item())
_dq = (c0.out_q != _q1)[:_nv]
_ds = c0.out_s != _s1
print(f"[gemm1] determinism: run 2 vs run 1 fp4 bytes differing {int(_dq.sum())} (rows {int(_dq.any(dim=1).sum())} of {_nv}), scale bytes differing {int(_ds.sum())}", flush=True)

# ---- correctness on sampled valid rows ----
if args.check_rows > 0 and not args.fake_dense:
    xq_ref, xs_ref = per_1x32_f4_quant(c0.x, quant_dtype=dtypes.fp4x2)
    xq_ref = u8(xq_ref).view(M, H // 2)
    xs_ref = u8(xs_ref).view(M, H // 32)
    same_q = torch.equal(u8(c0.a_q).view(M, H // 2), xq_ref)
    a_s_rows = u8(c0.a_s).numel() // (H // 32)
    a_s_unsh = e8m0_unshuffle(u8(c0.a_s).reshape(-1), a_s_rows, H // 32)
    sid = c0.sorted_ids[:nv]
    tok = (sid & 0xFFFFFF).long()
    real = tok < M
    same_s = torch.equal(a_s_unsh[:nv][real], xs_ref[tok[real]])
    print(f"[gemm1] prologue check: a_q == per_1x32 quant {same_q}; sorted+shuffled a_s == e8m0_shuffle(gathered) {same_s}")

    out_s_unsh = e8m0_unshuffle(c0.out_s, c0.num_m_blocks * BM, I // 32)
    rows_all = torch.nonzero(real).flatten()
    sel = rows_all[torch.randperm(rows_all.numel(), device=dev)[: args.check_rows]].sort().values
    w1_deq_cache = {}

    def w1_deq(e):
        if e not in w1_deq_cache:
            v = fp4_utils.mxfp4_to_f32(w1_q[e].reshape(-1)).view(2 * I, H)
            sc = fp4_utils.e8m0_to_f32(w1_s[e * 2 * I : (e + 1) * 2 * I].reshape(-1)).view(2 * I, H // 32)
            w1_deq_cache[e] = (v.view(2 * I, H // 32, 32) * sc.unsqueeze(-1)).view(2 * I, H)
        return w1_deq_cache[e]

    def quant_ref(h):
        """production's inter-stage quant: the stage-1 values are bf16, e8m0 =
        ceil_pow2(amax / 6) (aiter RoundUp), RNE fp4 of value / 2^(e8m0 - 127)"""
        hg = h.to(torch.bfloat16).float().view(-1, 32)
        amax = hg.abs().amax(dim=1)
        u = (amax * (1.0 / 6.0)).view(torch.int32)
        e8 = (u >> 23) & 0xFF
        e8 = e8 + (((u & 0x7FFFFF) != 0) & (e8 < 255)).int()
        scale = torch.ldexp(torch.ones_like(amax), (e8 - 127))
        q = hg / scale.unsqueeze(1)
        # RNE onto the fp4 grid {0,.5,1,1.5,2,3,4,6}
        grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=h.device)
        qa = q.abs().clamp(max=6.0)
        idx = torch.bucketize(qa, (grid[:-1] + grid[1:]) / 2)  # nearest below/above midpoints
        # ties -> even: midpoints hit exactly go to the even-mantissa neighbour
        mid = (grid[:-1] + grid[1:]) / 2
        tie = (qa.unsqueeze(-1) == mid).any(-1)
        qv = grid[idx]
        # even neighbour on ties: grid indices with even "mantissa" are 0,1(0.5? no) -> handle by
        # picking the lower value when the lower grid index is even
        lo = torch.clamp(idx - 1, min=0)
        qv = torch.where(tie & (lo % 2 == 0), grid[lo], qv)
        return torch.copysign(qv, q) * scale.unsqueeze(1), e8.to(torch.uint8)

    n_bad_q, n_bad_s, n_tot = 0, 0, 0
    worst = 0.0
    cos_all = []
    for r in sel.tolist():
        t = int(tok[r])
        e = int(c0.sorted_eids[r // BM])
        xv = fp4_utils.mxfp4_to_f32(xq_ref[t]).view(H)
        xs = fp4_utils.e8m0_to_f32(xs_ref[t]).view(H // 32)
        xd = (xv.view(H // 32, 32) * xs.unsqueeze(-1)).view(H)
        hh = xd @ w1_deq(e).T
        g = hh[:I].clamp(max=SWIGLU_LIMIT)
        uu = hh[I:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h_ref = g * torch.sigmoid(SWIGLU_ALPHA * g) * (uu + 1.0)
        q_ref, s_ref = quant_ref(h_ref)
        q_k = fp4_utils.mxfp4_to_f32(c0.out_q[r]).view(I)
        s_k = out_s_unsh[r]
        sk_f = fp4_utils.e8m0_to_f32(s_k).view(I // 32)
        h_k = (q_k.view(I // 32, 32) * sk_f.unsqueeze(-1)).view(I)
        n_bad_s += int((s_k != s_ref).sum())
        n_bad_q += int((h_k != q_ref.view(I)).sum())
        n_tot += I
        worst = max(worst, float((h_k - h_ref).abs().max() / (h_ref.abs().max() + 1e-6)))
        cos_all.append(float((h_k @ h_ref) / (h_k.norm() * h_ref.norm() + 1e-12)))
    print(
        f"[gemm1] check {len(sel)} rows: scale bytes mismatched {n_bad_s}/{n_tot // 32}, "
        f"fp4 values mismatched {n_bad_q}/{n_tot} ({n_bad_q / n_tot:.2%}); cos vs float ref "
        f"min {min(cos_all):.5f} mean {statistics.mean(cos_all):.5f}; worst |err|/max {worst:.3f}",
        flush=True,
    )

# ---- timing: graph of `copies` calls, each its own input ----
for c in cases:
    call(c)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for c in cases:
        call(c)
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
us = meds[len(meds) // 2]
flop_useful = 2.0 * (M * K) * (2 * I) * H
flop_padded = 2.0 * nv * (2 * I) * H
print(
    f"[gemm1:{args.kernel}] tokens={M} BM={BM} per call: median {us:.1f} us (range {meds[0]:.1f}..{meds[-1]:.1f}), "
    f"{flop_useful / us / 1e9:.2f} PF/s useful ({flop_padded / us / 1e9:.2f} incl. padding)",
    flush=True,
)
