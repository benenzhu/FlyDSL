"""Correctness + timing of ``m3_a4w4_moe.gemm2`` (MiniMax-M3 prefill MoE stage 2,
token-major output; ``--out bf16`` = production layout bf16(y * w), ``--out fp8`` =
mxfp8 route-out).

    PYTHONPATH=/flydsl python /flydsl/m3_a4w4_moe/bench_gemm2.py --tokens 16384

Inputs: the production prologue (aiter ``moe_sorting`` + fused fp4 quant) and
``gemm1.py`` (bit-exact with production stage 1) produce the sorted fp4 A tile;
W2 is random bf16 -> mxfp4 in the production preshuffled layouts.
Check (sampled valid rows, fp32 reference from the dequantised fp4 inputs):
bf16 mode counts the rows / values bit-identical to ``bf16(ref * w)`` (differences
are fp32 summation order only); fp8 mode compares bytes against the same
reference quantised in torch with the kernel's e8m0 rule.
Timing: HIP graph of ``--copies`` calls with different inputs.
"""

import argparse
import statistics
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--n-split", type=int, default=2, help="CTAs per m-tile (1, 2, 4, 6, 12)")
p.add_argument("--out", choices=["bf16", "fp8"], default="bf16", help="gemm2 output mode")
p.add_argument("--copies", type=int, default=8)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--check-rows", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dump-ir", action="store_true")
p.add_argument("--fake-expert0", action="store_true", help="timing only: every m-tile uses expert 0 (W2 slab stays in L2)")
args = p.parse_args()

import flydsl.compiler as flyc  # noqa: E402
from m3_a4w4_moe.gemm1 import compile_moe_gemm1  # noqa: E402
from m3_a4w4_moe.gemm2 import compile_moe_gemm2, gemm2_grid  # noqa: E402

import aiter  # noqa: E402,F401
from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort, per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K = args.tokens, args.hidden, args.inter, args.experts, args.topk
BM = 128


def u8(t):
    return t.view(torch.uint8)


def e8m0_unshuffle(flat, rows, cols):
    t = flat.reshape(rows // 32, cols // 8, 4, 16, 2, 2)  # (g, c8, s, r, q, h)
    return t.permute(0, 5, 3, 1, 4, 2).reshape(rows, cols)  # (g, h, r, c8, q, s)


# ---- weights ----
w1 = torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=dev) * 0.02
w1_q, w1_s = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(E, 2 * I, H // 2)
w1_s = w1_s.view(E * 2 * I, H // 32)
w1_k = shuffle_weight(w1_q, layout=(16, 16))
w1_sk = fp4_utils.e8m0_shuffle(w1_s)
del w1
w2 = torch.randn((E, H, I), dtype=torch.bfloat16, device=dev) * 0.02
w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
w2_q = w2_q.view(E, H, I // 2)
w2_s = w2_s.view(E * H, I // 32)
w2_k = shuffle_weight(w2_q, layout=(16, 16))
w2_sk = fp4_utils.e8m0_shuffle(w2_s)
del w2


def make_input():
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
    routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
    topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
    w_r = torch.rand((M, K - 1), device=dev)
    w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0
    topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32)
    return x, topk_ids, topk_w


def build_tile_map(sorted_eids, num_valid, num_m_blocks):
    nb = num_m_blocks
    NB_N = I // 128
    m_idx = torch.arange(nb, device=dev)
    valid_m = (m_idx * BM) < num_valid[0]
    e_m = torch.where(valid_m, sorted_eids[:nb].long(), torch.full_like(m_idx, E))
    hist = torch.zeros(E + 1, dtype=torch.long, device=dev).scatter_add_(0, e_m, torch.ones_like(e_m))
    starts = torch.cumsum(hist, 0) - hist
    rank = m_idx - starts[e_m]
    cnt = hist[e_m]
    grid = (nb * NB_N + 7) // 8 * 8
    n = torch.arange(NB_N, device=dev)
    idx = (NB_N * starts[e_m])[:, None] + n[None, :] * cnt[:, None] + rank[:, None]
    idx = torch.where(valid_m[:, None], idx, torch.full_like(idx, grid))
    val = (m_idx[:, None] << 3) | n[None, :]
    tm = torch.full((grid + 1,), -1, dtype=torch.long, device=dev)
    tm.scatter_(0, idx.reshape(-1), val.reshape(-1))
    tm[grid] = valid_m.sum() * NB_N
    return tm.to(torch.int32).contiguous(), grid


launch1 = compile_moe_gemm1(H=H, I=I, E=E, BLOCK_M=BM)
fn1 = None


class Case:
    def __init__(self):
        global fn1
        self.x, self.topk_ids, self.topk_w = make_input()
        self.sorted_ids, self.sorted_w, self.sorted_eids, self.num_valid, _buf = moe_sorting(
            self.topk_ids, self.topk_w, E, H, torch.bfloat16, block_size=BM
        )
        self.a_q, self.a_s = fused_dynamic_mx_quant_moe_sort(
            self.x, self.sorted_ids, self.num_valid, token_num=M, topk=K, block_size=BM
        )
        if args.fake_expert0:
            self.sorted_eids = torch.zeros_like(self.sorted_eids)
        self.num_m_blocks = int(self.sorted_eids.shape[0])
        self.tile_map, self.grid1 = build_tile_map(self.sorted_eids, self.num_valid, self.num_m_blocks)
        rows = self.num_m_blocks * BM
        self.h_q = torch.zeros((rows, I // 2), dtype=torch.uint8, device=dev)
        self.h_s = torch.zeros((rows * (I // 32),), dtype=torch.uint8, device=dev)
        a1 = (
            u8(self.a_q).contiguous().view(-1),
            u8(w1_k).contiguous().view(-1),
            self.h_q.view(-1),
            u8(self.a_s).contiguous().view(-1),
            u8(w1_sk).contiguous().view(-1),
            self.h_s,
            self.sorted_ids.contiguous(),
            self.sorted_eids.contiguous(),
            self.num_valid.contiguous(),
            M,
            self.num_m_blocks,
            int(u8(self.a_s).numel()),
            self.tile_map,
            self.grid1,
            torch.cuda.current_stream(),
        )
        if fn1 is None:
            fn1 = flyc.compile(launch1, *a1)
        fn1(*a1)
        # gemm2 output (token-major); NaN sentinel to detect unwritten rows
        if args.out == "fp8":
            self.out = torch.full((M * K, H), 0x7F, dtype=torch.uint8, device=dev)  # e4m3 NaN
        else:
            self.out = torch.full((M * K, H), float("nan"), dtype=torch.bfloat16, device=dev)
        self.out_s = torch.zeros((M * K, H // 32), dtype=torch.uint8, device=dev)
        self.grid2 = gemm2_grid(self.num_m_blocks, args.n_split)

    def args(self):
        return (
            self.h_q.view(-1),
            u8(w2_k).contiguous().view(-1),
            self.out.view(-1),
            self.h_s,
            u8(w2_sk).contiguous().view(-1),
            self.out_s.view(-1),
            self.sorted_ids.contiguous(),
            self.sorted_eids.contiguous(),
            self.sorted_w.contiguous(),
            self.num_valid.contiguous(),
            M,
            self.num_m_blocks,
            self.grid2,
            torch.cuda.current_stream(),
        )


t0 = time.time()
cases = [Case() for _ in range(max(1, args.copies))]
torch.cuda.synchronize()
c0 = cases[0]
nv = int(c0.num_valid[0].item())
print(
    f"[gemm2] M={M} topk={K}: sorted rows valid {nv} / alloc {c0.num_m_blocks * BM} (pad {nv / (M * K) - 1:+.1%}), "
    f"n_split {args.n_split} grid {c0.grid2}, setup x{len(cases)} {time.time() - t0:.1f}s",
    flush=True,
)

t0 = time.time()
launch2 = compile_moe_gemm2(H=H, I=I, E=E, topk=K, n_split=args.n_split, out_dtype=args.out)
fn2 = flyc.compile(launch2, *c0.args())
print(f"[gemm2] compile {time.time() - t0:.1f}s", flush=True)
fn2(*c0.args())
torch.cuda.synchronize()
_o1 = c0.out.clone()
fn2(*c0.args())
torch.cuda.synchronize()
_d = (c0.out.view(torch.int16) if args.out == "bf16" else c0.out) != (_o1.view(torch.int16) if args.out == "bf16" else _o1)
print(f"[gemm2] determinism: run 2 vs run 1 elements differing {int(_d.sum())}/{_d.numel()}, rows {int(_d.any(dim=1).sum())}", flush=True)

if args.check_rows > 0:
    sid = c0.sorted_ids[:nv]
    tok = (sid & 0xFFFFFF).long()
    slot = (sid >> 24).long()
    real = tok < M
    orow_all = tok * K + slot
    is_sent = (c0.out == 0x7F) if args.out == "fp8" else torch.isnan(c0.out)
    unwritten = int(is_sent.all(dim=1).sum())
    print(f"[gemm2] rows never written: {unwritten} / {M * K}", flush=True)
    h_s_unsh = e8m0_unshuffle(c0.h_s, c0.num_m_blocks * BM, I // 32)
    rows_all = torch.nonzero(real).flatten()
    sel = rows_all[torch.randperm(rows_all.numel(), device=dev)[: args.check_rows]].sort().values
    w2_deq_cache = {}

    def w2_deq(e):
        if e not in w2_deq_cache:
            v = fp4_utils.mxfp4_to_f32(w2_q[e].reshape(-1)).view(H, I)
            sc = fp4_utils.e8m0_to_f32(w2_s[e * H : (e + 1) * H].reshape(-1)).view(H, I // 32)
            w2_deq_cache[e] = (v.view(H, I // 32, 32) * sc.unsqueeze(-1)).view(H, I)
        return w2_deq_cache[e]

    def fp8_ref(y):
        """kernel's rule: e = floor(log2 amax) - 7 (floor 0), q = RNE e4m3 of y / 2^(e-127)"""
        yg = y.view(-1, 32)
        amax = yg.abs().amax(dim=1)
        e8 = ((amax.view(torch.int32) >> 23) - 7).clamp(min=0)
        scale = torch.ldexp(torch.ones_like(amax), e8 - 127)
        q = (yg / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
        return q.view(torch.uint8).view(-1), e8.to(torch.uint8), (q.float() * scale.unsqueeze(1)).view(-1)

    n_bad_q, n_bad_s, n_tot = 0, 0, 0
    rel, cos_all, rel_q = [], [], []
    bad_grp = torch.zeros(H // 32, dtype=torch.long, device=dev)  # mismatched scale groups per column group
    sentinel = is_sent[orow_all[real]].sum(dim=0)  # never-written elements per column
    if int(sentinel.sum()) > 0:
        cols = torch.nonzero(sentinel > 0).flatten()
        print(f"[gemm2] sentinel values left in {int(sentinel.sum())} places; columns {cols[:16].tolist()} .. {cols[-16:].tolist()}")
    n_exact_rows, n_exact_vals, max_ulp = 0, 0, 0
    for r in sel.tolist():
        if args.out == "bf16":
            e = int(c0.sorted_eids[r // BM])
            orow = int(orow_all[r])
            av = fp4_utils.mxfp4_to_f32(c0.h_q[r]).view(I)
            asc = fp4_utils.e8m0_to_f32(h_s_unsh[r]).view(I // 32)
            ad = (av.view(I // 32, 32) * asc.unsqueeze(-1)).view(I)
            y_ref = (ad @ w2_deq(e).T) * c0.sorted_w[r]  # [H] fp32, weighted like production
            ref_b = y_ref.to(torch.bfloat16)
            mine_b = c0.out[orow]
            same = mine_b.view(torch.int16) == ref_b.view(torch.int16)
            n_exact_vals += int(same.sum())
            n_exact_rows += int(same.all())
            ulp = (mine_b.view(torch.int16).int() - ref_b.view(torch.int16).int()).abs()
            max_ulp = max(max_ulp, int(ulp.max()))
            n_tot += H
            mine = mine_b.float()
            rel.append(float((mine - y_ref).norm() / (y_ref.norm() + 1e-12)))
            cos_all.append(float((mine @ y_ref) / (mine.norm() * y_ref.norm() + 1e-12)))
            continue
        e = int(c0.sorted_eids[r // BM])
        orow = int(orow_all[r])
        av = fp4_utils.mxfp4_to_f32(c0.h_q[r]).view(I)
        asc = fp4_utils.e8m0_to_f32(h_s_unsh[r]).view(I // 32)
        ad = (av.view(I // 32, 32) * asc.unsqueeze(-1)).view(I)
        y_ref = ad @ w2_deq(e).T  # [H] fp32
        q_bytes, s_bytes, y_q = fp8_ref(y_ref)
        mine_q = c0.out[orow].view(torch.float8_e4m3fn).float()
        mine_s = c0.out_s[orow]
        mine = (mine_q.view(H // 32, 32) * fp4_utils.e8m0_to_f32(mine_s).view(H // 32, 1)).view(H)
        bad_grp += (mine_s != s_bytes).long()
        n_bad_s += int((mine_s != s_bytes).sum())
        n_bad_q += int((c0.out[orow] != q_bytes).sum())
        n_tot += H
        rel.append(float((mine - y_ref).norm() / (y_ref.norm() + 1e-12)))
        rel_q.append(float((mine - y_q).norm() / (y_ref.norm() + 1e-12)))
        cos_all.append(float((mine @ y_ref) / (mine.norm() * y_ref.norm() + 1e-12)))
    if int(bad_grp.sum()) > 0:
        bg = bad_grp.view(H // 256, 8).cpu()
        print("[gemm2] mismatched scale groups per (n-tile row, 32-col group):")
        for t in range(H // 256):
            print(f"   n-tile {t:2d}: {bg[t].tolist()}")
    if args.out == "bf16":
        print(
            f"[gemm2] check {len(sel)} rows vs bf16(fp32 ref * w): rows bit-identical {n_exact_rows}/{len(sel)}, "
            f"values bit-identical {n_exact_vals}/{n_tot} ({n_exact_vals / n_tot:.4%}), max diff {max_ulp} bf16 ulp; "
            f"rel err mean {statistics.mean(rel):.5f} max {max(rel):.5f}, cos min {min(cos_all):.6f}",
            flush=True,
        )
    else:
        print(
            f"[gemm2] check {len(sel)} rows: scale bytes mismatched {n_bad_s}/{n_tot // 32}, fp8 bytes mismatched "
            f"{n_bad_q}/{n_tot} ({n_bad_q / n_tot:.2%}); vs fp32 ref: rel err mean {statistics.mean(rel):.4f} "
            f"max {max(rel):.4f}, cos min {min(cos_all):.5f}; vs torch-fp8 ref: rel err mean {statistics.mean(rel_q):.5f}",
            flush=True,
        )

for c in cases:
    fn2(*c.args())
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for c in cases:
        fn2(*c.args())
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
flop_useful = 2.0 * (M * K) * I * H
flop_padded = 2.0 * nv * I * H
out_gb = (M * K) * ((H + H // 32) if args.out == "fp8" else 2 * H) / 1e9
print(
    f"[gemm2] tokens={M} out={args.out} n_split={args.n_split} per call: median {us:.1f} us (range {meds[0]:.1f}..{meds[-1]:.1f}), "
    f"{flop_useful / us / 1e9:.2f} PF/s useful ({flop_padded / us / 1e9:.2f} incl. padding), "
    f"output {out_gb:.2f} GB = {out_gb / us * 1e6 / 1e3:.2f} TB/s",
    flush=True,
)
