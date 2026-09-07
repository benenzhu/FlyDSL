"""Whole MiniMax-M3 prefill MoE (a4w4) pipeline vs production aiter ``fused_moe``.

    PYTHONPATH=/flydsl python /flydsl/m3_a4w4_moe/bench_moe_a4w4.py --tokens 16384

Mine:  sort.py (3-stage)  ->  aiter fused_dynamic_mx_quant_moe_sort (x -> fp4 + sorted
       shuffled scales; the production HIP kernel)  ->  gemm1.py (fp4 out, expert-major
       tile map from tile_map.py)  ->  gemm2.py  ->  reduce.
       ``--out bf16`` (default): gemm2 writes the production [tokens, topk, H] bf16
       rows (weights applied) and the reduce is production's ``moe_reduction_kernel``.
       ``--out fp8``: mxfp8 route-out + ``reduce_fp8.py`` (weights in the reduce).
Prod:  aiter.fused_moe(quant_type per_1x32, Swiglu, gate_mode SEPARATED), the
       AITER_MXFP4_MXFP4 layouts (shuffle_weights + e8m0_shuffle).
Check: bit-identical fraction + max bf16-ulp diff and cos(mine, prod) over all tokens;
cos of both against an fp32 reference on ``--check-tokens`` sampled tokens.
Timing: HIP graphs with ``--copies`` inputs; the stages of my pipeline are also timed
as separate graphs.
"""

import argparse
import os
import statistics
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--n-split", type=int, default=2)
p.add_argument("--bm", type=int, choices=[0, 32, 64, 128, 256], default=0,
               help="sort block size = gemm1 tile rows (256 = half the W13 bytes per FLOP, 2x the padding; gemm2 stays "
                    "128-row tiles and skips all-padding tiles). 0 = auto: 256 from 16384 tokens up (chain 16384: 1059 vs "
                    "1090 us, 32768: 1929 vs 2020; 8192: 637 vs 614, 4096: 428 vs 402 -> 128 there)")
p.add_argument("--chain", choices=["prefill", "mid"], default="prefill")
p.add_argument("--no-check", action="store_true", help="timing-only diagnostics: do not assert on the reference cosines")
p.add_argument("--mid-g1", choices=["alds"], default="alds", help="mid chain gemm1 (gemm1_mid_alds.py; the other variants were dead ends, see MIDM_NOTES.md)")
p.add_argument("--mid-g2", choices=["atomic-alds"], default="atomic-alds", help="mid chain gemm2 (gemm2_mid_atomic_alds.py)")
p.add_argument("--loop-prod", action="store_true", help="with --loop: iterate aiter's chain instead of ours")
p.add_argument("--loop", type=int, default=0, help="eager whole-chain iterations for rocprofv3, after setup")
p.add_argument("--copies", type=int, default=4)
p.add_argument("--reps", type=int, default=10)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--check-tokens", type=int, default=64)
p.add_argument("--check-determinism", action="store_true", help="optional bitwise diagnostic; compare reference cosine by default")
p.add_argument("--diagnose-weight-power", action="store_true", help="compare aiter against a squared-route-weight reference and exit before timing")
p.add_argument("--prod-shuffled", action="store_true", help="mark the already-preshuffled aiter weight views with is_shuffled=True")
p.add_argument("--diagnose-aiter-fp8", action="store_true", help="capture aiter's fp8 partials, decode/reduce in torch, and exit before timing")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-prod", action="store_true")
p.add_argument("--tail", choices=["none", "residual", "prod"], default="none",
               help="append the layer's downstream consumer to BOTH chains so a cache-policy change is charged "
                    "where the model would pay it: 'residual' = res += y (one streaming read of the [M, H] output); "
                    "'prod' = what the M3 decoder layer runs after the MoE: the TP all-reduce (quick reduce reads y "
                    "once and writes the reduced tensor; single-GPU stand-in = one copy) followed by vLLM's Triton "
                    "gemma_fused_add_rmsnorm (residual add + GemmaRMSNorm, vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py)")
p.add_argument("--out", choices=["bf16", "fp8"], default="fp8" if os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1" else "bf16",
               help="gemm2 output mode; default follows aiter's AITER_FLYDSL_STAGE2_FP8 switch")
p.add_argument("--sim-fp8-formats", action="store_true",
               help="on the --check-tokens fp32 reference: cos of the final sum when the gemm2 rows are "
                    "bf16 / mxfp8 per 32 / mxfp8 per 8 / plain e4m3 / plain e4m3 of y*w / per-row-scaled e4m3")
args = p.parse_args()

import flydsl.compiler as flyc  # noqa: E402
# production kernels come from the vLLM tree (m3_a16w4_moe/vllm_ops.py)
from m3_a16w4_moe.vllm_ops import import_ops  # noqa: E402

import_ops("moe_a16w4_decode")
import_ops("moe_a4w4_prefill")
from moe_a4w4_prefill.gemm1 import SWIGLU_ALPHA, SWIGLU_LIMIT, compile_moe_gemm1  # noqa: E402
from moe_a4w4_prefill.gemm2 import compile_moe_gemm2, gemm2_grid  # noqa: E402
from moe_a4w4_prefill.sort import SortBuffers, compile_moe_sort  # noqa: E402
from moe_a4w4_prefill.tile_map import compile_tile_map, tile_map_grid  # noqa: E402
# lab-only kernels: fp8 route-out reduce, the mid-batch (512..1024 tokens) chain and its sort
from m3_a4w4_moe.reduce_fp8 import compile_moe_reduce_fp8  # noqa: E402
from m3_a4w4_moe.gemm1_mid_alds import compile_moe_gemm1_mid as compile_g1_alds  # noqa: E402
from m3_a4w4_moe.gemm2_mid_atomic_alds import compile_moe_gemm2_mid as compile_moe_gemm2_mid_atomic_alds  # noqa: E402
from m3_a16w4_moe.sort_decode_wave import compile_decode_sort, _ptr, _max_tokens_bucket  # noqa: E402

import aiter  # noqa: E402,F401
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction  # noqa: E402
if args.tail == "prod":
    from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
        gemma_fused_add_rmsnorm as _gemma_fused_add_rmsnorm,
    )
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort, per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K = args.tokens, args.hidden, args.inter, args.experts, args.topk
BM = args.bm if args.bm else (32 if args.chain == "mid" else (256 if args.tokens >= 16384 else 128))
fp4 = torch.float4_e2m1fn_x2


def u8(t):
    return t.view(torch.uint8)


# ---- weights: random bf16 -> mxfp4 per-1x32, production layouts ----
w13 = torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=dev) * 0.02
w2 = torch.randn((E, H, I), dtype=torch.bfloat16, device=dev) * 0.02
w13_q, w13_s = per_1x32_f4_quant(w13, quant_dtype=dtypes.fp4x2)
w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
del w13, w2
w13_q = w13_q.view(E, 2 * I, H // 2)
w2_q = w2_q.view(E, H, I // 2)
w13_s = u8(w13_s).view(E * 2 * I, H // 32)
w2_s = u8(w2_s).view(E * H, I // 32)
w13_k = shuffle_weight(w13_q, layout=(16, 16))
w2_k = shuffle_weight(w2_q, layout=(16, 16))
w13_sk = fp4_utils.e8m0_shuffle(w13_s.view(torch.float8_e8m0fnu))
w2_sk = fp4_utils.e8m0_shuffle(w2_s.view(torch.float8_e8m0fnu))


def make_input():
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
    routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
    topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
    w_r = torch.rand((M, K - 1), device=dev)
    w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0
    topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32).contiguous()
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


launch_sort = (compile_decode_sort(E=E, topk=K, block_m=BM, H=H, max_tokens=_max_tokens_bucket(M))
               if args.chain == "mid" else compile_moe_sort(E=E, topk=K, block_m=BM))
launch_tm = None if args.chain == "mid" else compile_tile_map(I=I, BM=BM)
fn_tm = None
mid_g1_builder = {"alds": compile_g1_alds}[args.mid_g1]
launch1 = (mid_g1_builder(H=H, I=I, E=E, BLOCK_M=BM) if args.chain == "mid"
           else compile_moe_gemm1(H=H, I=I, E=E, BLOCK_M=BM))
if args.chain == "mid":
    assert args.out == "bf16", "mid chain: bf16 atomic output only"
    launch2 = compile_moe_gemm2_mid_atomic_alds(H=H, I=I, E=E, topk=K, BLOCK_M=BM)
else:
    launch2 = compile_moe_gemm2(H=H, I=I, E=E, topk=K, n_split=args.n_split, out_dtype=args.out, sort_block_m=BM)
launch_r = compile_moe_reduce_fp8(H=H, topk=K) if args.out == "fp8" else None
fn1 = fn2 = fnr = None


class Case:
    def __init__(self):
        global fn1, fn2, fnr
        self.x, self.topk_ids, self.topk_w = make_input()
        self.bufs = SortBuffers.allocate(M, E, K, BM, dev)
        self.num_m_blocks = self.bufs.max_sorted // BM
        rows = self.num_m_blocks * BM
        self.num_m_blocks2 = self.num_m_blocks if args.chain == "mid" else rows // 128
        self.h_q = torch.zeros((rows, I // 2), dtype=torch.uint8, device=dev)
        self.h_s = torch.zeros((rows * (I // 32),), dtype=torch.uint8, device=dev)
        self.out = torch.zeros((M * K, H), dtype=torch.uint8 if args.out == "fp8" else torch.bfloat16, device=dev)
        self.out_s = torch.zeros((M * K, H // 32), dtype=torch.uint8, device=dev)
        self.y = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
        self.res = torch.randn((M, H), dtype=torch.bfloat16, device=dev)  # --tail residual
        if args.tail == "prod":
            self.ar_buf = torch.empty((M, H), dtype=torch.bfloat16, device=dev)  # all-reduce stand-in target
            self.norm_w = torch.zeros((H,), dtype=torch.bfloat16, device=dev)  # Gemma: scale = 1 + w
        self.grid2 = gemm2_grid(self.num_m_blocks2, args.n_split)
        self.grid1 = tile_map_grid(self.num_m_blocks, I)
        self.tile_map = torch.zeros((self.grid1 + 1,), dtype=torch.int32, device=dev)
        # run once eagerly (compiles on first use)
        self.stage_sort()
        self.stage_quant()
        self.stage_tile_map(compile_first=True)
        torch.cuda.synchronize()
        if args.chain != "mid":
            tm_ref, grid_ref = build_tile_map(self.bufs.sorted_expert_ids, self.bufs.num_valid_ids, self.num_m_blocks)
            assert grid_ref == self.grid1 and bool((tm_ref == self.tile_map).all()), "tile_map kernel != torch build"
        self.stage_gemm1(compile_first=True)
        self.stage_gemm2(compile_first=True)
        self.stage_reduce(compile_first=True)

    # ---- stages ----
    def stage_sort(self):
        if args.chain == "mid":
            b = self.bufs
            launch_sort(_ptr(self.topk_ids), _ptr(self.topk_w), _ptr(b.sorted_ids), _ptr(b.sorted_weights),
                        _ptr(b.sorted_expert_ids), _ptr(b.num_valid_ids), _ptr(self.y), M, torch.cuda.current_stream())
        else:
            launch_sort(*self.bufs.launch_args(self.topk_ids, self.topk_w, M))

    def stage_quant(self):
        b = self.bufs
        self.a_q, self.a_s = fused_dynamic_mx_quant_moe_sort(
            self.x, b.sorted_ids, b.num_valid_ids, token_num=M, topk=K, block_size=BM
        )

    def args1(self):
        b = self.bufs
        return (
            u8(self.a_q).view(-1),
            u8(w13_k).contiguous().view(-1),
            self.h_q.view(-1),
            u8(self.a_s).view(-1),
            u8(w13_sk).contiguous().view(-1),
            self.h_s,
            b.sorted_ids,
            b.sorted_expert_ids,
            M,
            self.num_m_blocks,
            int(u8(self.a_s).numel()),
            self.tile_map,
            self.grid1,
            torch.cuda.current_stream(),
        )

    def args_tm(self):
        b = self.bufs
        return (b.sorted_expert_ids, b.num_valid_ids, self.tile_map, self.grid1, torch.cuda.current_stream())

    def stage_tile_map(self, compile_first=False):
        global fn_tm
        if args.chain == "mid":
            return
        if compile_first and fn_tm is None:
            fn_tm = flyc.compile(launch_tm, *self.args_tm())
        fn_tm(*self.args_tm())

    def stage_gemm1(self, compile_first=False):
        global fn1
        if compile_first and fn1 is None:
            fn1 = flyc.compile(launch1, *self.args1())
        fn1(*self.args1())

    def args2(self):
        b = self.bufs
        return (
            self.h_q.view(-1),
            u8(w2_k).contiguous().view(-1),
            (self.y if args.chain == "mid" and args.mid_g2.startswith("atomic") else self.out).view(-1),
            self.h_s,
            u8(w2_sk).contiguous().view(-1),
            self.out_s.view(-1),
            b.sorted_ids,
            b.sorted_expert_ids,
            b.sorted_weights,
            b.num_valid_ids,
            M,
            self.num_m_blocks2,
            self.grid2,
            torch.cuda.current_stream(),
        )

    def stage_gemm2(self, compile_first=False):
        global fn2
        if compile_first and fn2 is None:
            fn2 = flyc.compile(launch2, *self.args2())
        fn2(*self.args2())

    def argsr(self):
        return (self.out.view(-1), self.out_s.view(-1), self.topk_w.view(-1), self.y.view(-1), M, torch.cuda.current_stream())

    def stage_reduce(self, compile_first=False):
        global fnr
        if args.chain == "mid" and args.mid_g2.startswith("atomic"):
            return
        if args.out == "bf16":
            # production's topk reduce (fp32 sum of the bf16 rows -> bf16)
            _run_moe_reduction(self.out.view(M, K, H), self.y, M, K, H)
            return
        if compile_first and fnr is None:
            fnr = flyc.compile(launch_r, *self.argsr())
        fnr(*self.argsr())

    def stage_tail(self, y=None):
        y = self.y if y is None else y
        if args.tail == "residual":
            self.res.add_(y)
        elif args.tail == "prod":
            yr = self.ar_buf.copy_(y)  # all-reduce stand-in: read y once, write the reduced tensor
            self.norm_out, self.res_out = _gemma_fused_add_rmsnorm(yr, self.res, self.norm_w, 1e-6)

    def mine(self):
        self.stage_sort()
        self.stage_quant()
        self.stage_tile_map()
        self.stage_gemm1()
        self.stage_gemm2()
        self.stage_reduce()
        self.stage_tail()
        return self.y

    def prod(self):
        y = self._prod()
        self.stage_tail(y)
        return y

    def _prod(self):
        pw1, pw2 = w13_k.view(fp4), w2_k.view(fp4)
        if args.prod_shuffled:
            pw1.is_shuffled = True
            pw2.is_shuffled = True
        return fused_moe(
            self.x,
            pw1,
            pw2,
            self.topk_w,
            self.topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            w1_scale=w13_sk,
            w2_scale=w2_sk,
            swiglu_limit=SWIGLU_LIMIT,
            gate_mode=GateMode.SEPARATED.value,
        )


t0 = time.time()
cases = [Case() for _ in range(max(1, args.copies))]
torch.cuda.synchronize()
c0 = cases[0]
nv = int(c0.bufs.num_valid_ids[0].item())
print(
    f"[moe] M={M} topk={K} E={E}: sorted rows valid {nv} / alloc {c0.num_m_blocks * BM}, setup x{len(cases)} "
    f"{time.time() - t0:.1f}s chain={args.chain} mid_g1={args.mid_g1} mid_g2={args.mid_g2} out={args.out}",
    flush=True,
)
if args.loop:
    for i in range(args.loop):
        (cases[i % len(cases)].prod if args.loop_prod else cases[i % len(cases)].mine)()
    torch.cuda.synchronize()
    raise SystemExit(0)

# ---- correctness ----
y_mine = c0.mine().clone()
torch.cuda.synchronize()
if not args.no_prod:
    if args.diagnose_aiter_fp8:
        import aiter.ops.flydsl.moe_kernels as _mk
        import inspect
        _orig_reduce = _mk._run_moe_reduction
        _captured_fp8 = {}

        def _capture_reduce(*a, **kw):
            bound = inspect.signature(_orig_reduce).bind(*a, **kw)
            bound.apply_defaults()
            if bound.arguments["is_fp8"]:
                _captured_fp8.update(bound.arguments)
            return _orig_reduce(*a, **kw)

        _mk._run_moe_reduction = _capture_reduce
        try:
            y_prod = c0.prod().clone()
        finally:
            _mk._run_moe_reduction = _orig_reduce
        assert _captured_fp8, "aiter did not call the fp8 reduction path"
        target = _captured_fp8["target"]
        blk = _captured_fp8["fp8_scale_blk"] or 32
        payload = target[:, :H].contiguous().view(torch.float8_e4m3fn).float()
        scales = target[:, H:H + H // blk].contiguous()
        sf = fp4_utils.e8m0_to_f32(scales.reshape(-1)).view(M * K, H // blk, 1)
        decoded = (payload.view(M * K, H // blk, blk) * sf).view(M, K, H)
        tw = _captured_fp8["topk_weights"]
        if tw is not None:
            decoded = decoded * tw.float().view(M, K, 1)
        y_redecoded = decoded.sum(dim=1).to(torch.bfloat16)
        print(f"[moe] captured aiter fp8 target={tuple(target.shape)} scale_blk={blk} weights_in_reduce={tw is not None}", flush=True)
    else:
        y_prod = c0.prod().clone()
    torch.cuda.synchronize()
# determinism: the same inputs again must give the same bits (a race shows up here)
for rep in range(2 if args.check_determinism else 0):
    y2 = c0.mine().clone()
    torch.cuda.synchronize()
    d = (y2.view(torch.int16) != y_mine.view(torch.int16)).any(dim=1)
    print(f"[moe] determinism: mine run {rep + 2} vs run 1: rows differing {int(d.sum())}/{M}", flush=True)
    assert not d.any(), "non-atomic chain is not bitwise deterministic"
if not args.no_prod and args.check_determinism:
    y2 = c0.prod().clone()
    torch.cuda.synchronize()
    d = (y2.view(torch.int16) != y_prod.view(torch.int16)).any(dim=1)
    print(f"[moe] determinism: prod run 2 vs run 1: rows differing {int(d.sum())}/{M}", flush=True)


def _stage_determinism():
    """which stage is racy: sort twice (row order may legitimately differ), then with the
    sort buffers frozen run quant -> tile_map -> gemm1 -> gemm2 -> reduce twice and
    compare every intermediate bitwise"""
    b = c0.bufs
    c0.stage_sort()
    torch.cuda.synchronize()
    s1 = b.sorted_ids.clone()
    c0.stage_sort()
    torch.cuda.synchronize()
    print(f"[moe] determinism: sort run 2 vs run 1: sorted_ids entries differing {int((b.sorted_ids != s1).sum())}/{s1.numel()}", flush=True)

    def _run():
        c0.stage_quant()
        c0.stage_tile_map()
        c0.stage_gemm1()
        c0.stage_gemm2()
        c0.stage_reduce()
        torch.cuda.synchronize()
        return {
            "a_q": u8(c0.a_q).clone(), "a_s": u8(c0.a_s).clone(), "tile_map": c0.tile_map.clone(),
            "h_q": c0.h_q.clone(), "h_s": c0.h_s.clone(), "y": c0.y.view(torch.int16).clone(),
            "out": (c0.out.clone() if args.out == "fp8" else c0.out.view(torch.int16).clone()),
            **({"out_s": c0.out_s.clone()} if args.out == "fp8" else {}),
        }

    r1, r2, r3 = _run(), _run(), _run()
    nv = int(b.num_valid_ids[0].item())
    sid = b.sorted_ids[:nv]
    tok, slot = (sid & 0xFFFFFF).long(), (sid >> 24).long()
    vrows = torch.nonzero(tok < M).flatten()  # valid sorted rows
    orow = (tok * K + slot)[tok < M]  # their token-major output rows
    rows_alloc = c0.num_m_blocks * BM

    def _unsh(flat, rows, cols):
        t = flat.reshape(rows // 32, cols // 8, 4, 16, 2, 2)
        return t.permute(0, 5, 3, 1, 4, 2).reshape(rows, cols)

    views = {
        "a_q": lambda r: r["a_q"],
        "a_s (valid rows)": lambda r: _unsh(r["a_s"], r["a_s"].numel() // (H // 32), H // 32)[vrows],
        "tile_map": lambda r: r["tile_map"],
        "h_q (valid rows)": lambda r: r["h_q"].view(rows_alloc, -1)[vrows],
        "h_s (valid rows)": lambda r: _unsh(r["h_s"], rows_alloc, I // 32)[vrows],
        "out (valid rows)": lambda r: r["out"].view(M * K, H)[orow],
        **({"out_s (valid rows)": lambda r: r["out_s"].view(M * K, H // 32)[orow]} if args.out == "fp8" else {}),
        "y": lambda r: r["y"],
    }
    for k, f in views.items():
        v1, v2 = f(r1), f(r2)
        diff = v1 != v2
        n = int(diff.sum())
        msg = f"[moe] determinism (sort frozen): {k:18s} elements differing {n}/{v1.numel()}"
        if n and v1.dim() == 2:
            rows = torch.nonzero(diff.any(dim=1)).flatten()
            msg += f"; rows {int(rows.numel())}, first {rows[:6].tolist()}"
        print(msg, flush=True)
        assert torch.equal(v1, v2) and torch.equal(v1, f(r3)), f"{k}: stage is not bitwise deterministic"


if args.check_determinism:
    _stage_determinism()


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _dequant(q, s, n_cols):
    v = fp4_utils.mxfp4_to_f32(q.reshape(-1)).view(q.shape[0], -1)
    sc = fp4_utils.e8m0_to_f32(s.reshape(-1)).view(q.shape[0], -1)
    return (v.view(q.shape[0], -1, 32) * sc.unsqueeze(-1)).view(q.shape[0], n_cols)


numerical_ok = True
if not args.no_prod:
    same = y_mine.view(torch.int16) == y_prod.view(torch.int16)
    numerical_ok = bool(torch.isfinite(y_mine).all())
    # bf16 ulp distance: the int16 patterns are monotonic in magnitude per sign; across
    # sign / zero use the value distance in units of the larger magnitude's ulp
    a16, b16 = y_mine.view(torch.int16).int(), y_prod.view(torch.int16).int()
    same_sign = (a16 < 0) == (b16 < 0)
    ulp_mag = torch.ldexp(torch.ones_like(y_mine, dtype=torch.float32), (torch.maximum(y_mine.abs(), y_prod.abs()).float().view(torch.int32) >> 23) - 127 - 7)
    ulp = torch.where(same_sign, (a16 - b16).abs().float(), (y_mine.float() - y_prod.float()).abs() / ulp_mag)
    n1 = int(((ulp > 0) & (ulp <= 1)).sum())
    n2 = int((ulp > 1).sum())
    print(
        f"[moe] mine vs prod over all {M} tokens: values bit-identical {float(same.float().mean()):.5%}, "
        f"rows bit-identical {int(same.all(dim=1).sum())}/{M}, values off by 1 bf16 ulp {n1}, by >1 ulp {n2} "
        f"(max {float(ulp.max()):.1f} ulp), max |diff| {float((y_mine.float() - y_prod.float()).abs().max()):.3g} "
        f"vs max |prod| {float(y_prod.float().abs().max()):.3g}, cos {_cos(y_mine, y_prod):.6f}",
        flush=True,
    )
if args.check_tokens > 0:
    xq, xs = per_1x32_f4_quant(c0.x, quant_dtype=dtypes.fp4x2)
    xq = u8(xq).view(M, H // 2)
    xs = u8(xs).view(M, H // 32)
    toks = torch.randperm(M, device=dev)[: args.check_tokens].tolist()
    w13d, w2d = {}, {}
    cm, cp, cp_squared, cp_unweighted, cp_decoded = [], [], [], [], []
    sim = {k: [] for k in ("bf16", "mx32", "mx8", "plain", "plain_w", "row")}

    def _e4m3(v):
        return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)

    def _mx(v, blk):
        """e8m0 = floor(log2 amax) - 7 per block (gemm2.py's rule): amax lands in [128, 256)"""
        vb = v.view(-1, blk)
        amax = vb.abs().amax(dim=1, keepdim=True)
        e = torch.floor(torch.log2(amax.clamp(min=1e-30))) - 7.0
        e = e.clamp(min=-127.0)
        sc = torch.exp2(e)
        return (_e4m3(vb / sc) * sc).view(-1)

    for t in toks:
        xd = _dequant(xq[t : t + 1], xs[t : t + 1], H).view(H)
        out = torch.zeros(H, dtype=torch.float32, device=dev)
        out_squared = torch.zeros_like(out)
        out_unweighted = torch.zeros_like(out)
        for j in range(K):
            e = int(c0.topk_ids[t, j])
            if e not in w13d:
                w13d[e] = _dequant(w13_q[e], w13_s[e * 2 * I : (e + 1) * 2 * I], H)
                w2d[e] = _dequant(w2_q[e], w2_s[e * H : (e + 1) * H], I)
            hh = xd @ w13d[e].T
            g = hh[:I].clamp(max=SWIGLU_LIMIT)
            uu = hh[I:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (uu + 1.0)
            ye = a @ w2d[e].T
            wj = float(c0.topk_w[t, j])
            out += wj * ye
            if args.diagnose_weight_power:
                out_squared += wj * wj * ye
                out_unweighted += ye
            if args.sim_fp8_formats:
                acc = sim.setdefault("_acc", {})
                acc.setdefault("bf16", torch.zeros_like(out)).add_((ye * wj).to(torch.bfloat16).float())
                acc.setdefault("mx32", torch.zeros_like(out)).add_(wj * _mx(ye, 32))
                acc.setdefault("mx8", torch.zeros_like(out)).add_(wj * _mx(ye, 8))
                acc.setdefault("plain", torch.zeros_like(out)).add_(wj * _e4m3(ye))
                acc.setdefault("plain_w", torch.zeros_like(out)).add_(_e4m3(ye * wj))
                acc.setdefault("row", torch.zeros_like(out)).add_(wj * _mx(ye, H))
        if args.sim_fp8_formats:
            acc = sim.pop("_acc")
            for k, v in acc.items():
                sim[k].append(_cos(v.to(torch.bfloat16), out))
        cm.append(_cos(y_mine[t], out))
        if not args.no_prod:
            cp.append(_cos(y_prod[t], out))
            if args.diagnose_aiter_fp8:
                cp_decoded.append(_cos(y_redecoded[t], out))
            if args.diagnose_weight_power:
                cp_squared.append(_cos(y_prod[t], out_squared))
                cp_unweighted.append(_cos(y_prod[t], out_unweighted))
    msg = f"[moe] cos vs fp32 reference on {len(toks)} tokens: mine min {min(cm):.5f} mean {statistics.mean(cm):.5f}"
    if cp:
        msg += f"; prod min {min(cp):.5f} mean {statistics.mean(cp):.5f}"
    print(msg, flush=True)
    assert args.no_check or min(cm) >= 0.97, "whole-chain mismatch vs fp32 reference"
    if cp:
        assert statistics.mean(cm) >= statistics.mean(cp) - 0.002, "mean cosine materially worse than aiter"
    if args.diagnose_weight_power:
        print(f"[moe] aiter cosine vs squared-route-weight reference: min {min(cp_squared):.5f} mean {statistics.mean(cp_squared):.5f}", flush=True)
        print(f"[moe] aiter cosine vs unweighted reference: min {min(cp_unweighted):.5f} mean {statistics.mean(cp_unweighted):.5f}", flush=True)
        raise SystemExit(0)
    if args.diagnose_aiter_fp8:
        print(f"[moe] torch-decoded aiter partials: vs aiter output cos {_cos(y_redecoded, y_prod):.8f}; vs reference min {min(cp_decoded):.5f} mean {statistics.mean(cp_decoded):.5f}", flush=True)
        raise SystemExit(0)
    if cp:
        assert args.no_check or min(cp) >= 0.97, "aiter comparison path fails its independent reference"
    if args.sim_fp8_formats:
        print(
            "[moe] fp8 format sim (cos of the bf16 final sum vs fp32 ref, min / mean): "
            + "; ".join(f"{k} {min(v):.5f} / {statistics.mean(v):.5f}" for k, v in sim.items()),
            flush=True,
        )


assert numerical_ok, "whole-chain mismatch vs aiter (see independent reference above)"

# ---- timing ----
def time_graph(fn_of_case, label):
    for c in cases:
        fn_of_case(c)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in cases:
            fn_of_case(c)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    if label == "mine (whole graph)" and args.check_determinism:
        snapshots = [c.y.clone() for c in cases]
        for trial in range(2):
            g.replay()
            torch.cuda.synchronize()
            for c, expected in zip(cases, snapshots):
                assert torch.equal(c.y.view(torch.int16), expected.view(torch.int16)), f"whole graph trial {trial + 2}: not bitwise deterministic"
        print(f"[moe] whole graph same inputs x3: bitwise identical ({len(cases)} inputs)", flush=True)
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
    print(f"[moe] {label:>22s}: median {us:7.1f} us (range {meds[0]:.1f}..{meds[-1]:.1f})", flush=True)
    return us


stages = [
    ("sort", lambda c: c.stage_sort()),
    ("quant+scale-sort", lambda c: c.stage_quant()),
    ("tile_map", lambda c: c.stage_tile_map()),
    ("gemm1", lambda c: c.stage_gemm1()),
    ("gemm2", lambda c: c.stage_gemm2()),
    ("reduce", lambda c: c.stage_reduce()),
]
if args.chain == "mid":
    stages = [(name, f) for name, f in stages if name != "tile_map"]
    if args.mid_g2.startswith("atomic"):
        stages = [(name, f) for name, f in stages if name != "reduce"]
if args.tail != "none":
    stages.append((f"tail {args.tail}", lambda c: c.stage_tail()))
tot = 0.0
for name, f in stages:
    tot += time_graph(f, name)
print(f"[moe] {'stage sum':>22s}: {tot:7.1f} us", flush=True)
mine_us = time_graph(lambda c: c.mine(), "mine (whole graph)")
if not args.no_prod:
    prod_us = time_graph(lambda c: c.prod(), "prod fused_moe")
    print(f"[moe] tokens={M}: mine {mine_us:.1f} us vs prod {prod_us:.1f} us -> {prod_us / mine_us:.2f}x", flush=True)
