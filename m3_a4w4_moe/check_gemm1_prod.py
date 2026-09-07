"""Compare ``gemm1.py``'s fused fp4 output with production stage 1 + quant on the
same tokens (a4w4 prefill path: v2 FlyDSL stage 1 -> bf16 rows -> aiter
``dynamic_per_group_scaled_quant`` with the RoundUp e8m0 rule).

    PYTHONPATH=/flydsl python /flydsl/m3_a4w4_moe/check_gemm1_prod.py --tokens 512

Reports, over the valid sorted rows: scale bytes identical, fp4 bytes identical
(overall and within blocks whose scale agrees), and checks the rule itself by
recomputing production's scales from its captured bf16 rows in torch.
"""

import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=512)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--inter", type=int, default=768)
p.add_argument("--experts", type=int, default=129)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--sort-ctas", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

import flydsl.compiler as flyc  # noqa: E402
from m3_a16w4_moe.vllm_ops import import_ops  # noqa: E402

import_ops("moe_a4w4_prefill")
from moe_a4w4_prefill.gemm1 import SWIGLU_LIMIT, compile_moe_gemm1  # noqa: E402
from moe_a4w4_prefill.sort import SortBuffers, compile_moe_sort  # noqa: E402

import aiter  # noqa: E402,F401
import aiter.fused_moe as FM  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort, per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, H, I, E, K = args.tokens, args.hidden, args.inter, args.experts, args.topk
BM = 128
fp4 = torch.float4_e2m1fn_x2
SCN = I // 32


def u8(t):
    return t.view(torch.uint8)


def e8m0_unshuffle(flat, rows, cols):
    t = flat.reshape(rows // 32, cols // 8, 4, 16, 2, 2)  # (g, c8, s, r, q, h)
    return t.permute(0, 5, 3, 1, 4, 2).reshape(rows, cols)  # (g, h, r, c8, q, s)


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

x = torch.randn((M, H), dtype=torch.bfloat16, device=dev)
routed = torch.stack([torch.randperm(E - 1, device=dev)[: K - 1] for _ in range(M)])
topk_ids = torch.cat([routed, torch.full((M, 1), E - 1, device=dev)], dim=1).to(torch.int32)
w_r = torch.rand((M, K - 1), device=dev)
w_r = w_r / w_r.sum(dim=1, keepdim=True) * 2.0
topk_w = torch.cat([w_r, torch.ones((M, 1), device=dev)], dim=1).to(torch.float32).contiguous()

# ---- production, capturing the stage-1 bf16 rows and their quant ----
# (every quant call goes through aiter.ops.quant.fused_dynamic_mx_quant_moe_sort; the
#  a2 one is the call whose input has I columns)
import aiter.ops.quant as AQ  # noqa: E402

calls = []
_orig = AQ.fused_dynamic_mx_quant_moe_sort


def _wrapped(input, **kw):
    out = _orig(input, **kw)
    calls.append((input.clone(), kw["sorted_ids"].clone(), out[0].clone(), out[1].clone()))
    return out


AQ.fused_dynamic_mx_quant_moe_sort = _wrapped
# which stage-2 wrapper runs tells the stage-1 path: v2 (bf16 rows + separate quant)
# or native mxmoe (fused fp4 quant); its inputs are production's fp4 a2 + scales.
s2cap = {}
for _name in ("_flydsl_stage2_wrapper", "_flydsl_v2_stage2_wrapper", "_mxfp4_a4w4_stage2_fw"):
    _f = getattr(FM, _name)

    def _mk(f, name):
        def w(a2, w1, w2, sids, seids, nvi, *a, **kw):
            s2cap.update(path=name, a2=a2.clone(), sorted_ids=sids.clone(), a2_scale=kw.get("a2_scale").clone())
            return f(a2, w1, w2, sids, seids, nvi, *a, **kw)

        return w

    setattr(FM, _name, _mk(_f, _name))
y_prod = fused_moe(
    x,
    w13_k.view(fp4),
    w2_k.view(fp4),
    topk_w,
    topk_ids,
    quant_type=QuantType.per_1x32,
    activation=ActivationType.Swiglu,
    w1_scale=w13_sk,
    w2_scale=w2_sk,
    swiglu_limit=SWIGLU_LIMIT,
    gate_mode=GateMode.SEPARATED.value,
)
AQ.fused_dynamic_mx_quant_moe_sort = _orig
if "a2" in s2cap:
    print(f"[g1] prod stage-2 path {s2cap['path']}: a2 {tuple(s2cap['a2'].shape)} {s2cap['a2'].dtype}, a2_scale {tuple(s2cap['a2_scale'].shape)}, sorted_ids {tuple(s2cap['sorted_ids'].shape)}", flush=True)
else:
    print("[g1] prod stage-2: none of the wrapped stage-2 wrappers ran", flush=True)
cap = {}
for inp, sids, q, sc in calls:
    print(f"[g1] prod quant call: input {tuple(inp.shape)} {inp.dtype}, q {tuple(q.shape)}, s {tuple(sc.shape)}, sorted_ids {tuple(sids.shape)}", flush=True)
    if inp.shape[-1] == I:
        cap = {"a2": inp, "sorted_ids": sids, "q": q, "s": sc}
torch.cuda.synchronize()
assert "q" in cap, "no production quant call with I columns was seen (stage 1 fused its quant?)"
a2 = cap["a2"].reshape(-1, I)[: M * K]  # bf16 rows tok*K + slot (the quant indexes rows token-major)
q_prod = u8(cap["q"]).reshape(-1, I // 2)[: M * K]
psid = cap["sorted_ids"]
n_prod = psid.numel()
s_prod_sorted = e8m0_unshuffle(u8(cap["s"]).reshape(-1), (n_prod + 31) // 32 * 32, SCN)
ptok = (psid & 0xFFFFFF).long()
pslot = (psid >> 24).long()
pvalid = ptok < M
ppos = torch.full((M * K,), -1, dtype=torch.long, device=dev)
ppos[(ptok * K + pslot)[pvalid]] = torch.nonzero(pvalid).flatten()

# ---- mine: sort.py -> production quant -> gemm1.py ----
bufs = SortBuffers.allocate(M, E, K, BM, args.sort_ctas, dev)
launch_sort = compile_moe_sort(E=E, topk=K, block_m=BM, sort_ctas=args.sort_ctas)
launch_sort(*bufs.launch_args(topk_ids, topk_w, M))
a_q, a_s = fused_dynamic_mx_quant_moe_sort(x, bufs.sorted_ids, bufs.num_valid_ids, token_num=M, topk=K, block_size=BM)
num_m_blocks = bufs.max_sorted // BM
rows = num_m_blocks * BM


def build_tile_map(sorted_eids, num_valid, nb):
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


tile_map, grid1 = build_tile_map(bufs.sorted_expert_ids, bufs.num_valid_ids, num_m_blocks)
h_q = torch.zeros((rows, I // 2), dtype=torch.uint8, device=dev)
h_s = torch.zeros((rows * SCN,), dtype=torch.uint8, device=dev)
launch1 = compile_moe_gemm1(H=H, I=I, E=E, BLOCK_M=BM)
a1 = (
    u8(a_q).view(-1),
    u8(w13_k).contiguous().view(-1),
    h_q.view(-1),
    u8(a_s).view(-1),
    u8(w13_sk).contiguous().view(-1),
    h_s,
    bufs.sorted_ids,
    bufs.sorted_expert_ids,
    M,
    num_m_blocks,
    int(u8(a_s).numel()),
    tile_map,
    grid1,
    torch.cuda.current_stream(),
)
fn1 = flyc.compile(launch1, *a1)
fn1(*a1)
torch.cuda.synchronize()

nv = int(bufs.num_valid_ids[0].item())
sid = bufs.sorted_ids[:nv]
tok = (sid & 0xFFFFFF).long()
slot = (sid >> 24).long()
real = tok < M
my_rows = torch.nonzero(real).flatten()
orow = (tok * K + slot)[real]
h_s_unsh = e8m0_unshuffle(h_s, rows, SCN)
mine_q = h_q[my_rows].view(-1, SCN, 16)
mine_s = h_s_unsh[my_rows]
prod_q = q_prod[orow].view(-1, SCN, 16)
prod_s = s_prod_sorted[ppos[orow]]
n_rows = mine_s.shape[0]

same_s = mine_s == prod_s
same_q = (mine_q == prod_q).all(dim=-1)  # per 32-col block
d = mine_s.int() - prod_s.int()
hist = {int(v): int((d == v).sum()) for v in torch.unique(d).tolist()}
print(
    f"[g1] {n_rows} valid rows x {SCN} blocks: scale bytes identical {float(same_s.float().mean()):.4%}; "
    f"fp4 blocks identical {float(same_q.float().mean()):.4%} overall, "
    f"{float(same_q[same_s].float().mean()):.4%} among blocks with the same scale; "
    f"rows fully identical {int((same_s & same_q).all(dim=1).sum())}/{n_rows}; "
    f"(mine - prod) exponent histogram {hist}",
    flush=True,
)

# ---- rule check: production's scales recomputed from its own bf16 rows ----
a2r = a2[orow].float().view(-1, SCN, 32)
amax = a2r.abs().amax(dim=-1)
u = (amax * (1.0 / 6.0)).view(torch.int32)
e_up = ((u >> 23) & 0xFF) + (((u & 0x7FFFFF) != 0) & (((u >> 23) & 0xFF) < 255)).int()  # RoundUp
r = (amax.view(torch.int32) + 0x400000) & -0x800000
e_even = ((r >> 23) - 2).clamp(min=0)  # Even (headroom 2), gemm1's old rule
print(
    f"[g1] rule check on production's bf16 rows: RoundUp reproduces prod scales {float((e_up == prod_s.int()).float().mean()):.4%}; "
    f"Even rule would match prod {float((e_even == prod_s.int()).float().mean()):.4%}; "
    f"Even rule matches MY scales {float((e_even == mine_s.int()).float().mean()):.4%}",
    flush=True,
)

# ---- the outlier blocks (|exponent diff| >= 2): who is off, mine or production? ----
_dequant = lambda q, s, n: (fp4_utils.mxfp4_to_f32(q.reshape(-1)).view(q.shape[0], -1, 32) * fp4_utils.e8m0_to_f32(s.reshape(-1)).view(q.shape[0], -1, 1)).view(q.shape[0], n)  # noqa: E731
bad = torch.nonzero(d.abs() >= 2)
if bad.numel() > 0:
    from moe_a4w4_prefill.gemm1 import SWIGLU_ALPHA

    xq, xs = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2)
    xq, xs = u8(xq).view(M, H // 2), u8(xs).view(M, H // 32)
    tok_all = tok[real]
    slot_all = slot[real]
    for r, blk in bad[:8].tolist():
        t, sl = int(tok_all[r]), int(slot_all[r])
        e = int(topk_ids[t, sl])
        xd = _dequant(xq[t : t + 1], xs[t : t + 1], H).view(H)
        w13d = _dequant(w13_q[e], w13_s[e * 2 * I : (e + 1) * 2 * I], H)
        hh = xd @ w13d.T
        g = hh[:I].clamp(max=SWIGLU_LIMIT)
        uu = hh[I:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h_ref = (g * torch.sigmoid(SWIGLU_ALPHA * g) * (uu + 1.0))[blk * 32 : (blk + 1) * 32]
        p_bf16 = a2[orow[r], blk * 32 : (blk + 1) * 32].float()
        m_val = _dequant(h_q[my_rows[r] : my_rows[r] + 1], h_s_unsh[my_rows[r] : my_rows[r] + 1], I).view(I)[blk * 32 : (blk + 1) * 32]
        p_val = _dequant(q_prod[orow[r] : orow[r] + 1], prod_s[r : r + 1], I).view(I)[blk * 32 : (blk + 1) * 32]
        print(
            f"[g1] outlier tok {t} slot {sl} expert {e} block {blk}: e8m0 mine {int(mine_s[r, blk])} prod {int(prod_s[r, blk])}; "
            f"amax ref {float(h_ref.abs().max()):.4g} prod-bf16 {float(p_bf16.abs().max()):.4g}; "
            f"max|mine - ref| {float((m_val - h_ref).abs().max()):.4g}, max|prod - ref| {float((p_val - h_ref).abs().max()):.4g}, "
            f"max|prod-bf16 - ref| {float((p_bf16 - h_ref).abs().max()):.4g}",
            flush=True,
        )
