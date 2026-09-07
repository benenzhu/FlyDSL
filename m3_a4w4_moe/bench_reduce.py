"""Correctness + timing of ``m3_a4w4_moe.reduce_fp8`` (weighted sum of gemm2's mxfp8
token-major rows -> bf16).

    PYTHONPATH=/flydsl python /flydsl/m3_a4w4_moe/bench_reduce.py --tokens 16384
"""

import argparse
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=16384)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--topk", type=int, default=5)
p.add_argument("--copies", type=int, default=4)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

import flydsl.compiler as flyc  # noqa: E402
from m3_a4w4_moe.reduce_fp8 import compile_moe_reduce_fp8  # noqa: E402

torch.manual_seed(args.seed)
dev = "cuda"
M, H, K = args.tokens, args.hidden, args.topk


class Case:
    def __init__(self):
        y = torch.randn((M * K, H), device=dev) * 3.0
        yg = y.view(-1, 32)
        amax = yg.abs().amax(dim=1)
        e8 = ((amax.view(torch.int32) >> 23) - 7).clamp(min=0)
        scale = torch.ldexp(torch.ones_like(amax), e8 - 127)
        self.q = (yg / scale.unsqueeze(1)).to(torch.float8_e4m3fn).view(M * K, H)
        self.s = e8.to(torch.uint8).view(M * K, H // 32)
        w = torch.rand((M, K - 1), device=dev)
        w = w / w.sum(dim=1, keepdim=True) * 2.0
        self.w = torch.cat([w, torch.ones((M, 1), device=dev)], dim=1).contiguous()
        self.y = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)

    def ref(self):
        v = self.q.float().view(M * K, H // 32, 32) * torch.ldexp(
            torch.ones((), device=dev), self.s.int() - 127
        ).unsqueeze(-1)
        v = v.view(M, K, H)
        return (v * self.w.view(M, K, 1)).sum(dim=1)

    def args(self):
        return (
            self.q.view(torch.uint8).view(-1),
            self.s.view(-1),
            self.w.view(-1),
            self.y.view(-1),
            M,
            torch.cuda.current_stream(),
        )


cases = [Case() for _ in range(args.copies)]
c0 = cases[0]
t0 = time.time()
launch = compile_moe_reduce_fp8(H=H, topk=K)
fn = flyc.compile(launch, *c0.args())
print(f"[reduce] compile {time.time() - t0:.1f}s", flush=True)
fn(*c0.args())
torch.cuda.synchronize()
ref = c0.ref()
mine = c0.y.float()
err = (mine - ref).abs()
ulp = (ref.abs() / 128).clamp(min=1e-30)  # bf16 half-ulp-ish scale
print(
    f"[reduce] max |err| {err.max().item():.4g}, mean rel {(err.sum() / ref.abs().sum()).item():.3e}, "
    f"rows exact-bf16 {(c0.y == ref.to(torch.bfloat16)).all(dim=1).float().mean().item():.3f}, "
    f"max err/ulp {(err / ulp).max().item():.2f}",
    flush=True,
)

for c in cases:
    fn(*c.args())
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for c in cases:
        fn(*c.args())
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
gb = (M * K * (H + H // 32) + M * H * 2) / 1e9
print(f"[reduce] tokens={M} per call: median {us:.1f} us (range {meds[0]:.1f}..{meds[-1]:.1f}), {gb:.2f} GB moved = {gb / us * 1e3:.2f} TB/s", flush=True)
