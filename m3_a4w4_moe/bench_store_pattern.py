"""Write-bandwidth ceiling of gemm2's output pattern (timing only, no GEMM).

gemm2 writes bf16 rows of [rows, H] token-major output: per n-tile a CTA writes
128 (sorted -> random) rows x BN cols, i.e. 128 pieces of BN*2 contiguous bytes at
random row addresses. This sweeps the piece width (COLS) and random vs sequential
rows so we know what the HBM/L2 write path can absorb before touching the kernel.
"""
import argparse, statistics, torch, triton, triton.language as tl

@triton.jit
def _scatter_rows(out_ptr, rows_ptr, H: tl.constexpr, COLS: tl.constexpr, CB: tl.constexpr, RB: tl.constexpr, CM: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = tl.load(rows_ptr + pr * RB + tl.arange(0, RB)).to(tl.int64)
    base = r * H + pc * COLS
    v = tl.full((RB, CB), 1.0, tl.bfloat16)
    for c0 in tl.static_range(0, COLS, CB):
        offs = base[:, None] + c0 + tl.arange(0, CB)[None, :]
        tl.store(out_ptr + offs, v, cache_modifier=CM)

def time_us(fn, reps):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=32768 * 5)
p.add_argument("--hidden", type=int, default=6144)
p.add_argument("--reps", type=int, default=10)
a = p.parse_args()
torch.manual_seed(0)
out = torch.empty(a.rows, a.hidden, dtype=torch.bfloat16, device="cuda")
gb = out.numel() * 2 / 1e9
t = time_us(lambda: out.fill_(1.0), a.reps)
print(f"torch fill_ (contiguous)                    : {t:8.1f} us  {gb / t * 1e6 / 1e3:5.2f} TB/s")
perm = torch.randperm(a.rows, device="cuda", dtype=torch.int32)
seq = torch.arange(a.rows, device="cuda", dtype=torch.int32)
for name, rows in (("random rows", perm), ("sequential rows", seq)):
    for RB in (128, 32):
        for COLS in (64, 128, 256, 512, 1024, 6144):
            if RB == 32 and COLS not in (64, 128):
                continue
            CB = min(COLS, 256)
            grid = (a.rows // RB, a.hidden // COLS)
            for cm in ("", ".cs", ".wt", ".cg"):
                if cm and (COLS != 128 or RB != 128):
                    continue
                try:
                    fn = lambda: _scatter_rows[grid](out, rows, a.hidden, COLS, CB, RB, cm, num_warps=4)
                    t = time_us(fn, a.reps)
                    print(f"{name:16s} RB={RB:3d} piece={COLS * 2:6d} B/row cm={cm or '-':4s}: {t:8.1f} us  {gb / t * 1e6 / 1e3:5.2f} TB/s")
                except Exception as ex:
                    print(f"{name:16s} RB={RB:3d} piece={COLS * 2:6d} B/row cm={cm}: failed: {str(ex)[:80]}")
