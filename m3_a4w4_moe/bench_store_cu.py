"""Per-CU store throughput: is the ~5 TB/s write ceiling a chip limit or a per-CU
store-issue limit? Contiguous dwordx4 stores from registers; P programs (<= P CUs busy),
W warps each; TB/s and B/clk per busy CU (2.4 GHz nominal)."""
import argparse, statistics, torch, triton, triton.language as tl

@triton.jit
def _store_loop(out_ptr, rows_per_prog, ROW: tl.constexpr, CB: tl.constexpr):
    pid = tl.program_id(0)
    row0 = pid * rows_per_prog
    v = tl.full((CB,), 1.0, tl.bfloat16)
    cols = tl.arange(0, CB)
    for r in range(rows_per_prog):
        rowbase = (row0 + r).to(tl.int64) * ROW
        for c0 in tl.static_range(0, ROW, CB):
            tl.store(out_ptr + rowbase + c0 + cols, v)

@triton.jit
def _scatter_store(out_ptr, rows_ptr, groups_per_prog, H: tl.constexpr, RB: tl.constexpr, CB: tl.constexpr):
    """per wave store: 8 random rows x 128 B (gemm2's pattern); the 96 pieces of a row
    group are written back to back (better DRAM locality than gemm2, same CU-side work)"""
    pid = tl.program_id(0)
    v = tl.full((RB, CB), 1.0, tl.bfloat16)
    cols = tl.arange(0, CB)
    for gi in range(groups_per_prog):
        g = pid * groups_per_prog + gi
        r = tl.load(rows_ptr + g * RB + tl.arange(0, RB)).to(tl.int64) * H
        for c0 in tl.static_range(0, H, CB):
            tl.store(out_ptr + r[:, None] + c0 + cols[None, :], v)

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
p.add_argument("--reps", type=int, default=5)
a = p.parse_args()
out = torch.empty(a.rows, a.hidden, dtype=torch.bfloat16, device="cuda")
gb = out.numel() * 2 / 1e9
perm = torch.randperm(a.rows, device="cuda", dtype=torch.int32)
for P in (32, 64, 128, 256, 1024):
    gpp = a.rows // 32 // P
    t = time_us(lambda: _scatter_store[(P,)](out, perm, gpp, a.hidden, 32, 64, num_warps=4), a.reps)
    tbs = gb / t * 1e6 / 1e3
    cus = min(P, 256)
    print(f"scatter 8x128B  programs={P:5d} (busy CUs<={cus:3d}): {t:8.1f} us  {tbs:5.2f} TB/s  {tbs * 1e12 / 2.4e9 / cus:6.1f} B/clk/CU", flush=True)
for W in (4,):
    CB = 512 * W  # 16 B per lane per store
    for P in (32, 64, 128, 256, 1024):
        rpp = a.rows // P
        t = time_us(lambda: _store_loop[(P,)](out, rpp, a.hidden, CB, num_warps=W), a.reps)
        tbs = gb / t * 1e6 / 1e3
        cus = min(P, 256)
        print(f"warps={W:2d} programs={P:5d} (busy CUs<={cus:3d}): {t:8.1f} us  {tbs:5.2f} TB/s  {tbs * 1e12 / 2.4e9 / cus:6.1f} B/clk/CU", flush=True)
