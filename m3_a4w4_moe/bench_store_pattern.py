"""HBM write cost vs. row-segment size. gemm2 writes each output row (token, slot) in
row-major [rows, H] as one SEG-byte contiguous piece per n-tile (bf16: 512 B, fp8: 256 B),
the rows of a tile scattered over the whole output (tokens sorted by expert). This writes
TOTAL bytes as SEG-byte pieces at random (row, column-block) positions of a [ROWS, H]
buffer, one wave per 1 KB store instruction (64 lanes x 16 B), and reports TB/s.
Usage: python -m m3_a4w4_moe.bench_store_pattern [--rows 163840] [--hbytes 12288]"""
import argparse, statistics, torch, triton, triton.language as tl

@triton.jit
def _scatter(out_ptr, rows_ptr, cols_ptr, n_pieces, SEG: tl.constexpr, PER_PROG: tl.constexpr, ROW_BYTES: tl.constexpr,
             STRIDED: tl.constexpr, CM: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, SEG // 16)  # one 16-B chunk per lane
    v = tl.full((SEG // 16, 4), 0x3F80, tl.int32)
    nprog = tl.num_programs(0)
    for i in range(PER_PROG):
        if STRIDED:
            p = pid + i * nprog  # neighbouring programs take neighbouring pieces
        else:
            p = pid * PER_PROG + i
        r = tl.load(rows_ptr + p)
        c = tl.load(cols_ptr + p)
        base = r.to(tl.int64) * ROW_BYTES + c.to(tl.int64) * SEG
        ptrs = out_ptr + base + lane[:, None] * 16 + tl.arange(0, 4)[None, :] * 4
        tl.store(ptrs.to(tl.pointer_type(tl.int32)), v, cache_modifier=CM)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=163840)
    ap.add_argument("--hbytes", type=int, default=12288)
    ap.add_argument("--total-gb", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--cm", default="", help="triton store cache_modifier, e.g. .cs (-> nt on gfx950)")
    ap.add_argument("--pair", action="store_true",
                    help="pieces 2k, 2k+1 are the two halves of one 2*SEG segment, written by neighbouring programs")
    a = ap.parse_args()
    dev = "cuda"
    out = torch.empty(a.rows * a.hbytes, dtype=torch.uint8, device=dev)
    total = int(a.total_gb * 2**30)
    torch.manual_seed(0)
    for seg in (128, 256, 512, 1024, 2048, 4096):
        n = total // seg
        if a.pair:
            rows = torch.randint(0, a.rows, (n // 2,), dtype=torch.int32, device=dev).repeat_interleave(2)
            cb = torch.randint(0, a.hbytes // (2 * seg), (n // 2,), dtype=torch.int32, device=dev)
            cols = (cb * 2).repeat_interleave(2) + torch.arange(n, dtype=torch.int32, device=dev) % 2
        else:
            rows = torch.randint(0, a.rows, (n,), dtype=torch.int32, device=dev)
            cols = torch.randint(0, a.hbytes // seg, (n,), dtype=torch.int32, device=dev)
        per_prog = max(1, 4096 // seg)  # ~4 KB per program
        grid = (n // per_prog,)
        kw = dict(SEG=seg, PER_PROG=per_prog, ROW_BYTES=a.hbytes, STRIDED=a.pair, CM=a.cm, num_warps=1)
        for _ in range(2):
            _scatter[grid](out, rows, cols, n, **kw)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.reps):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); _scatter[grid](out, rows, cols, n, **kw); e.record()
            torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
        t = statistics.median(ts)
        print(f"seg {seg:5d} B{' paired' if a.pair else ''}{' ' + a.cm if a.cm else ''}: {n/1e6:6.1f} M pieces, {t:8.1f} us, {total/t/1e6:5.2f} TB/s, {n/t/1e3:6.1f} G pieces/s", flush=True)

if __name__ == "__main__":
    main()
