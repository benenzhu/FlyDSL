# Softmax Kernel Performance: FlyDSL vs Quack Comparison

## Summary

This document compares the FlyDSL softmax kernel on AMD MI355X (CDNA4) against the Quack (CuTe-DSL) softmax kernel targeting NVIDIA H100/B200/B300, and analyzes the gap to theoretical peak HBM bandwidth on both platforms.

## Hardware Specifications

| | AMD MI355X (gfx950) | NVIDIA H100 SXM | NVIDIA B200/B300 |
|---|---|---|---|
| Architecture | CDNA4 | Hopper (SM90) | Blackwell (SM100) |
| HBM Type | HBM3E | HBM3 | HBM3E |
| HBM Capacity | 288 GB | 80 GB | 192 GB / 288 GB |
| **Peak HBM BW** | **8.0 TB/s** | **3.35 TB/s** | **8.0 TB/s** |
| Wave/Warp Size | 64 | 32 | 32 |
| Shared Memory | 160 KB LDS | 228 KB SMEM | ~228 KB SMEM |
| Cluster Support | N/A | Up to 16 CTAs | Up to 16 CTAs |

## FlyDSL Softmax Performance on MI355X

From CI run [#1259](https://github.com/ROCm/FlyDSL/actions/runs/24490203236/job/71573512108) on MI355X (gfx950):

| Op | Shape | Dtype | TB/s | % of Peak (8 TB/s) |
|---|---|---|---|---|
| softmax | 32768x8192 | bf16 | 5.833 | **72.9%** |
| layernorm | 32768x8192 | bf16 | 5.403 | 67.5% |
| rmsnorm | 32768x8192 | bf16 | 5.777 | 72.2% |

### Bandwidth Calculation

For softmax with shape (M=32768, N=8192) in bf16:
- Total data moved = 2 × M × N × 2 bytes = 2 × 32768 × 8192 × 2 = 1,073,741,824 bytes ≈ 1.07 GB
- Measured bandwidth = 5.833 TB/s → kernel time ≈ 1.07 GB / 5.833 TB/s ≈ 0.184 ms

## Quack Softmax Performance (NVIDIA)

### H100 (3.35 TB/s peak HBM)

From the Quack blogpost and benchmark infrastructure:
- For bf16, shape 32768x8192 (N=8192): the blogpost's chart shows ~**3.0 TB/s** model memory throughput (~90% of 3.35 TB/s peak), consistent across reduction dims 4K-262K
- For fp32, shape 16384x131072: NCU profiling shows 3.01 TB/s DRAM throughput (**89.7%** of peak)
- The Quack blog explicitly states: "our impl in CuTe DSL generally maintains a model memory throughput about 3 TB/s (~90% peak) for a reduction dimension larger than 4k"

### RTX 5090 / SM120 (~1.8 TB/s peak)

From `benchmarks/sm120_cluster_benchmark.md`:

| N | softmax fwd (bf16) GB/s | % of peak (~1.8 TB/s) |
|---|---|---|
| 4096 | 1,552 | ~86% |
| 8192 | 1,527 | ~85% |
| 16384 | 1,504 | ~84% |
| 32768 | 1,523 | ~85% |
| 65536 | 1,528 | ~85% |
| 131072 | 1,531 | ~85% |

### B200/B300 (8.0 TB/s peak HBM)

**No published softmax benchmark numbers for B200/B300 were found.** The Quack CI on B300 runs correctness tests only (13,319 test cases), not performance benchmarks. The README states B200/B300 support, and the kernel code has SM100 code paths, but no throughput numbers are published.

However, based on RMSNorm data from a [PyTorch PR](https://github.com/pytorch/pytorch/pull/175551) integrating CuTe-DSL kernels on B200:
- 65536x2048 bf16: ~5.1 TB/s (~64% of 8 TB/s peak)
- 32768x1024 bf16: ~4.1 TB/s (~51% of 8 TB/s peak)

These RMSNorm numbers suggest B200/B300 softmax would likely achieve similar 4-6 TB/s range.

## Head-to-Head Comparison

| Metric | FlyDSL (MI355X) | Quack (H100) | Quack (B200/B300 est.) |
|---|---|---|---|
| Peak HBM BW | 8.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| Softmax BW (bf16, 32Kx8K) | 5.833 TB/s | ~3.0 TB/s | ~5-6 TB/s (est.) |
| **% of Peak** | **72.9%** | **~90%** | **~63-75% (est.)** |
| Absolute throughput advantage | Baseline | 1.94x lower | Comparable |

### Key Observations

1. **FlyDSL achieves higher absolute bandwidth** (5.833 TB/s) than Quack on H100 (~3.0 TB/s) because MI355X has 2.4x the HBM bandwidth of H100.

2. **Quack achieves higher % of peak** (~90%) compared to FlyDSL (72.9%). This gap represents the optimization opportunity for FlyDSL.

3. **Both platforms have 8 TB/s peak** (MI355X and B200/B300), making the % of peak the fair comparison metric. No published Quack softmax numbers exist for B200/B300, but if they maintain 85-90% efficiency, that would be 6.8-7.2 TB/s.

## Gap Analysis: FlyDSL Path to 8 TB/s

Current FlyDSL softmax achieves 5.833 TB/s = 72.9% of 8 TB/s peak. To close the gap:

### Architecture-level Differences

| Feature | Quack Approach | FlyDSL Current |
|---|---|---|
| Load strategy | 128-bit vectorized async cp.async G→S→R | Scalar BufferCopy16b/32b (generic path) |
| Fast vectorized path | Always on (128-bit tiled copies) | **Disabled** (`if False and ...`) |
| Reduction hierarchy | Thread → Warp → Block → Cluster (4 tiers) | Thread → Warp → Block (3 tiers) |
| Online softmax | Supported (fuses max+sum in 1 pass) | Not used (separate max + sum passes) |
| SMEM reload trick | Reloads from SMEM between passes to reduce register pressure | Not implemented |
| Thread-per-row tuning | Adaptive (8-256 threads based on N) | Fixed 256 threads always |
| Cluster reduction | DSMEM cross-CTA reduction for large N | N/A (AMD has no cluster concept) |

### Specific Optimization Opportunities

1. **Enable the vectorized fast path** (BufferCopy128b, VEC_WIDTH=8): The code already exists but is disabled with `if False`. For the benchmark shape 32768x8192, `N % tile_cols == 0` holds (8192 % 2048 == 0). Enabling this alone could significantly improve throughput by loading 8 bf16 values per thread per memory transaction instead of 1.

2. **Adaptive threads-per-row**: Quack tunes `threads_per_row` per-N (8 for N≤64, up to 256 for N>16K). FlyDSL always uses 256 threads all reducing the same row. For N=8192 with bf16, each thread handles only 32 elements — more threads doing reduction means more shuffle overhead relative to useful work.

3. **Register pressure via SMEM reload**: Quack reloads `x` from SMEM between the max and sum phases. This halves peak register usage during reduction (the compiler can evict `x` registers). Blog data shows this improves throughput from 2266 GB/s to 3025 GB/s on H100 for RMSNorm — a 33% gain.

4. **Online softmax (single-pass max+sum)**: Quack supports fusing the max-finding and exp-sum phases into a single pass using the "online softmax" algorithm (Milakov & Gimelshein, 2018). This reduces the data touched from 3 passes (load for max, load for exp+sum, load for normalize) to 2 passes (load for online max+sum, load for normalize), a potential 33% bandwidth improvement for register-spill-limited cases.

5. **Multi-row processing per CTA**: Quack's `cols_per_block` can process multiple rows per CTA (e.g., when N is small enough that threads_per_row < num_threads). FlyDSL uses 1 row per CTA (grid.x = M).

## Conclusion

FlyDSL's softmax on MI355X achieves 5.833 TB/s (72.9% of 8 TB/s peak). The primary optimization opportunities are:

1. **Enable the vectorized load/store path** — highest impact, already implemented but disabled
2. **SMEM reload between reduction phases** — proven 33% gain on similar workloads
3. **Adaptive thread configuration** — match threads-per-row to problem size
4. **Online softmax** — fuse max+sum for fewer passes

Reaching 85-90% efficiency (6.8-7.2 TB/s) on MI355X would put FlyDSL on par with Quack's efficiency on H100, and would likely match or exceed whatever Quack achieves on B200/B300 (which has the same 8 TB/s peak HBM).
