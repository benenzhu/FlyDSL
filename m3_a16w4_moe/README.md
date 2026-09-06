# MiniMax-M3 decode MoE (M=4), a16w4 in FlyDSL

Goal: beat the CK-tile a16w4 MoE path vLLM uses for MiniMax-M3 on MI355X at decode
(M=4, TP4 shape: hidden 6144, inter 768, 128 routed + 1 shared expert, topk 4+1),
keeping bf16 activations (no fp4 activation quant, so accuracy matches production).

## Baseline (production image, GPU graph replay, M=4)

| path | kernels | per call |
|---|---|---|
| CK-tile a16w4 (`swiglu_mxfp4_bf16_cktile`, split-K 3) | sort 5.7 + zero 4.1 + gemm1 19.5 + swiglu 4.3 + gemm2 10.0 | **43.9 us** |
| aiter#3832 a4w4 FlyDSL port (fp4 activations, cos 0.97) | sort+quant 4.1 + gemm1 22.3 + gemm2 7.9 | 39.2 us |

Numbers from `m3-compare/scripts/bench_moe_m4.py` + rocprofv3 (see m3-compare NIGHTLOG 2026-09-04).

## What is here

Starting point is FlyDSL's `kernels/moe/moe_2stage_a16wmix` (PR #948, also vendored in
aiter main but only enabled for SiTUv2 there). Copied, then:

- `utils.py`: flydsl 0.2.4 shims (`buffer_ops` import, `s_waitcnt` raw encoding, static
  `crd2idx`) so it runs on the flydsl inside the production vLLM image.
- `gemm1.py`: `act="swigluoai"` epilogue (gpt-oss / MiniMax swiglu: alpha 1.702, limit 7).
- `host.py`: explicit-tile launch wrappers (no CSV lookup).
- `bench_m3.py`: M=4 bench, same inputs as `bench_moe_m4.py`, swigluoai reference,
  graph-replay timing; `--loop N` for rocprofv3.

Chain per call: aiter `moe_sorting` (sort + zero output) -> gemm1 -> gemm2 (atomic add).

## Run (inside the image)

```bash
docker exec -it m3cmp_kda bash
cd /flydsl && PYTHONPATH=/flydsl python3 m3_a16w4_moe/bench_m3.py --tile-m 16 --g1-tile-n 128 --g1-tile-k 256
# per-kernel breakdown
rocprofv3 --kernel-trace --stats -d /work/rp_flydsl -- python3 m3_a16w4_moe/bench_m3.py --no-check --loop 200
```

## Results log

| date | gemm1 tiles | gemm2 tiles | graph replay | note |
|---|---|---|---|---|
| 09-05 | bm16 tn128 tk256 kw1 | tn256 tk256 | 45.6 us | first run, untuned, on flydsl 0.2.4 |
| 09-05 | bm16 tn64 tk256 kw1 | tn256 tk256 | 43.0 us | |
| 09-05 | bm16 tn64 tk128 kw1 | tn256 tk256 | 45.5 us | |
| 09-05 | bm16 tn64 tk128 kw4 | tn256 tk256 | **40.9 us** | best so far; rocprof: sort 6.0 + gemm1 21.2 + gemm2 8.6 = 35.8 kernel us |
| 09-05 | bm16 tn64 tk256 kw4 | tn256 tk256 | 41.3 us | |
| 09-05 | bm16 tn64 tk128 kw2 | tn256 tk256 | 41.4 us | |
| 09-05 | bm16 tn128 tk128 kw1 | tn256 tk256 | 50.2 us | |
| 09-05 | bm16 tn128 tk128 kw2 | tn256 tk256 | 47.1 us | |
| 09-05 | bm16 tn128 tk256 kw1 xcd1 | tn256 tk256 | 46.5 us | xcd swizzle hurts at M=4 |
| 09-05 | bm16 tn64 tk128 kw4 nt2 | tn256 tk256 | 44.3 us | nt weight loads hurt at M=4 |
| 09-05 | bm16 tn32 tk256 kw1 | tn256 tk256 | (27.8 us) | INVALID: 8 cols/wave -> 0 accumulators, output NaN; now asserted |

| 09-05 | bm16 tn32 tk128 kw4 | tn256 tk256 | 36.3 us | 408 WGs; rocprof: sort 6.1 + gemm1 18.2 + gemm2 8.6 |
| 09-05 | bm16 tn32 tk256 kw4 | tn256 tk256 | 37.6 us | |
| 09-05 | bm16 tn16 tk128 kw4 | tn256 tk256 | 36.9 us | 816 WGs, no further gain |
| 09-06 | bm16 tn32 tk128 kw4 **a_direct** | tn256 tk256 | **34.6 us** | A straight to VGPR, no LDS/barrier in the K loop |
| 09-06 | bm16 tn64 tk128 kw4 a_direct | tn256 tk256 | 35.5 us | was 40.9 with the LDS A path |

Correctness gate: `cos vs swigluoai ref` >= 0.9999 (bf16-intermediate reference); 0.99999 measured.

### ATT trace of gemm1 (bm16 tn64 tk128 kw4, LDS A path), one CU, 4 waves

66% of cycles stalled: VMEM-wait 37%, VMEM-load (issue back-pressure) 23%, lgkmcnt-wait 23%,
barrier 5%. The K loop ISA has a `s_waitcnt vmcnt(0)` per tile: the backend inserts it
after the A LDS-DMA (`buffer_load ... lds`) before the ds_read, and it also drains the
W loads prefetched for the next tile, so the one-tile-ahead prefetch never overlaps.
The scalar W-scale reads (`buffer_load_dword`, 8 per tile) account for 21% of stall on
their own (issue back-pressure). Fix 1 = `a_direct` (above). Traces: `/work/att_g1*`
in the container; analyzer: `~/FlyDSL/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`.

Where the time goes (bm16 tn64 tk128 kw4, M=4): gemm1 streams the same 80 MB of W1 as
CK-tile but only 204 workgroups x 4 waves are in flight (17 expert blocks x 12 N tiles) on
256 CUs, one K tile ahead. The old a4w4 FlyDSL gemm1 (t32x64x256, "async" pipeline) moves
the same bytes in 15.2 us, so the gap is pipeline depth, not dequant cost.
