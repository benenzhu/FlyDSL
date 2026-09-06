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
| 09-06 | bm16 tn32 tk128 kw4 a_direct pf2/3/4 | tn256 tk256 | 35.0 / 34.7 / 35.0 us | deeper W prefetch is flat: per-wave depth is not the limiter now |
| 09-06 | bm16 tn32 tk128 kw4 a_direct + hoisted prologue | tn256 tk256 xcd1 | 34.15 us | gemm1 15.3 us (rocprof) |
| 09-06 | same | tn256 tk256 **xcd0** | **33.9 us** | gemm2 LDS path; kernel sum sort 6.1 + gemm1 15.3 + gemm2 8.6 |
| 09-06 | same | tn256 tk256 xcd0 a_direct | 35.9 us | gemm2 a_direct is slower: no k_wave in gemm2, so all 4 waves re-read the whole A block (4x L2 traffic) |
| 09-06 | same | tn256 tk768 / tk384 | 35.0 / 34.6 us | single/2-tile K does not help gemm2 |
| 09-06 | same | **tn128** tk256 xcd0 | 33.55 us | 816 gemm2 WGs |
| 09-06 | same, **sort = aiter#3832 single-CTA sort+zero** | tn256 xcd0 | 32.4 us | sort 2.9 us instead of 6.1 (`_adaptive_moe_sort`, `--sort mxfp4`) |
| 09-06 | g1 bm16 **tn16** tk128 kw4 a_direct, sort mxfp4 | tn128 tk256 xcd0 | **31.6 us** | current best; kernel sum sort 2.9 + gemm1 15.5 + gemm2 8.5 = 26.9, rest is 3 launch gaps |
| 09-06 | same + `--g1-ss 1` (share the 256-K scale dword across 8 K tiles) | same | 31.65 us | correct (cos 0.99999) but flat; scale loads were not the limiter |
| 09-06 | same, `--stages 1/2/3` | same | 8.75 / 23.35 / 31.62 us | one call per graph: the sort-only graph costs 8.75 us for a 2.9 us kernel, so ~5.5 us of the 31.6 is the per-graph launch cost, not our kernels |
| 09-06 | same, **`--graph-copies 10`** | same | **26.28 us/call** | 10 calls per graph amortise the launch cost: sort 3.25, +gemm1 14.94, +gemm2 8.09. CK-tile measured the same way: **39.94 us** -> 34% faster |

**Measurement rule from here on (09-06):** every graph captures **100 calls with different
tokens and routing** (`--graph-copies 100`, the default now). One-call graphs hide 5.5 us of
launch cost, and identical calls keep one 35 MB expert set in the 256 MB infinity cache, so
the weights stop coming from HBM. All rows above are one-call / same-input numbers.

| date | gemm1 tiles | gemm2 tiles | 100 inputs/graph | note |
|---|---|---|---|---|
| 09-06 | CK-tile production path (`bench_moe_m4.py --variant a16w4`) | | **42.44 us** | baseline measured the same way |
| 09-06 | bm16 tn16 tk128 kw4 a_direct ss, sort mxfp4 | tn128 tk256 xcd0 | **32.36 us** | -24%; stages: sort 2.67, +gemm1 18.66, +gemm2 11.03. W from HBM: gemm1 ~85 MB -> 4.6 TB/s, gemm2 ~43 MB -> 3.9 TB/s (17 distinct experts per call) |
| 09-06 | re-sweep, gemm1 tn16 tk128 kw4 a_direct pf1/2/3/4 | tn128 tk256 xcd0 | 32.68 / 32.68 / 32.34 / 33.73 | with W really coming from HBM, prefetch depth matters a little (pf3) |
| 09-06 | gemm1 **tn32** tk128 kw4 a_direct pf1/2/**3** | same | 32.13 / 32.11 / **31.95** | tn32 beats tn16 again under the new rule (408 WGs, 2 W loads per lane per tile) |
| 09-06 | gemm1 tn16 tk256 kw4 a_direct pf1/2 | same | 34.14 / 33.74 | |
| 09-06 | gemm1 tn16 tk128 pf1 + `--g1-ss 1` | same | 32.28 | vs 32.68 without: sharing the 256-K scale dword is now worth 0.4 us |
| 09-06 | gemm1 tn16 tk128 pf1 | tn128 **xcd1** / **tn256** / **tn64** / **tk768** / **tk384** | 32.67 / 32.95 / 33.60 / 32.64 / 33.13 | gemm2 tile shape is flat around tn128 tk256 |
| 09-06 | gemm1 tn16 tk128 pf1 | tn128 xcd0 **a_direct** pf1 / pf3 | 37.10 / 37.04 | gemm2 a_direct still loses (4x A traffic without a K split) |
| 09-06 | gemm1 tn32 tk128 kw4 a_direct pf3 ss | tn128 tk256 xcd0 | 31.83 | new gemm2 options below measured against this |
| 09-06 | same | + `--g2-pad-mask 1` | 31.66 | padding rows OOB-masked in the A LDS-DMA (zero fill, no L2 traffic) |
| 09-06 | same | + `--g2-hoist 1` | 31.85 | prologue hoist alone (identity tile map) is flat |
| 09-06 | same | + pad-mask + hoist | 31.13 | together -0.7: the row token ids come with cumsum0, so the mask costs no extra round trip |
| 09-06 | same | + `--g2-ksplit 3` | 31.50 | split-K over CTAs (2448 WGs, one K tile each; atomics sum the partials, cos 0.99998) |
| 09-06 | same | + ksplit 3 + pad-mask + hoist | **30.93** | |
| 09-06 | same | tn256 / tk128 ks3 / tk128 ks6 (+pm +hoist) | 31.12 / 31.48 / 32.17 | |
| 09-06 | gemm1 **pf2** | tn128 tk256 xcd0 ks3 pm hoist | **30.71 us** | current best, -27.6% vs CK-tile 42.44 |

Compare only numbers measured with 100 different inputs per graph: **30.7 us vs 42.4 us CK-tile**.

Current best command:

```bash
PYTHONPATH=/flydsl python3 m3_a16w4_moe/bench_m3.py --tile-m 16 --k-wave 4 --sort mxfp4 \
    --g1-tile-n 32 --g1-tile-k 128 --g1-a-direct 1 --g1-pf 2 --g1-ss 1 \
    --g2-tile-n 128 --g2-tile-k 256 --g2-xcd 0 --g2-ksplit 3 --g2-pad-mask 1 --g2-hoist 1
```

Sorting: `aiter.fused_moe._adaptive_moe_sort` (already in the image) launches aiter#3832's
`sort_quant_kernel_impl<..., kSkipQuant=true>`: block 0 sorts with LDS counters, the other
CTAs zero the output. Same contract as `moe_sorting` (token | slot<<24, padding token = M,
`num_valid_ids[0]` = padded rows, `sorted_expert_ids` per BM block, zeroed `moe_buf`).

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
