# Middle-batch fp4 MoE — checkpoint 1

The new chain is opt-in (`bench_moe_a4w4 --chain mid --bm 32`). It is correct
at the tested M512 shape but **does not yet beat aiter**. No vLLM integration.

Latest handoff: checkpoint 3 below contains the atomic G2 result and the
validated M2048 aiter FP8 producer repair. All speed targets remain open.

## Files and layout

- `gemm1_mid.py`: BM32/64, TN128, four N waves, direct VGPR fp4 prefetch;
  gate/up, swiglu-OAI, bf16 rounding and per-32 MXFP4 quant in one kernel.
- `gemm2_mid.py`: TN256, direct VGPR fp4 prefetch, bf16 weighted partials.
- `gemm2_mid_fp8.py`: independent MXFP8-partial variant, per-32-column scales;
  routing weights are applied by the existing `reduce_fp8.py`.
- Sort is `m3_a16w4_moe/sort_decode_wave.py`; input quantization is aiter's
  `fused_dynamic_mx_quant_moe_sort`. The chain needs no tile-map launch.

## Correctness

M512 isolated gemm1, 256 sampled rows: 0/196608 fp4 values and 0/6144 scale
bytes differ from the explicit quantized reference. Cosine against that
reference >= 0.99999988. Cosine against the *unquantized* intermediate is
0.98720 minimum / 0.99029 mean: this is expected fp4 quantization loss.
The benchmark now correctly rounds fp4 midpoint ties to the even code.

Both full-chain output modes pass same-input x3 eager and x3 graph checks;
graphs contain 100 independent input/routing sets. With sort frozen, input
quantization, gemm1 payload/scales, gemm2 partials/scales and reduction are
also checked individually over three executions.

| M512 mode | Cosine vs aiter | Reference cosine min / mean | Aiter reference min / mean |
|---|---|---|---|
| bf16 partial | 0.999367 | 0.98882 / 0.99023 | 0.98876 / 0.99021 |
| fp8 partial | 0.999014 | 0.98843 / 0.98988 | 0.98876 / 0.99021 |

## Timing (us)

Five rounds, 100 different input/routing sets per graph. Whole-chain numbers
include `--tail prod` (all-reduce stand-in + actual Gemma RMSNorm). All these
runs set `AITER_FLYDSL_STAGE2_FP8=1` for aiter; `--out bf16` explicitly selects
the new chain's bf16 reference variant.

| M512 mode | Sort | Input quant | G1 | G2 | Reduce | Tail | Whole chain | Aiter whole chain |
|---|---|---|---|---|---|---|---|---|
| bf16 | 6.9 | 7.2 | 129.2 | 77.5 | 8.4 | 10.0 | 240.6 [240.6,240.7] | 213.0 [213.0,213.1] |
| fp8 | 7.0 | 7.2 | 129.2 | 75.9 | 7.9 | 10.0 | 240.3 [240.2,240.3] | 212.6 [212.6,212.7] |

Isolated G1: 127.9 [125.4,129.2] us. The station's original BM128 G1 was
145.5 us at M512; the new G1 improves this stage but leaves a substantial gap.
The station's full-MoE weight-read floor is 138.6 us at 7 TB/s and excludes
the downstream tail. Its 204.1 us aiter number also excludes that tail.

The user's standing preference is to keep `AITER_FLYDSL_STAGE2_FP8=1` in
future comparable A/B runs. It applies to supported FlyDSL stage-2 paths;
the currently deployed M3 prefill wrapper hardcodes bf16 and needs wiring
when the new chain is integrated. Atomic-add rounding variation is accepted
for the small decode path, but these new partial-buffer variants are deterministic.

## Next diagnostic

Profile G1 rather than extrapolating the isolated G2/reduce saving to the
whole chain. ATT collection with `sys_trace: true` hit a rocprofv3 finalizer
segfault while serializing HIP API arguments; GPU work and the kernel CSV
completed first. Retry uses the established ATT-only configuration in a new
output directory. No kernel failure is inferred from this tool finalizer error.

## Checkpoint 2: AGPR / wider tiles; aiter fp8 reference issue at M2048

Independent G1 files, M512 isolated latency (us, 100 inputs, five rounds):

| Variant | Median [range] |
|---|---|
| mid, depth 1 | 131.3 [129.2,133.0] |
| mid, depth 2 | 128.2 [125.7,129.4] |
| scalar K offsets | 129.2 [127.4,131.4] |
| shallow A / deep W | 130.2 [127.8,131.7] |
| pinned AGPR, TN128 | 129.8 [128.0,131.8] |
| pinned AGPR, TN256 | 124.4 [119.8,126.1] |
| TN256, shallow A / deep W | 124.6 [120.3,126.0] |
| TN256, rotate K quarter per wave | 127.3 [123.6,129.5] |

All passed the explicit quantized reference (zero sampled payload/scale
differences) and the then-required eager/graph bitwise diagnostics. Depth
1/2 reduced register counts to 116/162 without a speedup; simply increasing
occupancy was not enough. TN256 AGPR uses 204 VGPR + 64 AGPR; splitting A/W
rings reduces that to 180 + 64, still with similar time.

`--chain mid --mid-g1 agpr-wide-splitring --bm 32 --copies 100 --tail prod`,
fp8 enabled for both sides: M512 235.7 [235.7,235.8] vs aiter 212.7
[212.6,212.7]; M1024 306.1 [306.0,306.1] vs 242.3 [242.2,242.3]. Still slower.

M2048 could not be accepted against the fp8-enabled aiter path: mine vs aiter
cosine 0.909868. Independent fp32 reference on 64 tokens gives mine
min/mean 0.98787/0.98979, aiter 0.89509/0.90107. The error is therefore in the
aiter comparison path or its invocation, not evidence of a new-kernel loss.
The squared-route-weight hypothesis was tested and rejected (aiter cosine
against that reference only 0.87526 mean). Investigate before using its
performance as a baseline. The suspect path is the M2048 reduce epilogue
with `AITER_FLYDSL_STAGE2_FP8=1` / `MXFP4_G2_KSTATIC=1`; small-M atomic
aiter paths do not activate this fp8-output mode.

Latest user instruction: approximate cosine parity is sufficient; bitwise
checks are optional diagnostics now (`--check-determinism`). The benchmark
prints pairwise cosine but accepts by finite output and independent reference
cosine, requiring a healthy aiter reference as well.

The `sys_trace` finalizer left one sleeping profiler Python process with a
CUDA context. Its exact PID/command was verified and the process was cleaned
up with TERM then KILL (no pattern kills). Subsequent GPU selection reported
zero process VRAM and an idle device. Ordinary benchmark defaults are unchanged.

## Checkpoint 3: atomic G2 and verified aiter FP8 producer repair

User requested a handoff on 2026-09-07. No new performance scan was started
while closing this checkpoint. All whole-chain timings below include
`--tail prod`, 100 independent input/routing sets per graph, five rounds.
The 138.6 us weight-read floor excludes the tail, as does the station's
aiter table; do not mix those timing scopes.

| Configuration | M | Ours, median [range] us | Aiter, median [range] us |
|---|---|---|---|
| Wide split-ring AGPR G1, persistent FP8 G2, BM32 | 512 | 242.4 [242.3,242.4] | 212.9 [212.9,213.0] |
| Wide split-ring AGPR G1, atomic G2, BM32 | 512 | 222.4 [222.3,222.4] | 213.0 [212.9,213.1] |
| TN128 AGPR G1, atomic G2, BM64 | 1024 | 304.4 [304.4,304.5] | 242.6 [242.6,242.6] |

`gemm2_mid_persist.py` holds A across six output tiles. It passes reference
checks but is slower. `gemm2_mid_atomic.py` uses the existing packed-BF16
atomic epilogue and sort's output zeroing. There is no FP8 partial/reduce
on this path, even when the benchmark prints `out=fp8`; `--mid-g2 atomic`
takes precedence. M512 G2 is 69.4 [69.4,69.5] us, and its chain eliminates
the separate ~8 us reduction. Reference cosine min/mean is
0.98883/0.99023 vs aiter 0.98876/0.99021. M1024 is 0.98818/0.99015 vs
0.98820/0.99014. The best M512 candidate still loses to aiter by 4.4%.

Logs under `m3-compare/work/moe_midm/`:
`chain_mid_g2persist_m512.log`, `chain_mid_atomic_m512.log`,
`chain_mid_bm64_atomic_m1024.log`.

### M2048 aiter accuracy: producer bug, not reduction or routing weights

The active path is `moe_kernels.compile_flydsl_moe_stage2` ->
`mixed_moe_gemm_2stage.py` -> `mixed_moe_gemm_2stage_common.py`.
The separate `mxmoe_dispatcher` / `_flydsl_v2_stage2_wrapper` path is not
active in this benchmark. The captured buffer is `[10240, 6912]`: each row
has 6144 FP8 values plus 768 E8M0 scales (groups of 8); routing weights are
already applied by G2. `MXFP4_G2_KSTATIC=0/1` and `is_shuffled=True` did not
repair this path. Squared or omitted routing weights were also ruled out.

At TN128, `e_vec = min(body_tile_n // 32, 8)` selected four values per
thread. FP8 `store_pair` wrote the scale at `col_g0 // 8`, so two independently
quantized groups of four overwrote the same scale byte. The repair is
`e_vec=8` for `need_fp8_out`, together with
`cshuffle_nlane=min(32, body_tile_n // e_vec)` (16 at TN128).
Changing only e_vec fails the c-shuffle layout constraint.

| M2048 case | Independent-reference cosine min / mean |
|---|---|
| Our FP8-partial chain | 0.98787 / 0.98979 |
| Aiter FP8 before repair | 0.89509 / 0.90107 |
| Aiter FP8 disabled | 0.98822 / 0.99014 |
| Aiter FP8 after two-part repair | 0.98787 / 0.98980 |

Independent Torch decoding/reduction of aiter's partials gives cosine
~1.000000 against its output, both before and after repair. This isolates
the error to the producer. The repaired run completed successfully in
`aiter_fp8_fixed_diag.log`; it exits before timing. **No accepted M2048
timing with the repaired aiter baseline exists yet.** Repair validation is
limited to this M2048 configuration, not a general aiter regression suite.

The patch is checked in as `aiter_fp8_group8.patch`. It is already applied
only to container `m3cmp_new1` at
`/usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`.
Host `~/aiter` remains clean on its unrelated `ar-rms-2stage-pull` branch.
Container file SHA256:
`dd85d82f3fedccf1ccd0a367b588b0ca7663147b1eb3fa064d5616eeff31b875`.
Original and candidate copies are in
`m3-compare/work/moe_midm/aiter_fp8_fix/{before,after}.py`.
Record patched/unpatched aiter explicitly in any later timing table.

### Unfinished G1 prototype

`gemm1_mid_k2.py` changes G1 to two N waves x two K waves, TN128, AGPR C,
with one LDS exchange to combine K halves. Its goal is to halve repeated
A reads relative to the four-N-wave TN128 variant. The initial GPU compile
failed because local `ir` shadowed the imported IR module. The import is
now `_ir`; **only Python syntax was checked after this edit**. GPU compile,
cosine and performance remain unvalidated. Do not select it in vLLM.

First command for the next agent (not rerun during handoff):

```bash
docker exec -w /flydsl -e HIP_VISIBLE_DEVICES=1 -e PYTHONPATH=/flydsl \
  -e FLYDSL_RUNTIME_ENABLE_CACHE=0 -e AITER_FLYDSL_STAGE2_FP8=1 \
  m3cmp_new1 python3 -m m3_a4w4_moe.bench_gemm1 \
  --tokens 512 --bm 32 --kernel mid-k2 --prefetch 2 \
  --copies 100 --rounds 5 --check-rows 256
```
