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
