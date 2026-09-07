# Persistent decode experiment — checkpoint 1

This is work in progress, **not a validated fast path**. The production kernel
and the benchmark's default implementation remain `gemm1.py`.

## Validation infrastructure

`bench_m3.py --stages 2` now validates gate/up + swiglu against the bf16
reference. It canonicalizes outputs by `(token, topk slot)`, checks routing
coverage, excludes unwritten padding, and checks three eager executions and
three graph replays bitwise. Validation is outside the timed graph.

The existing kernel passed at M=32: cosine approximately 1.0, three runs
bitwise equal in both modes. Sort + gemm1 measured 71.45 us [69.82, 71.51]
over five rounds, 100 different input/routing sets per graph. This is a
validation baseline, not a performance improvement or a full-layer time.

## New kernel status

`gemm1_persist.py` + `host_persist.py`, selected with `--g1-impl persist`,
implement BM16/TN32/TK128, four K waves and a CTA-strided item list. The last
K steps prefetch the next item's first steps before reduction/activation.

The first compilable version failed the numeric check (NaNs); no performance
result is accepted. ISA inspection found register copies of shared scale
dwords before the explicit wait. The current checkpoint deduplicates wait
operands and forwards ready scale values; this repair still needs validation.
Do not integrate this experiment into vLLM until checks and timing pass.

The user explicitly accepts rounding variation from bf16 atomic addition.
This does not waive deterministic gemm1 checks or justify attributing an
unexplained discrepancy to atomics. Gemm2's contributions before atomic
accumulation must be isolated when investigating such discrepancies.

## References

- `m3-compare/STATION_MOE_MIDM.md`, especially sections 2–5. Prior conclusions
  in section 3 are inputs, not experiments to repeat.
- Existing `gemm1.py`: weight addressing, scale sharing, slice-K reduction,
  and swigluoai math.
- `m3_a4w4_moe/gemm2_persist.py`: item loop and explicit wait conventions.
- ROCm/aiter PR #3832, merged 2026-07-09; reviewed head
  `aa65f59a10c38face177e08c86d0069ed5f930cd`. Small BM a4w4 layouts are useful
  references for the later mid-batch chain. Its LDS aliasing bug illustrates
  why one cosine pass cannot establish freedom from timing-dependent races.

Resume with the station's decode command plus `--stages 2 --g1-impl persist`.
Container `/work` currently maps to `m3-compare/work/moe_bench`, while the
host-side station logs are under `m3-compare/work/moe_midm`.

## Checkpoint 2: M32 correctness passes; performance does not improve yet

The scale hazard is fixed by deduplicating tied wait operands and forwarding
ready scale SSA values to adjacent tiles. FlyDSL numeric/vector wrappers
overload equality and formatting: bookkeeping must explicitly use MLIR value
identity, not DSL `==` or `str(value)`.

M32: reference cosine ~1.0, eager x3 and graph x3 bitwise equal. Sort + gemm1
is 75.55 us [73.61, 75.57], five rounds x 100 different inputs per graph,
versus the original 71.45 us [69.82, 71.51]. There is **no speedup** yet.
ISA still serializes the next descriptor's vector loads at every item start.
Next: issue row look-ahead under the current item's K loop and use a scalar
expert-id load. Other M values still need validation.

## Checkpoint 3: tested persistent variants do not beat the existing kernel

All numbers below are **sort + gemm1**, not the whole layer. Five rounds,
100 independent inputs/routings per graph. Every listed run passed reference
cosine >= 0.9999 and eager/graph same-input x3 bitwise checks.

| M | Existing | Descriptor look-ahead |
|---|---|---|
| 32 | 71.45 [69.82, 71.51] | 74.29 [72.31, 74.29] |
| 64 | 91.82 [91.80, 93.85] | 94.83 [94.78, 97.36] |
| 128 | 107.60 [107.53, 109.68] | 110.85 [110.78, 114.53] |
| 256 | 123.91 [123.88, 126.91] | 132.22 [132.18, 136.65] |

Independent files preserve each design:

- `gemm1_persist_lookahead.py`: scalar expert id + asynchronous row look-ahead.
- `gemm1_persist_fused_reduce.py`: separate gate/up scratch, two barriers
  instead of four. M32 74.12 [72.10, 74.14]; M256 130.42 [130.37, 133.83].
  M256 depth 2 / 512 CTAs: 136.50 [136.48, 140.77]; depth 1 / 768 CTAs:
  128.73 [128.63, 131.37]; depth 1 / 1024 CTAs: 130.08 [130.06, 135.51].
- `gemm1_persist_interleave.py`: defer next-tile loads into individual MFMA
  groups, with scheduling fences. M256 137.90 [137.81, 141.78].

None is selected by default or copied into vLLM. Pause this parameter scan;
the next substantial experiment is the middle-batch fp4 GEMM design.

### ATT evidence and reusable profiling entry points

The existing and new M256 traces were analyzed with
`.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`.
New trace: `att_traces/dec_gemm1_lookahead_256`, dispatch 129395.
VMEM-load accounts for 76.4% of its stalls; VMEM-wait 5.0%. The existing
trace reports 61.9% / 15.6%. Per-wave cycles are not additive GPU time:
eliminating explicit waits has moved much of the waiting to load issue.
This is not, by itself, proof of an HBM bandwidth ceiling.

`att_summary.py` complements the supplied analyzer with persistent-item
intervals (first/last MFMA), not another hotspot classifier. Each item executes
192 MFMAs. The sampled new waves execute six or seven items each.

The supplied analyzer heuristically labels bf16-only ISA as gfx942. These
traces are on **gfx950 / MI355X**; its architecture label must not override
the actual GPU or be used for the LDS limit. The ISA-only register estimates
are 208 allocated for the old kernel and 248 for look-ahead, both allowing
two waves/SIMD under the combined 512-entry budget.

### Sort experiment: small, verified saving

`sort_decode_wave.py` replaces the eight CTA-wide prefix-scan rounds with
wave-local scans and one cross-wave exchange. Its sort-only M32/M64/M128
times are 3.01 [3.01, 3.01] / 3.50 [3.50, 3.50] / 3.73 [3.72, 3.74] us,
versus 3.24 [3.24, 3.24] / 3.73 [3.73, 3.73] / 3.92 [3.92, 3.93].
With the old gemm1, sort+gemm1 is 71.30 [69.61, 71.31] /
91.57 [91.53, 93.59] / 107.41 [107.38, 109.39]. These all pass routing
coverage, reference and bitwise repeat checks. Select via `--sort decode-wave`.
The ~0.2 us saving is far smaller than the full-layer target gap.

Full-layer context from the station (not newly measured here): aiter
M32/64/128/256 = 121.5/153.3/181.0/200.3 us; one-weight-read floor at
7 TB/s = 88.8/120.7/136.3/138.6 us. No full-layer speedup is claimed.
