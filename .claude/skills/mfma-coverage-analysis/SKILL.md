---
name: mfma-coverage-analysis
description: From an ATT trace, find which hot-loop instructions are NOT hidden behind MFMA execution -- i.e. what is keeping cyc/mfma above the MFMA execute floor (16 for fp4, 32 for fp8 16x16x128). Tiles each MFMA's execute window, finds the exposed gaps, and attributes each gap's cycles to the instruction that blocked the next MFMA. Use when a GEMM/attention kernel is MFMA-bound and you want to push cyc/mfma toward the floor, or the user asks "what isn't hidden behind MFMA", "哪些指令没被 mfma 掩盖", "暴露的指令", or which non-MFMA op is the biggest exposed stall.
---

# MFMA Coverage Analysis

## The model (user-validated)

Each MFMA occupies a fixed **execute window** of EXEC cycles from its issue cycle.
Back-to-back MFMAs tile `[c, c+EXEC)` windows; while the matrix unit is busy, any
co-issued scalar / VMEM / LDS instruction is *free* (hidden in the shadow). Cycles
**outside** the union of those windows are EXPOSED — the matrix unit idles, so
`cyc/mfma > EXEC`. The goal of MFMA-bound optimization is to shrink the exposed
fraction toward 0 so cyc/mfma → EXEC (see [[feedback-mfma-bound-reduce-scalar-toward-cyc16]]).

Attribute each exposed gap to the **first non-MFMA instruction in it** = what
blocked the next MFMA from issuing on time. That ranks the real targets by cycles,
which is very different from ranking by instruction *count* or by the generic ATT
stall-type breakdown (a scalar op that issues alone for 1 cycle and an op that
gates a 50-cycle VMEM wait look identical by count but not by exposed cycles).

EXEC is the MFMA **execute** latency, NOT issue latency:
- fp4 `mfma_scale_f32_16x16x128_f8f6f4` → **16**
- fp8 `mfma..16x16x128` → **32** (its issue latency)
Pass the right `--exec`.

## How to run

```bash
# 1. capture ATT (see att-hotloop-benchmark skill) on an idle GPU, long-K shape
# 2. run, first without --range to print the cycle span:
python3 .claude/skills/mfma-coverage-analysis/scripts/mfma_coverage.py <dispatch_dir>
# 3. pick a mid steady slice spanning ~10 outer-loop iters, re-run:
python3 .claude/skills/mfma-coverage-analysis/scripts/mfma_coverage.py \
    <dispatch_dir> --range 219000,245000 --exec 16
```

Output: MFMA-covered %, EXPOSED %, cyc/mfma vs floor, and the exposed-gap cycles
bucketed by blocking instruction.

## Reading it

- **A named instruction with high exposed cycles is the real target**, even if it
  is a tiny fraction of the instruction count and not in the generic stall top-N.
  Example (fp4_gemm_4wave, m0-incr commit, cyc/mfma 20.25, exposed 23%): the
  biggest blocker was `buffer_load_dword` (scale) at 30% of exposed cycles — only
  3.3% of instructions and absent from the stall-type top-25, yet the #1 thing
  keeping cyc/mfma above 16. This reversed an earlier "scale load isn't worth it"
  call that was based on the wrong (count / stall-type) metric.
- **`idle` gaps** = pure latency stalls (waitcnt drain / dependency) with no issuing
  instruction — usually not directly cuttable, attack the named buckets first.
- `s_waitcnt` high → the wait itself gates; consider relaxing vmcnt/lgkmcnt or
  moving the producing load earlier (deeper prefetch).

## Caveats

- Uses one wave file (se0_sm0_sl0_wv0.json by default). Different waves are
  equivalent repeats; pass `--wave` to check another.
- The "first non-MFMA in gap" heuristic attributes the whole gap to one op; a gap
  containing a producer + its consumer is charged to the producer. Good enough to
  rank targets, not exact per-op liveness.
- Data structures: instruction record = `[cycle, dur, _, _, code_id]`; code_id ->
  `code.json["code"][cid][0]` is the asm text. (See att-hotloop-benchmark for the
  complementary barrier-window cyc/mfma summary.)
