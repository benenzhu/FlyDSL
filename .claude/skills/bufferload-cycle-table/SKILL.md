---
name: bufferload-cycle-table
description: Build a per-tile buffer_load cycle/stall comparison table from two ATT traces, anchored at the K-tile start barrier. Each load shows gap-since-previous-load and its stall, with a trailing gap-to-tile-end. Use when comparing why one GEMM kernel's buffer_load stalls and another's does not (e.g. fp8 vs bf16, our kernel vs HipKittens), or when the user asks for a "buffer_load cycle table" / "每个 buffer_load 卡了多久".
---

# buffer_load Cycle Table

Compare two kernels' global-load (buffer_load) behavior inside one steady-state K
tile, to see whether stalls come from spacing (排布) or from VMEM-queue
back-pressure / DMA retire rate.

## What it prints

For each `buffer_load` in one K tile, anchored at the tile-start `s_barrier`:

```
       fp8 gap(stall)   bf16 gap(stall)
 L 0      80(4)            284(0)
 L 2     208(0)          128(216)   <- bf16 stalls here
 ...
end->   140(end)          128(end)  <- last load -> tile-end barrier
 sum    stall=24         stall=740
 len      2468             3324
```

- **gap** = cycles since the previous buffer_load (L0's gap = offset from the tile
  barrier).
- **stall** = ATT stall cycles on that load.
- **end->** = cycles from the last load to the tile-end barrier.
- **sum/len** = total stall and tile length (cycles) per kernel.

## How to read it (the key insight)

If the stalling kernel has **larger gaps but more stall**, spacing is NOT the
problem — it's VMEM-queue back-pressure (DMAs not retiring fast enough), which
correlates with **occupancy** and **dtype arithmetic intensity**, not scheduling.
A big empty gap (e.g. a 576-cyc hole at a half boundary) followed by zero-stall
loads confirms the queue had to fully drain before new loads could issue.

Example finding: fp8 4wave moved the same bytes/tile as our bf16 4wave but stalled
24 vs 740 cyc — fp8's loads spread evenly with small gaps and never stalled, while
bf16's first-half loads (L2-L7) stalled despite *larger* gaps. Root cause: VMEM
queue turnover, tied to occupancy=1 and bf16's lower FLOP/byte, not load spacing.

## Run

1. Capture ATT for both kernels (see att-hotloop-benchmark skill), pick a
   steady-state `--bar-start` (e.g. 40).
2. Determine how many barrier intervals make one K tile (`--tiles`): RP1 bf16 and
   fp8 4wave both use 2 barriers per tile.

```bash
python .claude/skills/bufferload-cycle-table/scripts/bufferload_table.py \
  <ui_dir_fp8> <ui_dir_bf16> --labels fp8,bf16 --bar-start 40 --tiles 2
```

Uses wave `se0_sm0_sl0_wv0.json` by default (override with `--wave`). The ATT inst
tuple is `[cycle, type, stall, dur, code_id]`; barriers are located via code.json
asm text. See [[feedback-bufferload-per-tile-cycle-analysis]],
[[bf16-hgemm-vmem-bound-arith-intensity]], [[fp8-4wave-8buffer-no-lgkmcnt-drain]].
