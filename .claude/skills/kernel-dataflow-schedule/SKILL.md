---
name: kernel-dataflow-schedule
description: Write a compact, source-derived steady-state SCHEDULE for one wave of a GEMM/attention kernel, by hand-reading the kernel source (NOT the ISA). Each line is one logical op (mfma / dsrd / bfld / barrier) in program order, with C blocks in GLOBAL coordinates, A/B operands as double-buffer positions (A0/A1/B0/B1), explicit vmcnt on every barrier, and an end-of-iteration count check. Use when the user asks to "describe/draw the data flow", "画数据流 / 流水排布", "compact schedule", "走一遍这个 kernel 的流水", or to explain how mfma/ds_read/buffer_load interleave and which buffer each load fills. This is the SOURCE-LEVEL companion to the ISA-level hotloop-table skill.
---

# Kernel Dataflow Schedule

Produce a human-verifiable, source-level schedule of one wave's steady-state
main loop. The goal is a table the user can read line-by-line to (1) confirm
correctness of the pipeline (which buffer/block each load touches, which slice it
holds) and (2) reason about latency hiding (how dsrd/bfld are co-issued in the
MFMA shadow). This is NOT the ISA hotloop-table (that collapses real AMDGCN to
M/r/L/B); here we hand-derive from the Python/C++ kernel source and keep full
operand positions.

Reference examples (read these first, match their style EXACTLY):
- `resource_inspect/hgemm_wave0_compact_schedule.txt` (bf16, K=32 mfma -> ops
  carry a k-subtile index like `A0[5,1]`)
- `resource_inspect/fp8_4wave_compact_schedule.txt` (fp8, K=128 mfma -> one mfma
  consumes a whole block, no k index on mfma)

## The four ops (only these, one per line, in program order)

| op | meaning | source it comes from |
|----|---------|----------------------|
| `mfma C[r,c] += A0[a]*B0[b]` | one MMA instruction | mfma.call_one / mma_atom_call |
| `dsrd <buf>[blk,h]` | LDS -> reg, one load granule | s2r.load_one / ds_read |
| `bfld <buf>[lo:hi]` | global -> LDS chunk | g2s.load_one / buffer_load_lds |
| `barrier vmcnt(N)+s_barrier` | wait_barrier(N) | wait_barrier / gpu.barrier |

Drop everything else (address math, salu/valu, packing). Keep blank lines between
logical groups (e.g. per output quadrant) so the structure is scannable.

## Notation conventions (the agreed style)

- **One wave only** (wave0). Steady-state main-loop body = one outer K iteration.
- **C blocks: GLOBAL coordinates.** The output tile is a grid of 16x16 MFMA
  blocks; `C[r,c]` uses the block's global (row,col) index over the whole
  BLOCK_M x BLOCK_N tile (e.g. a 256x256 tile -> r,c in 0..15). If the wave owns a
  strided set, say so in the header and use the real global indices (e.g. wave0 =
  rows {0..3,8..11} x cols {0..3,8..11}).
- **A/B operands: double-buffer view `A0/A1/B0/B1`.** Merge the kernel's physical
  sub-buffers into two logical buffers per operand: `A0`=current K-slice,
  `A1`=next K-slice (prefetch), same for B. Each logical buffer is the full
  BLOCK_M (or BLOCK_N) x BLOCK_K, indexed by block: `A0[blk]`. If the kernel
  splits M/N into halves, map them into one block range (e.g. half0 -> blk 0..7,
  half1 -> blk 8..15) so block numbers line up with the global C rows/cols.
- **mfma operand index has NO K subscript when one mfma eats the whole BLOCK_K**
  (e.g. fp8 16x16x128). It DOES carry a k-subtile index when BLOCK_K needs
  multiple mfma along K (e.g. bf16 16x16x32 -> `A0[5,1]` = block5, k-half1).
- **dsrd second index `h` is the LOAD granularity**, not necessarily a K subtile:
  if one ds_read moves half a block, write `B0[8,0]` / `B0[8,1]` for the two
  halves that pack into one mfma operand. State this in the header.
- **bfld writes a BLOCK RANGE** `<buf>[lo:hi]` (which row/col blocks of the K+P
  prefetch slice this g2s step fills). Compute the range from the loader's
  per-step row stride. Name the destination buffer explicitly (A0/B0 if the
  kernel refills the current buffer in place + swaps, A1/B1 if it writes the
  other buffer). Note which convention this kernel uses -- it is a real semantic
  difference between kernels.
- **Every barrier shows its vmcnt** as `barrier vmcnt(N)+s_barrier`. N = max
  outstanding VMEM allowed (smaller = waits for more to land). Derive N from the
  wait_barrier argument symbolically AND give the number.
- **Ignore swizzle.** It relabels LDS addresses, not pipeline order. Say so once.

## How to produce it (steps)

1. Read the kernel source: launch config (threads, waves, BLOCK_M/N/K), the
   per-wave tile ownership, the loader classes (G2S, S2R), the mma wrapper, and
   the main loop body. Do NOT read ISA -- this is a source-derived schedule.
2. Fix the wave0 -> C-block mapping (compute it from store offsets / wave_i,j).
3. Define the logical A0/A1/B0/B1 merge and the block-index convention; write the
   header block (layout, buffers, notation, K-slice bookkeeping) like the
   examples.
4. Walk the main loop in program order. For each compute cluster emit mfma lines
   interleaved with the exact dsrd/bfld the source issues between them. Put the
   barriers where the source calls wait_barrier, with the resolved vmcnt.
5. Track K-slice bookkeeping: which slice each buffer holds on entry, what the
   bfld writes (K+P), and the end-of-iter swap. State entry/exit so the user can
   confirm the pipeline is consistent.
6. **Count check** at the end: mfma / dsrd / bfld counts per iteration, and
   confirm bfld count == the main-loop vmcnt value (they usually match by
   construction). List the barrier split points.

## Output

- Write to `resource_inspect/<kernel>_wave0_compact_schedule.txt` (or answer
  inline if the user only wants to read it).
- Keep the header explanatory but put NO inline comments on the op lines unless
  asked -- the user reads the ops directly. Blank lines separate quadrants/groups.
- After writing, surface the 1-2 load-buffer-convention assumptions you made
  (in-place refill+swap vs write-other-buffer; strided vs contiguous wave
  footprint) and ask the user to confirm, since those are the easy things to get
  backwards.
