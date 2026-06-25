HGEMM register-prefetch / K32-slice pipeline idea
=================================================

Goal
----

Use extra VGPRs to keep a whole K32 slice in registers, then reuse the just-read
LDS stage earlier.  The current AK32 kernel is at about 2589 cycles per BLOCK_K=64
tile.  It uses 380 VGPR, 256 AGPR, no spill:

  vgpr_count=380
  agpr_count=256
  private_segment_fixed_size=0
  vgpr_spill_count=0

So for this specific kernel it is reasonable to spend more VGPR if we stay below
the no-spill limit.


Current naming
--------------

Use logical block coordinates:

  C[i,j]       one 16x16 output block
  A0[i,k]      current LDS stage, A block i, K32 slice k
  B0[j,k]      current LDS stage, B block j, K32 slice k
  A1/B1        the other LDS stage
  k=0          K[0:32]
  k=1          K[32:64]

Current loop entry already carries:

  A0[0,0]
  B0[0:4,0]

Current loop then mostly does:

  compute A0[:,0] * B0[:,0]
  prefetch A0[:,1] / B0[:,1] from LDS late in the first K32 half
  issue all next-tile bfld in the first half
  compute A0[:,1] * B0[:,1]
  barrier
  tail-prefetch A1[0,0] and B1[0:4,0]


Proposed high-level schedule
----------------------------

The idea is to make each K32 half a scheduling block.  Inside one block, dsrd,
bfld, and mfma are independent enough to interleave, as long as the data hazards
below are respected.

Sketch, normalized to stage names:

  Block A, current stage S, write stage W:

    s_waitcnt vmcnt(N)               # ideally N=4, only if safe by VMEM order
    dsrd S.A[0,1]
    dsrd S.B[0:8,1]
    dsrd S.A[1:8,1]
    bfld W.A[:,1]
    bfld W.B[:,1]
    mfma S.A[:,0] * S.B[:,0]

    s_waitcnt lgkmcnt(0)
    s_barrier                        # all waves finished reading S.K1 from LDS

  Block B:

    bfld S.A[:,0]                    # S can now be reused for tile t+2
    bfld S.B[:,0]
    mfma S.A[:,1] * S.B[:,1]         # uses VGPR-resident K1, not LDS

    s_waitcnt vmcnt(N)
    s_barrier                        # next current stage W is readable
    dsrd W.B[0:4,0]
    dsrd W.A[0,0]
    yield current=W, write=S

Then the next loop flips S/W.

This is the main conceptual change: before overwriting S with future data, we
must have copied all S.K1 fragments needed by all waves into VGPRs.


Safety rules
------------

1. Do not overwrite an LDS stage while another wave may still read it.

   If we want to `bfld S.A[:,0]` / `bfld S.B[:,0]` while computing `S.K1`,
   every wave must first finish all `dsrd S.A[:,1]` and `dsrd S.B[:,1]`.
   That needs:

     s_waitcnt lgkmcnt(0)
     s_barrier

   A wave-local data dependency is not enough, because another wave in the CTA
   may still be behind and reading the same LDS stage.

2. `vmcnt(4)` is only safe if the last 4 outstanding VMEM ops are not for the
   fragments about to be read from LDS.

   For a first implementation, use `vmcnt(0)` at the stage handoff.  After the
   schedule is correct, we can order bfld so the final outstanding copies belong
   to future slices that are not read yet, then relax to `vmcnt(4)`.

3. Inside a block, operations can be interleaved if they target independent
   resources:

   - `mfma S.K0` can overlap with `dsrd S.K1` because MFMA consumes K0 regs.
   - `mfma S.K0` can overlap with `bfld W.K1` because W is the other LDS stage.
   - `mfma S.K1` can overlap with `bfld S.K0` only after the mid-block
     `lgkmcnt(0)+s_barrier`, because S.K1 is then in VGPR.

4. Register pressure increases.

   To free S early, a wave needs to hold a full K1 slice:

     A K1 fragments: A[0:8,1]  -> 8 vector fragments
     B K1 fragments: B[0:8,1]  -> 8 vector fragments

   On gfx950 BF16 K32, `WMMA_A_FRAG_VALUES = 8` and `WMMA_B_FRAG_VALUES = 8`.
   The current 380 VGPR count gives room to try this, but the kernel must keep
   the existing no-spill assertion.


LDS layout implications
-----------------------

A is already mostly in the shape we want when `A_LDS_K32_BLOCKING=True`.

Current A AK32 LDS view:

  as_k32_: (stage, k_group, M, K32)

So one async copy chunk can logically fill:

  bfld A1[0:4,0]
  bfld A1[4:8,0]
  bfld A1[8:12,0]
  bfld A1[12:16,0]
  bfld A1[0:4,1]
  bfld A1[4:8,1]
  bfld A1[8:12,1]
  bfld A1[12:16,1]

That part is already aligned with the proposed schedule.

B is not yet K32-blocked.  Current B LDS view is:

  bs_: (stage, N, K64)

So one async copy chunk currently fills a 32x64 logical region:

  bfld B1[0:2,0:2]
  bfld B1[2:4,0:2]
  ...

If we want true `bfld B1[:,1]` independent from `bfld B1[:,0]`, we need a B
K32-blocked LDS view too:

  bs_k32_: (stage, k_group, N, K32)

Then B chunks can become symmetric with A:

  bfld B1[0:4,0]
  bfld B1[4:8,0]
  bfld B1[8:12,0]
  bfld B1[12:16,0]
  bfld B1[0:4,1]
  bfld B1[4:8,1]
  bfld B1[8:12,1]
  bfld B1[12:16,1]

This would make the proposed `bfld A1[:,1] * B1[:,1]` schedule much easier to
express and reason about.


Implementation route
--------------------

Step 1: Register-prefetch K1, keep current B layout.

  - Add a variant of `ldmatrix_compute_tile_streaming` that can load all
    `A[:,1]` and `B[:,1]` fragments into Python lists before the K0 MFMAs finish.
  - Add a mid-loop `s_waitcnt lgkmcnt(0); s_barrier`.
  - After that barrier, allow bfld into the just-freed current stage while K1
    MFMAs consume VGPR-resident fragments.
  - Keep B bfld as `[row_pair,0:2]` chunks for this first pass.
  - Benchmark with ATT; check no spill.

Step 2: Add `B_LDS_K32_BLOCKING`.

  - Add `bs_k32_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, 2, BLOCK_N, 32))`.
  - Change `ldg_sts_b_async_one` to map `i` as `(k_group, row_group)` like A.
  - Change `load_b_frag_from` to read through `bs_k32_` when enabled.
  - Recheck correctness, bank conflicts, and spills.

Step 3: Tune waitcnt.

  - Start with conservative `vmcnt(0)` at stage handoff.
  - Once correctness is stable, order VMEM issues so the last outstanding copies
    are future-stage data not read immediately.
  - Try `vmcnt(4)` and validate in ATT by checking the exact cycle window and
    `s_waitcnt vmcnt` stalls.


Expected win / risk
-------------------

Expected win:

  - Less exposed VMEM wait at loop boundary.
  - More useful MFMA distance between bfld issue and LDS read.
  - Ability to launch future-stage global->LDS copies while K1 compute is still
    running, instead of clustering all bfld in the first K32 half.

Main risks:

  - Extra mid-loop barrier may cost cycles, although 4-wave CTA barrier should be
    relatively cheap compared with long vmcnt stalls.
  - Extra VGPR may push register allocation over a cliff if address temporaries
    also grow.
  - B K32 layout may change LDS bank behavior; ATT must check `ds_read` and
    `lgkmcnt` stalls, not just total cycles.

Success criterion:

  - Correctness unchanged.
  - `private_segment_fixed_size=0`, `vgpr_spill_count=0`, `sgpr_spill_count=0`.
  - ATT hot-loop cycles below the current AK32 ~2589 cycles/tile.
  - Ideally beat the older tail-prefetch baseline around 2493 cycles/tile.
