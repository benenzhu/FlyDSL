---
name: hotloop-table
description: Generate a compact hot-loop instruction table from a kernel's AMDGCN ISA, reduced to the three GEMM-scheduling ops (v_mfma=M, ds_read=r, buffer_load=L) plus s_barrier=B, one per line with a single-char stream summary. Use when inspecting or comparing how MFMA / LDS-read / global-load prefetch are interleaved in a GEMM/attention main loop, or when the user asks for a "hotloop table" / "指令排布表".
---

# Hot-loop Table

Standardize how we look at a kernel's steady-state main loop: collapse the ISA to
only the ops that matter for GEMM scheduling and show their interleaving.

## Tags (only these three ops + barrier)

| Tag | Instruction | Role |
|-----|-------------|------|
| `M` | `v_mfma_*` | matrix multiply-accumulate (compute) |
| `r` | `ds_read_*` | LDS -> register (S2R operand load) |
| `L` | `buffer_load_*` | global -> LDS/VGPR prefetch (G2S) |
| `B` | `s_barrier` | half / loop boundary |

Everything else (address calc, `s_waitcnt`, salu, valu) is dropped so the M/r/L
interleaving and the barrier-delimited halves are obvious.

## Input: a "pure" ISA

Best run on a cleaned ISA. Produce one with benenzhu/learn-hip `gen_pure.py`
(strips comments/debug, inlines `.loc`, labels `s_cbranch` targets, expands
`v[a:b]` ranges, zero-pads vgpr numbers):

```bash
curl -sL https://raw.githubusercontent.com/benenzhu/learn-hip/main/gen_pure.py -o /tmp/gen_pure.py
python /tmp/gen_pure.py <kernel>_final_isa.s   # writes <...>final_isa.spure.s
```

Get the raw ISA from a FlyDSL compile with `FLYDSL_DUMP_IR=1` (file
`<dump>/<kernel>_0/21_final_isa.s`).

## Generate the table

```bash
python .claude/skills/hotloop-table/scripts/hotloop_table.py <pure.s> \
  --out <kernel>_hotloop.txt --compact
```

- Auto-detects the main loop as the most-MFMA-heavy backward `s_cbranch` body.
  Override with `--loop-label .LBB0_1` if needed.
- `--compact` appends a one-char-per-op `stream:` line, e.g.
  `MLrMMMMrMMMMLrMMMMrMMMM...B...B` — read it to judge interleaving at a glance:
  a leading run like `LLLLrrrr` before any `M` means a front-loaded load/read
  wall (bad); `MLrMMMM` repeating means loads/reads are tucked into MFMA
  co-issue shadows (good).

## Header reports per-loop counts

`Per loop: M=128 r=32 L=16 B=2` — sanity-check against the kernel: e.g. a
256x256 bf16 tile, 4 waves, BK64 (2 K-steps) gives 2x64 MFMA, 2x16 ds_read,
2x8 buffer_load, 2 barriers per loop.

## Reporting

Lead with the `stream:` line and the per-loop counts; point out any leading wall
or uneven half. Commit the generated `<kernel>_hotloop.txt` next to the ISA
snapshot under `resource_inspect/snapshots/` so loops can be diffed across
optimization steps.
