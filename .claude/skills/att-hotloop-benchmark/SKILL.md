---
name: att-hotloop-benchmark
description: Capture and analyze ROCm rocprofv3 Advanced Thread Trace (ATT) for FlyDSL or AMD GPU kernels, especially when the user asks to dump hot-loop cycles, compare kernel schedules using barrier-to-barrier cycle counts, inspect dispatch ATT traces, report MFMA utilization from hot-loop cycles, or use ATT rather than wall-clock TFLOPS as the benchmark.
---

# ATT Hotloop Benchmark

## Goal

Use ATT as the primary benchmark for scheduling work. Prefer cycle windows from the ATT UI output over wall-clock microbenchmarks when the user is optimizing hot loops, MFMA/VMEM/DSRD interleaving, waitcnt stalls, or barrier-to-barrier loop bodies.

## Capture Workflow

0. Pick an idle GPU first with `gpu_select.py` (repo root), then bind it for every
   subsequent command. ATT capture is invalid if another process shares the CU.

```bash
GPU=$(python3 -c "import gpu_select; print(gpu_select.select_empty_gpu().index)")
echo "selected GPU $GPU"
export HIP_VISIBLE_DEVICES=$GPU
```

Prefer a long-K shape so the steady-state hot loop dominates over prologue/tail
(e.g. fp8_gemm_4wave reaches ~3000 TFLOPS at 8192x8192x16384 vs ~2380 at
5120x5120x8320). Use the same shape for capture and for any before/after compare.

1. Build or run the target once with debug info and dumps enabled so source lines and ISA are available:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR=resource_inspect/<dump-name> \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
<python command that compiles/runs the kernel>
```

2. Capture ATT with `rocprofv3`. Use a narrow `kernel_include_regex`, keep only 1-2 iterations, and enable UI-compatible CSV output:

```bash
cat >/tmp/att.yaml <<'YAML'
jobs:
  - kernel_include_regex: <kernel-regex>
    kernel_iteration_range: "[1, [1-2]]"
    output_file: att
    output_directory: profiler_traces_grid/<trace-name>
    output_format: [csv]
    truncate_kernels: true
    sys_trace: true
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0xf"
    att_simd_select: "0xf"
    att_buffer_size: "0x6000000"
YAML

FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
rocprofv3 -i /tmp/att.yaml -- <python command that launches the kernel>
```

3. Locate the UI output directory:

```bash
find profiler_traces_grid/<trace-name> -maxdepth 2 -type d -name 'ui_output_agent_*_dispatch_*'
```

### Which dispatch to open

There are usually several `dispatch_<n>` dirs (one per kernel launch in the
`kernel_iteration_range`, e.g. `--iters 2` -> two dispatches). **They are
equivalent repeats of the same kernel** — pick any *valid* one and tell the user
exactly which you used. A dispatch is valid iff `--list-barriers` returns a
non-zero barrier count (and the same count across dispatches); skip any whose
`code.json` is empty/`null` (it yields 0 barriers / unparseable output). Default
to the lowest-numbered valid dispatch. When handing a zip to the user, name the
single dispatch+wave to open so they don't have to guess.

## Hot-Loop Cycle Policy

Use one wave JSON file from the selected dispatch directory, usually `se*_sm*_sl*_wv0.json` or the wave with the most instructions. Treat `s_barrier` timestamps as loop boundaries when the kernel synchronizes once per outer loop.

For stable comparisons:

- Prefer a middle barrier window, for example barrier 30 to barrier 50, instead of loop-top setup or tail loops.
- Report `total_cycles`, `cycles_per_loop`, start/end barrier ordinals, and start/end cycles.
- Also report instruction mix inside the same window: `v_mfma`, `ds_read`, `buffer_load`, `s_waitcnt`, `s_barrier`, and other instructions.
- For MFMA throughput sanity, compute `cycles_per_mfma16 = total_cycles / (mfma_count * 16)` when the window contains MFMA instructions. Lower is better; `1.0` means the selected window averages one 16-cycle MFMA issue slot per MFMA.

### MFMA cycle cost depends on the instruction

The script's `cycles_per_mfma16` assumes a 16-cycle MFMA. Many wide MFMAs cost
more. Pick the issue latency of the actual atom before computing utilization:

- `MFMA_Scale 16x16x128` (fp8, e.g. fp8_gemm_4wave / Mfma16x16x128) issues every
  **32 cycles**, not 16.

For those, compute by hand from `total_cycles` and `mfma_count`:

```text
mfma_cycle   = 32                                  # the atom's issue latency
ideal_cycles = mfma_count * mfma_cycle
utilization  = ideal_cycles / total_cycles         # closer to 1.0 is better
cycles_per_mfma = total_cycles / mfma_count         # compare against mfma_cycle
```

Example (fp8_gemm_4wave, 8192x8192x16384, barrier window 30:50): total 24916,
mfma_count 645 -> ideal 645*32=20640 -> utilization 20640/24916 = 0.83 (~83%).

If the user gives explicit UI cycle coordinates, use a cycle range directly instead of barrier ordinals.

### Isolating exactly one loop iteration

A kernel that issues N `s_barrier` per outer-loop iteration spans N barrier
intervals per iteration (fp8_gemm_4wave issues 2: `wait -> block1+block2 ->
wait -> block3+block4`). To get the cycle->cycle bounds of a single iteration,
list barriers and take a window `N` apart starting on an even iteration boundary:

```bash
python scripts/att_hotloop_cycles.py <ui_dir> --list-barriers | sed -n '40,46p'
# pick barrier k that starts an iteration, window k:k+N (N=barriers per iter)
python scripts/att_hotloop_cycles.py <ui_dir> --barrier-window 38:40
```

Report it as `cycle_start -> cycle_end` with `total_cycles` = one iteration, and
name the exact wave file the numbers came from, e.g.
`se0_sm0_sl0_wv0.json`. The user jumps to `cycle_start` in the UI to land on the
iteration's first `s_barrier`.

## Script

Use `scripts/att_hotloop_cycles.py` for deterministic parsing of ATT UI output:

```bash
python ~/.codex/skills/att-hotloop-benchmark/scripts/att_hotloop_cycles.py \
  profiler_traces_grid/<trace-name>/ui_output_agent_<agent>_dispatch_<n> \
  --list-barriers
```

```bash
python ~/.codex/skills/att-hotloop-benchmark/scripts/att_hotloop_cycles.py \
  profiler_traces_grid/<trace-name>/ui_output_agent_<agent>_dispatch_<n> \
  --barrier-window 30:50 \
  --per-interval
```

```bash
python ~/.codex/skills/att-hotloop-benchmark/scripts/att_hotloop_cycles.py \
  profiler_traces_grid/<trace-name>/ui_output_agent_<agent>_dispatch_<n> \
  --cycle-range 123552:125600
```

Important output fields:

- `cycle_start`, `cycle_end`: ATT cycle coordinates to jump to in the UI.
- `total_cycles`: `cycle_end - cycle_start`.
- `cycles_per_loop`: `total_cycles / number_of_barrier_intervals`.
- `mfma_count`: one-wave MFMA instruction count in the selected window.
- `cycles_per_mfma16`: `total_cycles / (mfma_count * 16)`.

## Reporting

When answering the user, lead with the cycle result and the exact dispatch/wave/window:

```text
dispatch: ui_output_agent_123_dispatch_7
wave: se0_sm0_sl0_wv0.json
barrier window: 30 -> 50
cycle range: 123552 -> 176084
hot-loop cycles: 52532 total, 2626.6 per loop
mfma_count: 1280
cycles_per_mfma16: 2.565
```

Then summarize the instruction mix and any obvious stall source. Avoid using wall-clock TFLOPS as the main comparison unless the user explicitly asks for it.

## Packaging the trace for the user

When asked to hand off the trace, zip the whole trace-name directory, keeping its
structure intact. The UI viewer needs the dispatch dir's `filenames.json` to sit
next to the sibling raw `.att` files and code-object `.out` files at the
trace-name level; do NOT zip only the `ui_output_*_dispatch_*` dir or only its
contents -- both break the viewer (`Invalid or inaccessible path: .../filenames.json`).

```bash
zip -rq <name>_att.zip profiler_traces_grid/<trace-name>
# sanity: the dispatch dir + its filenames.json are nested under trace-name
unzip -l <name>_att.zip | grep filenames.json
```

Always report the handoff in exactly this three-line format (nothing else needed):

```text
1. /root/FlyDSL/<name>_att.zip
2. ui_output_agent_<agent>_dispatch_<n>/se0_sm0_sl0_wv0.json
3. cycle <start> -> <end> (共 <total>)
```

Where line 3 is one complete for-loop iteration (the cycle range to jump to in
the UI, and its total cycle count). Keep this exact ordering and labels.
