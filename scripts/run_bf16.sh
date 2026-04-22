set -ex
#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Sweep bf16 async preshuffle GEMM tile sizes for an 8192x8192x8192 shape.
# Mirrors the table format used by run_benchmark.sh.
#
# Usage:
#   bash scripts/run_bf16.sh
#
# Tunable env vars:
#   M, N, K             -- problem size (default 8192,8192,8192)
#   WAVES_PER_EU        -- amdgpu waves-per-EU hint (default 2)
#   TILE_M_LIST         -- space-separated list of tile_m (default "128 256")
#   TILE_N_LIST         -- space-separated list of tile_n (default "256")
#   TILE_K_LIST         -- space-separated list of tile_k (default "128 256")
#   BENCH_LOG_DIR       -- where per-shape logs go (default /tmp/flydsl_bench)
set -eu
if (set -o pipefail) 2>/dev/null; then set -o pipefail; fi
cd "$(dirname "$0")/.."

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Locate the build directory (default: build-fly; fallback: build/).
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi
PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages"
export PYTHONPATH="${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
MLIR_LIBS_DIR="${PYTHON_PACKAGE_ROOT}/flydsl/_mlir/_mlir_libs"
if [ -d "${MLIR_LIBS_DIR}" ]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

BENCH_LOG_DIR="${BENCH_LOG_DIR:-/tmp/flydsl_bench}"
mkdir -p "${BENCH_LOG_DIR}"

# Auto-select GPU with the most free VRAM (skip if HIP_VISIBLE_DEVICES is set).
if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && command -v python3 >/dev/null 2>&1; then
  _best_gpu=$(python3 -c "
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    print(best)
" 2>/dev/null || true)
  if [ -n "${_best_gpu}" ]; then
    export HIP_VISIBLE_DEVICES="${_best_gpu}"
    echo "[run_bf16] Auto-selected GPU ${_best_gpu} (most free VRAM)"
  fi
fi

GPU_ARCH=$(python3 -c "from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())" 2>/dev/null || echo "unknown")
echo "[run_bf16] GPU arch: ${GPU_ARCH}"

# Sweep config
M="${M:-8192}"
N="${N:-8192}"
K="${K:-8192}"
WAVES_PER_EU="${WAVES_PER_EU:-2}"
TILE_M_LIST="${TILE_M_LIST:-128 256}"
TILE_N_LIST="${TILE_N_LIST:-256}"
TILE_K_LIST="${TILE_K_LIST:-128 256}"

SUCCESS_COUNT=0
FAIL_COUNT=0

# Re-use the parser style from run_benchmark.sh
_py_parse_and_emit() {
  python3 - "$@" <<'PY'
import re, sys
op, shape, dtype, path = sys.argv[1:5]
tbps = tflops = None
try:
    txt = open(path, "r", errors="ignore").read()
except Exception:
    txt = ""
m = None
for m in re.finditer(r"Throughput:.*?([0-9.]+)\s*TFLOPS.*?BW:\s*([0-9.]+)\s*TB/s", txt):
    pass
if m:
    tflops = float(m.group(1))
    tbps = float(m.group(2))
def fmt(x): return "-" if x is None else f"{x:.3f}"
print(f"{op}\t{shape}\t{dtype}\t{fmt(tbps)}\t{fmt(tflops)}")
PY
}

_emit_row() {
  printf "%-14.14s %-34.34s %-10.10s %10s %10s\n" "$1" "$2" "$3" "$4" "$5"
}

_show_fail_log() {
  log_path="$1"; op_name="${2:-unknown}"
  if [ -f "${log_path}" ]; then
    echo "" >&2
    echo "-------------------- ${op_name} log (tail) --------------------" >&2
    tail -n 200 "${log_path}" >&2 || true
    echo "-------------------- end of ${op_name} log --------------------" >&2
  else
    echo "[warn] ${op_name} log missing: ${log_path}" >&2
  fi
}

echo "========================================================================"
echo "BF16 async preshuffle GEMM tile sweep (logs under ${BENCH_LOG_DIR})"
echo "  M=${M} N=${N} K=${K}  waves_per_eu=${WAVES_PER_EU}"
echo "  tile_m in {${TILE_M_LIST}}  tile_n in {${TILE_N_LIST}}  tile_k in {${TILE_K_LIST}}"
echo "========================================================================"
printf "\n%-14.14s %-34.34s %-10.10s %10s %10s\n" "op" "shape" "dtype" "TB/s" "TFLOPS"
printf "%-14.14s %-34.34s %-10.10s %10s %10s\n" "--------------" "----------------------------------" "----------" "----------" "----------"

dtype="bf16"
for tile_m in ${TILE_M_LIST}; do
  for tile_n in ${TILE_N_LIST}; do
    for tile_k in ${TILE_K_LIST}; do
      log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}_async_copy_${WAVES_PER_EU}.log"
      if python3 tests/kernels/test_preshuffle_gemm.py \
          --in_dtype "${dtype}" \
          --num_warmup 10 \
          --num_iters 100 \
          -M "${M}" -N "${N}" -K "${K}" \
          --tile_m "${tile_m}" --tile_n "${tile_n}" --tile_k "${tile_k}" \
          --use_async_copy \
          --waves_per_eu "${WAVES_PER_EU}" >"${log}" 2>&1; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "gemm_async bf16 failed. Log: ${log}" >&2
        _show_fail_log "${log}" "gemm_async_bf16"
      fi
      shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${WAVES_PER_EU}tg"
      row="$(_py_parse_and_emit gemm_async "${shape_tag}" "${dtype}" "${log}")"
      # row is tab-separated; default IFS includes tabs.
      # shellcheck disable=SC2086
      set -- $row
      _emit_row "$1" "$2" "$3" "$4" "$5"
    done
  done
done

TOTAL=$((SUCCESS_COUNT + FAIL_COUNT))
echo ""
echo "========================================================================"
echo "Summary: ${SUCCESS_COUNT}/${TOTAL} passed (${FAIL_COUNT} failed)"
echo "Logs: ${BENCH_LOG_DIR}"
echo "========================================================================"

if [ "${FAIL_COUNT}" -eq 0 ]; then
  exit 0
else
  exit 1
fi
