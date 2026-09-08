#!/bin/bash
# Deploy the M3 MoE ops from the vLLM worktree into the bench container and run the
# unit tests there (the tests run against the vLLM installed in the image, so the
# two ops packages and the test files are copied in first).
#
#   m3_a16w4_moe/vllm_tests.sh            # decode (24 cases)
#   m3_a16w4_moe/vllm_tests.sh prefill    # prefill (5 cases)
#   m3_a16w4_moe/vllm_tests.sh a16w8      # MXFP8 decode (13 cases)
#   m3_a16w4_moe/vllm_tests.sh a8w8       # MXFP8 prefill (3 cases)
#   m3_a16w4_moe/vllm_tests.sh all        # both
#   m3_a16w4_moe/vllm_tests.sh bench      # decode chain timing (vllm_dec_bench.py)
#
# Env: M3_VLLM_WT (worktree, default /dev/shm/m3-compare/wt-m3-05), M3_CTR (container,
# default m3cmp_new1), M3_GPU (default 1). Prints one summary line per suite.
set -u
WHAT=${1:-decode}
WT=${M3_VLLM_WT:-/dev/shm/m3-compare/wt-m3-05}
CTR=${M3_CTR:-m3cmp_new1}
GPU=${M3_GPU:-1}
OPS=vllm/models/minimax_m3/amd/ops
DST=/usr/local/lib/python3.12/dist-packages/$OPS

for pkg in moe_a16w4_decode moe_a4w4_prefill moe_a8w8_decode moe_a8w8_prefill; do
  docker cp "$WT/$OPS/$pkg" "$CTR:$DST/" || exit 1
done
docker cp "$WT/tests/kernels/moe/test_m3_flydsl_decode_moe.py" "$CTR:/work/tests/" || exit 1
docker cp "$WT/tests/kernels/moe/test_m3_flydsl_prefill_moe.py" "$CTR:/work/tests/" || exit 1
docker cp "$WT/tests/kernels/moe/test_m3_flydsl_a16w8_decode_moe.py" "$CTR:/work/tests/" || exit 1
docker cp "$WT/tests/kernels/moe/test_m3_flydsl_a8w8_prefill_moe.py" "$CTR:/work/tests/" || exit 1
docker exec "$CTR" bash -c "rm -rf $DST/moe_a16w4_decode/__pycache__ $DST/moe_a4w4_prefill/__pycache__ $DST/moe_a8w8_decode/__pycache__ $DST/moe_a8w8_prefill/__pycache__"

run_suite() {  # label file
  docker exec -e HIP_VISIBLE_DEVICES=$GPU "$CTR" bash -c \
    "cd /work/tests && timeout 2400 python3 -m pytest -q -p no:cacheprovider $2 2>&1 | grep -v Warning | tail -1" \
    | sed "s/^/[$1] /"
}
case "$WHAT" in
  decode)  run_suite decode test_m3_flydsl_decode_moe.py ;;
  prefill) run_suite prefill test_m3_flydsl_prefill_moe.py ;;
  a16w8)   run_suite a16w8 test_m3_flydsl_a16w8_decode_moe.py ;;
  a8w8)    run_suite a8w8 test_m3_flydsl_a8w8_prefill_moe.py ;;
  all)     run_suite decode test_m3_flydsl_decode_moe.py; run_suite prefill test_m3_flydsl_prefill_moe.py; run_suite a16w8 test_m3_flydsl_a16w8_decode_moe.py; run_suite a8w8 test_m3_flydsl_a8w8_prefill_moe.py ;;
  bench)   docker exec -e HIP_VISIBLE_DEVICES=$GPU "$CTR" bash -c \
             "cd /work && timeout 900 python3 vllm_dec_bench.py --tokens 4 32 64 128 256 --layout standard 2>&1 | grep 'per call' | sed 's/\[vllm decode pkg\] //'" ;;
  *) echo "usage: $0 [decode|prefill|a16w8|a8w8|all|bench]"; exit 2 ;;
esac
