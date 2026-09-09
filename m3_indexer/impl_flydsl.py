"""IDX_IMPL shim for work/idx_bench/bench_upstream.py: the FlyDSL prefill scorer
(top-k and decode fall back to the vllm Triton kernels)."""
import sys

sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops  # noqa: E402

import_ops("index_bf16")
from index_bf16.host import index_score_prefill as minimax_m3_index_score  # noqa: E402,F401
from index_bf16.host import index_topk_prefill as minimax_m3_index_topk  # noqa: E402,F401
from index_bf16.host import index_decode as minimax_m3_index_decode  # noqa: E402,F401
