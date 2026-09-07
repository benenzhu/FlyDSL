# SPDX-License-Identifier: Apache-2.0
"""Opt-in host wrapper for the independent persistent decode kernels."""

import functools
import torch

from .gemm1_persist import compile_gemm1_persist
from .host import _run_compiled


get_gemm1_persist = functools.cache(compile_gemm1_persist)


def a16w4_gemm1_persist(
    *, x_bf16, w1_u8, w1_scale_u8, inter_sorted_bf16, n_tokens,
    NE, D_HIDDEN, D_INTER, topk, tile_m, tile_n, tile_k,
    sorted_expert_ids, num_valid_ids, sorted_token_ids, k_wave=4,
    b_nt=2, xcd_swizzle=0, waves_per_eu=None, act="swigluoai",
    alpha=1.702, swiglu_limit=7.0, w_layout="standard", a_direct=True,
    prefetch=3, scale_share=True, a_rows4=False, n_ctas=512, stream=None,
):
    assert k_wave == 4 and a_direct and scale_share and not a_rows4
    assert xcd_swizzle == 0 and waves_per_eu is None and act == "swigluoai"
    assert n_ctas > 0
    launch = get_gemm1_persist(
        D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, NE=NE, TOPK=topk,
        BM=tile_m, TILE_N=tile_n, TILE_K=tile_k, prefetch=prefetch,
        b_cache_mod=b_nt, w_layout=w_layout,
    )
    _run_compiled(
        launch, x_bf16.data_ptr(), w1_u8.data_ptr(), w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_token_ids.data_ptr(),
        int(n_tokens), int(n_ctas), float(alpha), 1.0 / float(alpha), 1.0, 1.0,
        float(swiglu_limit), inter_sorted_bf16.data_ptr(), 0, 0,
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16
