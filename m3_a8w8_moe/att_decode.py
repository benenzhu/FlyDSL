# SPDX-License-Identifier: Apache-2.0
"""Eager launches of the a16w8 decode chain (moe_a8w8_decode.a16w8_decode_moe) for
rocprofv3 ATT capture, ``--iters`` calls with different inputs (no HIP graph).

    FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i att.yaml -- \\
        python3 -m m3_a8w8_moe.att_decode --tokens 32 --iters 4
"""
import argparse

import torch

from m3_a8w8_moe.ref_mxfp8 import HIDDEN, make_weights, routing
from m3_a8w8_moe.test_chain import ours


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--iters", type=int, default=4)
    a = p.parse_args()
    dev = torch.device("cuda")
    _, shuffled = make_weights(dev)
    for i in range(a.iters):
        torch.manual_seed(100 + i)
        x = torch.randn((a.tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
        ids, w = routing(a.tokens, dev)
        ours(x, shuffled, w, ids)
        torch.cuda.synchronize()
    print("done", flush=True)


if __name__ == "__main__":
    main()
