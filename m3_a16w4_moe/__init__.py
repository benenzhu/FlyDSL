# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 decode (M=4) a16w4 MoE on MI355X: bf16 activation x mxfp4 weight, FlyDSL.

Working copy of FlyDSL's ``kernels/moe/moe_2stage_a16wmix`` (#948) trimmed to run on the
flydsl 0.2.4 that ships in the production vLLM image, plus the MiniMax-M3 swigluoai
activation and an M=4 bench against the CK-tile a16w4 path vLLM uses today.
"""
