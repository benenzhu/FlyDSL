# SPDX-License-Identifier: Apache-2.0
"""Import the MiniMax-M3 indexer kernels from the vLLM tree (the single copy).

The kernels live in vLLM (branch ``m3/08-flydsl-indexer``):
``vllm/models/minimax_m3/amd/ops/index_bf16`` (bf16 index cache, prefill + decode).
``M3_VLLM_OPS`` points at that ``ops``
directory; the default is the worktree on /dev/shm, which the bench containers
see as-is (they run with ``--ipc=host``).  Whatever is checked out there is what
the benches run, so both repos simply track their latest commit.

Usage::

    from m3_indexer.vllm_ops import import_ops
    import_ops("index_bf16")
    from index_bf16.host import index_score_prefill
"""

import importlib
import os
import pkgutil
import sys

OPS_DIR = os.environ.get(
    "M3_VLLM_OPS", "/dev/shm/m3-compare/wt-m3-08/vllm/models/minimax_m3/amd/ops"
)
_VLLM_PREFIX = "vllm.models.minimax_m3.amd.ops."


def import_ops(name):
    """Import ops package ``name`` (and all its submodules) from ``OPS_DIR``.

    The package-level entry points 
    import their submodules by the vLLM dotted name; those names are aliased to
    the modules loaded here so nothing falls back to the vLLM installed in the
    image."""
    if not os.path.isdir(os.path.join(OPS_DIR, name)):
        raise ImportError(f"{name} not found under M3_VLLM_OPS={OPS_DIR}")
    if OPS_DIR not in sys.path:
        sys.path.insert(0, OPS_DIR)
    pkg = importlib.import_module(name)
    for info in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{name}.{info.name}")
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        sys.modules.setdefault(_VLLM_PREFIX + key, sys.modules[key])
    return pkg
