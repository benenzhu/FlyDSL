"""Which XCD runs which workgroup? out[pid] = XCC_ID (HW_REG 20, bits 3:0) | CU/SE id bits of HW_ID.
grid order = dispatch order; tells whether consecutive pids land on the same XCD."""
import sys, torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
sys.path.insert(0, "/flydsl")
from m3_indexer.vllm_ops import import_ops
import_ops("index_bf16")
from index_bf16.utils import _global_i32_ptr, _run_compiled

N = 2048

def _getreg(imm):
    return fx.Int32(_llvm.call_intrinsic(T.i32, "llvm.amdgcn.s.getreg", [fx.Int32(imm).ir_value()], [], []))

@flyc.kernel(name="probe_xcc", known_block_size=[256, 1, 1])
def kernel(arg_out: fx.Int64):
    pid = fx.block_idx.x
    tx = fx.thread_idx.x
    xcc = _getreg(20 | (3 << 11))          # HW_REG_XCC_ID[3:0]
    hwid = _getreg(4 | (31 << 11))         # HW_REG_HW_ID full 32 bits
    if tx == 0:
        _global_i32_ptr(arg_out)[pid * 2] = xcc
        _global_i32_ptr(arg_out)[pid * 2 + 1] = hwid

@flyc.jit
def launch(arg_out: fx.Int64, stream: fx.Stream):
    kernel(arg_out).launch(grid=(N, 1, 1), block=(256, 1, 1), stream=stream)

out = torch.zeros(N * 2, dtype=torch.int32, device="cuda")
_run_compiled(launch, out.data_ptr(), torch.cuda.current_stream()); torch.cuda.synchronize()
o = out.view(N, 2).cpu()
xcc = (o[:, 0] & 0xF).tolist(); hw = o[:, 1].tolist()
print("xcc of pid 0..31:", xcc[:32])
print("xcc of pid 256..287:", xcc[256:288])
import collections
print("count per xcc:", sorted(collections.Counter(xcc).items()))
# pid % 8 == xcc ?
print("pid%8==xcc for all:", all((p % 8) == x for p, x in enumerate(xcc)), " pid//256%8==xcc:", all(((p // 256) % 8) == x for p, x in enumerate(xcc)))
# HW_ID: wave_id[3:0], simd_id[5:4], pipe[7:6], cu_id[11:8], sh_id[12], se_id[15:13]
cu = [(h >> 8) & 0xF for h in hw]; se = [(h >> 13) & 0x7 for h in hw]
print("pid 0..15 (xcc, se, cu):", list(zip(xcc[:16], se[:16], cu[:16])))
