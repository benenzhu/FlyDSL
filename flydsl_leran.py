def exp(xx):
    return xx


def log(xx):
    return xx
def zeros(xx) -> List[List[float]]:
    return xx
def inf(xx) -> List[List[float]]:
    return xx

warp = 0
thread = 0
BLOCK_N = 64
BLOCK_T = 64
from typing import List
class WorkInfo: 
    output_slot:int
    qo_start:int
    kv_start:int
    kv_end:int
softmax_scale = xx
def mla(query,
        start_idx:int,
        kv_cache: List[List[float]],
        kv_page_indices: List[int],
        end_idx:int, 
        work_info_set:List[WorkInfo],
        ):

    shared_Q = List[128][128]
    shared_KV = List[2][128][128] # TODO.
    for work_idx in range(start_idx, end_idx):
        info = work_info_set[work_idx]
        output_slot = info.output_slot
        query_id = info.qo_start
        kv_start = info.kv_start
        kv_end = info.kv_end
        
        head_begin = warp * 16
        head_end = head_begin + 16
        
        shared_Q[head_begin:head_end, :] = query[query_id, head_begin:head_end, :] # [16, 576]
        q_frag = shared_Q[head_begin:head_end, :]
        q = query[query_id]
        row_max = -inf(16)
        row_sum = zeros(16)
        output_acc = zeros(16, 512)
        for tile_start in range(kv_start, kv_end, BLOCK_T):
            physical_rows = kv_page_indices[tile_start: tile_start+64] # logicial positions
            kv_tile = kv_cache[physical_rows]
            k_tile = kv_tile
            v_tile = kv_tile[:, :512]
            
            scores = q @ k_tile.T
            scores *= softmax_scale
            
            new_row_max = max(row_max, max(scores, dim=-1))
            old_rescale = exp(row_max - new_row_max)
            p = exp(scores - new_row_max[:, None])
            row_sum = row_sum * old_rescale + sum(p, dim=-1)
            output_acc = output_acc * old_rescale[:, None]
            
            output_acc += p @ v_tile
            row_max = new_row_max
        partial_output = output_acc / row_sum[:, None]
        partial_lse = row_max + log(row_sum)

