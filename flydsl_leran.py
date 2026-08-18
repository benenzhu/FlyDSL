def exp(xx):
    return xx


def __syncthreads():
    return xx
def log(xx):
    return xx
def zeros(xx) -> List[List[float]]:
    return xx
def inf(xx) -> List[List[float]]:
    return xx
def mfma(xx) -> :
    return xx

float4 = 0
float2 = 0
warp_idx = 0
lane_idx = 0
thread = 0
ping = 0
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

    shared_Q = List[128][64+4]
    shared_KV = List[16 * 9][256]
    q_frag = List[18][8]
    for work_idx in range(start_idx, end_idx):
        info = work_info_set[work_idx]
        output_slot = info.output_slot
        query_id = info.qo_start
        kv_start = info.kv_start
        kv_end = info.kv_end
        
        head_begin = warp_idx * 16
        head_end = head_begin + 16
        
        for d64 in range(0, 576, 64):
            with thread():
                # 忽略一下 shared_Q的pad先.
                # float4* 表示一次float4*加载, 写法对齐cpp.
                shared_Q[head_begin + lane_idx//4][(lane_idx%4)*16] = float4* query[query_id][head_begin + lane_idx//4][d64 + (lane_idx%4)*16]
                __syncthreads()
                q_frag[d64//32] =  float2* shared_Q[head_begin + lane_idx % 16][lane_idx//16 * 8]
                q_frag[d64//32+1] =float2* shared_Q[head_begin + lane_idx % 16][lane_idx//16 * 8 + 32]

        row_max = -inf(16)
        row_sum = zeros(16)
        output_acc = zeros(16, 512)
        for tile_start in range(kv_start, kv_end, BLOCK_T):
            with thread(): # load kv.
                row_base = (warp_idx % 4) * 4  + (lane_idx / 32) * 16 + (lane_idx % 32) // 8
                col_base = (warp_idx / 4) * 32 + (lane_idx % 8) * 4
                
                for row_pass in range(2):
                    physical_row = kv_page_indices[tile_start + row_pass * 32 + row_base]
                    for block_id in range(9): 
                        slot_id = row_pass * 8 + warp_idx
                        shared_KV[block_id*4224 + slot_id * 264 + lane_idx * 4]=kv_cache[physical_row][block_id * 64 + col_base]
            
            with thread(): # calculate S=Q@K^T?
                score_acc = zeros(4, 4)
                for k_step in range(18):
                    # col_offset = k_step * 32
                    q = q_frag[k_step]
                    for n_sub in range(4):
                        # row_offset = n_sub * 16
                        # k_row = lane_idx % 16
                        # k_col = (lane_idx // 16) * 8
                        # 逻辑上. shared_KV[row_offset + k_row][col_offset + k_col]
                        lane_offset = (
                            (lane_idx % 4) * 32
                            + ((lane_idx // 4) % 4) * 256
                            + (lane_idx // 16) * 8
                        )
                        n_offset = (
                            (n_sub % 2) * 128
                            + (n_sub // 2) * 8 * 256
                        )
                        k_offset = (
                            (k_step % 2) * 4 * 256
                            + (k_step // 2) * 8 * 256 * 2
                        )
                        k = float2 * shared_KV[lane_offset + n_offset + k_offset] # TODO: offset not right.
                        score_acc[n_sub] += mfma(q, k)

            with thread(): # row_max? P=exp(S)
                ...
            with thread(): # calculate P@V
                ...
            
                

            physical_rows = kv_page_indices[tile_start: tile_start+64] # logicial positions
            kv_tile = kv_cache[physical_rows]
            k_tile = kv_tile
            v_tile = kv_tile[:, :512]
            
            scores = q_frag @ k_tile.T
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

