def ceil_div(a, b):
    return
def atomic_add():
    pass
def launch():
    pass
def round_up():
    return
from typing import List

bid = 0
tid = 0
M = 32768
topk = 8
NE = 384 + 1
BM = 128



topk_ids: [M, topk]
topk_weights: [M, topk]
max_num_tokens_padded = M*topk + num_experts * (BM - 1)
max_num_m_blocks = ceil_div(max_num_tokens_padded, BM)
max_sorted = max_num_m_blocks * BM

sorted__token_ids: [max_sorted] int
sorted_experts_ids: [max_sorted // BM] int
sorted_weights: [max_num_tokens_padded] float


...


"""
launch:
moe_3stage_sort.cuh:168
kernel:
moe_3stage_sort.cuh:10

# gen_instances.py:190):N_SORT_CTAS = kSplitSortCtas = 16,THREADS_PER_CTA = kThreadsSort = 1024
"""



# cumsum? 分布式的每个kernel多少个.
@launch(16, 1024)
def sort_count_kernel(
    M,
    topk_ids,
    block_offsets
):
    s_counts = [0] * NE
    per_cta = ceil_div(M * topk, 16)
    for i in range(bid * per_cta + tid, bid * per_cta + per_cta, 1024):
        atomic_add(s_counts[topk_ids[i]], 1)
    if tid < NE:
        e = tid
        block_offsets[e * 16 + bid] = s_counts[e]

# ---------------------------------------------------------------------------
# STAGE 2/3  moe_3stage_sort.cuh:42   单 block 串行归约
#   ① 每 expert 的 16 个局部计数跨 block 求和 -> total
#   ② padding 到 BM 倍数,前缀和 -> 每 expert 起始偏移 expert_starts
#   ③ block_offsets 原地: 计数 -> "每 (expert,cta) 段的写入起始行"(给 place_pad)
#   ④ sorted_expert_ids(eids): 每个 m_block 属于哪个 expert
# ---------------------------------------------------------------------------
@launch(1, 1024)
def sort_cumsum_kernel(M, 
                       block_offsets, 
                       real_counts: [NE], # 真实行数. 给place_pad填充用.
                       cumsum_tensor: [2], # 
                       sorted_expert_ids: [max_num_m_blocks]):
    s_total_count = [0] * NE # 累加sum
    s_padded_count  = [0] * NE #  padded sum
    s_expert_starts = [0] * (NE + 1) # 存放每个expert的开始位置, padded的cumsum

    # ① 跨 16 block 求和 + ② 每 expert padding 到 BM 倍数
    for e in range(tid, NE, 1024):
        s = sum(block_offsets[e * 16 + c] for c in range(16))
        s_total_count[e]  = s
        s_padded_count[e] = round_up(s, BM)   # 向上取整到 BM
        real_counts[e]  = s                  # 真实(未 pad)行数,给 place_pad 填哨兵用 

    # ② 前缀和(单线程,NE 个)
    if tid == 0:
        acc = 0
        for e in range(NE):
            s_expert_starts[e] = acc
            acc += s_padded_count[e]
        s_expert_starts[NE] = acc
        cumsum_tensor[0] = acc    # = max_sorted 有效总行数(含 padding)
        cumsum_tensor[1] = M      # 有效 token 数 (non-EP == M)

    # ③ block_offsets[e,c]: 计数 -> 该 (expert,cta) 段写入起始行
    for e in range(tid, NE, 1024):
        acc = s_expert_starts[e]
        for c in range(16):
            cnt = block_offsets[e * 16 + c] # 本来存的是cnt
            block_offsets[e * 16 + c] = acc # 改成padded以后+cnt?
            acc += cnt

    # ④ eids: 第 b 个 m_block(每 BM 行)属于哪个 expert
    for e in range(tid, NE, 1024):
        for b in range(s_expert_starts[e] // BM, s_expert_starts[e + 1] // BM):
            sorted_expert_ids[b] = e


# ---------------------------------------------------------------------------
# STAGE 3/3  moe_3stage_sort.cuh:97
#   ① 每个 (token,slot) 原子占位到它 expert 桶的下一空位,写 4 个输出数组
#   ② 每 expert 桶 [real_end, padded_end) 的 padding 空槽填哨兵
# ---------------------------------------------------------------------------
@launch(16, 1024)
def sort_place_pad_kernel(M, 
                          topk_ids: [M, topk], 
                          topk_weight: [M, topk],
                          block_offsets: [NE, 16], #每个expert, cta的写入起始行
                          real_counts: [NE], # 真实行数
                          cumsum_tensor: [2], # 0 = 有效行数 1 = M?
                          sorted_token_ids, 
                          reverse_sorted, 
                          sorted_weights, 
                          m_indices):
    s_local_offsets = [0] * NE
    s_row_starts    = [0] * (NE + 1)

    e = tid
    s_local_offsets[e] = block_offsets[e * 16 + bid]
    s_row_starts[e]    = block_offsets[e * 16 + 0]
    if tid == 0:
        s_row_starts[NE] = cumsum_tensor[0] # 有效总行数.

    # ① 放置: 本 cta 负责 [start,end) 段
    per_cta = ceil_div(M * topk, 16)
    start = bid * per_cta
    end   = min(start + per_cta, M * topk)
    for i in range(start + tid, end, 1024): # [M*topk]的映射到sorted_token_ids
        eid = topk_ids[i]
        sp  = atomic_add(s_local_offsets[eid], 1)   # 该 expert 桶落位行号
        token_id = i // topk
        topk_id  = i %  topk
        sorted_token_ids[sp] = (token_id & 0x00FFFFFF) | ((topk_id & 0xFF) << 24)  # 高8位=topk_id
        sorted_weights[sp]   = topk_weight[i]
        m_indices[sp]        = token_id & 0x00FFFFFF
        reverse_sorted[i]    = sp    # 原 pair i -> sorted 行 sp (scatter_reduce 反查用)

    # ② 填 padding 哨兵: pad_val=M 会让 gemm buffer_load voffset 越界 -> HW 丢弃该 load
    experts_per_cta = ceil_div(NE, 16)
    e_lo = bid * experts_per_cta
    e_hi = min(e_lo + experts_per_cta, NE)
    pad_val = M & 0x00FFFFFF
    for e in range(e_lo, e_hi):
        real_end   = s_row_starts[e] + real_counts[e]
        padded_end = s_row_starts[e + 1]
        for j in range(real_end + tid, padded_end, 1024):
            sorted_token_ids[j] = pad_val
            m_indices[j]        = pad_val
            sorted_weights[j]   = 0.0


# ---------------------------------------------------------------------------
# quant_kernel  moe_sort_quant.cuh:301 (quant_impl :215)
#   ★ 它【不排序】!量化的是原始 hidden_states(按原 token 顺序),
#     输出 a_quant / a_scale 也是【未排序】。sort 由后面 sort_scales + gemm gather 完成。
#   量化粒度: 32 个 hidden 元素共享一个 e8m0 scale (= 一个 quant-block)。
#   D_HIDDEN=7168 -> 每 token 224 个 quant-block。
# ---------------------------------------------------------------------------
@launch(N_QCTAS, 1024)   # grid=N_QCTAS(codegen 常量), block=1024=16 waves
def quant_kernel(M, hidden_states,   # [M, 7168] bf16 (未排序)
                 a_quant,            # [M, 7168//2] u8  (fp4, 未排序)
                 a_scale):           # [M, 7168//32] u8 (e8m0, 未排序)
    BLOCKS_PER_HIDDEN = 7168 // 32    # 224 quant-block / token
    # 4 个 lane 协作处理一个 32-元素 block,每 lane 8 个元素
    TOTAL_BLOCKS = M * BLOCKS_PER_HIDDEN
    for my_block in assigned_blocks(bid, tid):   # grid-stride, 见源码 wi/wave/block 映射
        kb = my_block * 32 + lane_in_block * 8
        h[0:4] = load_int4(hidden_states[kb : kb+8])    # 一次载 8 个 bf16

        local_amax = max(abs(h[j]) for j in range(4))   # 本 lane 4 元素
        amax = dpp_reduce_max_over_4lanes(local_amax)    # 4-lane DPP 规约 -> 32 元素 amax

        scale = e8m0_from(amax * (1/6))                  # -> e8m0, clamp 0..254
        qs = 2.0 ** (scale - 127)
        pk = cvt_scalef32_pk_fp4_bf16(h[0:4], qs)        # gfx950 硬件: 8 bf16 -> 8 fp4(4B)
        a_quant[my_block*16 + lane_in_block*4 : +4] = pk
        if lane_in_block == 0:
            a_scale[my_block] = scale                    # 每 block 一个 scale


# ---------------------------------------------------------------------------
# sort_scales_kernel  moe_sort_scales.cuh:14
#   ★ 真正对 A 做 sort 的地方 —— 但只 sort【scale】。
#   读未排序 a_scale,按 sorted_token_ids gather 重排 + shuffle 成 MFMA 读 scale
#   的交织布局 (MN_PACK/K_PACK/K_LANE=4/N_LANE=16)。
#   数据本体 a_quant 不在此排,gemm1 读时用 sorted_token_ids 动态 gather。
# ---------------------------------------------------------------------------
@launch(N_CTAS, 1024)    # kNCtasScales=512, kThreadsScales=1024
def sort_scales_kernel(M, max_sorted,
                       a_scale,                    # [M, 7168//32] u8 (未排序)
                       sorted_token_ids,           # [max_sorted]
                       cumsum,                     # [2], cumsum[0]=有效总行数
                       a_scale_sorted_shuffled):   # 输出: 交织布局 sorted scale
    A_SCALE_COLS = 7168 // 32     # 224
    actual_sorted = cumsum[0]
    for work_id in grid_stride(bid, tid):          # 每 work_id = 输出一个 4B chunk
        chunk, mi, ku, k_lane, n_lane = decode(work_id)  # -> MFMA 布局坐标
        bytes = [0, 0, 0, 0]
        if chunk < actual_n_chunks:
            for im_a in range(2):                  # MN_PACK=2
                sorted_row = chunk*BM + (mi*2 + im_a)*16 + n_lane
                tok = sorted_token_ids[sorted_row] & 0x00FFFFFF   # 反查原 token
                tok = tok if tok < M else 0                        # 哨兵 -> 0
                for ikxdl in range(K_PACK):
                    k_idx = ku*K_PACK*4 + ikxdl*4 + k_lane
                    bytes[...] = a_scale[tok * A_SCALE_COLS + k_idx]  # gather 重排
        a_scale_sorted_shuffled[work_id*4 : +4] = bytes