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
bx = 0
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
# constexpr int DPP_QUAD_PERM(int a,int b,int c,int d){ return (a&3) | ((b&3)<<2) | ((c&3)<<4) | ((d&3)<<6); } 怎么交换.
@launch(512, 1024)   # grid=kNCtasSort=512, block=kThreadsSort=1024 (dispatch.h:12-13)
def quant_kernel(M, 
                 hidden_states,      # [M, 7168] bf16 (未排序)
                 a_quant,            # [M, 7168//2] u8  (fp4, 未排序)
                 a_scale):           # [M, 7168//32] u8 (e8m0, 未排序)
    WARP__64 = 64
    BLOCKS_PER_HIDDEN__224 = 7168 // 32     # 224 quant-block / token
    LANES_PER_BLOCK__4   = 4              # 4 lane 协作一个 32-元素 block,每 lane 8 元素
    BLOCKS_PER_WAVE__16   = WARP__64 // LANES_PER_BLOCK__4      # 16 (一个 wave 管 16 个 block)
    WAVES_PER_CTA     = 1024 // WARP__64                 # 16
    BLOCKS_PER_CTA__256    = BLOCKS_PER_WAVE__16 * WAVES_PER_CTA   # 256 (一个 CTA 一批做 256 个)

    # 线程 -> (wave, block_in_wave, lane_in_block) 的拆分
    wave_id       = tid // WARP__64           # 0..15
    lane          = tid %  WARP__64           # 0..63
    block_in_wave = lane // LANES_PER_BLOCK__4   # 0..15
    lane_in_block = lane %  LANES_PER_BLOCK__4   # 0..3

    # 把 TOTAL_BLOCKS 个 quant-block 按 "批(BLOCKS_PER_CTA=256)" 分给 512 个 CTA
    TOTAL_BLOCKS  = M * BLOCKS_PER_HIDDEN__224
    N_BATCHES     = ceil_div(TOTAL_BLOCKS, BLOCKS_PER_CTA__256)
    BATCH_PER_CTA = ceil_div(N_BATCHES, 512)          # 每个 CTA 分几批
    wi_start = bid * BATCH_PER_CTA
    wi_end   = min(wi_start + BATCH_PER_CTA, N_BATCHES)

    for wi in range(wi_start, wi_end):
        # 本线程这一批负责哪个 quant-block(← 这就是原来偷懒的 assigned_blocks)
        my_block = wi * BLOCKS_PER_CTA__256 + wave_id * BLOCKS_PER_WAVE__16 + block_in_wave
        if my_block >= TOTAL_BLOCKS:
            continue

        kb = my_block * 32 + lane_in_block * 8
        h[0:4] = load_int4(hidden_states[kb : kb+8])    # 一次载 8 个 bf16

        local_amax = max(abs(h[j]) for j in range(4))   # 本 lane 4 元素
        amax = dpp_reduce_max_over_4lanes(local_amax)    # 4-lane DPP 规约 -> 32 元素 amax

        scale = e8m0_from(amax * (1/6))                  # -> e8m0, clamp 0..254
        qs = 2.0 ** (scale - 127)
        pk = cvt_scalef32_pk_fp4_bf16(h[0:4], qs)        # gfx950 硬件: 8 bf16 -> 8 fp4(4B)
        a_quant[my_block*16 + lane_in_block*4 : +4] = pk
        if lane_in_block == 0:
            a_scale[my_block] = scale                    # 每 block 一个 scale (4 lane 只写一次)


# ---------------------------------------------------------------------------
# sort_scales_kernel  moe_sort_scales.cuh:14
#   ★ 真正对 A 做 sort 的地方 —— 但只 sort【scale】。
#   读未排序 a_scale,按 sorted_token_ids gather 重排 + shuffle 成 MFMA 读 scale
#   的交织布局 (MN_PACK/K_PACK/K_LANE=4/N_LANE=16)。
#   数据本体 a_quant 不在此排,gemm1 读时用 sorted_token_ids 动态 gather。
# ---------------------------------------------------------------------------
@launch(512, 1024)    # grid=kNCtasScales=512, block=kThreadsScales=1024
def sort_scales_kernel(M, max_sorted,
                       a_scale,                    # [M, 7168//32] u8 (未排序)
                       sorted_token_ids,           # [max_sorted]
                       cumsum,                     # [2], cumsum[0]=有效总行数
                       a_scale_sorted_shuffled):   # 输出: 交织布局 sorted scale
    # MFMA 读 scale 的布局常量 (BM=128, BK=256, D_HIDDEN=7168)
    A_SCALE_COLS = 7168 // 32        # 224 = 每 token 的 scale 列数
    MN_PACK = 2                       # 每输出 dword 打包 2 行(M 方向)
    K_PACK  = 256 // 128              # BK/128 = 2
    C_M1    = BM // (16 * MN_PACK)    # 128/32 = 4
    C_K1    = (7168 // 32) // (4 * K_PACK)   # 224/8 = 28
    K_LANE  = 4
    N_LANE  = 16
    DWORDS_PER_CHUNK = C_M1 * C_K1 * K_LANE * N_LANE   # 4*28*4*16 = 7168

    n_chunks        = max_sorted // BM                 # 全部(含 padding)chunk 数
    actual_sorted   = cumsum[0]
    actual_n_chunks = ceil_div(actual_sorted, BM)      # 有效 chunk 数

    # 显式 grid-stride:512*1024 个线程扫 total_work 个 4B 输出 dword
    total_work    = n_chunks * DWORDS_PER_CHUNK
    total_threads = 512 * 1024
    global_tid    = bid * 1024 + tid
    for work_id in range(global_tid, total_work, total_threads):
        # 把线性 work_id 拆成 MFMA 布局坐标(低维在前)
        r = work_id
        n_lane = r % N_LANE; r //= N_LANE     # 0..15
        k_lane = r % K_LANE; r //= K_LANE     # 0..3
        ku     = r % C_K1;   r //= C_K1       # 0..27  (K 方向 chunk)
        mi     = r % C_M1;   r //= C_M1       # 0..3   (M 方向 pack)
        chunk  = r                             # 第几个 BM 行块

        bytes = [0, 0, 0, 0]
        if chunk < actual_n_chunks:
            tok_ids = [0, 0]
            for im_a in range(MN_PACK):                     # 2 行
                sorted_row = chunk*BM + (mi*MN_PACK + im_a)*16 + n_lane
                if sorted_row < actual_sorted:
                    v = sorted_token_ids[sorted_row] & 0x00FFFFFF   # 反查原 token
                    tok_ids[im_a] = v if v < M else 0                # 哨兵 -> 0
            for ikxdl in range(K_PACK):                     # 2
                for im_a in range(MN_PACK):                  # 2
                    k_idx = ku*K_PACK*4 + ikxdl*4 + k_lane
                    bytes[ikxdl*MN_PACK + im_a] = a_scale[tok_ids[im_a]*A_SCALE_COLS + k_idx]  # gather
        a_scale_sorted_shuffled[work_id*4 : work_id*4 + 4] = bytes   # 写成交织布局


# ===========================================================================
# GEMM1 / stage1  (gate+up projection)   FlyDSL: mxfp4_gemm1.py:748
#   数学: 每个 expert  out[m, 2*inter] = A[m, H] @ w1[e]^T ,  A=gemm1 输入激活
#         w1 拼了 gate 和 up 两半 -> N_OUT = 2*inter。算完做 SiLU(gate)*up
#         再重新量化成 fp4 -> 喂给 gemm2 的 A。
#   ★ A 的数据本体 a_quant 是【未排序】的(quant_kernel 按原 token 序产出)。
#     gemm1 在这里用 m_indices(=sorted 行->原 token)【动态 gather】读 A。
#     而 a_scale 已被 sort_scales 预排+交织,gemm1 直接顺序读。
#   下面只写我们 TP 实际走的主干: BM=128, cached(非 inline_quant), 非 interleave。
#
#   维度(TP32768): H=7168(K), inter=512, N_OUT=2*512=1024, NE=385
#   BM=128, BN=256, BK=256 -> K_TILES = H/BK = 28, NUM_N_BLOCKS = N_OUT/BN = 4
#   kStages=2 (A 的 LDS 双缓冲深度)
# ===========================================================================
#  max_m_blocks = (n_tokens*TOPK + NE*(BM-1) + BM-1)//BM;  grid = max_m_blocks*NUM_N_BLOCKS
# launch: grid=(total_m_blocks*NUM_N_BLOCKS, 1, 1) 一维, block=(256,1,1) (mxfp4_gemm1.py:941)
#   bx 一维线性, kernel 内拆: m_block_idx = bx//NUM_N_BLOCKS, n_block_idx = bx%NUM_N_BLOCKS
@launch(grid="total_m_blocks * NUM_N_BLOCKS", block=256)   # 一维 grid, block=256=4 waves; 每 WG 一个 (m_block,n_block) 瓦片
def gemm1_kernel(arg_aq,        # [ntok, H//2]   u8 fp4  A 数据本体(未排序!)
                 arg_ascale,    # sort_scales 输出: 已排序+交织的 A scale
                 arg_bq,        # [NE, N_OUT, H//2] u8 fp4  w1 权重(gate|up 拼一起)
                 arg_bscale,    # w1 scale
                 eids_in,      # [m_blocks]  每个 m_block 的 expert
                 cumsum_in,    # [2] cumsum[0]=有效总行数
                 mindices_in,      # m_indices: sorted 行 -> 原 token(gather A 用)
                 i32_ntok,
                 arg_aqout,     # 输出: [ntok, inter//2] u8 fp4  (SiLU 后 requant, 喂 gemm2)
                 arg_ascaleout, # 输出: gemm2 要的 A scale
                 arg_hidden):   # inline_quant 变体才用,主干忽略
    H, INTER, NE = 7168, 512, 385
    N_OUT = 2 * INTER              # 1024 = gate(512) | up(512)
    # ★ gemm1 的 N = N_OUT = 2*inter (不是 H!). NUM_N_BLOCKS = 1024/256 = 4.
    #   (gemm2 才是 N=H=7168 -> 28 个 n-block; 别和 gemm1 混)
    NUM_N_BLOCKS = N_OUT // 256    # 4  (gemm1; gemm2 是 7168//256=28)
    K_TILES = H // 256             # 28  (K = H = 7168 是 gemm1 的规约维)
    kStages = 2
    total_m_blocks = cumsum_in[0] // BM   # total blocks
    bound = total_m_blocks * NUM_N_BLOCKS
    if bx >= bound:                        # early return
        return
    # 即: grid=host上界(偏多) + kernel内 if bx<bound 剪掉多余; 用空转换掉 d2h 同步

    # ---- ① block -> 负责哪个 (m_block, n_block) 瓦片 + 查 expert ----
    #   (xcd_swizzle>0 时先过 _xcd() 重映射做 XCD 负载均衡; 主干 SW=0 直接用 bx)
    n_block_idx = bx % NUM_N_BLOCKS
    m_block_idx = bx // NUM_N_BLOCKS
    e     = readfirstlane(eids_in[m_block_idx])   # 本瓦片所有行的 expert
    m_row = m_block_idx * BM

    # ---- ② 预取 gather 索引: 本 wave 负责的 sorted 行 -> 原 token 行 ----
    #   A 未排序,所以要用 m_indices 把 "sorted 行" 翻译成 a_quant 里的原 token 行。
    cached_actual_row = [ mindices_in[m_row + wave*(BM//4) + sub*8 + lane//8]
                          for sub in range(kSubBlocks) ]   # 每 sub 一个原 token 行号

    # ---- ③ B(权重)/scale 的 per-wave 基址(readfirstlane 成标量)----
    b_load_s_base = [ (e*N_OUT + col_of(j)) * K_HALF   for j in range(4) ]
    b_scale_s_base = ...   # e*kBS_per_expert + ...

    # A -> LDS: 用 gather 行号 cached_actual_row 去 a_quant 里 buffer_load_lds
    def issue_a_load_lds(slot, kt):
        for sub in range(kSubBlocks):
            voffset = swizzle(...) + cached_actual_row[sub] * K_HALF   # ← gather 原 token 行
            buffer_load_lds(aq_rsrc, s_aq[slot], voffset, soffset=kt*KH_TILE)

    def issue_a_ds_read(slot):        # 从 LDS 读 A 片给 MFMA(带 xor swizzle 解 bank 冲突)
        return [[ lds_load_vec4(s_aq[slot], row=i, k) for k in range(2)] for i in range(kMChunks)]

    def mfma_cluster(b, a, a_sc, b_sc, J, init):   # 4 个 J,每个多条 16x16x128 fp4 MFMA 累加
        accm[.][J] = mfma_scale_f32_16x16x128_f8f6f4(a, b, accm, a_sc, b_sc)

    # ---- ④ 软流水主循环 (kStages=2 双缓冲): 边算当前 tile 边预取下一 tile 的 A ----
    issue_a_scale_load()                       # A scale 一次性载进 LDS
    for K_C in range(kStages):                  # prologue: 预热前 2 个 K-tile
        issue_a_load_lds(K_C, K_C)
    for K_C in range(kStages):                  # B 权重/scale 预取
        for j in range(4): issue_b_load_j(b[K_C], K_C, j)
        issue_b_scale_load(b_scale_v[K_C], K_C)

    for OFFSET in range(kUnroll):               # 主体: K_TILES-kStages 轮
        K_C = kStages + OFFSET
        gpu.barrier()
        asc = issue_a_scale_ds_read(K_C - kStages)
        a   = issue_a_ds_read(OFFSET % kAStages)     # 读当前 tile 的 A
        issue_a_load_lds(K_C % kAStages, K_C)         # 预取下一 tile 的 A(重叠)
        for J in range(4):
            mfma_cluster(b[OFFSET%kStages], a, asc, b_scale_v[OFFSET%kStages], J, init=(OFFSET==0))
            issue_b_load_j(b[OFFSET%kStages], K_C, J)  # 顺带预取下一 tile 的 B
        issue_b_scale_load(b_scale_v[OFFSET%kStages], K_C)

    for S in range(kStages):                    # epilogue drain 最后 2 个 tile
        kt = K_TILES - kStages + S
        gpu.barrier()
        asc = issue_a_scale_ds_read(kt); a = issue_a_ds_read(kt % kAStages)
        for J in range(4): mfma_cluster(b[kt%kStages], a, asc, b_scale_v[kt%kStages], J, init=False)

    # ---- ⑤ epilogue: accm -> LDS 重排 -> SiLU(gate)*up -> requant fp4 -> 写出 ----
    store accm to lds_acc                         # [BM, BN] f32 经 LDS 重排
    gpu.barrier()
    for mr in range(M_REPS):                       # 每线程负责的输出行
        gate[0:8] = lds_acc[row, gate_cols]        # 从 LDS 取 gate 半
        up[0:8]   = lds_acc[row, up_cols]          #          up 半
        result = silu(gate) * up                   # _silu_mul_batch
        amax   = dpp_quad_amax(|result|)           # 8 元素 blockwise absmax
        e8m0, qs = e8m0_from_amax(amax)
        packed = cvt_scalef32_pk_fp4_f32(result, qs)  # 8 f32 -> 8 fp4 (4B)
        arg_aqout[out_row, byte_pos]   = packed    # 写 gemm2 的 A 数据本体
        arg_ascaleout[...]             = e8m0       # 写 gemm2 的 A scale
    # → gemm2 的输入 (a_quant/a_scale) 就是这里产出的,已是 sorted 布局(按 m_row 写)