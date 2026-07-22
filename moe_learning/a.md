# MoE 代码地图 —— 我们实际 bench 的 MXFP4 FlyDSL port

聚焦 Kimi-K2.5-MXFP4 在 MI355X(gfx950)上实际跑的这条路径:
`fused_moe_` → moe_sorting(aux) → gemm1(stage1) → gemm2(stage2) → scatter_reduce。
下面路径都相对 `/app/aiter-test/`(本目录下有 `aiter` 软链)。

---

## 调用链(按调用顺序,行号 = aiter/fused_moe.py 除非另注)


topk_ids: [M, topk]
max_num_tokens_padded = M*topk + num_experts * BM - topk ?? why - topk?
max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)


### A) 生产路径
```
fused_moe (:441)                         # 公开 API,薄封装
└─ fused_moe_ (:535)                      # 真正的编排函数
   ├─ moe_sorting (:724 调用 → def :335)  # ★先 sort(只排路由索引)
   │  └─ _flydsl_moe_sorting (:285)
   │     └─ flydsl_moe_sorting_fwd  (ops/flydsl/moe_sorting.py:19)
   │        └─ device: csrc/kernels/mxfp4_moe/moe_3stage_sort.cuh
   │                   moe_sort_quant.cuh / moe_sort_scales.cuh
   │  (stage1/stage2 fn 由 metadata 绑定, :1978-1983 →
   │   _mxfp4_a4w4_stage1_fw / _mxfp4_a4w4_stage2_fw)
   ├─ [run_1stage] metadata.stage1(...) (:766)
   └─ [2-stage]    fused_moe_2stages (:795)   # 我们的路径:先 g1 后 g2
      ├─ stage1 → _mxfp4_a4w4_stage1_fw (:1526)
      │           └─ _mxfp4_a4w4_stage1 (:1288)
      │              └─ flydsl_mxfp4_gemm1 (ops/flydsl/mxfp4_gemm1_kernels.py:70)
      │                 └─ compile_gemm1_a4w4_port (kernels/mxfp4_gemm1.py:748)  ← DSL kernel
      └─ stage2 → _mxfp4_a4w4_stage2_fw (:1591)
                  └─ _mxfp4_a4w4_stage2 (:1386)
                     ├─ flydsl_mxfp4_gemm2 (ops/flydsl/mxfp4_gemm2_kernels.py:102)
                     │  └─ compile_gemm2_a4w4_port (kernels/mxfp4_gemm2.py:88)   ← DSL kernel
                     └─ aiter.mxfp4_moe_scatter_reduce (:1513)
                        └─ device: csrc/kernels/mxfp4_moe/moe_scatter_reduce.cuh
```

### B) 我们实际 bench 的路径(tuner,不是 fused_moe_)
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` 的 `Mxfp4FlydslTuner`:
```
_port_e2e (Mxfp4FlydslTuner)
├─ moe_sorting            (fused_moe.py:335,同上 aux sort)
├─ _mxfp4_a4w4_stage1_fw  (fused_moe.py:1526) → gemm1 DSL kernel
└─ _mxfp4_a4w4_stage2_fw  (fused_moe.py:1591) → gemm2 DSL kernel + scatter_reduce
```
→ 我们的 `split_gemm_tflops.py` 直接复用这三步,分别给 stage1/stage2 计时。
两条链的 sort/gemm1/gemm2 device kernel 完全一样,只是编排壳不同。

---

> 说明:aiter 里有很多 MoE 后端(ck、cktile、asm、opus、通用 flydsl…),
> 我们 bench 的**只有** `--mxfp4-flydsl` 这条 "a4w4 FlyDSL port"
> (device kernel 名 `gemm1_a4w4_port_*` / `gemm2_a4w4_port_*`)。
> 本文只标这条,其它后端忽略。

---

## 0. 顶层入口
- `aiter/fused_moe.py:535` **`fused_moe_`** — 生产主入口,串起全流程。
  顺序:sort(`:724`/`:736`)→ stage1(`:766`)→ stage2(`:795`)。先读它看骨架。
- tuner 侧等价入口(我们 bench 用的):
  `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` 的
  `Mxfp4FlydslTuner._port_e2e`(moe_sorting → stage1_fw → stage2_fw)。

---

## 1. ROUTER(topk gating:每个 token 选 topk 个 expert)
- `aiter/fused_moe.py:3390` `fused_topk` — 通用 topk 入口。
- `aiter/ops/topk.py`:
  - `:138 biased_grouped_topk` / `:92 grouped_topk` — Kimi/DSV 这类 grouped 路由。
  - `:179 biased_grouped_topk_torch` / `:224 grouped_topk_torch` — **torch 参考,先读这个懂语义**。
  - `:125 moe_fused_gate` — 融合门控 kernel。
- `aiter/ops/moe_op.py:16 topk_softmax` — softmax+topk device 封装。
- 产出:`topk_ids [M, topk]`、`topk_weights [M, topk]`。

## 2. SORT(按 expert 排序 + padding 到 BM,只排索引,不依赖 gemm1 输出)

### ★ 我们实际走的 sort(output_aux=True 分支)
tuner `_port_e2e` 和生产 `fused_moe_` 都用 `output_aux=True` 调 moe_sorting,
所以走 **adaptive/aux sort**(不是 flydsl_moe_sorting_fwd,那条被 output_aux 跳过):
```
moe_sorting (:335, output_aux=True → 跳过 :350 flydsl 分支)
└─ _moe_sorting_impl (:169)
   └─ (:187 output_aux and backend not in opus/ck) _adaptive_moe_sort (:112)
      └─ aiter.mxfp4_moe_sort (:144)          ← 真正的 sort kernel
         └─ def: aiter/ops/moe_mxfp4_aux.py:48
         └─ device: csrc/kernels/mxfp4_moe/ (aux sort + quant + scales + scatter_reduce)
                    codegen/gen_instances.py  ← 按 NE 生成实例(handoff 加 NE96 那行,SHAPES)
```
这就是 EP-mask 注释里说的 "routes through the adaptive/aux sort" = `_adaptive_moe_sort`,
它没接 `expert_mask`。

**max_sorted 公式(adaptive 版, :137-138,和之前问的 -topk 那条不同!)**:
```python
active = min(num_experts, M * topk)
max_sorted = (((M*topk + active*(BM-1)) + BM-1) // BM) * BM
```
- `active*(BM-1)` = 精确最坏 padding(每个有 token 的 expert 最多浪费 BM-1)。
- `active = min(num_experts, M*topk)` 再 cap:token 对数 < expert 数时,活跃 expert
  不可能超过 token 对数。→ 这是**更紧**的上界。

### 传统路径(我们不走,仅参考)
- `_flydsl_moe_sorting`(:285)→ `flydsl_moe_sorting_fwd`(ops/flydsl/moe_sorting.py:19)
  → `moe_3stage_sort.cuh` / `moe_sort_quant.cuh` / `moe_sort_scales.cuh`。
- 传统 `max_num_tokens_padded`(:203)= `M*topk + num_experts*BM - topk`:
  `-topk` 是**安全过分配**;精确硬上界是 `M*topk + num_experts*(BM-1)`(即 -num_experts)。
  num_experts(384) > topk(8),故 `-num_experts` 更紧,`-topk` 更松但安全。

### 产出(两条都一样)
`sorted_token_ids`、`sorted_expert_ids(eids)`、`cumsum`、`sorted_weights`、
`reverse_sorted`、`m_indices`。

### ⚠️ EP mask 的坑(你贴的那段注释,`fused_moe.py:704-713`)
```python
# The a4w4 FlyDSL port routes through the adaptive/aux sort, which does
# not thread expert_mask into moe_sorting below -- EP masking would be
# silently ignored and tokens routed to the wrong experts. Fail loudly
# until EP support is added to the port.
if expert_mask is not None:
    raise NotImplementedError(...)
```
含义:真正的 EP(专家并行)里,每卡只持有 routed 专家的一个子集,靠
`expert_mask` 告诉 sort "哪些 expert 是本卡的、其余 token 要丢/转发"。
但 a4w4 FlyDSL port 走的是 **aux/adaptive sort**,这条 sort **没有把
`expert_mask` 传下去**。若硬塞 mask,会被**静默忽略**→ token 被路由到错的
expert → 结果错但不报错。所以 port 里遇到 `expert_mask is not None` 直接
`raise NotImplementedError` **主动报错**,而不是给你错数据。

→ 对我们的意义:我们现在 bench 的 "EP" 是**用 NE=96 的 shape 模拟每卡的
专家数**(把每卡当成一个 96-expert 的独立 MoE 来量 GEMM 算力),并**没有**
走真正的跨卡 expert_mask 分发。要落地真 EP,得先给这条 aux sort 补上
expert_mask 支持。我们目前只关心 GEMM 本身的 TFLOP/s,所以这个限制不影响
测算,但落地时是必须解的。

### expert_mask 本身是个啥(开了 EP 时)
- **形状**:`[global_E]` 一维,长度 = **全局 expert 总数**(Kimi routed=384),
  元素 0/1。`fused_moe.py:581 global_E = expert_mask.numel()` 印证。
- **语义**:`expert_mask[g]==1` = "全局第 g 个 expert **住本卡**",`0` = 在别卡。
  就是每张卡的一份"专家花名册"。EP4:每卡 96 个 1、288 个 0。
- **用法 1 — 挑出本卡的 token-slot**(`:403 get_topk_valid_mask`):
  `valid_mask = expert_mask[topk_ids]` → `[M, topk]`,1=本卡 0=外卡;
  外卡 slot 丢弃(由持有该 expert 的卡负责)。
- **用法 2 — 全局 id → 本地 id 重映射**(`:2850`,torch 参考最清楚):
  ```python
  local_expert_hash = expert_mask.cumsum(0) - 1   # 前缀和 = 本卡第几个
  local_expert_hash[expert_mask == 0] = -1        # 外卡标 -1
  topk_ids = local_expert_hash[topk_ids]          # 全局 0..383 -> 本地 0..95 / -1
  ```
  原因:本卡权重只存自己 96 个 expert(`w2[96,...]`,本地下标 0..95),但
  router 的 `topk_ids` 是全局 0..383,必须压成本卡紧凑下标(-1=不是我的,跳过)。
- 一句话:expert_mask = 标记"哪些 expert 在本卡"的 0/1 向量;EP 用它
  ① 挑本卡 token-slot ② 把全局 expert 号压成本地 0..95 下标。aux sort 没接这
  两步 → 硬跑会拿全局 id 去 index 96 槽的本地权重 → 越界/错位 → 所以 port 直接 raise。

## 3. GEMM1 / stage1(gate+up projection)
- `aiter/fused_moe.py:1526 _mxfp4_a4w4_stage1_fw` — tuner 封装(算 shape、建 buffer)。
- `aiter/fused_moe.py:1288 _mxfp4_a4w4_stage1` — 核心 host 逻辑。
- `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:70 flydsl_mxfp4_gemm1` — launch 封装。
- `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:748 compile_gemm1_a4w4_port`
  — **device kernel(DSL 源码)**,主循环 ~`:567`。
- 中间有 SiLU 激活 + 把中间激活重新量化成 mxfp4(gemm2 的 A 要 fp4 输入)。
  这就是 gemm1/gemm2 无法融成一个 kernel 的原因。

## 4. GEMM2 / stage2(down projection)—— 我们在优化的
- `aiter/fused_moe.py:1591 _mxfp4_a4w4_stage2_fw` — tuner 封装。
- `aiter/fused_moe.py:1386 _mxfp4_a4w4_stage2` — 核心 host 逻辑(non-atomic 后接 scatter_reduce)。
- `aiter/ops/flydsl/mxfp4_gemm2_kernels.py:102 flydsl_mxfp4_gemm2` — launch(grid = m_blocks×28)。
- `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:88 compile_gemm2_a4w4_port`
  — **device kernel(DSL 源码)**。要点行:
  - `:144 gemm2_kernel` 签名(12 个 arg:aq/ascale/bq/bscale/eids/cumsum/stids/sweights/M/max_m_blocks/out/out_scale)
  - `:224` persistent 分支 / `:270` 非 persistent(cshuffle 走这条)
  - `:390-443` block→expert/A/B 寻址(`m_block_idx=bx/28, n_block_idx=bx%28, e=eids[m_block_idx]`)
  - `:573-630` K-loop + `_kloop_fence`(**放松 vmcnt 的地方**)
  - `:659 _flat_bf16_epilog` / `:676 _cshuffle_flat_bf16_epilog`
- 共用维度 helper:`aiter/ops/flydsl/kernels/mxfp4_gemm_common.py`
  (`tiling / k_tiles_total_for / num_n_blocks_for / kStages=2 / kmchunks_for`)。

## 5. 归并(gemm2 之后)
- `csrc/kernels/mxfp4_moe/moe_scatter_reduce.cuh` + host `aiter.mxfp4_moe_scatter_reduce`
  (`fused_moe.py:1513`)— 按 `sorted_weights` 加权、按 `reverse_sorted` 把
  `[max_sorted,H]` 求和回每 token `[M,H]`(topk 个 slot 相加)。

---

## 数据流(一句话)
router 出 topk_ids/weights → moe_sorting(排序+padding,只排索引)→
gemm1 按 sorted 顺序读 hidden,出**已 sorted** 的中间激活(+SiLU+requant fp4)→
gemm2 用同一套 sorted_ids/eids/cumsum 直接接着算(**不再 sort**)→
scatter_reduce 归并回每 token。

## gemm2 维度速查(EP, token=32768)
- a=[max_sorted,2048] mxfp4, a_scale=[max_sorted,64] e8m0
- w2=[NE,7168,2048] mxfp4, w2_scale=[NE,7168,64]
- H=7168=输出维N; inter=2048=K; NE=96(EP,shared 未融)/ 385(TP,含 shared)
- BM=128(M行,需 padding), BN=256(N=H→28 个 n-block), BK=256(K=2048→8 个 K-tile)
- grid.x = total_m_blocks × 28;每 WG 出一个 [128,256] 输出瓦片

## 建议阅读顺序
`fused_moe_`(535 骨架)→ `topk.py` torch 参考(路由语义)→
`moe_sorting`(335)+ `moe_3stage_sort.cuh`(padding/eids 怎么来)→
gemm1 DSL(sorted 布局输入)→ gemm2 DSL(最熟)。
