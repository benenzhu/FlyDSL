# Kimi FP4 MoE 16384 MXFP4 Flow

This note is a map for the current all-FlyDSL MXFP4 path in
`kimi_fp4_moe_16384_opt_simplify.py`, centered on:

```python
run_kimi_fp4_mxfp4_moe_16384_all_flydsl(...)
```

It is meant as a reading guide. The detailed GEMM scheduling, MFMA tiling, and
LDS pipeline are still in the kernel code; this document focuses on which
buffers exist, how rows are rearranged, and why the final scatter/reduce is
needed.

## Fixed Shape

| name | value |
| --- | ---: |
| `TOKEN` | 16384 |
| `TOPK` | 9 |
| `EXPERTS` | 385 |
| `MODEL_DIM` | 7168 |
| `INTER_DIM` | 512 |
| `MXFP4_BLOCK_M` | 128 |
| routed pairs before padding | `TOKEN * TOPK = 147456` |
| worst-case padded sorted rows | `max_sorted = 196352` |
| sorted expert blocks | `max_sorted / 128 = 1534` |

MXFP4 storage convention in this file:

| logical data | packed value shape | scale shape |
| --- | --- | --- |
| hidden `[TOKEN, MODEL_DIM]` | `[16384, 3584]` uint8 | `[16384, 224]` uint8 |
| intermediate `[sorted, INTER_DIM]` | `[196352, 256]` uint8 | logical `[196352, 16]`, but stored in GEMM2 shuffled layout |
| output per routed row `[sorted, MODEL_DIM]` | `[196352, 3584]` uint8 | `[196352, 224]` uint8 |

## Stage Count vs Kernel Count

There are 6 high-level Python stages in `run_kimi_fp4_mxfp4_moe_16384_all_flydsl`,
but 8 actual GPU kernel launches. The difference is that the first stage,
`kimi_mxfp4_sort_16384`, launches three kernels internally.

| high-level stage | actual GPU launches |
| --- | --- |
| 1. `kimi_mxfp4_sort_16384` | `sort_count`, `sort_cumsum`, `sort_place_pad` |
| 2. `kimi_mxfp4_quant_16384` | `quant` |
| 3. `kimi_mxfp4_sort_scales_16384` | `sort_scales` |
| 4. `kimi_mxfp4_gemm1_16384` | `gemm1` |
| 5. `kimi_mxfp4_gemm2_mxfp4out_16384` | `gemm2_mxfp4out` |
| 6. `kimi_mxfp4_scatter_reduce_q_16384` | `scatter_reduce_q` |

So if "kernel" means Python wrapper stage, there are 6. If it means GPU launch,
there are 8.

## Main Dataflow

```mermaid
flowchart TD
    HS["hidden_states<br/>(16384, 7168) bf16"] --> Q["kimi_mxfp4_quant_16384<br/>per 32 model values:<br/>fp4x2 bytes + e8m0 scale"]
    Q --> AQ["a_quant<br/>(16384, 3584) uint8"]
    Q --> AS["a_scale<br/>(16384, 224) uint8"]

    TOPKID["topk_ids<br/>(16384, 9) int32"] --> SORT
    TOPKW["topk_weight<br/>(16384, 9) f32"] --> SORT

    subgraph SORT["kimi_mxfp4_sort_16384<br/>expert-major sort + BM128 padding"]
        direction TB
        SC["sort_count<br/>count routed pairs per expert<br/>block_offsets: (385, 16)"]
        SP["sort_cumsum<br/>sum counts, round each expert to BM128,<br/>prefix starts, fill sorted_expert_ids"]
        SPL["sort_place_pad<br/>place real rows, write inverse map,<br/>append padding rows"]
        SC --> SP --> SPL
    end

    SORT --> STI["sorted_token_ids<br/>(196352) int32<br/>packed token/topk; pad token = TOKEN"]
    SORT --> SEI["sorted_expert_ids<br/>(1534) int32<br/>one expert id per BM128 block"]
    SORT --> CS["cumsum_tensor(0)<br/>actual padded sorted rows"]
    SORT --> REV["reverse_sorted<br/>(147456) int32<br/>original token/topk pair -> sorted row"]
    SORT --> SW["sorted_weights<br/>(196352) f32<br/>topk weight in sorted-row order"]
    SORT --> MI["m_indices<br/>(196352) int32<br/>token id in sorted-row order"]

    AS --> SS["kimi_mxfp4_sort_scales_16384<br/>gather hidden scales by sorted_token_ids<br/>and swizzle for GEMM1"]
    STI --> SS
    CS --> SS
    SS --> ASS["a_scale_sorted_shuffled<br/>flat uint8<br/>196352 * 224 * 2 bytes"]

    AQ --> G1["kimi_mxfp4_gemm1_16384<br/>expert-major GEMM1:<br/>hidden @ W_gate/up,<br/>SiLU(gate) * up,<br/>quantize intermediate to MXFP4"]
    ASS --> G1
    W1["w1 packed<br/>(385, 1024, 3584) uint8"] --> G1
    W1S["w1_scale<br/>(385, 1024, 224) uint8"] --> G1
    SEI --> G1
    MI --> G1
    CS --> G1
    G1 --> IQ["inter_sorted_quant<br/>(196352, 256) uint8"]
    G1 --> IS["inter_sorted_shuffled_scale<br/>GEMM2-ready scale layout<br/>(785408, 16) uint8"]

    IQ --> G2["kimi_mxfp4_gemm2_mxfp4out_16384<br/>expert-major GEMM2:<br/>intermediate @ W_down,<br/>quantize output rows to MXFP4"]
    IS --> G2
    W2["w2 packed<br/>(385, 7168, 256) uint8"] --> G2
    W2S["w2_scale<br/>(385, 7168, 16) uint8"] --> G2
    SEI --> G2
    CS --> G2
    G2 --> OQ["flat_out_q<br/>(196352, 3584) uint8"]
    G2 --> OS["flat_out_scale<br/>(196352, 224) uint8"]

    OQ --> SR["kimi_mxfp4_scatter_reduce_q_16384<br/>for each token and topk slot:<br/>reverse lookup sorted row,<br/>dequantize, multiply topk weight,<br/>sum 9 routes"]
    OS --> SR
    REV --> SR
    SW --> SR
    SR --> OUT["out<br/>(16384, 7168) bf16"]
```

## Sort Internals

The sort stage converts row order from token-major routing:

```text
original pair index = token_id * TOPK + topk_slot
```

to expert-major, BM128-padded rows:

```text
sorted rows = expert 0 rows, expert 0 padding,
              expert 1 rows, expert 1 padding,
              ...
```

```mermaid
flowchart LR
    P["original routed pairs<br/>147456 = 16384 * 9"] --> C["sort_count<br/>per expert counts split over 16 CTAs"]
    C --> BO["block_offsets<br/>(385, 16) int32<br/>temporary per-expert partial counts"]
    BO --> X["sort_cumsum<br/>total count per expert<br/>padded = round_up(count, 128)<br/>exclusive prefix"]
    X --> RC["real_counts<br/>(385) int32"]
    X --> MM["masked_m<br/>(385) int32<br/>padded counts"]
    X --> CS2["cumsum_tensor(0)<br/>actual padded total"]
    X --> EBLK["sorted_expert_ids<br/>expert id per 128-row block"]
    X --> P2["sort_place_pad<br/>atomic local offset per expert"]
    TOPKID2["topk_ids"] --> P2
    TOPKW2["topk_weight"] --> P2
    P2 --> STI2["sorted_token_ids(row)<br/>low 24 bits: token id<br/>high bits: topk slot"]
    P2 --> MI2["m_indices(row)<br/>token id only"]
    P2 --> SW2["sorted_weights(row)<br/>route weight"]
    P2 --> REV2["reverse_sorted(token*TOPK+topk)<br/>sorted row"]
    P2 --> PAD["padding rows<br/>token id = TOKEN<br/>weight = 0"]
```

Three buffers are easy to confuse:

| buffer | direction | used by |
| --- | --- | --- |
| `sorted_token_ids[row]` | sorted row -> packed `(token, topk)` | `sort_scales`, padding checks |
| `m_indices[row]` | sorted row -> token id | GEMM1 A-row lookup |
| `reverse_sorted[token * TOPK + topk]` | original routed pair -> sorted row | final scatter/reduce |

## Kernel 1: `sort_count`

The first actual GPU launch is:

```python
sort_count(topk_ids, block_offsets).launch(
    grid=(16, 1, 1),
    block=(1024, 1, 1),
)
```

Its job is only to count how many routed pairs in each CTA chunk go to each
expert. It does not place rows yet.

Fixed constants:

| item | value |
| --- | ---: |
| total routed pairs | `16384 * 9 = 147456` |
| `sort_ctas` | 16 |
| threads per CTA | 1024 |
| routed pairs per CTA chunk | `147456 / 16 = 9216` |
| exact coverage check | `147456 % 16 == 0`, `9216 % 1024 == 0` |
| loop expression in code | `range_constexpr(div_up(per_cta, threads))` |
| loop iterations per thread, fixed shape | `ceil(9216 / 1024) = 9` |
| experts | 385 |
| output entries | `385 * 16 = 6160` |

```mermaid
flowchart TD
    IN["topk_ids flattened<br/>(147456) int32<br/>idx = token * 9 + topk"] --> CTA0

    subgraph CTA["sort_count CTA work"]
        direction TB
        CTA0["one CTA<br/>bx in 0..15<br/>1024 threads"]
        R["CTA range<br/>start = bx * 9216<br/>end = start + 9216<br/>exactly covers valid idxs"]
        Z["threads tx < 385<br/>local_count(tx) = 0<br/>shared memory: 385 x i32"]
        B0["barrier"]
        LOOP["for it in range_constexpr(div_up(per_cta, threads))<br/>fixed shape: it = 0..8<br/>idx = start + tx + it * 1024"]
        LOAD["eid = topk_ids(idx)"]
        ATOM["atomic_add<br/>local_count(eid) += 1"]
        B1["barrier"]
        WR["threads tx < 385<br/>cnt = local_count(tx)<br/>block_offsets(tx * 16 + bx) = cnt"]

        CTA0 --> R --> Z --> B0 --> LOOP --> LOAD --> ATOM --> LOOP
        LOOP --> B1 --> WR
    end

    WR --> OUT["block_offsets<br/>(385, 16) int32<br/>partial count per expert per CTA"]
```

Equivalent scalar pseudocode:

```python
for bx in parallel_range(16):
    local_count = [0] * 385

    start = bx * 9216
    end = start + 9216

    for tx in parallel_range(1024):
        for it in range_constexpr(div_up(per_cta, threads)):  # 9 for this shape
            idx = start + tx + it * 1024
            eid = topk_ids[idx]
            atomic_add(local_count[eid], 1)

    for expert in parallel_range(385):
        block_offsets[expert, bx] = local_count[expert]
```

The output layout is expert-major:

```text
block_offsets[e, bx] = count of routes to expert e inside CTA chunk bx
flat offset           = e * 16 + bx
```

## Kernel 2: `sort_cumsum`

The second actual GPU launch is:

```python
sort_cumsum(
    block_offsets,
    masked_m,
    real_counts,
    cumsum_tensor,
    sorted_expert_ids,
).launch(
    grid=(1, 1, 1),
    block=(1024, 1, 1),
)
```

Its job is to turn the per-CTA partial counts from `sort_count` into global
row ranges for each expert. This is the kernel that decides the padded sorted
layout.

Inputs and outputs:

| buffer | before `sort_cumsum` | after `sort_cumsum` |
| --- | --- | --- |
| `block_offsets(e, c)` | count of expert `e` in CTA chunk `c` | absolute row start for expert `e`, CTA chunk `c` |
| `real_counts(e)` | uninitialized | true routed count for expert `e` |
| `masked_m(e)` | uninitialized | BM128-padded count for expert `e` |
| `cumsum_tensor(0)` | uninitialized | total padded sorted rows |
| `sorted_expert_ids(b)` | uninitialized | expert id for BM128 block `b` |

```text
real_count(e) = sum(block_offsets(e, 0:16))
padded(e)     = round_up(real_count(e), 128)
start(e)      = sum_{k < e} padded(k)
end(e)        = start(e) + padded(e)
```

```mermaid
flowchart TD
    BOIN["block_offsets<br/>(385, 16) int32<br/>partial counts from sort_count"] --> K2

    subgraph K2["sort_cumsum single CTA"]
        direction TB
        T0["1024 threads<br/>only tx < 385 are expert lanes"]

        SUM["per expert lane tx<br/>load 16 partial counts<br/>4 vector loads x 4 lanes<br/>total = sum counts"]
        PAD["padded = round_up(total, 128)<br/>store s_total_count(tx)<br/>store s_padded_count(tx)<br/>real_counts(tx) = total<br/>masked_m(tx) = padded"]
        B0["barrier"]

        SCAN["tx == 0 serial prefix scan<br/>for e = 0..384<br/>s_expert_starts(e) = acc<br/>acc += s_padded_count(e)"]
        FINAL["s_expert_starts(385) = acc<br/>cumsum_tensor(0) = acc"]
        B1["barrier"]

        REWRITE["per expert lane tx<br/>acc = s_expert_starts(tx)<br/>for c = 0..15:<br/>cnt = old block_offsets(tx, c)<br/>block_offsets(tx, c) = acc<br/>acc += cnt"]
        FILL["fill sorted_expert_ids<br/>start = s_expert_starts(tx)<br/>end = s_expert_starts(tx + 1)<br/>for b in start/128 .. end/128 - 1:<br/>sorted_expert_ids(b) = tx"]

        T0 --> SUM --> PAD --> B0 --> SCAN --> FINAL --> B1 --> REWRITE --> FILL
    end

    PAD --> RC["real_counts<br/>(385) int32"]
    PAD --> MM["masked_m<br/>(385) int32"]
    FINAL --> CS["cumsum_tensor(0)<br/>actual padded sorted row count"]
    REWRITE --> BOOUT["block_offsets<br/>(385, 16) int32<br/>now per-chunk absolute starts"]
    FILL --> SEI["sorted_expert_ids<br/>(1534) int32<br/>one id per BM128 block"]
```

Equivalent scalar pseudocode:

```python
# Parallel across expert lanes tx = 0..384.
for e in parallel_range(385):
    total = 0
    for c in range_constexpr(16):
        total += block_offsets[e, c]

    padded = round_up(total, 128)
    s_total_count[e] = total
    s_padded_count[e] = padded
    real_counts[e] = total
    masked_m[e] = padded

barrier()

# One thread computes the expert prefix ranges.
acc = 0
for e in range(385):
    s_expert_starts[e] = acc
    acc += s_padded_count[e]
s_expert_starts[385] = acc
cumsum_tensor[0] = acc

barrier()

# Parallel across expert lanes again.
for e in parallel_range(385):
    acc = s_expert_starts[e]
    for c in range_constexpr(16):
        cnt = block_offsets[e, c]
        block_offsets[e, c] = acc
        acc += cnt

    for b in range(s_expert_starts[e] // 128, s_expert_starts[e + 1] // 128):
        sorted_expert_ids[b] = e
```

The key mutation is `block_offsets`:

```text
before: block_offsets(e, c) = local count from sort_count CTA c
after:  block_offsets(e, c) = absolute start row for sort_place_pad CTA c
```

## Kernel 3: `sort_place_pad`

The third actual GPU launch is:

```python
sort_place_pad(
    topk_ids,
    topk_weight,
    block_offsets,
    real_counts,
    cumsum_tensor,
    sorted_token_ids,
    reverse_sorted,
    sorted_weights,
    m_indices,
).launch(
    grid=(16, 1, 1),
    block=(1024, 1, 1),
)
```

This is the kernel that actually writes the sorted row buffers. It consumes the
absolute starts produced by `sort_cumsum`, places all real routed pairs, then
fills each expert's BM128 padding rows.

Fixed constants:

| item | value |
| --- | ---: |
| `sort_ctas` | 16 |
| threads per CTA | 1024 |
| routed pairs per CTA chunk | 9216 |
| place-loop iterations per thread | 9 |
| `experts_per_cta` for padding | `ceil(385 / 16) = 25` |
| padding lanes | 128 |

```mermaid
flowchart TD
    TOPK["topk_ids<br/>(147456) int32"] --> K3
    TOPKW["topk_weight<br/>(147456) f32"] --> K3
    BO["block_offsets<br/>(385, 16) int32<br/>absolute starts from sort_cumsum"] --> K3
    RC["real_counts<br/>(385) int32"] --> K3
    CS["cumsum_tensor(0)<br/>total padded sorted rows"] --> K3

    subgraph K3["sort_place_pad CTA work"]
        direction TB
        CTA0["one CTA<br/>bx in 0..15<br/>1024 threads"]

        INIT["init expert metadata<br/>threads tx < 385:<br/>local_offsets(tx) = block_offsets(tx, bx)<br/>row_starts(tx) = block_offsets(tx, 0)<br/>thread 0: row_starts(385) = cumsum_tensor(0)"]
        B0["barrier"]

        PLACE["place real rows loop<br/>for idx = bx*9216 + tx; idx < end; idx += 1024"]
        EID["eid = topk_ids(idx)<br/>sp = atomic_add(local_offsets(eid), 1)"]
        PACK["token_id = idx / 9<br/>topk_id = idx % 9<br/>packed = token_id | topk_id << 24"]
        STORE["sorted_token_ids(sp) = packed<br/>m_indices(sp) = token_id<br/>sorted_weights(sp) = topk_weight(idx)<br/>reverse_sorted(idx) = sp"]
        B1["barrier"]

        PADLANE["padding phase<br/>only tx < 128"]
        EXPLOOP["for ee = 0..24<br/>e = bx * 25 + ee<br/>skip if e >= 385"]
        PADRANGE["real_end = row_starts(e) + real_counts(e)<br/>padded_end = row_starts(e + 1)<br/>for j = real_end + tx; j < padded_end; j += 1024"]
        PADSTORE["sorted_token_ids(j) = TOKEN<br/>m_indices(j) = TOKEN<br/>sorted_weights(j) = 0"]

        CTA0 --> INIT --> B0 --> PLACE --> EID --> PACK --> STORE
        STORE --> B1 --> PADLANE --> EXPLOOP --> PADRANGE --> PADSTORE
    end

    STORE --> STI["sorted_token_ids<br/>(196352) int32<br/>real rows packed token/topk"]
    STORE --> MI["m_indices<br/>(196352) int32<br/>real rows token id"]
    STORE --> SW["sorted_weights<br/>(196352) f32<br/>real rows route weight"]
    STORE --> REV["reverse_sorted<br/>(147456) int32<br/>original pair -> sorted row"]
    PADSTORE --> PADOUT["padding rows<br/>token id TOKEN<br/>weight 0"]
```

Equivalent scalar pseudocode:

```python
for bx in parallel_range(16):
    # Shared state for this CTA.
    for e in parallel_range(385):
        local_offsets[e] = block_offsets[e, bx]
        row_starts[e] = block_offsets[e, 0]
    row_starts[385] = cumsum_tensor[0]

    barrier()

    # Place real routed pairs from this CTA chunk.
    start = bx * 9216
    end = min(start + 9216, 147456)
    for tx in parallel_range(1024):
        for idx in range(start + tx, end, 1024):
            eid = topk_ids[idx]
            sp = atomic_add(local_offsets[eid], 1)

            token_id = idx // 9
            topk_id = idx % 9
            packed = (token_id & 0x00FFFFFF) | (topk_id << 24)

            sorted_token_ids[sp] = packed
            m_indices[sp] = token_id
            sorted_weights[sp] = topk_weight[idx]
            reverse_sorted[idx] = sp

    barrier()

    # Fill padding rows for up to 25 experts assigned to this CTA.
    for tx in parallel_range(128):
        for ee in range_constexpr(25):
            e = bx * 25 + ee
            if e < 385:
                real_end = row_starts[e] + real_counts[e]
                padded_end = row_starts[e + 1]
                for j in range(real_end + tx, padded_end, 1024):
                    sorted_token_ids[j] = TOKEN
                    m_indices[j] = TOKEN
                    sorted_weights[j] = 0.0
```

After this kernel, the sort stage is complete:

| output | meaning |
| --- | --- |
| `sorted_token_ids(row)` | sorted-row order, packed token/topk; padding rows use token `TOKEN` |
| `m_indices(row)` | sorted-row order token id, consumed by GEMM1 |
| `sorted_weights(row)` | sorted-row order route weight, consumed by scatter/reduce |
| `reverse_sorted(orig)` | token-major original route index -> sorted row, consumed by scatter/reduce |

## Kernel 4: `quant`

The fourth actual GPU launch is:

```python
quant(hidden, a_quant, a_scale).launch(
    grid=(512, 1, 1),
    block=(1024, 1, 1),
)
```

This kernel quantizes the original bf16 hidden matrix into MXFP4. It does not
use the sorted row order yet.

Work unit:

```text
one MXFP4 block = 32 consecutive hidden values
one MXFP4 block produces 16 packed fp4 bytes + 1 e8m0 scale byte
```

Fixed constants:

| item | value |
| --- | ---: |
| hidden shape | `(16384, 7168)` bf16 |
| blocks per token | `7168 / 32 = 224` |
| total MXFP4 blocks | `16384 * 224 = 3670016` |
| CTAs | 512 |
| threads per CTA | 1024 |
| waves per CTA | `1024 / 64 = 16` |
| MXFP4 blocks per wave | `64 / 4 = 16` |
| MXFP4 blocks per CTA iteration | `16 * 16 = 256` |
| CTA iterations | `3670016 / (512 * 256) = 28` |

```mermaid
flowchart TD
    HS["hidden_states<br/>(16384, 7168) bf16"] --> K4

    subgraph K4["quant CTA work"]
        direction TB
        CTA["one CTA<br/>bx in 0..511<br/>1024 threads"]
        WILOOP["wi range for this CTA<br/>wi = bx*28 .. bx*28+27"]
        MAP["per thread mapping<br/>wave_id = tx / 64<br/>lane = tx % 64<br/>block_in_wave = lane / 4<br/>lane_in_block = lane % 4"]
        BLOCK["my_block = wi*256 + wave_id*16 + block_in_wave<br/>fixed shape covers every block exactly once"]
        LOAD["4-lane group handles one 32-value MXFP4 block<br/>each lane loads 8 bf16 values<br/>as 4 i32 words"]
        AMAX["compute abs max<br/>local lane max over 8 values<br/>quad reduce with shuffle_xor offsets 1 and 2"]
        SCALE["derive e8m0 scale<br/>bexp from max exponent<br/>e8m0 = clamp(bexp - 2, 0, 254)<br/>quant_scale = e8m0 as f32 power-of-two"]
        PACK["pack 8 values for this lane<br/>llvm.amdgcn.cvt.scalef32.pk.fp4.f32<br/>4 pair conversions -> one i32"]
        STOREQ["a_quant word store<br/>q_word = my_block*4 + lane_in_block"]
        STORES["lane_in_block == 0<br/>a_scale byte store<br/>a_scale(my_block) = e8m0"]

        CTA --> WILOOP --> MAP --> BLOCK --> LOAD --> AMAX --> SCALE --> PACK --> STOREQ
        SCALE --> STORES
    end

    STOREQ --> AQ["a_quant<br/>(16384, 3584) uint8"]
    STORES --> AS["a_scale<br/>(16384, 224) uint8"]
```

Equivalent scalar pseudocode:

```python
for bx in parallel_range(512):
    for wi in range(bx * 28, bx * 28 + 28):
        for tx in parallel_range(1024):
            wave_id = tx // 64
            lane = tx % 64
            block_in_wave = lane // 4
            lane_in_block = lane % 4

            my_block = wi * 256 + wave_id * 16 + block_in_wave

            # The code still has a my_block < total_blocks guard, but this
            # fixed shape covers exactly 3670016 blocks.
            base = my_block * 32 + lane_in_block * 8
            vals = load_8_bf16_values(hidden_states, base)

            local_amax = max(abs(vals))
            block_amax = quad_reduce_max(local_amax)
            e8m0 = clamp(exponent(block_amax) - 2, 0, 254)
            quant_scale = e8m0_to_f32_scale(e8m0)

            packed_i32 = pack_8_values_to_fp4(vals, quant_scale)
            a_quant_words[my_block * 4 + lane_in_block] = packed_i32

            if lane_in_block == 0:
                a_scale[my_block] = e8m0
```

Output layout:

| output | meaning |
| --- | --- |
| `a_quant(token, model_byte)` | packed fp4 hidden values; two fp4 values per byte |
| `a_scale(token, model_block32)` | one e8m0 scale per 32 hidden values |

## Kernel 5: `sort_scales`

The fifth actual GPU launch is:

```python
sort_scales(
    a_scale,
    sorted_token_ids,
    cumsum_tensor,
    a_scale_sorted_shuffled,
).launch(
    grid=(512, 1, 1),
    block=(1024, 1, 1),
)
```

This kernel gathers the token-major activation scales into the shuffled layout
that GEMM1 expects. `a_quant` itself is not sorted; GEMM1 uses `m_indices` to
load token rows. The scales are sorted and swizzled ahead of time because the
MFMA scale path consumes them in a compact packed order.

Fixed constants:

| item | value |
| --- | ---: |
| input `a_scale` | `(16384, 224)` uint8 |
| sorted chunks | `max_sorted / 128 = 1534` |
| scale columns per token | `7168 / 32 = 224` |
| row groups per BM128 chunk | `c_m1 = 4` |
| K groups | `c_k1 = 28` |
| output dwords per chunk | `4 * 28 * 4 * 16 = 7168` |
| total work dwords | `1534 * 7168 = 10995712` |
| launch threads | `512 * 1024 = 524288` |
| loop trips per thread upper bound | `ceil(10995712 / 524288) = 21` |
| output bytes | `196352 * 224 * 2 = 87965696` |

```mermaid
flowchart TD
    AS["a_scale<br/>(16384, 224) uint8<br/>token-major"] --> K5
    STI["sorted_token_ids<br/>(196352) int32<br/>sorted row -> packed token/topk"] --> K5
    CS["cumsum_tensor(0)<br/>actual padded sorted rows"] --> K5

    subgraph K5["sort_scales CTA work"]
        direction TB
        CTA["one thread owns a strided sequence of work dwords<br/>global_tid = bx*1024 + tx<br/>work += 512*1024"]
        ZERO["zero output dword first<br/>a_scale_sorted_shuffled(work) = 0"]
        DECODE["decode flat work id<br/>n_lane = work % 16<br/>k_lane = work / 16 % 4<br/>ku = work / 64 % 28<br/>mi = work / 1792 % 4<br/>chunk = work / 7168"]
        VALID["valid chunk check<br/>chunk < ceil(cumsum_tensor(0) / 128)"]
        ROWS["select two sorted rows<br/>row0 = chunk*128 + (mi*2 + 0)*16 + n_lane<br/>row1 = chunk*128 + (mi*2 + 1)*16 + n_lane"]
        TOK["load sorted_token_ids(row0,row1)<br/>extract token id<br/>padding token TOKEN maps to token 0 for safe scale load"]
        GATHER["gather four scale bytes<br/>two rows x two k-pack positions<br/>k_idx = ku*8 + ikxdl*4 + k_lane"]
        PACK["pack bytes into one i32<br/>byte0 row0 k0<br/>byte1 row1 k0<br/>byte2 row0 k1<br/>byte3 row1 k1"]
        STORE["store packed dword<br/>a_scale_sorted_shuffled(work) = packed"]

        CTA --> ZERO --> DECODE --> VALID --> ROWS --> TOK --> GATHER --> PACK --> STORE
    end

    STORE --> ASS["a_scale_sorted_shuffled<br/>flat uint8<br/>GEMM1 scale layout"]
```

Equivalent scalar pseudocode:

```python
actual_sorted = cumsum_tensor[0]
actual_n_chunks = ceil(actual_sorted / 128)

for work in parallel_strided_range(total_work, stride=512 * 1024):
    a_scale_sorted_shuffled_dwords[work] = 0

    n_lane = work % 16
    k_lane = (work // 16) % 4
    ku = (work // 64) % 28
    mi = (work // (16 * 4 * 28)) % 4
    chunk = work // (16 * 4 * 28 * 4)

    if chunk < actual_n_chunks:
        packed = 0

        tok_ids = []
        for im_a in range_constexpr(2):
            sorted_row = chunk * 128 + (mi * 2 + im_a) * 16 + n_lane
            packed_token_topk = sorted_token_ids[sorted_row]
            token = packed_token_topk & 0x00FFFFFF
            token = token if token < TOKEN else 0
            tok_ids.append(token)

        for ikxdl in range_constexpr(2):
            for im_a in range_constexpr(2):
                k_idx = ku * 8 + ikxdl * 4 + k_lane
                byte = a_scale[tok_ids[im_a], k_idx]
                packed |= byte << ((ikxdl * 2 + im_a) * 8)

        a_scale_sorted_shuffled_dwords[work] = packed
```

The packed output is deliberately not a readable `(sorted_row, scale_col)`
matrix. It is a GEMM1 input stream arranged to match the later
`mfma_scale_f32_16x16x128_f8f6f4` scale operand pattern.

## Stage Notes

### 1. Quantize Hidden

`kimi_mxfp4_quant_16384(hidden_states)` reads bf16 hidden states in original
token order and writes:

```text
a_quant: (TOKEN, MODEL_DIM / 2)  = (16384, 3584)
a_scale: (TOKEN, MODEL_DIM / 32) = (16384, 224)
```

Each scale covers 32 bf16 values. The data is still token-major at this point.

### 2. Sort Scales

`a_quant` is not physically sorted before GEMM1. GEMM1 uses `m_indices` to
lookup the source token row. The scale layout is different: GEMM1 expects a
GEMM-friendly shuffled scale stream, so `kimi_mxfp4_sort_scales_16384` gathers
`a_scale[token]` through `sorted_token_ids` and writes
`a_scale_sorted_shuffled`.

The output is not a plain matrix. The wrapper allocates:

```text
padded_rows * (MODEL_DIM / 32) * 2
= 196352 * 224 * 2 bytes
```

### 3. GEMM1

`kimi_mxfp4_gemm1_16384` works in sorted expert-major row order:

```text
for each sorted BM128 block:
    expert = sorted_expert_ids[block]
    token rows = m_indices[sorted rows]
    load hidden MXFP4 from a_quant[token]
    load W_gate and W_up for expert
    compute gate/up
    compute SiLU(gate) * up
    quantize intermediate to MXFP4
```

Outputs:

```text
inter_sorted_quant:          (196352, 256) uint8
inter_sorted_shuffled_scale: (785408, 16) uint8
```

`inter_sorted_shuffled_scale` is already in the scale layout expected by GEMM2,
so it is larger than the logical `(196352, 16)` scale matrix.

### 4. GEMM2 MXFP4OUT

`kimi_mxfp4_gemm2_mxfp4out_16384` keeps the sorted row order:

```text
inter_sorted_quant @ W_down[expert]
```

It does not reduce TOPK and it does not write bf16 staging. It writes an MXFP4
output row per sorted routed row:

```text
flat_out_q:     (196352, 3584) uint8
flat_out_scale: (196352, 224) uint8
```

This is the important difference from the older bf16 staging path. The older
path materialized `[TOKEN, TOPK, MODEL_DIM]` bf16 and then called `torch.sum`.
This path keeps output packed and delays the weighted TOPK reduce to the next
kernel.

### 5. Scatter/Reduce

`kimi_mxfp4_scatter_reduce_q_16384` switches back to token-major output order.
For each token and output column group, it walks `TOPK=9` routes:

```text
orig = token * TOPK + topk_slot
sorted_row = reverse_sorted[orig]
weight = sorted_weights[sorted_row]
value = dequant(flat_out_q[sorted_row], flat_out_scale[sorted_row])
acc += value * weight
```

The final result is:

```text
out: (16384, 7168) bf16
```

## Reading Order In Code

Use this order if you want to understand or edit the file:

1. `run_kimi_fp4_mxfp4_moe_16384_all_flydsl`
2. `kimi_mxfp4_sort_16384`
3. `compile_kimi_mxfp4_sort_16384`
4. `kimi_mxfp4_quant_16384`
5. `kimi_mxfp4_sort_scales_16384`
6. `kimi_mxfp4_gemm1_16384`
7. `kimi_mxfp4_gemm2_mxfp4out_16384`
8. `kimi_mxfp4_scatter_reduce_q_16384`

When debugging correctness, keep these invariants in mind:

| invariant | why it matters |
| --- | --- |
| `cumsum_tensor[0] <= max_sorted` | actual padded rows must fit allocated buffers |
| `sorted_expert_ids` is per BM128 block | GEMM kernels pick expert weights by block |
| padding token id is `TOKEN` | GEMM kernels can guard invalid rows |
| `reverse_sorted` maps original pair to sorted row | scatter/reduce needs original token-major route order |
| `sorted_weights` is sorted-row aligned | final reduce multiplies each decoded row by the route weight |
