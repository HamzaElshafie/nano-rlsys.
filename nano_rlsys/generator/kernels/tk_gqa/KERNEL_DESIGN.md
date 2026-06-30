# Kernel Design

## 1. Prefill Full Flow

<!-- PUT PREFILL FULL FLOW visual -->
<img width="4929" height="2739" alt="prefill-full-flow excalidraw" src="https://github.com/user-attachments/assets/473427fd-6f5c-490d-98cf-22fc89dc443f" />

<br><br>

During prefill, the model receives a prompt such as:

```text
"What is the largest ocean?"
```

The input hidden states have shape:

```text
X: [seq_len, d_model]
```

The attention layer first projects the hidden states into query, key, and value tensors:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

Conceptually, attention computes:

```text
S = Q @ K^T
P = softmax(S)
O = P @ V
```

For decoder-only models, this attention is causal, so each token can only attend to itself and previous tokens.

In an efficient FlashAttention style kernel, the full score matrix `S` and probability matrix `P` are **not materialised in global memory**. Instead, the kernel processes one query tile `Q_i` against one key/value tile `K_j, V_j` at a time.

For each query tile `Q_i` (outer loop), the kernel streams over KV tiles (inner loop):

```text
for each K_j, V_j tile:
    S_ij = Q_i @ K_j^T
    apply causal mask if needed
    update online softmax state
    update O_i accumulator with P_ij @ V_j
```

The running state per query row is:

```text
m_i = running max score
l_i = running softmax denominator
O_i = running output numerator / accumulator
```

At the end of the KV loop:

```text
O_i = O_i / l_i
```

The important idea is that `S_ij` and `P_ij` are temporary tile-level register objects. The kernel keeps only the online softmax statistics and the output accumulator.

## 2. Prefill CTA Parallelism Over Query Tiles

<!-- PUT PREFILL CTA PARALLELISM OVER Q TILES (PERSISTENT KV TILES) visual -->
<img width="4211" height="1914" alt="prefill-parallelism excalidraw" src="https://github.com/user-attachments/assets/b55fdf06-8e4a-468f-b67a-177de8f53e3a" />

<br><br>


In prefill, there are many query tokens. Therefore the natural parallelisation strategy is over query tiles.

A simple mental model is:

```text
CTA 0 owns query tile Q_0 and output tile O_0
CTA 1 owns query tile Q_1 and output tile O_1
CTA 2 owns query tile Q_2 and output tile O_2
...
```

Each CTA streams through the K/V sequence:

```text
CTA 0:
    load Q_0
    loop over K_j/V_j tiles
    accumulate O_0

CTA 1:
    load Q_1
    loop over K_j/V_j tiles
    accumulate O_1
```

So the CTA ownership is:

```text
CTA owns a Q tile / O tile.
CTA does not uniquely own the K/V tensors.
```

K/V tiles are streamed by each CTA as needed. The same K/V tile may be loaded by multiple CTAs while computing different query tiles.

This is why prefill has enough parallelism: there are usually many query tiles, so many CTAs can work independently.

## 3. Decode Full Flow

<!-- PUT DECODE FULL FLOW visual -->
<img width="4929" height="2870" alt="decode-full-flow excalidraw" src="https://github.com/user-attachments/assets/b901bcfe-fef7-4264-a635-d137a8508881" />

<br><br>

During decode, the model generates one or a few new tokens at a time. For a single new token, the hidden state is projected into:

```text
q_new = x_new @ Wq
k_new = x_new @ Wk
v_new = x_new @ Wv
```

The new `k_new` and `v_new` are appended to the KV cache. The query `q_new` is used immediately to attend over the cached sequence.

The decode attention computation is:

```text
O = softmax(q_new @ K_cache^T) @ V_cache
```

The important difference from prefill is:

- Q is tiny. Usually 1, unless speculative decoding is used perhaps. (Acc I am not sure a kernel used after speculative decoding should be designed more like a prefill with paralellsim over queries or still with the split-KV decode design, maybe I can explore this later).
- K/V cache can be very long.

For example:

```text
Q:       [1, q_heads, head_dim]
K_cache: [kv_seq_len, kv_heads, head_dim]
V_cache: [kv_seq_len, kv_heads, head_dim]
```

The query block is usually small, but the KV cache sequence is long and paged. The kernel therefore streams K/V tiles from the cache while maintaining online softmax state, just like FlashAttention.

However, a single CTA walking the full KV sequence may not expose enough parallelism, especially when batch size is small. This motivates split-KV parallelism.

## 4. Decode CTA KV-Split Parallelism

<!-- PUT DECODE CTA KV-SPLIT PARALLELISM VISUAL -->
<img width="2262" height="1189" alt="decode-parallelism excalidraw" src="https://github.com/user-attachments/assets/4bbc2596-8e13-4d42-a7b2-4de2b00f9eb6" />

<br><br>

Split-KV parallelism adds a new parallel dimension: the KV sequence length.

Instead of one CTA processing the entire KV cache sequence, the KV sequence is divided into larger ranges:

```text
KV range 0 (split 0) -> CTA 0
KV range 1 (split 1) -> CTA 1
KV range 2 (split 2) -> CTA 2
...
```

Each CTA receives the same query block but attends only over its assigned KV range.

For each split:

```text
CTA s:
    load Q
    loop over K/V tiles inside KV range s
    compute partial attention over that range
    write partial output O_s
    write log-sum-exp / lvec_s
```

Inside each split, the CTA still has an inner loop over smaller K/V tiles just like with flash attention:

```text
for each K/V tile in this split:
    S_tile = Q @ K_tile^T
    update online softmax state
    update O accumulator with P_tile @ V_tile
```

So there are two granularities:

```text
large split range:
    assigned to a CTA / partial task

small K/V tile:
    streamed inside that CTA's inner loop
```

Each partial output is not enough by itself. A split only saw part of the KV sequence, so it produces:

```text
O_s    = attention output over split s
lvec_s = log-sum-exp / normalization state for split s
```

The final output is computed by reducing the partial outputs using the log-sum-exp values:

```text
l_total = logsumexp(lvec_0, lvec_1, ..., lvec_n)

O_final =
    exp(lvec_0 - l_total) * O_0
  + exp(lvec_1 - l_total) * O_1
  + ...
  + exp(lvec_n - l_total) * O_n
```

This reduction is necessary because softmax normalization is global over the full KV sequence.

In our kernel (inspired by ThunderGQA):

```text
partial_template:
    computes one KV split/range
    writes O_scratch and Lvec_scratch
    marks Semaphore ready

reduction_template:
    waits for partial outputs
    loads O_scratch and Lvec_scratch
    performs logsumexp-weighted reduction
    writes final O or another scratch output
```

The high-level algorithm is:

```text
same Q block
    -> multiple KV splits in parallel
    -> partial O + lvec per split
    -> reduction over splits
    -> final O
```

## 5. Initial Kernel Design

### Partial Split Algorithm

Require:
    Query block Q in R^{T_q x G x d}
    Key cache K in R^{N x d}
    Value cache V in R^{N x d}
    KV split range [s, e)
    KV tile size B_c
    Number of KV tiles T_c = ceil((e - s) / B_c)
    Softmax scale tau

where:
    T_q <= 4 is the number of packed query tokens handled by this partial CTA.
    G is the number of query heads sharing one KV head.
    d is the head dimension.
    N is the current KV cache length.
    [s, e) is the contiguous KV range assigned to this CTA.
    B_c is the number of cached KV tokens loaded per iteration.

Implementation view:
    Q is packed into Q_tile in R^{64 x d}.
    Each query token occupies 16 row slots.
    Row slots 0..G-1 within each token slice are real.
    Remaining row slots are padding/inactive.