#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int head_dim = 128;
static constexpr int kv_tile_tokens = 32;

// smem tile for Q
using q_tile = st_bf<64, head_dim>;
// gmem descriptor for Q
// shape: (batch, num_new_tokens, num_q_heads, head_dim)
using q_global = gl<bf16, -1, -1, -1, head_dim, q_tile>;
// smem tiles for K and V
using kcache_tile = st_bf<kv_tile_tokens, head_dim>;
using vcache_tile = st_bf<kv_tile_tokens, head_dim>;
// gmem descriptor for K/V
// shape: (batch, cached_seq_len, head_dim)
using kcache_global = gl<bf16, -1, -1, head_dim, kcache_tile>;
using vcache_global = gl<bf16, -1, -1, head_dim, vcache_tile>;
// smem tile for O (output tile) 
using o_tile = st_bf<64, head_dim>;
// gmem descriptor for O
// shape: (batch, num_new_tokens, num_q_heads, head_dim)
using o_store_tile = st_bf<16, head_dim>;
using o_global = gl<bf16, -1, -1, -1, head_dim, o_partial_store_tile, o_store_tile>;