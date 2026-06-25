#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int head_dim = 128;
static constexpr int half_head_dim = head_dim/2;
static constexpr int kv_tile_tokens = 32;
static constexpr int page_size = 256;

// smem tile for Q
using q_tile = st_bf<64, head_dim>;
// gmem descriptor for Q
// shape: (batch, num_new_tokens, num_q_heads, head_dim)
using q_global = gl<bf16, -1, -1, -1, head_dim, q_tile>;

