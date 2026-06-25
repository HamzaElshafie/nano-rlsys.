import torch
import torch.nn as nn

def apply_rotary_emb(x, cos, sin):
    pass

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 2 == 0
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # shape: (head_dim/2,)
        inv_freqs = 1.0 / (base**(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # shape: (max_position_embeddings,)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        # shape: (max_position_embeddings,1) * (1,head_dim/2) = (max_position_embeddings, head_dim/2)
        freqs = t[:, None] * inv_freqs[None, :]
        # shape: (max_position_embeddings, head_dim/2)
        cos = freqs.cos()
        sin = freqs.sin()
        # shape: (max_position_embeddings, 1, head_dim)
        cache = torch.cat((cos, sin), dim=-1).unsqueeze(1)
        self.register_buffer("cos_sin_cache", cache)
    
    def forward(
        self, 
        positions: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # retrieve cache - shape: (max_position_embeddings, 1, head_dim)
        cos_sin = self.cos_sin_cache[positions]
        # each has shape: (max_position_embeddings, 1, head_dim // 2)
        cos, sin = cos_sin.chunk(2, dim=-1)
        # rotate query and key
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


        
