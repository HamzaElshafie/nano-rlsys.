import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-06,
    ) -> None:
        """See RMSNorm paper https://arxiv.org/pdf/1910.07467
        
        Formula: 
                RMSNorm(a) = (a / RMS(a)) * scale 
                where RMS(a) = sqrt(mean(x^2) + eps)
                mean is across the model `hidden_size` dimension
        """
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # shape: (batch, seq_len, hidden_size) (BF16)
        assert x.shape[-1] == self.hidden_size
        # cast to fp32 for numerical stability
        x_fp32, dtype = x.float(), x.dtype
        # mathematically, x/sqrt(v) can be written as x * 1/sqrt(v)
        x_fp32 = x_fp32 * torch.rsqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)
        # shape: (batch, seq_len, hidden_size) (BF16)
        return (x_fp32 * self.weight).to(dtype)