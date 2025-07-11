import math
from einops import einsum
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()

        mean = 0.0
        std = math.sqrt(2 / (out_features + in_features))
        a = -3 * std
        b = 3 * std

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=mean, std=std, a=a, b=b)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, 'd_out d_in, ... d_in -> ... d_out')
    
class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()

        mean = 0.0
        std = 1.0
        a = -3
        b = 3

        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=mean, std=std, a=a, b=b)
        self.weight = nn.Parameter(w)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    
    # (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x * self.scale / rms

        return result.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SWiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.SiLU = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.SiLU(self.W1(x)) * self.W3(x))