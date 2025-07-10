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
        self.W = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, 'd_out d_in, ... d_in -> ... d_out')
    
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
        self.W = nn.Parameter(w)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]