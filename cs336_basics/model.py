import math
from einops import einsum, rearrange
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
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def silu_activation(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu_activation(self.W1(x)))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(SiLU.silu_activation(self.W1(x)) * self.W3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None=None):
        super().__init__()
        
        # pre-compute the cos and sin values
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freqs = 1.0 / (theta**freqs)
        angles = positions * inv_freqs

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = rearrange([x_rot_even, x_rot_odd], "two ... d_k -> ... (d_k two)")

        return x_rot
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Stable softmax implementation."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute scaled dot-product attention."""
    d_k = Q.size(-1)
    scores = einsum(Q, K, '... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k')
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, '... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v')

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_QKV = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self,
                x: torch.Tensor, 
                rope: RotaryPositionalEmbedding | None = None,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(-2)
        
        # qkv projection
        qkv = self.W_QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        # apply rotary positional embedding for q, k
        if rope is not None and token_positions is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)
            
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        y = scaled_dot_product_attention(q, k, v, mask)
        y = rearrange(y, "... h s d -> ... s (h d)")

        return self.W_O(y)