import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union
from jaxtyping import Float, Int, Bool
from torch.nn.attention import SDPBackend, sdpa_kernel

_NO_CACHE = object()  # sentinel: no KV-caching at all

def softmax(x: torch.Tensor, dim: int, temperature: Optional[float] = None) -> torch.Tensor:
    """Implement numerically stable softmax from scratch"""
    max_val = x.max(dim=dim, keepdim=True).values
    if temperature is None:
        z = x - max_val
    else:
        t = torch.as_tensor(temperature, dtype=x.dtype, device=x.device)
        z = (x - max_val) / (t + 1e-5)
    x_exp = z.exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Float[Tensor, "batch_size ... seq_len head_dim"],
                                 K: Float[Tensor, "batch_size ... seq_len head_dim"],
                                 V: Float[Tensor, "batch_size ... seq_len head_dim"],
                                 mask: Optional[Tensor] = None) -> Float[Tensor, "batch_size ... seq_len head_dim"]:
    """Implement scaled dot product attention from scratch"""
    d_k = Q.shape[-1]
    scale = 1.0 / (d_k ** 0.5)

    # Affinities: [..., seq_len, seq_len]
    attn_logits = Q @ K.transpose(-2, -1)

    if mask is not None:
        # Expect mask shape [seq_len, seq_len] (True = keep), broadcast to leading dims
        mask = mask.to(torch.bool)
        while mask.dim() < attn_logits.dim():
            mask = mask.unsqueeze(0)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

    attn = softmax(attn_logits * scale, dim=-1) # [..., seq_len, seq_len]
    out = attn @ V
    return out


class Linear(nn.Module):
    """Implement a linear layer from scratch"""
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.weights: Float[Tensor, "out_features in_features"] = \
            nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        with torch.no_grad():
            init_std = (2 / (out_features + in_features)) ** 0.5
            nn.init.trunc_normal_(self.weights, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return x @ self.weights.T


class Embedding(nn.Module):
    """Implement an embedding layer from scratch"""
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.weights: Float[Tensor, "vocab_size d_model"] = \
            nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        with torch.no_grad():
            init_std = 1.0 / (d_model ** 0.5)
            nn.init.trunc_normal_(self.weights, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

    def forward(self, token_ids: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        return self.weights[token_ids.long()]
    

class RMSNorm(nn.Module):
    """Implement a RMSNorm layer from scratch"""
    def __init__(self,
                 d_model: int,
                 eps: Optional[float] = 1e-5,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights: Float[Tensor, "d_model"] \
            = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        # Upcast to prevent overflow when squaring
        x = x.to(torch.float32)

        inv_rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # Cast back before weight multiply to reduce peak memory
        x = (x * inv_rms).to(in_dtype) * self.weights
        return x
    
class SiLU(nn.Module):
    """Implement the SiLU activation function from scratch"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.silu(x)


class SwiGLU(nn.Module):
    """Implement a SwiGLU feedforward network from scratch"""
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        up_projection = self.up(x)
        silu_activations = self.silu(up_projection)
        gated_projection = self.gate(x)
        swiglu_activations = silu_activations * gated_projection
        down_projection = self.down(swiglu_activations)
        return down_projection
    

class RotaryPositionalEmbedding(nn.Module):
    """Implement RoPE from scratch"""
    def __init__(self, 
                 theta: float,
                 head_dim: int,
                 max_seq_len: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, d_k/2]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)  # [max_seq_len, d_k/2]
        self.register_buffer("sin_cached", sin, persistent=False)  # [max_seq_len, d_k/2]

    def forward(self, 
                x: Float[Tensor, "... seq_len d_k"],
                token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        # Bounds check (tensor-only; compile-friendly)
        max_pos_i64 = token_positions.to(torch.int64).max()
        limit = torch.tensor(self.max_seq_len, device=token_positions.device, dtype=torch.int64)
        torch._assert(max_pos_i64 < limit, "token_positions exceed max_seq_len used to precompute RoPE tables.")

        cos = self.cos_cached[token_positions.long()] # [..., seq, d_k/2]
        sin = self.sin_cached[token_positions.long()] # [..., seq, d_k/2]
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x_even = x[..., 0::2] # [..., seq, d_k/2]
        x_odd  = x[..., 1::2] # [..., seq, d_k/2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_odd  * cos + x_even * sin

        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2) # [..., seq, d_k]
        return out
    

class MultiHeadSelfAttention(nn.Module):
    """Implement causal multi-head attention from scratch"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int = 2048,
                 use_sdpa: bool = False,
                 rope: Optional["RotaryPositionalEmbedding"] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)

        self.rope = rope
        if self.rope is not None:
            assert self.rope.head_dim == self.head_dim, "RoPE d_k must equal head_dim"

        self.use_sdpa = use_sdpa

        # Pre-allocate causal mask (avoids per-forward allocation)
        causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self,
                x: Float[Tensor, "batch seq_len d_model"],
                token_positions: Optional[Int[Tensor, "batch seq_len"]] = None,
                kv_cache = _NO_CACHE) -> Union[Float[Tensor, "batch seq_len d_model"], tuple]:
        b, s, _ = x.shape
        use_cache = kv_cache is not _NO_CACHE

        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]

        if self.rope is not None:
            pos = token_positions
            if pos is None:
                offset = kv_cache[0].shape[2] if (use_cache and kv_cache is not None) else 0
                pos = torch.arange(offset, offset + s, device=x.device)
            if pos.dim() == 1: # [s] to [b, s]
                pos = pos.unsqueeze(0).expand(b, s)
            pos = pos.to(device=x.device)
            pos_b1s = pos.unsqueeze(1) # [b, 1, s], broadcasts across heads
            q = self.rope(q, pos_b1s)
            k = self.rope(k, pos_b1s)

        # KV-cache: concatenate with cached keys/values
        if use_cache and kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
                ) # [b, h, s, d]
        else:
            total_len = k.shape[2]
            mask = self.causal_mask[total_len - s : total_len, :total_len]
            attn_out = scaled_dot_product_attention(q, k, v, mask) # [b, h, s, d]

        out = attn_out.transpose(1, 2).reshape(b, s, self.d_model)
        out = self.o_proj(out)

        if use_cache:
            return out, (k, v)
        return out
        

class TransformerBlock(nn.Module):
    """Implement a transformer block from scratch"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 max_seq_len: int,
                 use_sdpa: bool = False,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.rope = RotaryPositionalEmbedding(theta=theta, head_dim=self.head_dim, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, rope=self.rope, device=device, dtype=dtype, use_sdpa=use_sdpa)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, kv_cache=_NO_CACHE):
        if kv_cache is not _NO_CACHE:
            attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, new_kv
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    """Implement a complete Transformer LM from scratch"""
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 use_sdpa: bool = True,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.num_layers = num_layers
        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                theta=self.theta,
                max_seq_len=self.context_length,
                device=device,
                dtype=dtype,
                use_sdpa=use_sdpa
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)
        # Weight tying
        self.lm_head.weights = self.token_embeddings.weights

    def forward(self, in_tokens: Float[Tensor, "batch_size seq_len"],
                kv_caches: Optional[list] = None) -> Union[Float[Tensor, "batch_size seq_len vocab_size"], tuple]:
        x = self.token_embeddings(in_tokens)

        if kv_caches is not None:
            new_caches = []
            for i, block in enumerate(self.layers):
                cache = kv_caches[i] if i < len(kv_caches) else None
                x, new_cache = block(x, kv_cache=cache)
                new_caches.append(new_cache)
            logits = self.lm_head(self.ln_final(x))
            return logits, new_caches

        for block in self.layers:
            x = block(x)
        logits = self.lm_head(self.ln_final(x))
        return logits