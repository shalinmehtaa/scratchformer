import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union
from jaxtyping import Float, Int, Bool
from torch.nn.attention import SDPBackend, sdpa_kernel

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
        x = x * inv_rms
        x = x * self.weights
        x = x.to(in_dtype)
        return x
    
class SiLU(nn.Module):
    """Implement the SiLU activation function from scratch.

    NOTE: For production use, prefer F.silu() which fuses into a single
    CUDA kernel. This class is kept for educational reference.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.silu(x)


class SwiGLU(nn.Module):
    """Implement a SwiGLU feedforward network from scratch.

    The up and gate projections are fused into a single Linear(d_model, 2*d_ff)
    so they execute as one matmul, saving a kernel launch and memory round-trip.
    """
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_ff = d_ff
        self.up_gate = Linear(d_model, 2 * d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        up_and_gate = self.up_gate(x)  # [..., 2*d_ff]
        up_projection, gated_projection = up_and_gate.split(self.d_ff, dim=-1)
        swiglu_activations = F.silu(up_projection) * gated_projection
        return self.down(swiglu_activations)
    

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
        # Bounds check removed from hot path — the precomputed buffer size
        # already constrains valid positions; an out-of-bounds index will
        # raise an IndexError naturally.

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
    """Implement causal multi-head attention from scratch.

    Supports an optional KV cache for efficient autoregressive inference.
    During training, pass ``kv_cache=None``. During inference, pass the
    previous cache (or an empty tuple ``(None, None)`` on the first step)
    and receive the updated cache back.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 use_sdpa: bool = True,
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

    def forward(self,
                x: Float[Tensor, "batch seq_len d_model"],
                token_positions: Optional[Int[Tensor, "batch seq_len"]] = None,
                kv_cache: Optional[tuple] = None) -> tuple:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model].
            token_positions: Absolute positions for RoPE [batch, seq_len].
            kv_cache: Tuple of (cached_k, cached_v) from previous step, or None.

        Returns:
            (output, new_kv_cache) where new_kv_cache is None during training.
        """
        b, s, _ = x.shape

        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) # [b, h, s, d]

        if self.rope is not None:
            pos = token_positions
            if pos is None:
                pos = torch.arange(s, device=x.device)
            if pos.dim() == 1: # [s] to [b, s]
                pos = pos.unsqueeze(0).expand(b, s)
            pos = pos.to(device=x.device)
            pos_b1s = pos.unsqueeze(1) # [b, 1, s], broadcasts across heads
            q = self.rope(q, pos_b1s)
            k = self.rope(k, pos_b1s)

        # KV cache: append current k, v to cached values
        new_kv_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            new_kv_cache = (k, v)

        if self.use_sdpa:
            # Prefer Flash Attention and Efficient Attention kernels;
            # MATH backend is excluded to avoid falling back to the slow path.
            # is_causal is only valid when q and k have the same seq length
            is_causal = (kv_cache is None) and (q.shape[2] == k.shape[2])
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                ) # [b, h, s, d]
        else:
            # Fallback: use cached causal mask to avoid per-call allocation
            total_s = k.shape[2]
            if not hasattr(self, '_causal_mask') or self._causal_mask.shape[0] < total_s:
                self._causal_mask = torch.ones(total_s, total_s, dtype=torch.bool, device=x.device).tril()
            # For KV cache: query attends to all past + current keys
            if kv_cache is not None:
                mask = self._causal_mask[total_s - s:total_s, :total_s]
            else:
                mask = self._causal_mask[:s, :s]
            attn_out = scaled_dot_product_attention(q, k, v, mask) # [b, h, s, d]

        out = attn_out.transpose(1, 2).reshape(b, s, self.d_model)
        return self.o_proj(out), new_kv_cache
        

class TransformerBlock(nn.Module):
    """Implement a transformer block from scratch"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 max_seq_len: int,
                 use_sdpa: bool = True,
                 device: Optional[Union[str, torch.device]] = None, 
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.rope = RotaryPositionalEmbedding(theta=theta, head_dim=self.head_dim, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=self.rope, device=device, dtype=dtype, use_sdpa=use_sdpa)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache

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
            ) for _ in torch.arange(num_layers, device=device)
        ])
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)
        # Weight tying
        self.lm_head.weights = self.token_embeddings.weights  

    def forward(self, in_tokens: Float[Tensor, "batch_size seq_len"],
                kv_caches: Optional[list] = None) -> tuple:
        """
        Args:
            in_tokens: Token IDs [batch, seq_len].
            kv_caches: List of per-layer KV caches, or None for training.

        Returns:
            (logits, new_kv_caches) — new_kv_caches is None during training.
        """
        x = self.token_embeddings(in_tokens)
        new_kv_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=layer_cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)
        x_norm = self.ln_final(x)
        logits = self.lm_head(x_norm)

        return logits, new_kv_caches