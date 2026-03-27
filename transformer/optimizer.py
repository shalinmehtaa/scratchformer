import math
import torch
import torch.distributed as dist
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Tuple


def cross_entropy(logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """Numerically stable cross-entropy in float32."""
    # Upcast to float32 for stable exp/log even if logits came from bf16 autocast
    logits_f32 = logits.float()
    # logsumexp across vocab
    lse = torch.logsumexp(logits_f32, dim=-1)
    # Gather the target logits
    tgt = logits_f32.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = lse - tgt
    return nll.mean()


def learning_rate_scheduler(t, lr_max, lr_min, t_warmup, t_cos):
    """Cosine annealing learning rate schedule"""
    if t <= t_warmup:
        lr = (t / t_warmup) * lr_max
    elif t_warmup < t < t_cos:
        lr = lr_min + 0.5 * (1 + math.cos(((t - t_warmup) / (t_cos - t_warmup)) * math.pi)) * (lr_max - lr_min)
    else:
        lr = lr_min
    return lr


def gradient_clipping(params, max_l2_norm: float, eps: Optional[float] = 1e-6):
    """Implement gradient clipping"""
    total_norm_sq = torch.tensor(0.0, device=params[0].device)
    for p in params:
        if p.grad is None:
            continue
        total_norm_sq += p.grad.detach().float().pow(2).sum()

    # Use global norm across ranks if distributed
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)

    total_norm = total_norm_sq.sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)
    return total_norm


class AdamW(torch.optim.Optimizer):
    """Implement AdamW optimizer from scratch"""
    def __init__(self, 
                 params: Float[Tensor, "..."],
                 lr: float, 
                 betas: Tuple[float],
                 weight_decay: float,
                 eps: Optional[float] = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be > 0.")
        self.eps = eps
        defaults = {
            "lr": lr, 
            "betas": betas,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Get the optimizer params
            lr = group["lr"] 
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Get state associated with p
                state = self.state[p]
                # Initialize state tensors on first step (avoids scalar→tensor overhead)
                if "t" not in state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # Get the gradient for p
                grad = p.grad.data
                # Compute moments and lr (in-place where possible)
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))

                # Update parameter in place with weight decay
                p.data.add_(m / (v.sqrt() + self.eps), alpha=-lr_t)
                p.data.mul_(1 - lr * weight_decay)

                state["t"] = t + 1
                
        return loss
    