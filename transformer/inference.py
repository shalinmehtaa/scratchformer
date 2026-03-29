import torch
import argparse
from typing import List, Optional
from transformer.optimizer import AdamW
from transformer.checkpoint import load_checkpoint
from transformer.model import Transformer, softmax
from tokenization.bpe.tokenizer import Tokenizer


def _map_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = dtype_str.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    # logits: [B, V]
    if top_k is not None and top_k > 0:
        topk_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        kth_vals = topk_vals[..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)

        # Create a mask for tokens to remove: everything after the nucleus
        remove = cumprobs > top_p
        # Shift right to always keep the first token above threshold
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # Scatter back to original indices
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits


def generate(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    num_samples: int = 1,
    eos_token: Optional[str] = "<|endoftext|>",
    device: str = "cpu",
) -> List[str]:
    model.eval()
    context_len = model.context_length

    # Encode prompt and prepare batch
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = tokenizer.encode(" ")

    eos_id: Optional[int] = None
    if eos_token is not None:
        eos_encoded = tokenizer.encode(eos_token)
        if len(eos_encoded) == 1:
            eos_id = eos_encoded[0]

    tokens_full = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    tokens_full = tokens_full.repeat(num_samples, 1)  # [B, S]

    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

    with torch.inference_mode():
        # Prefill: process the full prompt to populate KV caches
        input_tokens = tokens_full[:, -context_len:]
        logits, kv_caches = model(input_tokens)
        next_logits = logits[:, -1, :]  # [B, V]
        cur_pos = input_tokens.shape[1]  # track absolute position for RoPE

        for _ in range(max_new_tokens):
            # Apply top-k/top-p on raw logits, then temperature in softmax
            filtered_logits = _top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = softmax(filtered_logits, dim=-1, temperature=temperature)  # [B, V]
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            if eos_id is not None:
                finished |= (next_token.squeeze(1) == eos_id)

            tokens_full = torch.cat([tokens_full, next_token], dim=1)

            if finished.all():
                break

            # Decode step: only process the new token using KV cache
            # If we've exceeded context_len, drop the oldest cache entries
            if cur_pos >= context_len:
                # Trim KV caches to context_len - 1 to make room
                kv_caches = [
                    (k[:, :, -(context_len - 1):, :], v[:, :, -(context_len - 1):, :])
                    for k, v in kv_caches
                ]
                cur_pos = context_len - 1

            logits, kv_caches = model(next_token, kv_caches=kv_caches)
            next_logits = logits[:, -1, :]  # [B, V]
            cur_pos += 1

    # Decode per sample (trim at first EOS if present)
    outputs: List[str] = []
    for b in range(num_samples):
        ids = tokens_full[b].tolist()
        if eos_id is not None and eos_id in ids[len(prompt_ids):]:
            # Only trim in the generated suffix
            suffix = ids[len(prompt_ids):]
            cut = suffix.index(eos_id)
            ids = ids[: len(prompt_ids) + cut]
        outputs.append(tokenizer.decode(ids))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Text generation with scratchformer Transformer.")
    # I/O
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to consolidated checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Text prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional file containing the prompt")

    # Sampling
    parser.add_argument("--max_tokens", type=int, default=64, help="Max new tokens to sample")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) filtering (1.0 = disabled)")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>", help="EOS token text; set empty to disable")

    # System
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--dtype", type=str, default=None, help="float32, bfloat16, float16")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Model config (must match the checkpoint)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--use_sdpa", action="store_true", help="Use PyTorch SDPA attention kernels")

    args = parser.parse_args()

    # Resolve device/dtype
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = _map_dtype(args.dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_filepath=args.tokenizer)

    # Build model with config matching the checkpoint
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device,
        dtype=dtype,
        use_sdpa=args.use_sdpa,
    )
    model.eval()

    # Create an optimizer instance to satisfy checkpoint loading API
    optimizer = AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )

    # Load checkpoint (weights and, if present, optimizer state)
    _ = load_checkpoint(
        src=args.checkpoint,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    # Read prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

    eos_text = None if (args.eos_token is None or args.eos_token == "") else args.eos_token

    # Generate
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.samples,
        eos_token=eos_text,
        device=device,
    )

    # Print
    for i, text in enumerate(outputs, 1):
        print(f"\n=== Sample {i} ===")
        print(text)


if __name__ == "__main__":
    main()