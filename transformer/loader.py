import os
import torch
import numpy as np
from torch import Tensor
from typing import Tuple
from jaxtyping import Int
from numpy.typing import NDArray


def get_batch(x: NDArray,
              batch_size: int, 
              context_length: int, 
              device: str) -> Tuple[Int[Tensor, "batch_size seq_len"], 
                                    Int[Tensor, "batch_size seq_len"]]:
    """Sample a batch of contiguous token sequences from a 1D numpy array (or np.memmap)."""
    if x.ndim != 1:
        raise ValueError("Expected a 1D array of token IDs.")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for token IDs, got {x.dtype}.")
    n = x.shape[0]
    if n <= context_length:
        raise ValueError(
            f"Dataset too small: n={n} must be > context_length={context_length}."
        )

    # Sample start indices so that i+context_length is within bounds for inputs
    starts = np.random.randint(0, n - context_length, size=batch_size, dtype=np.int64)

    # Build index matrix for inputs and targets using broadcasting
    offsets = np.arange(context_length, dtype=np.int64)[None, :]  # (1, m)
    idx_in = starts[:, None] + offsets                            # (B, m)
    idx_tg = idx_in + 1                                           # (B, m)

    # Fancy-index into x; works for memmap and in-memory arrays
    inputs_np  = x[idx_in]
    targets_np = x[idx_tg]

    # Convert to torch.long; use non-blocking transfer to overlap with compute on CUDA
    inputs  = torch.from_numpy(inputs_np.astype(np.int64))
    targets = torch.from_numpy(targets_np.astype(np.int64))
    if device and "cuda" in str(device):
        inputs  = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs  = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets


def load_bin_array(path: str, dtype: str = "uint16") -> NDArray:
    """
    Load a 1D array of token IDs from a .bin file as a memmap.
    Assumes raw uint16 (or provided dtype) little-endian data.
    """
    dt = np.dtype(dtype)
    if dt.kind not in ("i", "u"):
        raise ValueError(f"dtype must be integer for .bin, got {dt}")
    n_bytes = os.path.getsize(path)
    if n_bytes % dt.itemsize != 0:
        raise ValueError(f"File size {n_bytes} is not a multiple of dtype size {dt.itemsize}")
    length = n_bytes // dt.itemsize
    return np.memmap(path, mode="r", dtype=dt, shape=(length,))


if __name__ == "__main__":

     # Train/valid paths (see data/inputs)
    train_path = "data/inputs/tinystories-train-tokens.bin"
    valid_path = "data/inputs/tinystories-valid-tokens.bin"

    # Memory-map the arrays to avoid loading into RAM
    x_train = load_bin_array(train_path, dtype="uint16")
    x_valid = load_bin_array(valid_path, dtype="uint16")

    # Quick sanity check on dtype
    assert np.issubdtype(x_train.dtype, np.integer)

    # Sample a batch
    batch_size = 32
    context_length = 256
    device = "cpu"  # or "cuda:0"
    inputs, targets = get_batch(x_train, batch_size, context_length, device)

    print(f"{inputs.shape}")
    print(f"{targets.shape}")
    print("Inputs sample:", inputs[0, :10])
    print("Targets sample:", targets[0, :10])
