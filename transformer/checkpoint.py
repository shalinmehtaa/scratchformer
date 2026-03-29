import os
import json
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, 
    StateDictType, 
    FullStateDictConfig,
    LocalStateDictConfig,
)
from typing import Union, BinaryIO, IO

# Only available in recent PyTorch
try:
    from torch.distributed.fsdp import OptimStateDictConfig, OptimStateDictType
    _HAS_SHARDED_OPTIM = True
except Exception:
    _HAS_SHARDED_OPTIM = False

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def _is_fsdp_model(m: Module) -> bool:
    return isinstance(m, FSDP)

def save_checkpoint(model: Module, optimizer: Optimizer, iteration: int, out: PathOrFile, weights_only: bool = False) -> None:
    """Save consolidated model/optimizer on rank 0 (large). Use weights_only=True to skip optimizer."""
    obj = {"iteration": int(iteration), "pytorch_version": torch.__version__}
    if _is_fsdp_model(model):
        if not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
            return  # only rank 0 writes
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            obj["model"] = model.state_dict()
        if not weights_only:
            obj["optimizer"] = optimizer.state_dict()
        torch.save(obj, out)
    else:
        obj["model"] = model.state_dict()
        if not weights_only:
            obj["optimizer"] = optimizer.state_dict()
        torch.save(obj, out)


def load_checkpoint(src: PathOrFile, model: Module, optimizer: Optimizer, device: Union[str, torch.device]) -> int:
    """Load consolidated checkpoint (single file)."""
    device = torch.device(device)
    ckpt = torch.load(src, map_location=device, weights_only=False)

    # Normalize state_dict keys saved from torch.compile (prefixed with _orig_mod.)
    model_sd = ckpt.get("model", {})
    if any(k.startswith("_orig_mod.") for k in model_sd.keys()):
        model_sd = {k.replace("_orig_mod.", "", 1): v for k, v in model_sd.items()}

    # Migrate old SwiGLU checkpoints: fuse separate up/gate into up_gate
    keys_to_migrate = [k for k in model_sd if ".ffn.up.weights" in k]
    for up_key in keys_to_migrate:
        gate_key = up_key.replace(".ffn.up.weights", ".ffn.gate.weights")
        fused_key = up_key.replace(".ffn.up.weights", ".ffn.up_gate.weights")
        if gate_key in model_sd:
            model_sd[fused_key] = torch.cat([model_sd.pop(up_key), model_sd.pop(gate_key)], dim=0)
        # If gate_key is missing, leave up_key as-is (shouldn't happen)

    if _is_fsdp_model(model):
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        ):
            model.load_state_dict(model_sd)
    else:
        model.load_state_dict(model_sd)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    state[k] = v.to(device)
    return int(ckpt["iteration"])


def save_checkpoint_sharded(model: Module, optimizer: Optimizer, iteration: int, out_dir: str, save_optimizer: bool = True) -> None:
    """Save sharded model and optimizer per rank; small per-file size, scalable."""
    assert dist.is_available() and dist.is_initialized(), "Sharded save requires distributed init."
    rank = dist.get_rank()
    os.makedirs(out_dir, exist_ok=True)
    step_dir = os.path.join(out_dir, f"step_{iteration}")
    os.makedirs(step_dir, exist_ok=True)

    # Model: local sharded state per rank
    if _is_fsdp_model(model):
        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT,
            state_dict_config=LocalStateDictConfig(),
        ):
            model_sd = model.state_dict()
        torch.save(model_sd, os.path.join(step_dir, f"model.rank{rank}.pt"))
    else:
        # Non-FSDP: just save full model on rank 0
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(step_dir, "model.full.pt"))

    # Optimizer: local sharded state per rank (if available)
    if save_optimizer and _HAS_SHARDED_OPTIM and _is_fsdp_model(model):
        osd = FSDP.optim_state_dict(
            model, optimizer,
            optim_state_dict_config=OptimStateDictConfig(
                optim_state_dict_type=OptimStateDictType.SHARDED_STATE_DICT
            ),
        )
        torch.save(osd, os.path.join(step_dir, f"optim.rank{rank}.pt"))

    # Rank 0 meta
    if rank == 0:
        with open(os.path.join(step_dir, "meta.json"), "w") as f:
            json.dump({"iteration": int(iteration)}, f)


def load_checkpoint_sharded(src_dir: str, model: Module, optimizer: Optimizer, device: Union[str, torch.device]) -> int:
    """Load sharded model/optimizer per rank; returns iteration."""
    assert dist.is_available() and dist.is_initialized(), "Sharded load requires distributed init."
    device = torch.device(device)
    rank = dist.get_rank()

    # Find latest step dir
    steps = [d for d in os.listdir(src_dir) if d.startswith("step_")]
    if not steps:
        raise FileNotFoundError(f"No step_* directories in {src_dir}")
    step_dir = os.path.join(src_dir, sorted(steps, key=lambda s: int(s.split('_')[-1]))[-1])

    # Model
    if _is_fsdp_model(model):
        model_shard = torch.load(os.path.join(step_dir, f"model.rank{rank}.pt"), map_location=device)
        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT,
            state_dict_config=LocalStateDictConfig(),
        ):
            model.load_state_dict(model_shard)
    else:
        if rank == 0:
            full_sd = torch.load(os.path.join(step_dir, "model.full.pt"), map_location=device)
            model.load_state_dict(full_sd)
        dist.barrier()

    # Optimizer (optional)
    optim_path = os.path.join(step_dir, f"optim.rank{rank}.pt")
    if os.path.exists(optim_path) and _HAS_SHARDED_OPTIM and _is_fsdp_model(model):
        osd = torch.load(optim_path, map_location="cpu")
        FSDP.load_optim_state_dict(model, optimizer, osd)

    # Iteration
    meta = {"iteration": 0}
    meta_path = os.path.join(step_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return int(meta.get("iteration", 0))
	