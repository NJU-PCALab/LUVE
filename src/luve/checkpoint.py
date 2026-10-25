"""Strict runtime loading of the HCR component from a complete checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch

from .configuration import LuveConfig
from .modeling import LuveDiT


def _load_hcr_state(path: Path, config: LuveConfig,
                    backend: Any | None = None) -> dict[str, torch.Tensor]:
    # Validate shapes against a meta model instead of allocating a second
    # complete backbone. VLU tensors are never materialized by this loader.
    template = LuveDiT(config, backend=backend, device="meta", dtype=torch.bfloat16).state_dict()
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = tuple(checkpoint.keys())
        unknown = [key for key in keys if not key.startswith(("hcr.", "vlu."))]
        if unknown:
            raise ValueError(f"Unexpected complete-checkpoint tensor prefixes: {unknown[:8]}")
        component_keys = {key[4:]: key for key in keys if key.startswith("hcr.")}
        missing = sorted(set(template) - set(component_keys))
        unexpected = sorted(set(component_keys) - set(template))
        if missing or unexpected:
            raise ValueError(f"HCR checkpoint key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        for key, stored_key in component_keys.items():
            if tuple(checkpoint.get_slice(stored_key).get_shape()) != tuple(template[key].shape):
                raise ValueError(f"HCR checkpoint shape mismatch for {key}")
        for key, stored_key in component_keys.items():
            value = checkpoint.get_tensor(stored_key)
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{stored_key} must contain finite floating values")
            if key.endswith(".alpha_scale") and (
                    value.dtype != torch.float32 or value.ndim != 0 or float(value) <= 0):
                raise ValueError(f"{stored_key} must be a positive FP32 zero-dimensional scale")
            state[key] = value
    return state


def load_pretrained(directory: str | Path, *, backend: Any | None = None,
                    device: torch.device | str = "cpu", dtype: torch.dtype | None = None,
                    verify_hashes: bool = False) -> LuveDiT:
    """Load HCR from config.json and the shared model.safetensors.

    Only hcr. tensors are read into memory. The file may additionally contain
    vlu. tensors; all other tensor prefixes are rejected. Header metadata
    does not determine tensor routing. Optional file verification uses the
    model directory's integrity manifest through luve.assets. The stored
    patch-embedding dtype is used unless dtype is supplied; stage scale
    buffers remain FP32 CPU scalars.
    """
    directory = Path(directory).expanduser().resolve()
    if verify_hashes:
        from .assets import validate_model_directory
        directory, _ = validate_model_directory(directory, verify_hashes=True)
    config = LuveConfig.from_json(directory / "config.json")
    state = _load_hcr_state(directory / "model.safetensors", config, backend)
    if dtype is None:
        dtype = state["patch_embedding.weight"].dtype
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Unsupported inference dtype")
    model = LuveDiT(config, backend=backend, device="cpu", dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)
    return model
