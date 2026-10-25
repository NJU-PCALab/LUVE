"""Serializable architecture and frequency routing configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import json
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class LuveConfig:
    """Architecture, native frequency branches and raw-timestep routing.

    Cutoffs are Gaussian bandwidths measured relative to spatial Nyquist.
    ``switch_timestep`` is normalized; equality selects the LF stage.
    """
    dim: int = 1536
    in_dim: int = 16
    ffn_dim: int = 8960
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_heads: int = 12
    num_layers: int = 30
    has_image_input: bool = False
    lf_rank: int = 128
    hf_rank: int = 96
    lf_alpha: float = 32.0
    hf_alpha: float = 32.0
    lf_cutoff: float = 0.25
    hf_cutoff: float = 0.25
    lf_multiplier: float = 1.0
    hf_multiplier: float = 1.0
    text_policy: str = "legacy"
    frame_chunk_size: int = 4
    switch_timestep: float = 0.875
    num_train_timesteps: int = 1000

    def __post_init__(self) -> None:
        for name in ("dim", "in_dim", "ffn_dim", "out_dim", "text_dim", "freq_dim", "num_heads", "num_layers", "lf_rank", "hf_rank", "frame_chunk_size", "num_train_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 2:
            raise ValueError("dim/num_heads must be an even integer for rotary embeddings")
        if self.freq_dim % 2:
            raise ValueError("freq_dim must be even")
        patch = tuple(self.patch_size)
        if len(patch) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in patch):
            raise ValueError("patch_size must contain three positive integers")
        object.__setattr__(self, "patch_size", patch)
        if self.has_image_input is not False:
            raise ValueError("This release currently supports the validated T2V architecture only")
        for name in ("eps", "lf_alpha", "hf_alpha", "lf_cutoff", "hf_cutoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        for name in ("lf_multiplier", "hf_multiplier"):
            if isinstance(getattr(self, name), bool) or not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(float(self.switch_timestep)) or not 0 < self.switch_timestep < 1:
            raise ValueError("switch_timestep must lie between zero and one")
        if self.text_policy not in ("legacy", "identity"):
            raise ValueError("text_policy must be legacy or identity")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible architecture description without source paths."""
        return {"model_type": "luve", "format_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LuveConfig:
        """Validate a configuration, rejecting unknown fields and format versions."""
        values = dict(values)
        if values.pop("model_type", "luve") != "luve" or values.pop("format_version", 1) != 1:
            raise ValueError("Unsupported LUVE architecture/checkpoint format")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown LUVE configuration fields: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> LuveConfig:
        """Read a saved ``config.json`` without loading model weights."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def projection_specs(self) -> dict[str, dict[str, str | int | bool]]:
        """Native target paths, without constructing or modifying a model."""
        specs = {}
        for index in range(self.num_layers):
            for attention in ("self_attn", "cross_attn"):
                for name in ("q", "k", "v", "o"):
                    path = f"blocks.{index}.{attention}.{name}"
                    specs[path] = dict(stage="lf", band="low", rank=self.lf_rank,
                                       in_features=self.dim, out_features=self.dim,
                                       is_text=attention == "cross_attn" and name in ("k", "v"))
            for name, in_features, out_features in (("0", self.dim, self.ffn_dim), ("2", self.ffn_dim, self.dim)):
                specs[f"blocks.{index}.ffn.{name}"] = dict(stage="hf", band="high", rank=self.hf_rank,
                                                          in_features=in_features, out_features=out_features, is_text=False)
        return specs
