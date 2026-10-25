"""Native frequency-modulated Wan DiT; no runtime adapter installation."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch import nn

from .configuration import LuveConfig
from .filters import legacy_frequency_filter, spatial_frequency_filter, validate_grid_size


Stage = Literal["lf", "hf", "base"]
ExpertStage = Literal["lf", "hf"]
Band = Literal["low", "high"]


def resolve_backend(backend: Any | None = None) -> Any:
    """Use an injected Wan helper namespace or the bundled backend."""
    return backend if backend is not None else importlib.import_module("luve_diffsynth.models.wan_video_dit")


@dataclass
class FrequencyContext:
    """Ephemeral route and post-patch geometry shared by native projections."""

    stage: Stage | None = None
    grid_size: tuple[int, int, int] | None = None
    batch_size: int | None = None


def _projection_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    value = getattr(module, "computation_dtype", None)
    if isinstance(value, torch.dtype):
        return value
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else fallback


class FrequencyProjection(nn.Module):
    """A native base projection and a stage-specific frequency correction.

    The low-rank factorization is an internal model parameterization. All
    parameters are loaded from the same complete model checkpoint.
    """
    def __init__(self, in_features: int, out_features: int, *, rank: int,
                 stage: ExpertStage, band: Band, context: FrequencyContext,
                 cutoff: float = 0.25, alpha: float = 32.0, multiplier: float = 1.0,
                 is_text: bool = False, text_policy: str = "legacy",
                 frame_chunk_size: int = 4, bias: bool = True,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if stage not in ("lf", "hf") or band != {"lf": "low", "hf": "high"}[stage]:
            raise ValueError("Frequency stage and band must be lf/low or hf/high")
        self.stage, self.band, self.context = stage, band, context
        self.cutoff, self.multiplier = float(cutoff), float(multiplier)
        self.is_text, self.text_policy = is_text, text_policy
        self.frame_chunk_size, self.rank = frame_chunk_size, rank
        self.base = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        self.register_buffer("alpha_scale", torch.tensor(float(alpha) / rank, dtype=torch.float32, device="cpu"))

    def _apply(self, function: Any, recurse: bool = True) -> FrequencyProjection:
        # Model dtype/device changes must never round the rank-96 scale 1/3
        # into BF16 or turn its CPU scalar into a device tensor.
        scalar = self._buffers.pop("alpha_scale")
        try:
            result = super()._apply(function, recurse=recurse)
        finally:
            self._buffers["alpha_scale"] = scalar.to(device="cpu", dtype=torch.float32)
        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the active correction; inactive stages return only the base."""
        stage = self.context.stage
        if stage not in ("lf", "hf", "base"):
            raise RuntimeError("Enter model.stage_context(...) before calling native LUVE blocks")
        if stage != self.stage:
            return self.base(x, *args, **kwargs)
        if not self.is_text:
            grid = self.context.grid_size
            if grid is None or x.ndim != 3 or x.shape[1] != math.prod(grid) or x.shape[0] != self.context.batch_size:
                raise ValueError("Active video frequency projection requires the complete patchify grid; sequence shards/padding are unsupported")
        original = self.base(x, *args, **kwargs)
        down_dtype = _projection_dtype(self.down, x.dtype)
        if self.is_text and self.text_policy == "legacy":
            branch = self.down(legacy_frequency_filter(x, self.band).to(dtype=down_dtype))
        else:
            branch = self.down(x.to(dtype=down_dtype))
            if not self.is_text:
                branch = spatial_frequency_filter(branch, grid, self.band, self.cutoff, self.frame_chunk_size)
        branch = self.up(branch.to(dtype=_projection_dtype(self.up, branch.dtype)))
        # Preserve the established BF16 expression and FP32 CPU scalar.
        return original + branch.to(dtype=original.dtype) * self.multiplier * self.alpha_scale


def _frequency_projection(config: LuveConfig, context: FrequencyContext, stage: ExpertStage,
                          in_features: int, out_features: int,
                          is_text: bool = False) -> FrequencyProjection:
    return FrequencyProjection(in_features, out_features, rank=getattr(config, stage + "_rank"),
                               stage=stage, band="low" if stage == "lf" else "high", context=context,
                               cutoff=getattr(config, stage + "_cutoff"), alpha=getattr(config, stage + "_alpha"),
                               multiplier=getattr(config, stage + "_multiplier"), is_text=is_text,
                               text_policy=config.text_policy, frame_chunk_size=config.frame_chunk_size)


class _SelfAttention(nn.Module):
    def __init__(self, config: LuveConfig, context: FrequencyContext, backend: Any) -> None:
        super().__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.q = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.k = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.v = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.o = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.norm_q = backend.RMSNorm(config.dim, eps=config.eps)
        self.norm_k = backend.RMSNorm(config.dim, eps=config.eps)
        self.attn = backend.AttentionModule(config.num_heads)
        self._rope_apply = backend.rope_apply

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(x)), self.v(x)
        q = self._rope_apply(q, freqs, self.num_heads)
        k = self._rope_apply(k, freqs, self.num_heads)
        return self.o(self.attn(q, k, v))


class _CrossAttention(nn.Module):
    def __init__(self, config: LuveConfig, context: FrequencyContext, backend: Any) -> None:
        super().__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        self.head_dim, self.has_image_input = config.dim // config.num_heads, False
        self.q = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.k = _frequency_projection(config, context, "lf", config.dim, config.dim, is_text=True)
        self.v = _frequency_projection(config, context, "lf", config.dim, config.dim, is_text=True)
        self.o = _frequency_projection(config, context, "lf", config.dim, config.dim)
        self.norm_q = backend.RMSNorm(config.dim, eps=config.eps)
        self.norm_k = backend.RMSNorm(config.dim, eps=config.eps)
        self.attn = backend.AttentionModule(config.num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        return self.o(self.attn(q, k, v))


class _Block(nn.Module):
    def __init__(self, config: LuveConfig, context: FrequencyContext, backend: Any) -> None:
        super().__init__()
        self.dim, self.num_heads, self.ffn_dim = config.dim, config.num_heads, config.ffn_dim
        self.self_attn = _SelfAttention(config, context, backend)
        self.cross_attn = _CrossAttention(config, context, backend)
        self.norm1 = nn.LayerNorm(config.dim, eps=config.eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(config.dim, eps=config.eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(config.dim, eps=config.eps)
        self.ffn = nn.Sequential(_frequency_projection(config, context, "hf", config.dim, config.ffn_dim),
                                 nn.GELU(approximate="tanh"),
                                 _frequency_projection(config, context, "hf", config.ffn_dim, config.dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)
        self.gate = backend.GateModule()
        self._modulate = backend.modulate

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_mod: torch.Tensor,
                freqs: torch.Tensor) -> torch.Tensor:
        has_seq = len(t_mod.shape) == 4
        parts = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2 if has_seq else 1)
        if has_seq:
            parts = tuple(part.squeeze(2) for part in parts)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        input_x = self._modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(input_x))


class LuveDiT(nn.Module):
    """One HCR backbone with both native frequency stages in its architecture."""
    def __init__(self, config: LuveConfig, *, backend: Any | None = None,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = config
        backend = resolve_backend(backend)
        self._backend = backend
        self.frequency_context = FrequencyContext()
        self.dim, self.freq_dim = config.dim, config.freq_dim
        self.patch_size = config.patch_size
        self.has_image_input = self.has_image_pos_emb = self.has_ref_conv = False
        self.seperated_timestep = False
        self.require_vae_embedding = self.require_clip_embedding = True
        self.fuse_vae_embedding_in_latents = False
        self.control_adapter = None
        with torch.device(device):
            self.patch_embedding = nn.Conv3d(config.in_dim, config.dim, kernel_size=config.patch_size, stride=config.patch_size)
            self.text_embedding = nn.Sequential(nn.Linear(config.text_dim, config.dim), nn.GELU(approximate="tanh"), nn.Linear(config.dim, config.dim))
            self.time_embedding = nn.Sequential(nn.Linear(config.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
            self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.dim * 6))
            self.blocks = nn.ModuleList([_Block(config, self.frequency_context, backend) for _ in range(config.num_layers)])
            self.head = backend.Head(config.dim, config.out_dim, config.patch_size, config.eps)
        self.head_dim = config.dim // config.num_heads
        self.freqs = backend.precompute_freqs_cis_3d(self.head_dim)
        self.to(dtype=dtype)

    def stage_for_timestep(self, timestep: torch.Tensor | float | list[float]) -> ExpertStage:
        """Select LF/HF from an unrounded raw timestep shared by the batch."""
        values = torch.as_tensor(timestep).detach().float().reshape(-1)
        if values.numel() < 1 or not bool(torch.isfinite(values).all()) or not bool((values == values[0]).all()):
            raise ValueError("A native stage context requires one finite timestep shared by the batch")
        return "lf" if float(values[0]) >= self.config.switch_timestep * self.config.num_train_timesteps else "hf"

    @contextmanager
    def stage_context(self, stage: Stage) -> Iterator[LuveDiT]:
        """Set an explicit route and fresh geometry, restoring both on exit.

        Use this around external pipeline code that calls ``patchify`` and
        blocks directly. ``forward`` enters its own context automatically.
        """
        if stage not in ("lf", "hf", "base"):
            raise ValueError("stage must be lf, hf or base")
        previous = (self.frequency_context.stage, self.frequency_context.grid_size, self.frequency_context.batch_size)
        self.frequency_context.stage, self.frequency_context.grid_size, self.frequency_context.batch_size = stage, None, None
        try:
            yield self
        finally:
            self.frequency_context.stage, self.frequency_context.grid_size, self.frequency_context.batch_size = previous

    def patchify(self, x: torch.Tensor,
                 control_camera_latents_input: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Embed a video and record its actual (F, H, W) patch grid."""
        if control_camera_latents_input is not None:
            raise ValueError("Camera-control inputs are not supported by this T2V release")
        context = self.frequency_context
        context.grid_size, context.batch_size = None, None
        x = self.patch_embedding(x)
        grid = validate_grid_size(x.shape[2:])
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        context.grid_size, context.batch_size = grid, tokens.shape[0]
        return tokens, grid

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """Restore latent video layout from output patch predictions."""
        from einops import rearrange
        return rearrange(x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
                         f=grid_size[0], h=grid_size[1], w=grid_size[2],
                         x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2])

    def set_ntk(self, ntk_alpha: tuple[float, float, float] = (1.0, 20.0, 20.0)) -> None:
        """Retain the backend's positional-frequency compatibility operation."""
        function = self._backend.precompute_freqs_cis_3d
        self.freqs = function(self.head_dim, ntk_alpha=list(ntk_alpha)) + function(self.head_dim, ntk_alpha=[1.0, 1.0, 1.0])

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor,
                *, stage: Stage | None = None) -> torch.Tensor:
        """Denoise T2V latents with an explicit stage or raw-timestep routing."""
        active = stage if stage is not None else self.stage_for_timestep(timestep)
        with self.stage_context(active):
            t = self.time_embedding(self._backend.sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            context = self.text_embedding(context)
            x, (frames, height, width) = self.patchify(x)
            freqs = torch.cat([self.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                               self.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                               self.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1)], dim=-1)
            freqs = freqs.reshape(frames * height * width, 1, -1).to(x.device)
            for block in self.blocks:
                x = block(x, context, t_mod, freqs)
            return self.unpatchify(self.head(x, t), (frames, height, width))

    @classmethod
    def from_pretrained(cls, directory: str | Path, **kwargs: Any) -> LuveDiT:
        """Load a complete model without consulting legacy expert files."""
        from .checkpoint import load_pretrained
        return load_pretrained(directory, **kwargs)
