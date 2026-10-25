"""Spatial Gaussian frequency filters for patchified video features.

Video features use [batch, frames * height * width, channels].  Frequencies
are normalized per spatial axis so that the Nyquist frequency is one.
"""

from __future__ import annotations

from functools import lru_cache
import math
import numbers
from typing import Sequence

import torch
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_cutoff(cutoff: float) -> float:
    if isinstance(cutoff, bool):
        raise ValueError("cutoff must be finite and greater than zero")
    try:
        cutoff = float(cutoff)
    except (ValueError, TypeError) as exc:
        raise ValueError("cutoff must be finite and greater than zero") from exc
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and greater than zero")
    return cutoff


def _validate_band(band: str) -> str:
    if band not in ("low", "high"):
        raise ValueError(f"band must be 'low' or 'high', got {band!r}")
    return band


def validate_grid_size(grid_size: Sequence[int]) -> tuple[int, int, int]:
    """Validate the post-patchification (F, H, W), not the RGB dimensions."""
    if isinstance(grid_size, torch.Tensor):
        if grid_size.ndim != 1 or grid_size.numel() != 3:
            raise ValueError("grid_size must contain exactly (frames, height, width)")
        grid_size = grid_size.tolist()
    try:
        values = tuple(grid_size)
    except TypeError as exc:
        raise ValueError("grid_size must contain exactly (frames, height, width)") from exc
    if len(values) != 3:
        raise ValueError("grid_size must contain exactly (frames, height, width)")
    return tuple(_positive_integer(v, n) for v, n in zip(values, ("frames", "height", "width")))


def gaussian_frequency_mask(
    height: int,
    width: int,
    cutoff: float = 0.25,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the unshifted FP32 [H, W//2+1] Gaussian rFFT mask.

    ``cutoff`` is the standard deviation of the Gaussian in normalized
    frequency units.  The radial frequency may reach sqrt(2) at a corner;
    each individual frequency coordinate lies in [-1, 1].
    """
    height = _positive_integer(height, "height")
    width = _positive_integer(width, "width")
    cutoff = _validate_cutoff(cutoff)
    # Computing the tiny mask in FP64 keeps all finite positive cutoffs
    # well-defined, including values that would underflow to zero in FP32.
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float64) * 2
    fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float64) * 2
    return torch.exp(-0.5 * ((fy[:, None] / cutoff).square() + (fx[None, :] / cutoff).square())).float()


@lru_cache(maxsize=32)
def _cached_mask(height: int, width: int, cutoff: float, device: str) -> torch.Tensor:
    return gaussian_frequency_mask(height, width, cutoff, torch.device(device))


def spatial_frequency_filter(
    x: torch.Tensor,
    grid_size: Sequence[int],
    band: str,
    cutoff: float = 0.25,
    frame_chunk_size: int = 4,
) -> torch.Tensor:
    """Filter H/W independently for each frame, preserving input dtype.

    The full spatial grid is required.  A sequence-parallel token shard or
    padded token sequence is rejected, rather than interpreted as an image.
    FFT and high-pass subtraction run in FP32, even under autocast.  Frame
    chunks bound the temporary FP32/complex allocation without changing the
    frequency response or introducing temporal filtering.
    """
    band = _validate_band(band)
    cutoff = _validate_cutoff(cutoff)
    frame_chunk_size = _positive_integer(frame_chunk_size, "frame_chunk_size")
    frames, height, width = validate_grid_size(grid_size)
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
        raise ValueError("video features must be a floating tensor of shape [B, F*H*W, C]")
    if x.shape[0] < 1 or x.shape[2] < 1:
        raise ValueError("video feature batch and channel dimensions must be nonempty")
    expected_length = frames * height * width
    if x.shape[1] != expected_length:
        raise ValueError(
            f"video token length {x.shape[1]} does not match complete patch grid "
            f"{(frames, height, width)} ({expected_length} tokens); "
            "sequence-parallel shards and padding are unsupported"
        )
    batch, _, channels = x.shape
    # A frame-major view makes the chunk limit independent of batch size.
    images = x.reshape(batch, frames, height, width, channels).permute(0, 1, 4, 2, 3)
    images = images.reshape(batch * frames, channels, height, width)
    mask = _cached_mask(height, width, cutoff, str(x.device))
    outputs = []
    with torch.autocast(device_type=x.device.type, enabled=False):
        for start in range(0, batch * frames, frame_chunk_size):
            chunk = images[start:start + frame_chunk_size].float()
            spectrum = torch.fft.rfft2(chunk, dim=(-2, -1), norm="ortho")
            low = torch.fft.irfft2(spectrum * mask, s=(height, width), dim=(-2, -1), norm="ortho")
            outputs.append((low if band == "low" else chunk - low).to(dtype=x.dtype))
    result = torch.cat(outputs, dim=0)
    return result.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2).reshape_as(x)


def legacy_frequency_filter(x: torch.Tensor, band: str) -> torch.Tensor:
    """Reproduce the original flattened-token 0.25x interpolation path.

    This deliberately filters the original full-channel input before its
    LoRA projection.  It is the comparison baseline and also the default
    policy for text K/V adapters, where a video spatial FFT is undefined.
    """
    band = _validate_band(band)
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
        raise ValueError("legacy features must be a floating tensor of shape [B, L, C]")
    if x.shape[1] < 4:
        raise ValueError("legacy 0.25x interpolation requires at least four tokens")
    down = F.interpolate(x.permute(0, 2, 1), scale_factor=0.25, mode="linear", align_corners=False)
    low = F.interpolate(down, size=x.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
    return low if band == "low" else x - low
