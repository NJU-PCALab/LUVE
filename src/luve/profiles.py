"""Checkpoint-selected generation defaults, without importing torch or CUDA."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Literal


Profile = Literal["2k", "4k"]

_COMMON_DEFAULTS = {
    "num_frames": 49, "low_height": 720, "low_width": 1280,
    "low_steps": 50, "steps": 50, "skip_steps": 10,
    "cfg_scale": 5, "seed": 0, "fps": 15,
}
_PROFILE_DEFAULTS = {
    "2k": {**_COMMON_DEFAULTS, "height": 1472, "width": 2560, "video_quality": 6},
    "4k": {**_COMMON_DEFAULTS, "height": 2176, "width": 3840, "video_quality": 10},
}


def _profile(value: Any) -> Profile:
    if value not in ("2k", "4k"):
        raise ValueError("LUVE checkpoint profile must be '2k' or '4k'")
    return value


def profile_defaults(profile: Profile = "2k") -> dict[str, Any]:
    """Return a fresh default mapping for generation.

    The 4K profile runs LMG for 50 steps, followed directly by 4K HCR.
    There is no intermediate 2K denoising stage.
    """
    return dict(_PROFILE_DEFAULTS[_profile(profile)])


def resolve_profile(index: Mapping[str, Any]) -> Profile:
    """Read the resolution profile stored with the checkpoint."""
    return _profile(index.get("profile"))


def validate_generation_settings(profile: Profile, settings: Mapping[str, Any]) -> None:
    """Check dimensions and controls before allocating generation models."""
    profile = _profile(profile)
    integer_keys = set(_COMMON_DEFAULTS) - {"cfg_scale"}
    integer_keys.update(("height", "width", "video_quality"))
    for key in sorted(integer_keys):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    height, width = settings["height"], settings["width"]
    if profile == "2k":
        if min(height, width) <= 0 or max(height, width) != 2560 or height % 64 or width % 64:
            raise ValueError("2K requires long side 2560 and dimensions divisible by 64 for VLU")
    elif min(height, width) <= 0 or max(height, width) != 3840 or height % 64 or width % 128:
        raise ValueError("4K requires long side 3840, height divisible by 64 and width divisible by 128 for split VLU")
    low_height, low_width = settings["low_height"], settings["low_width"]
    if min(low_height, low_width) < 64 or low_height % 16 or low_width % 16:
        raise ValueError("Low-resolution dimensions must be multiples of 16 and at least 64")
    if profile == "4k" and low_width < 128:
        raise ValueError("4K split VLU requires low_width of at least 128")
    num_frames = settings["num_frames"]
    if num_frames < 5 or (num_frames - 1) % 4:
        raise ValueError("num_frames must be 4n+1 and at least 5")
    if not 0 <= settings["skip_steps"] < settings["steps"] or settings["low_steps"] < 1:
        raise ValueError("Require 0 <= skip_steps < steps and low_steps >= 1")
    cfg_scale = settings["cfg_scale"]
    if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale):
        raise ValueError("cfg_scale must be finite")
    if settings["fps"] < 1:
        raise ValueError("fps must be a positive integer")
    if not 0 <= settings["video_quality"] <= 10:
        raise ValueError("video_quality must lie between zero and ten")


def resolve_generation_settings(index: Mapping[str, Any],
                                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve explicit values over checkpoint defaults over profile defaults.

    ``None`` in an override means that the caller accepts the checkpoint's
    value.  Resolution is shared by the command line and the Python pipeline.
    """
    profile = resolve_profile(index)
    result = profile_defaults(profile)
    defaults = index.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("Checkpoint defaults must be a mapping")
    explicit = overrides if overrides is not None else {}
    for label, values in (("checkpoint defaults", defaults), ("generation overrides", explicit)):
        if not isinstance(values, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"Unknown {label}: {sorted(unknown)}")
        for key, value in values.items():
            if value is not None:
                result[key] = value
    validate_generation_settings(profile, result)
    return result
