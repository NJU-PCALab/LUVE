"""Resolve one LUVE checkpoint and an explicitly supplied public Wan model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .profiles import resolve_generation_settings


_MODEL_FILES = frozenset({
    "config.json", "model_index.json", "vlu_config.yaml", "model.safetensors",
})
_WAN_FILES = {
    "lmg": "diffusion_pytorch_model.safetensors",
    "text_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
    "vae": "Wan2.1_VAE.pth",
}


def sha256_file(path: str | Path) -> str:
    """Calculate a content hash with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def validate_model_directory(model_dir: str | Path, *,
                             verify_hashes: bool = False) -> tuple[Path, dict[str, Any]]:
    """Validate checkpoint metadata, file sizes and optionally its full SHA256.

    The model directory contains a single tensor file for HCR and VLU. No
    executable code or external asset paths are read from its metadata.
    """
    root = Path(model_dir).expanduser().resolve()
    index = _read_object(root / "model_index.json")
    if index.get("format") != "luve-inference" or index.get("format_version") != 1:
        raise ValueError("Unsupported LUVE inference checkpoint format")
    if index.get("profile") not in ("2k", "4k"):
        raise ValueError("The LUVE checkpoint must specify its 2k or 4k profile")
    resolve_generation_settings(index)
    manifest = _read_object(root / "weights_manifest.json")
    if manifest.get("format") != "luve-inference-weights" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported LUVE weights manifest")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or set(entries) != _MODEL_FILES:
        raise ValueError("The manifest must cover all four LUVE checkpoint files")
    for name, entry in entries.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing LUVE checkpoint file: {path}")
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid manifest record: {name}")
        size, expected_hash = entry.get("size"), entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid file size in manifest: {name}")
        if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError(f"Invalid SHA256 in manifest: {name}")
        if path.stat().st_size != size:
            raise ValueError(f"LUVE file size mismatch: {name}")
        # Metadata is small; always check it before trusting architecture and
        # profile settings. Full tensor hashing is an explicit option.
        if (verify_hashes or name != "model.safetensors") and sha256_file(path) != expected_hash:
            raise ValueError(f"LUVE file SHA256 mismatch: {name}")
    return root, index


def load_model_assets(model_dir: str | Path, wan_model_dir: str | Path, *,
                      verify_hashes: bool = False) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    """Locate LUVE and original-format Wan2.1-T2V-1.3B components locally.

    Public Wan files stay in their own directory and are never copied into
    the LUVE checkpoint. Missing files raise an error without downloading.
    """
    root, index = validate_model_directory(model_dir, verify_hashes=verify_hashes)
    wan_root = Path(wan_model_dir).expanduser().resolve()
    paths = {name: wan_root / filename for name, filename in _WAN_FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing public Wan {name} weights: {path}")
    tokenizer = wan_root / "google" / "umt5-xxl"
    for filename in ("tokenizer_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json"):
        if not (tokenizer / filename).is_file():
            raise FileNotFoundError(f"Missing public Wan tokenizer file: {tokenizer / filename}")
    paths.update(hcr=root, tokenizer=tokenizer, vlu_config=root / "vlu_config.yaml",
                 vlu_checkpoint=root / "model.safetensors")
    return root, index, paths
