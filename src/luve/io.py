"""Convert decoded video frames and write an MP4."""

from __future__ import annotations

from pathlib import Path


def frames_from_tensor(decoded):
    """Convert Wan VAE output in [-1, 1] to RGB PIL frames."""
    import torch
    from PIL import Image

    if decoded.ndim == 5:
        if decoded.shape[0] != 1:
            raise ValueError("Video export supports batch size one")
        decoded = decoded[0]
    if decoded.ndim != 4 or decoded.shape[0] != 3:
        raise ValueError("Expected decoded video shape [3, T, H, W]")
    # The reference performs this transform in the decoder output dtype.
    # Promoting BF16 to FP32 here changes integer RGB values after rounding.
    pixels = ((decoded.permute(1, 2, 3, 0) + 1) * 127.5)
    pixels = pixels.clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(frame) for frame in pixels]


def save_video(frames, path: str | Path, *, fps: int = 15, quality: int = 6) -> None:
    """Encode an RGB frame sequence with the reference encoder settings."""
    import imageio.v2 as imageio
    import numpy as np

    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video output must be an MP4 file")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    if not frames:
        raise ValueError("Cannot export an empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the new filename exclusively before handing it to the encoder.
    with path.open("xb"):
        pass
    try:
        with imageio.get_writer(str(path), fps=fps, quality=quality) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    except Exception:
        path.unlink(missing_ok=True)
        raise
