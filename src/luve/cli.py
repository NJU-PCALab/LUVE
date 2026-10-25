"""Generate a 2K or 4K MP4 with LUVE and local public Wan components."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .assets import load_model_assets
from .profiles import resolve_generation_settings, resolve_profile

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

_CONTROLS = (
    "seed", "height", "width", "num_frames", "low_height", "low_width",
    "low_steps", "steps", "skip_steps", "cfg_scale", "fps", "video_quality",
)


def main() -> None:
    """Run the single public prompt-to-video entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"luve {__version__}")
    parser.add_argument("--model", required=True, type=Path,
                        help="LUVE 2K or 4K checkpoint directory")
    parser.add_argument("--wan-model-dir", required=True, type=Path,
                        help="Local public Wan2.1-T2V-1.3B directory")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=Path, help="New output MP4 file")
    parser.add_argument("--gpu", default="0", help="One physical CUDA GPU index or UUID")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--low-height", type=int)
    parser.add_argument("--low-width", type=int)
    parser.add_argument("--low-steps", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--video-quality", type=int)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate local model paths without loading CUDA models")
    parser.add_argument("--verify-weights", action="store_true",
                        help="Also verify the full LUVE tensor-file SHA256")
    args = parser.parse_args()
    if not args.check_only:
        if args.prompt is None or not args.prompt.strip() or args.output is None:
            parser.error("Generation requires a nonempty --prompt and --output")
        if args.output.suffix.lower() != ".mp4":
            parser.error("--output must name an MP4 file")
        if args.output.exists() or args.output.is_symlink():
            parser.error("Refusing to overwrite an existing output file")
    if not args.gpu.strip() or "," in args.gpu:
        parser.error("--gpu must select exactly one CUDA GPU")

    # Select the GPU before importing the inference runtime or torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    root, index, paths = load_model_assets(args.model, args.wan_model_dir,
                                           verify_hashes=args.verify_weights)
    settings = resolve_generation_settings(index, {key: getattr(args, key) for key in _CONTROLS})
    if args.check_only:
        print(json.dumps({"status": "passed", "model": str(root),
                          "profile": resolve_profile(index),
                          "generation_settings": settings,
                          "components": sorted(paths),
                          "luve_weight_hashes_verified": args.verify_weights}, indent=2))
        return

    import torch
    from .io import save_video
    from .pipeline import LUVEPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LUVE generation")
    pipeline = LUVEPipeline.from_pretrained(root, args.wan_model_dir, device="cuda:0")
    frames = pipeline(args.prompt, negative_prompt=args.negative_prompt,
                      **{key: settings[key] for key in _CONTROLS
                         if key not in ("fps", "video_quality")})
    if len(frames) != settings["num_frames"] or any(
            frame.size != (settings["width"], settings["height"]) for frame in frames):
        raise RuntimeError("Generated frame count or dimensions differ from the request")
    save_video(frames, args.output, fps=settings["fps"], quality=settings["video_quality"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
