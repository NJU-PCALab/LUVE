"""Prompt-to-video inference with integrated LUVE weights and public Wan assets."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .assets import load_model_assets
from .io import frames_from_tensor
from .profiles import resolve_generation_settings, resolve_profile


class LUVEPipeline:
    """Generate 2K or direct 720p-to-4K video from a prompt.

    One HCR backbone contains both experts. The scheduler selects the active
    expert before timestep conversion to BF16. Wan supplies the low-resolution
    generator, text encoder, tokenizer and VAE from a separate local directory.
    """

    def __init__(self, root: Path, index: dict, paths: dict[str, Path], *,
                 device: str, dtype: torch.dtype):
        self.root = root
        self.index = index
        self.paths = paths
        self.profile = resolve_profile(index)
        self.device = torch.device(device)
        self.dtype = dtype
        self.models = {}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, wan_model_dir: str | Path, *,
                        device: str = "cuda:0",
                        dtype: torch.dtype = torch.bfloat16,
                        verify_hashes: bool = False) -> "LUVEPipeline":
        """Resolve LUVE weights and explicitly supplied public Wan components."""
        if dtype != torch.bfloat16:
            raise ValueError("LUVE generation uses torch.bfloat16")
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("LUVE generation requires a CUDA GPU")
        root, index, paths = load_model_assets(model_dir, wan_model_dir,
                                              verify_hashes=verify_hashes)
        pipeline = cls(root, index, paths, device=device, dtype=dtype)
        return pipeline

    def _path(self, name: str) -> Path:
        return self.paths[name]

    def _activate(self, *names: str) -> None:
        from luve_diffsynth.runtime import activate_model, offload_model

        for name, model in self.models.items():
            if name not in names:
                offload_model(model)
        torch.cuda.empty_cache()
        for name in names:
            activate_model(self.models[name])

    def _load(self, name: str):
        from luve_diffsynth import runtime

        if name in self.models:
            return self.models[name]
        if name == "hcr":
            from .checkpoint import load_pretrained
            model = load_pretrained(self._path("hcr"), dtype=self.dtype)
        else:
            loader = {"lmg": runtime.load_lmg,
                      "text_encoder": runtime.load_text_encoder,
                      "vae": runtime.load_vae}[name]
            model = loader(self._path(name), dtype=self.dtype)
        model.eval().requires_grad_(False)
        runtime.configure_offload(model, kind=name, device=str(self.device), dtype=self.dtype)
        self.models[name] = model
        return model

    def _contexts(self, prompt: str, negative_prompt: str, cfg_scale: float):
        from luve_diffsynth.runtime import make_prompter

        text_encoder = self._load("text_encoder")
        prompter = make_prompter(text_encoder, self._path("tokenizer"))
        self._activate("text_encoder")
        positive = prompter.encode_prompt(prompt, positive=True, device=self.device)
        negative = (prompter.encode_prompt(negative_prompt, positive=False, device=self.device)
                    if cfg_scale != 1 else None)
        return positive, negative

    def _predict(self, model, latent, timestep, context, stage: str):
        prediction = (model(latent, timestep, context, stage=stage) if stage in ("lf", "hf")
                      else model(latent, timestep, context))
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError(f"Non-finite denoiser output in {stage}")
        return prediction

    def _low_stage(self, contexts, *, height, width, num_frames, steps, seed, cfg_scale):
        from luve_diffsynth.schedulers.flow_match import FlowMatchScheduler

        model = self._load("lmg")
        self._activate("lmg")
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(steps, denoising_strength=1, shift=5)
        # The reference LMG draws FP32 noise with a local CPU generator.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn((1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                             generator=generator, device="cpu", dtype=torch.float32)
        latent = latent.to(device=self.device, dtype=self.dtype)
        for timestep in tqdm(scheduler.timesteps, desc="LMG"):
            t = timestep.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            positive = self._predict(model, latent, t, contexts[0], "lmg")
            if cfg_scale != 1:
                negative = self._predict(model, latent, t, contexts[1], "lmg")
                prediction = negative + cfg_scale * (positive - negative)
            else:
                prediction = positive
            latent = scheduler.step(prediction, timestep, latent)
        result = latent.cpu()
        self._activate()
        del self.models["lmg"]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _upscale_latent(self, vlu, low_latent: torch.Tensor, *,
                        height: int, width: int) -> torch.Tensor:
        """Apply the profile's original VLU geometry and BF16 cast order."""
        # Preserve the reference's BF16 HCR-input cast before VLU uses FP32.
        vlu_input = low_latent.to(device=self.device, dtype=self.dtype).float()
        if self.profile == "2k":
            return vlu(vlu_input,
                       H_target=height // 8, W_target=width // 8).to(self.dtype)
        # The original 4K path encodes and decodes the two halves separately.
        # This is not whole-image VLU with chunked coordinate queries.
        midpoint = vlu_input.shape[-1] // 2
        blocks = (vlu_input[..., :midpoint], vlu_input[..., midpoint:])
        outputs = [vlu(block, H_target=height // 8, W_target=width // 16).to(self.dtype)
                   for block in blocks]
        return torch.cat(outputs, dim=4)

    @torch.inference_mode()
    def __call__(self, prompt: str, *, negative_prompt: str = "", seed: int | None = None,
                 height: int | None = None, width: int | None = None, num_frames: int | None = None,
                 low_height: int | None = None, low_width: int | None = None, low_steps: int | None = None,
                 steps: int | None = None, skip_steps: int | None = None,
                 cfg_scale: float | None = None) -> list:
        """Run LMG → VLU → frequency-aware HCR → VAE decoding.

        Return RGB PIL frames. Omitted controls use the checkpoint's profile
        and generation defaults.
        """
        settings = resolve_generation_settings(self.index, {
            "seed": seed, "height": height, "width": width, "num_frames": num_frames,
            "low_height": low_height, "low_width": low_width, "low_steps": low_steps,
            "steps": steps, "skip_steps": skip_steps, "cfg_scale": cfg_scale,
        })
        seed, height, width = settings["seed"], settings["height"], settings["width"]
        num_frames = settings["num_frames"]
        low_height, low_width, low_steps = settings["low_height"], settings["low_width"], settings["low_steps"]
        steps, skip_steps, cfg_scale = settings["steps"], settings["skip_steps"], settings["cfg_scale"]
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("Prompts must be strings")
        contexts = self._contexts(prompt, negative_prompt, cfg_scale)
        low_latent = self._low_stage(contexts, height=low_height, width=low_width,
                                     num_frames=num_frames, steps=low_steps,
                                     seed=seed, cfg_scale=cfg_scale)
        expected = (1, 16, (num_frames - 1) // 4 + 1, low_height // 8, low_width // 8)
        if tuple(low_latent.shape) != expected or not torch.isfinite(low_latent).all():
            raise ValueError(f"Expected finite low latent with shape {expected}")
        low_cpu = low_latent.detach().cpu()
        self._activate()
        from luve_diffsynth.vlu import LatentVSR
        vlu = LatentVSR(config=self._path("vlu_config"),
                        checkpoint=self._path("vlu_checkpoint"), device=str(self.device),
                        checkpoint_prefix="vlu.")
        # VLU remains FP32, matching the original pipeline.
        high_latent = self._upscale_latent(vlu, low_cpu, height=height, width=width)
        del vlu
        gc.collect()
        torch.cuda.empty_cache()
        from luve_diffsynth.schedulers.flow_match import FlowMatchScheduler
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(steps, denoising_strength=1, shift=5)
        # Preserve the reference HCR's BF16 CUDA noise shape and generator.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(high_latent.shape[1:], generator=generator,
                            device=self.device, dtype=self.dtype).unsqueeze(0)
        latent = scheduler.add_noise(high_latent, noise,
                                     scheduler.timesteps[skip_steps].unsqueeze(0))
        scheduler._step_index = 0
        model = self._load("hcr")
        self._activate("hcr")
        for index, timestep in enumerate(tqdm(scheduler.timesteps, desc="HCR")):
            if index < skip_steps:
                scheduler._step_index += 1
                continue
            # Route on the unrounded scheduler value, not BF16 timestep t.
            stage = model.stage_for_timestep(timestep)
            t = timestep.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            positive = self._predict(model, latent, t, contexts[0], stage)
            if cfg_scale != 1:
                negative = self._predict(model, latent, t, contexts[1], stage)
                prediction = negative + cfg_scale * (positive - negative)
            else:
                prediction = positive
            latent = scheduler.step(prediction, timestep, latent)
        self._load("vae")
        self._activate("vae")
        decoded = self.models["vae"].decode(latent, device=self.device, tiled=True,
                                             tile_size=(30, 52), tile_stride=(15, 26))
        frames = frames_from_tensor(decoded)
        self._activate()
        return frames
