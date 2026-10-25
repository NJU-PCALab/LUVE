### <div align="center"> (ICML'2026) LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts <div> 
<div align="center">
  <a href="https://unicornanrocinu.github.io/LUVE_web/"><img src="https://img.shields.io/static/v1?label=LUVE&message=Project&color=purple"></a> &ensp;
  <a href="https://arxiv.org/pdf/2602.11564"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/zhihefang/LUVE"><img src="https://img.shields.io/static/v1?label=Weights&message=HuggingFace&color=blue"></a> &ensp;
  <a href="https://huggingface.co/datasets/zhihefang/UltraHR-100K"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow"></a> &ensp;
</div>


## LUVE
Recent advances in video diffusion models have significantly improved visual quality, yet ultra-high-resolution (UHR) video generation remains a formidable challenge due to the compounded difficulties of motion modeling, semantic planning, and detail synthesis. To address these limitations, we propose **LUVE**, a **L**atent-cascaded **U**HR **V**ideo generation framework built upon dual frequency **E**xperts. LUVE employs a three-stage architecture comprising low-resolution motion generation for motion-consistent latent synthesis, video latent upsampling that performs resolution upsampling directly in the latent space to mitigate memory and computational overhead, and high-resolution content refinement that integrates low-frequency and high-frequency experts to jointly enhance semantic coherence and fine-grained detail generation. Extensive experiments demonstrate that our LUVE achieves superior photorealism and content fidelity in UHR video generation, and comprehensive ablation studies further validate the effectiveness of each component.

## Inference

Inference code for **LUVE**, supporting 2K and 4K video generation.
Each resolution checkpoint also contains the trained VLU parameters.


## Installation

Use Linux, Python 3.10 and an NVIDIA GPU supporting BF16. Clone the repository first:

```bash
git clone https://github.com/NJU-PCALab/LUVE.git
cd LUVE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cu124.txt
```

`requirements-cu124.txt` selects the official PyTorch 2.6.0 CUDA 12.4 wheel
for Linux x86_64 / Python 3.10. With a different compatible PyTorch installation,
use `python -m pip install -r requirements.txt` instead. The requirements install
both the runtime dependencies and this package. FlashAttention is optional;
PyTorch attention is available without it.

## Weights

Download the integrated checkpoints from [zhihefang/LUVE](https://huggingface.co/zhihefang/LUVE):

```bash
python - <<'PYDOWNLOAD'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="zhihefang/LUVE",
    local_dir=".",
    allow_patterns=["weights/2k/*", "weights/4k/*"],
)
PYDOWNLOAD
```

To download one resolution, keep only its `weights/2k/*` or `weights/4k/*`
pattern. Each checkpoint directory contains one tensor file and its configuration:

```text
weights/
  2k/
    model.safetensors
    config.json
    vlu_config.yaml
    model_index.json
    weights_manifest.json
  4k/
    model.safetensors
    config.json
    vlu_config.yaml
    model_index.json
    weights_manifest.json
```

The tensor file contains **HCR + low-frequency attention expert + high-frequency
FFN expert + trained VLU**. HCR parameters use the `hcr.` namespace and VLU
parameters use `vlu.`; both load from this same file. Frequency filtering is an
operation on activations, so the frequency projections remain native model
layers rather than being folded into ordinary linear weights.

Download the original-format
[Wan2.1-T2V-1.3B public components](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main)
separately. Pass the directory containing these files to `--wan-model-dir`:

```text
Wan2.1-T2V-1.3B/
  diffusion_pytorch_model.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/
    tokenizer_config.json
    special_tokens_map.json
    spiece.model
    tokenizer.json
```

Use the native Wan files shown above. The loader reads local files and does not
download components during generation. Public Wan weights are not redistributed
in this release.

## Generate a video

The same command supports both resolutions; `--model` selects the checkpoint
and its inference defaults. Run these examples from this directory after
replacing the Wan path:

```bash
luve-generate \
  --model ./weights/2k \
  --wan-model-dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A cinematic landscape with mountains and a slow-moving river." \
  --output ./outputs/luve-2k.mp4 \
  --gpu 0
```

```bash
luve-generate \
  --model ./weights/4k \
  --wan-model-dir /path/to/Wan2.1-T2V-1.3B \
  --prompt "A cinematic landscape with mountains and a slow-moving river." \
  --output ./outputs/luve-4k.mp4 \
  --gpu 0
```

| Setting | 2K | 4K |
| --- | --- | --- |
| Output height × width | 1472 × 2560 | 2176 × 3840 |
| Low-resolution height × width | 720 × 1280 | 720 × 1280 |
| Frames / FPS | 49 / 15 | 49 / 15 |
| LMG steps | 50 | 50 |
| HCR scheduler steps / skipped prefix | 50 / 10 | 50 / 10 |
| CFG / seed | 5 / 0 | 5 / 0 |

The command saves one MP4. Use `luve-generate --help` for generation controls.
Generation requires a single GPU with sufficient free memory. Memory use
depends on output resolution, frame count and the available attention kernel.

To check checkpoint integrity and component paths without generating a video:

```bash
luve-generate \
  --model ./weights/2k \
  --wan-model-dir /path/to/Wan2.1-T2V-1.3B \
  --check-only --verify-weights
```
