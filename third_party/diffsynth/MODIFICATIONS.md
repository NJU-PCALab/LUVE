# Local changes to the supplied backend subset

This directory contains only the backend modules needed for LUVE inference.

- Package initializers import only the selected Wan/VLU components. They
  do not import DiffSynth's global model manager, unrelated models,
  downloader, prompt refiner registry, or large pipeline-unit framework.
- `models/wan_video_dit.py`: CPU tensors use PyTorch SDPA even when optional
  CUDA attention packages are installed. The CUDA attention selection and
  core model calculation remain unchanged.
- `models/utils.py` and `vram_management/layers.py`: use native PyTorch
  device contexts and explicit meta constructors. They do not replace
  `Module.register_parameter`, `Module.register_buffer`, tensor constructors,
  or model methods globally. Existing Linear/LayerNorm parameters are
  retained when installing VRAM wrappers.
- `prompters/wan_prompter.py`: tokenizer loading defaults to
  `local_files_only=True`. The existing token cleaning, 512-token policy,
  and single-prompt padded-context zeroing are retained.
- `prompters/base_prompter.py`: retains the local prompt-processing loop
  without importing a model manager or network-loaded refiners.
- `vlu/models/{liif,mlp,network_vrt}.py`: package-relative registry imports;
  LIIF coordinate tensors follow the input feature device instead of using
  `.cuda()`. VRT uses `packaging.version` instead of removed `distutils`
  APIs, lazily imports optional torchvision deformable convolution, and
  refuses implicit SpyNet downloads. The supplied `pa_frames=0` model does
  not use those optional flow/deformable-convolution paths.
- Camera control code, gradient-checkpoint training branches, scheduler
  training targets/weights and unused state-dictionary discovery tools are
  omitted. Inference attention, scheduler and block calculations are retained.
- New `runtime.py`: explicit local checkpoint loading, original state-dict
  converters, local tokenizer construction, and component-specific original
  Wan VRAM policies. Tensor loading uses `weights_only=True` for PyTorch
  files and accepts safetensors. Original FP32 adapter buffers are not
  globally converted by offload configuration.
- New `vlu/__init__.py`: explicit config/checkpoint/device; safe YAML with
  relative optional `_base_` inheritance and cycle detection; training
  resume paths are ignored. Preserves original FP32 VLU math and spatial
  input crop, with `local_ensemble=False` by default. Releases retained
  LIIF activations after calls; offloading also releases cached VRT masks.
  The inference checkpoint uses the `vlu.` namespace in the same tensor file
  as the integrated HCR model, with strict selective loading.
- `vlu/utils.py`: only the original coordinate-grid utility is retained.

The subset exposes only the supplied T2V/Wan and `pa_frames=0` 3D LIIF VLU
configurations through its integration API. Broader camera, flow, image,
parallel, or caching configurations have not been validated by this release.
