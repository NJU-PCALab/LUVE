"""Explicit, local-only component loading and original Wan VRAM policies.

No global model manager, legacy pipeline imports, private source paths or
runtime replacement of model methods is required.
"""
from pathlib import Path

import torch
from torch import nn

from .models.utils import load_state_dict
from .models.wan_video_dit import WanModel, RMSNorm, precompute_freqs_cis_3d
from .models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from .models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from .prompters.wan_prompter import WanPrompter
from .vram_management import (
    AutoTorchModule, AutoWrappedModule, AutoWrappedLinear,
    WanAutoCastLayerNorm, enable_vram_management,
)


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = load_state_dict(str(path), device='cpu')
    if not isinstance(state, dict):
        raise ValueError(f'Component checkpoint must contain a tensor mapping: {path}')
    return state


def _assign(model, state, dtype):
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device='cpu', dtype=dtype)


def load_lmg(path, dtype=torch.bfloat16):
    """Load the original Wan T2V LMG DiT, on CPU, using its native converter."""
    state, config = WanModel.state_dict_converter().from_civitai(_read(path))
    if not config:
        raise ValueError('Unrecognized Wan checkpoint structure; expected the exported native Wan T2V weights')
    if config.get('has_image_input'):
        raise ValueError('This release supports text-to-video LMG weights only')
    with torch.device('meta'):
        model = WanModel(**config)
    # RoPE tables are ordinary attributes, not assigned state_dict buffers.
    # Rebuild them on CPU after native meta construction.
    model.freqs = precompute_freqs_cis_3d(model.head_dim)
    return _assign(model, state, dtype)


def load_text_encoder(path, dtype=torch.bfloat16):
    state = WanTextEncoder.state_dict_converter().from_civitai(_read(path))
    with torch.device('meta'):
        model = WanTextEncoder()
    return _assign(model, state, dtype)


def load_vae(path, dtype=torch.bfloat16):
    state = _read(path)
    # Support both the original Wan .pth schema and an exported native state.
    if not state or not all(isinstance(key, str) and key.startswith('model.') for key in state):
        state = WanVideoVAE.state_dict_converter().from_civitai(state)
    # Keep the original CPU FP32 latent mean/std attributes.  They are not
    # state_dict buffers and must not be created as value-less meta tensors.
    model = WanVideoVAE()
    return _assign(model, state, dtype)


def make_prompter(model, tokenizer_path):
    path = Path(tokenizer_path)
    if not path.is_dir():
        raise FileNotFoundError(f'Local tokenizer directory missing: {path}')
    prompter = WanPrompter(tokenizer_path=str(path))
    prompter.fetch_models(model)
    return prompter


def configure_offload(model, kind, device, dtype=torch.bfloat16):
    """Install the former low/high Wan component-specific VRAM wrappers.

    ``lmg`` uses ordinary LayerNorm wrapping and eagerly onloads its DiT;
    ``hcr`` uses FP32-aware WanAutoCastLayerNorm and the original VRAM
    budget policy.  Adapter containers are retained while their Linear
    children are wrapped.  Existing FP32 adapter scale buffers are untouched.
    """
    if kind not in ('lmg', 'hcr', 'text_encoder', 'vae'):
        raise ValueError(f'Unknown component kind: {kind}')
    if getattr(model, 'vram_management_enabled', False):
        raise ValueError('VRAM management has already been configured for this model')
    device = str(torch.device(device))
    storage_dtype = next(model.parameters()).dtype
    vram_limit = None
    if kind != 'lmg' and torch.device(device).type == 'cuda':
        vram_limit = torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - .5
    if kind == 'text_encoder':
        module_map = {nn.Linear: AutoWrappedLinear, nn.Embedding: AutoWrappedModule,
                      T5RelativeEmbedding: AutoWrappedModule, T5LayerNorm: AutoWrappedModule}
        onload_device = 'cpu'
    elif kind == 'vae':
        module_map = {nn.Linear: AutoWrappedLinear, nn.Conv2d: AutoWrappedModule,
                      RMS_norm: AutoWrappedModule, CausalConv3d: AutoWrappedModule,
                      Upsample: AutoWrappedModule, nn.SiLU: AutoWrappedModule,
                      nn.Dropout: AutoWrappedModule}
        onload_device = device
    else:
        module_map = {nn.Linear: AutoWrappedLinear, nn.Conv3d: AutoWrappedModule,
                      nn.LayerNorm: WanAutoCastLayerNorm if kind == 'hcr' else AutoWrappedModule,
                      RMSNorm: AutoWrappedModule, nn.Conv2d: AutoWrappedModule}
        onload_device = device if kind == 'lmg' or vram_limit is None else 'cpu'
    enable_vram_management(model, module_map=module_map, module_config={
        'offload_dtype': storage_dtype, 'offload_device': 'cpu',
        'onload_dtype': storage_dtype, 'onload_device': onload_device,
        'computation_dtype': dtype, 'computation_device': device,
    }, vram_limit=vram_limit)
    model.eval()
    return model


def activate_model(model):
    if not getattr(model, 'vram_management_enabled', False):
        raise ValueError('Call configure_offload before activate_model')
    for module in model.modules():
        if isinstance(module, AutoTorchModule):
            module.onload()
    return model


def offload_model(model):
    if getattr(model, 'vram_management_enabled', False):
        for module in model.modules():
            if isinstance(module, AutoTorchModule):
                module.offload()
    else:
        model.to(device='cpu')
    return model
