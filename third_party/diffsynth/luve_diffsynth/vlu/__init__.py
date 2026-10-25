"""Portable, explicit-device wrapper for the supplied 3D latent VSR model."""
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from . import models
from .utils import make_coord_3d


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (_merge(result[key], value)
                       if key in result and isinstance(result[key], dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def load_config(path, _ancestors=()):
    """Safe local YAML with optional relative ``_base_`` inheritance.

    The checkpoint always comes from the explicit constructor argument;
    training-only resume_path/map_loc/save_path fields are never consulted.
    """
    path = Path(path).resolve()
    if path in _ancestors:
        raise ValueError('Cyclic VLU YAML inheritance: ' + ' -> '.join(map(str, (*_ancestors, path))))
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'VLU YAML must contain a mapping: {path}')
    bases = config.pop('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) for base in bases):
        raise ValueError('_base_ must be a relative YAML path or list of relative paths')
    inherited = {}
    for base in bases:
        relative = Path(base)
        if relative.is_absolute():
            raise ValueError('VLU YAML inheritance must use relative paths')
        inherited = _merge(inherited, load_config(path.parent / relative, (*_ancestors, path)))
    result = _merge(inherited, config)
    for field in ('resume_path', 'map_loc', 'save_path'):
        result.pop(field, None)
    return result


class LatentVSR:
    def __init__(self, config, checkpoint, device, local_ensemble=False,
                 checkpoint_prefix='vlu.'):
        self.device = torch.device(device)
        self.config = load_config(config) if not isinstance(config, dict) else deepcopy(config)
        specification = deepcopy(self.config['model'])
        if specification.get('name') != 'liif':
            raise ValueError('This release supports the supplied 3D LIIF VLU only')
        arguments = specification.setdefault('args', {})
        arguments['local_ensemble'] = bool(local_ensemble)
        for stage in ('encoder_spec', 'decoder_spec'):
            settings = arguments[stage]
            if settings.get('name') != 'vrt' or settings.get('args', {}).get('pa_frames', 2) != 0:
                raise ValueError('The portable VLU requires its supplied pa_frames=0 VRT configuration')
        self.model = models.make(specification)
        if checkpoint_prefix not in ('vlu.', ''):
            raise ValueError("VLU checkpoint_prefix must be 'vlu.' or empty for a standalone reference")
        expected = self.model.state_dict()
        with safe_open(str(checkpoint), framework='pt', device='cpu') as checkpoint_file:
            keys = tuple(checkpoint_file.keys())
            if checkpoint_prefix:
                unknown = [key for key in keys if not key.startswith(('hcr.', 'vlu.'))]
                if unknown:
                    raise ValueError(f'Unexpected complete-checkpoint tensor prefixes: {unknown[:8]}')
            selected = {key[len(checkpoint_prefix):]: key for key in keys
                        if key.startswith(checkpoint_prefix)}
            missing = sorted(set(expected) - set(selected))
            unexpected = sorted(set(selected) - set(expected))
            if missing or unexpected:
                raise ValueError(f'VLU checkpoint key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}')
            for key, stored_key in selected.items():
                if tuple(checkpoint_file.get_slice(stored_key).get_shape()) != tuple(expected[key].shape):
                    raise ValueError(f'VLU checkpoint shape mismatch for {key}')
            state = {key: checkpoint_file.get_tensor(stored_key) for key, stored_key in selected.items()}
        for key, value in state.items():
            if expected[key].is_floating_point():
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'VLU checkpoint requires finite floating values for {key}')
            elif value.dtype != expected[key].dtype:
                raise ValueError(f'VLU checkpoint buffer dtype mismatch for {key}')
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, latent, H_target, W_target):
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 16:
            raise ValueError('VLU input must have shape [1, 16, latent_frames, H, W]')
        if latent.shape[2] < 2:
            raise ValueError('The supplied VRT mutual attention requires at least two latent frames')
        if min(H_target, W_target) < 8 or H_target % 8 or W_target % 8:
            raise ValueError('VLU target spatial dimensions must be positive multiples of eight')
        original_dtype = latent.dtype
        height, width = latent.shape[-2:]
        height, width = height // 8 * 8, width // 8 * 8
        if min(height, width) < 8:
            raise ValueError('VLU source latent spatial dimensions must be at least eight')
        # Preserve the original input crop, frame ordering and FP32 VLU math.
        inputs = latent[:, :, :, :height, :width].to(self.device, torch.float32).permute(0, 2, 1, 3, 4)
        frames = inputs.shape[1]
        coordinates = make_coord_3d((frames, H_target, W_target), flatten=False, device=self.device).unsqueeze(0)
        cell = torch.ones_like(coordinates)
        cell[..., 0] *= 2 / frames
        cell[..., 1] *= 2 / H_target
        cell[..., 2] *= 2 / W_target
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                output = self.model(inputs, coordinates, cell).permute(0, 2, 1, 3, 4)
            return output.to(dtype=original_dtype)
        finally:
            # LIIF retains these as ordinary attributes rather than buffers.
            # A persistent pipeline must release them after each completed
            # call; model.to('cpu') alone would retain GPU activations.
            for name in ('feat', 'inp', 'feat_coord'):
                if hasattr(self.model, name):
                    delattr(self.model, name)

    def offload(self):
        self.model.to(device='cpu')
        models.network_vrt.compute_mask.cache_clear()
        return self

    def activate(self):
        self.model.to(device=self.device)
        return self
