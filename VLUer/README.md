# VLUer: Video Latent Upsampler

VLUer is a module for upsampling video latent spaces. It combines the continuous representation capabilities of **LIIF** (Learning Continuous Image Representation) with the video processing power of **VRT** (Video Restoration Transformer), enabling video latent space scaling at arbitrary ratios.

## Project Structure

```text
.
├── vlu/                # Core inference code
│   ├── models.py       # Model factory
│   ├── liif.py         # LIIF module implementation
│   ├── mlp.py          # MLP module implementation
│   ├── network_vrt.py  # VRT network implementation
│   ├── utils.py        # Utility functions
│   └── config.yaml     # Default model configuration
├── vlu_training/       # Training code and configurations
└── test.py             # Inference example code
```

## Quick Start

### Inference Example

You can refer to `test.py` to learn how to load the model and upsample tensors.

```python
import torch
import numpy as np
import vlu

# 1. Prepare input tensor (Format: THWC, e.g., from a latent extractor)
test_tensor = np.load("tensor.npy") 
test_tensor = torch.from_numpy(test_tensor)
test_tensor = test_tensor.permute(3, 0, 1, 2) # Convert to CTHW format
H, W = test_tensor.shape[-2:]

# 2. Load pre-trained model
vlu_path = "vlu/model.pth"
checkpoint = torch.load(vlu_path)
vlu_model = vlu.models.make(checkpoint["model"], load_sd=True).cuda()

# 3. Perform upsampling (e.g., scale by 1.5x)
# output_tensor format: CTHW
output_tensor = vlu.utils.latent_upsampler(vlu_model, test_tensor, H * 1.5, W * 1.5)

# 4. Save results
output_tensor = output_tensor.permute(1, 2, 3, 0) # Convert back to THWC
np.save("output.npy", output_tensor.cpu().numpy())
```

### Training

For detailed training instructions, please refer to [vlu_training/README.md](vlu_training/README.md).

## Acknowledgments

This module is based on [LSRNA](https://github.com/3587jjh/LSRNA) and [VRT](https://github.com/JingyunLiang/VRT). Thanks for their awesome works.