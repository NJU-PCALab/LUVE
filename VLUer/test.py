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