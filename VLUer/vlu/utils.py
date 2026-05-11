import torch

def make_coord_3d(shape, ranges=None, flatten=True, device='cpu'):
    """
    Make coordinates at grid centers for 3D grid.
    shape: (D, H, W)
    ranges: [(z0,z1), (y0,y1), (x0,x1)] or None
    flatten: if True, returns (D*H*W, 3), else (D, H, W, 3)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    # meshgrid for 3D
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)  # (D,H,W,3)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # (D*H*W,3)
    return ret

def latent_upsamplar(model, latent, H_target, W_target):
    model.eval()
    device = next(model.parameters()).device
    latent = latent.unsqueeze(0).to(device)

    with torch.no_grad():
        H, W = latent.shape[-2:]
        H, W = H // 8 * 8, W // 8 * 8
        H_target, W_target = H_target // 8 * 8, W_target // 8 * 8 
        latent = latent[:, :, :, :H, :W]
        # latent -> (1, T, C, H, W)
        latent = latent.permute(0, 2, 1, 3, 4)
        H, W = latent.shape[-2:]
        T = latent.shape[1]
        coord = make_coord_3d((T, H_target, W_target), flatten=False, device='cuda').unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[:,:,:,:,0] *= 2 / T
        cell[:,:,:,:,1] *= 2 / H_target
        cell[:,:,:,:,2] *= 2 / W_target
        pred_latent = model(latent, coord, cell).squeeze(0).permute(1, 0, 2, 3)
        # (C, T, H, W)

        return pred_latent