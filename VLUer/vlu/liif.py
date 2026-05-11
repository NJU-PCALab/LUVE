import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models
from .models import register
from .utils import make_coord_3d


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec, decoder_spec,feat_unfold=True, local_ensemble=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.encoder = models.make(encoder_spec)
        self.decoder = models.make(decoder_spec)

        imnet_in_dim = self.encoder.out_chans
        if self.feat_unfold:
            imnet_in_dim *= 27
        imnet_in_dim += 6 # attach coord, cell
        self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        self.conv = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=1, stride=1, padding=0)
        
    def gen_feat(self, inp):
        self.inp = inp
        feat = self.encoder(inp)
        feat = feat.squeeze(0).permute(1, 0, 2, 3)  # (t, c, h, w)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            feat = F.pad(feat, (0, 0, 0, 0, 0, 0, 1, 1)) # pad t (t+2, c*9, h, w)
            feat = torch.stack([
                feat[0:feat.shape[0] - 2],       # t-1
                feat[1:feat.shape[0] - 1],     # t
                feat[2:feat.shape[0]]      # t+1
            ], dim=1).view(feat.shape[0] - 2, feat.shape[1] * 3, feat.shape[2], feat.shape[3])   # (t, c*27, h, w)
        self.feat = feat
        self.feat_coord = make_coord_3d(feat.permute(1,0,2,3).shape[-3:], flatten=False).cuda() \
            .permute(3, 0, 1, 2)
        
    def query_rgb(self, coord, cell):
        feat = self.feat
        feat_coord = self.feat_coord
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            vt_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, vt_lst, eps_shift = [0], [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        rt = 2 / feat.shape[0] / 2

        preds = []
        areas = []

        coord_list = []
        for vt in vt_lst:
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:, :, :, 0] += vt * rt + eps_shift
                    coord_[:, :, :, 1] += vx * rx + eps_shift
                    coord_[:, :, :, 2] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(feat.unsqueeze(0).permute(0,2,1,3,4), coord_.flip(-1),
                        mode='nearest', align_corners=False).permute(0, 2, 3, 4, 1)
                    q_coord = F.grid_sample(feat_coord.unsqueeze(0), coord_.flip(-1),
                        mode='nearest', align_corners=False).permute(0, 2, 3, 4, 1)

                    coord_list.append(q_coord)

                    rel_coord = coord - q_coord
                    rel_coord[:, :, :, :, 0] *= feat.shape[0]
                    rel_coord[:, :, :, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, :, :, 2] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    rel_cell = cell.clone()
                    rel_cell[:, :, :, :, 0] *= feat.shape[0]
                    rel_cell[:, :, :, :, 1] *= feat.shape[-2]
                    rel_cell[:, :, :, :, 2] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                    pred = self.imnet(inp.contiguous())
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, :, :, 0] * rel_coord[:, :, :, :, 1] * rel_coord[:, :, :, :, 2])
                    areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble: 
            t = areas[0]; areas[0] = areas[7]; areas[7] = t
            t = areas[1]; areas[1] = areas[6]; areas[6] = t
            t = areas[2]; areas[2] = areas[5]; areas[5] = t
            t = areas[3]; areas[3] = areas[4]; areas[4] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret = ret.permute(0, 1, 4, 2, 3)

        if ret.shape[1] != self.inp.shape[1]:
            ret += F.grid_sample(self.inp.permute(0,2,1,3,4), coord.flip(-1), mode='bilinear',\
                padding_mode='border', align_corners=False).permute(0,2,1,3,4)
        else:
            ret += F.grid_sample(self.inp.squeeze(0), coord[0, :, :, :, 1:].flip(-1), mode='bilinear',\
                padding_mode='border', align_corners=False).unsqueeze(0)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        x = self.query_rgb(coord, cell)
        return self.conv(self.decoder(x).squeeze(0).permute(1,0,2,3)).unsqueeze(0) + x
    
    def batched_predict(self, inp, coord, cell, bsize=512*512):
        self.gen_feat(inp)
        H,W = coord.shape[1:3]
        n = H*W
        coord = coord.view(1,1,n,2)
        cell = cell.view(1,1,n,2)

        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(coord[:,:,ql:qr,:], cell[:,:,ql:qr,:])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=-1).view(1,-1,H,W)
        return pred