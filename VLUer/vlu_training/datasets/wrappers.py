import numpy as np
from torch.utils.data import Dataset
import torchvision.io as io
from datasets import register
from utils import *


@register('sr-explicit-paired')
class SRExplicitPaired(Dataset):

    def __init__(self, dataset, inp_size, augment=[], max_frames=None, sample_size=None, num_channels=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.max_frames = max_frames
        self.sample_size = inp_size if sample_size is None else sample_size
        self.num_channels = num_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_path, lr_paths, gt_path = self.dataset[idx]
        lr_path = lr_paths[np.random.randint(len(lr_paths))]

        max_frames = self.max_frames
        # video: (T,C,H,W), numpy, range [-3,3] or [0,1]
        hr, lr = read_video(hr_path, max_frames=max_frames), read_video(lr_path, max_frames=max_frames)
        gt, _, _ = io.read_video(gt_path, pts_unit="sec") # video: [T, H, W, C]
        if max_frames is not None:
            gt = gt[:(max_frames - 1) * 4 + 1] / 255 * 2 - 1 # to [-1,1]
        else:
            gt = gt / 255 * 2 - 1 # to [-1,1]

        if self.num_channels:
            assert hr.shape[-1] == lr.shape[-1] == self.num_channels
        hr, lr, gt = random_crop_together(hr, lr, gt, self.inp_size)

        # augmentation
        hflip = (np.random.random() < 0.5) if 'hflip' in self.augment else False
        vflip = (np.random.random() < 0.5) if 'vflip' in self.augment else False
        dflip = (np.random.random() < 0.5) if 'dflip' in self.augment else False

        def base_augment(video):
            if hflip:
                video = video[:, ::-1, :, :]
            if vflip:
                video = video[:, :, ::-1, :]
            if dflip:
                video = np.array([np.transpose(frame, (1,0,2)) for frame in video])
            return video.copy()
        hr = torch.from_numpy(base_augment(hr)).permute(0,3,1,2).float() # (T,C,H,W)
        lr = torch.from_numpy(base_augment(lr)).permute(0,3,1,2).float() # (T,C,h,w)

        coord = make_coord_3d(hr.permute(1,0,2,3).shape[-3:], flatten=False) # (T,H,W,3)
        cell = torch.ones_like(coord) # (T,H,W,3)
        cell[:,:,:,0] *= 2 / hr.shape[0]
        cell[:,:,:,1] *= 2 / hr.shape[-2]
        cell[:,:,:,2] *= 2 / hr.shape[-1]

        # P = self.sample_size
        # hr, pos = random_crop(hr, P, return_pos=True) # (C,P,P)
        # coord = coord[pos[0]:pos[0]+P, pos[1]:pos[1]+P] # (P,P,2)
        # cell = cell[pos[0]:pos[0]+P, pos[1]:pos[1]+P] # (P,P,2)

        return {
            'lr': lr,
            'coord': coord,
            'cell': cell,
            'hr': hr,
            'gt': gt
        }
