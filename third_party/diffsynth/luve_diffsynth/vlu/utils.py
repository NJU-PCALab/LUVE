"""Coordinate utility retained from the supplied 3D LIIF implementation."""
import torch


def make_coord_3d(shape, ranges=None, flatten=True, device='cpu'):
    coordinates = []
    for index, length in enumerate(shape):
        lower, upper = (-1, 1) if ranges is None else ranges[index]
        radius = (upper - lower) / (2 * length)
        coordinates.append(lower + radius + 2 * radius * torch.arange(length, device=device).float())
    result = torch.stack(torch.meshgrid(*coordinates, indexing='ij'), dim=-1)
    return result.reshape(-1, result.shape[-1]) if flatten else result
