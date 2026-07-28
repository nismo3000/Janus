"""uint8 frame tensors -> normalized float tensors on device, plus clip augmentation."""

import torch
import torch.nn.functional as F


def to_model_input(frames_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float in [-1, 1]."""
    x = frames_u8.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
    return x.movedim(-1, -3).contiguous()


def augment_clips(x: torch.Tensor, gen: torch.Generator, scale_min: float = 0.65) -> torch.Tensor:
    """Random crop / flip / photometric jitter on (B, T, 3, R, R) clips.

    The transform is drawn per *clip* and applied identically to every frame in
    it. Jittering frames independently would destroy the very motion the model is
    supposed to predict -- the augmentation has to move the camera, not the world.
    """
    b, t = x.shape[:2]
    dev = x.device
    r = lambda lo, hi: (torch.rand(b, generator=gen, device=dev) * (hi - lo) + lo)

    s = r(scale_min, 1.0)
    flip = torch.where(torch.rand(b, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    tx = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * (1 - s)
    ty = (torch.rand(b, generator=gen, device=dev) * 2 - 1) * (1 - s)

    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = s * flip
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    flat = x.flatten(0, 1)
    grid = F.affine_grid(theta.repeat_interleave(t, 0), flat.shape, align_corners=False)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="reflection",
                        align_corners=False)

    contrast = r(0.8, 1.2).repeat_interleave(t).view(-1, 1, 1, 1)
    bright = r(-0.15, 0.15).repeat_interleave(t).view(-1, 1, 1, 1)
    out = (out * contrast + bright).clamp_(-1.0, 1.0)
    return out.view_as(x)
