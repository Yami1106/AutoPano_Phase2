#!/usr/bin/env python3

'''
Unsupervised Learning - Step 4

Differentiable homography warp

input :
  PA: (B,1,H,W) source patch
  H : (B,3,3) homography mapping CA -> CB (source -> target)

Output:
  PA_warp = warp(PA, H) that aligns with PB.

NEW:
  apply_warp_with_mask() returns (PA_warp, valid_mask) WITHOUT doing a second warp.
'''

from typing import Tuple
import torch
import torch.nn.functional as F


def homogeneous_pixel_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0)  # (3, H*W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1)   # (B,3,H*W)
    return grid


def normalize_grid(xy: torch.Tensor, H: int, W: int, align_corners: bool) -> torch.Tensor:
    x = xy[..., 0]
    y = xy[..., 1]

    if align_corners:
        x_norm = 2.0 * (x / (W - 1)) - 1.0
        y_norm = 2.0 * (y / (H - 1)) - 1.0
    else:
        x_norm = 2.0 * ((x + 0.5) / W) - 1.0
        y_norm = 2.0 * ((y + 0.5) / H) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, H*W, 2)
    return grid


def apply_warp(
    PA: torch.Tensor,
    H_mat: torch.Tensor,
    out_hw: Tuple[int, int] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:

    if PA.ndim != 4 or PA.shape[1] != 1:
        raise ValueError(f"PA must be (B,1,H,W). Got {PA.shape}")
    if H_mat.ndim != 3 or H_mat.shape[1:] != (3, 3):
        raise ValueError(f"H_mat must be (B,3,3). Got {H_mat.shape}")

    B, _, H, W = PA.shape
    if out_hw is None:
        outH, outW = H, W
    else:
        outH, outW = out_hw

    device = PA.device
    dtype = PA.dtype

    tgt_grid = homogeneous_pixel_grid(B, outH, outW, device=device, dtype=dtype)  # (B,3,outH*outW)

    # Inverse homography: target -> source
    H_inv = torch.linalg.inv(H_mat)  # (B,3,3)

    # Map target pixels back to source pixels
    src = H_inv @ tgt_grid  # (B,3,N)
    xs = src[:, 0, :] / (src[:, 2, :] + 1e-8)
    ys = src[:, 1, :] / (src[:, 2, :] + 1e-8)

    # (B, N, 2)
    src_xy = torch.stack([xs, ys], dim=-1)

    # Normalize to [-1,1] for grid_sample and reshape to (B,outH,outW,2)
    grid_norm = normalize_grid(src_xy, H, W, align_corners=align_corners)
    grid_norm = grid_norm.view(B, outH, outW, 2)

    # Sample
    PA_warp = F.grid_sample(
        PA,
        grid_norm,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return PA_warp


def apply_warp_with_mask(
    PA: torch.Tensor,
    H_mat: torch.Tensor,
    out_hw: Tuple[int, int] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same warp as apply_warp(), but ALSO returns a binary valid_mask (B,1,outH,outW)
    that is 1 where the sampling location lies inside the source image bounds.

    This avoids the expensive "warp ones" trick and is much faster.
    """
    if PA.ndim != 4 or PA.shape[1] != 1:
        raise ValueError(f"PA must be (B,1,H,W). Got {PA.shape}")
    if H_mat.ndim != 3 or H_mat.shape[1:] != (3, 3):
        raise ValueError(f"H_mat must be (B,3,3). Got {H_mat.shape}")

    B, _, H, W = PA.shape
    if out_hw is None:
        outH, outW = H, W
    else:
        outH, outW = out_hw

    device = PA.device
    dtype = PA.dtype

    tgt_grid = homogeneous_pixel_grid(B, outH, outW, device=device, dtype=dtype)  # (B,3,N)

    H_inv = torch.linalg.inv(H_mat)
    src = H_inv @ tgt_grid
    xs = src[:, 0, :] / (src[:, 2, :] + 1e-8)
    ys = src[:, 1, :] / (src[:, 2, :] + 1e-8)

    # valid in source bounds
    valid = (xs >= 0) & (xs <= (W - 1)) & (ys >= 0) & (ys <= (H - 1))
    valid_mask = valid.view(B, 1, outH, outW).float()

    src_xy = torch.stack([xs, ys], dim=-1)
    grid_norm = normalize_grid(src_xy, H, W, align_corners=align_corners)
    grid_norm = grid_norm.view(B, outH, outW, 2)

    PA_warp = F.grid_sample(
        PA,
        grid_norm,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return PA_warp, valid_mask


if __name__ == "__main__":
    B, H, W = 2, 128, 128
    PA = torch.rand(B, 1, H, W)
    I = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    out = apply_warp(PA, I)
    out2, m = apply_warp_with_mask(PA, I)
    print("PA:", PA.shape, "out:", out.shape, "out2:", out2.shape, "mask:", m.shape)
    print("mean abs diff apply_warp:", (PA - out).abs().mean().item())
    print("mean abs diff apply_warp_with_mask:", (PA - out2).abs().mean().item())
    print("mask unique:", torch.unique(m))
