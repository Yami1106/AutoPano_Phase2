#!/usr/bin/env python3
"""
Evaluation metrics for homography corner regression.
"""

import torch


def reshape_h4pt(h4pt: torch.Tensor) -> torch.Tensor:
    """
    h4pt: (B,8) or (8,)
    returns: (B,4,2) or (4,2)
    """
    if h4pt.ndim == 1:
        if h4pt.numel() != 8:
            raise ValueError("Expected (8,) for single sample")
        return h4pt.view(4, 2)
    if h4pt.ndim == 2:
        if h4pt.shape[1] != 8:
            raise ValueError("Expected (B,8)")
        return h4pt.view(-1, 4, 2)
    raise ValueError("h4pt must be 1D or 2D")


def corner_l2_errors_px(CA: torch.Tensor, h4_pred: torch.Tensor, h4_gt: torch.Tensor) -> torch.Tensor:
    """
    CA:      (B,4,2)
    h4_pred: (B,8)
    h4_gt:   (B,8)

    returns:
      per-corner L2 error: (B,4)
    """
    pred = reshape_h4pt(h4_pred)  # (B,4,2)
    gt = reshape_h4pt(h4_gt)      # (B,4,2)

    if CA.ndim != 3 or CA.shape[1:] != (4, 2):
        raise ValueError(f"CA must be (B,4,2), got {CA.shape}")

    corners_pred = CA + pred
    corners_gt = CA + gt
    err = torch.linalg.norm(corners_pred - corners_gt, dim=-1)  # (B,4)
    return err


def summarize_errors(err_per_corner: torch.Tensor) -> dict:
    """
    err_per_corner: (N,4) or (N,)
    returns dict of stats
    """
    e = err_per_corner.reshape(-1)  # flatten all corners
    mean = e.mean()
    rmse = torch.sqrt((e ** 2).mean())
    median = e.median()
    p90 = torch.quantile(e, 0.90)
    p95 = torch.quantile(e, 0.95)

    return {
        "mean_px": float(mean.item()),
        "rmse_px": float(rmse.item()),
        "median_px": float(median.item()),
        "p90_px": float(p90.item()),
        "p95_px": float(p95.item()),
        "count_corners": int(e.numel()),
    }
