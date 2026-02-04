#!/usr/bin/env python3
"""
Unsupervised Learning - Step 5.1

Masked Photometric Loss (Robust)
Supports:
  - L1
  - Charbonnier (recommended)
"""

import torch
import torch.nn as nn


class PhotometricLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, mode: str = "charbonnier", charbonnier_alpha: float = 0.5):
        """
        mode:
          - "l1"
          - "charbonnier"  (sqrt(x^2 + eps^2))^alpha ; alpha=0.5 gives standard charbonnier
        """
        super().__init__()
        self.eps = eps
        self.mode = mode
        self.alpha = charbonnier_alpha

    def forward(
        self,
        PA_warp: torch.Tensor,    # (B,1,H,W)
        PB: torch.Tensor,         # (B,1,H,W)
        valid_mask: torch.Tensor  # (B,1,H,W), values in {0,1}
    ) -> torch.Tensor:

        if PA_warp.shape != PB.shape:
            raise ValueError(f"Shape mismatch: PA_warp={PA_warp.shape}, PB={PB.shape}")

        if valid_mask.shape != PA_warp.shape:
            raise ValueError(f"Mask shape mismatch: mask={valid_mask.shape}, PA_warp={PA_warp.shape}")

        diff = PA_warp - PB

        if self.mode == "l1":
            per_pix = torch.abs(diff)
        elif self.mode == "charbonnier":
            # robust: (diff^2 + eps^2)^(alpha)
            per_pix = (diff * diff + (self.eps * self.eps)).pow(self.alpha)
        else:
            raise ValueError("PhotometricLoss mode must be one of: l1, charbonnier")

        # apply mask
        per_pix = per_pix * valid_mask

        # per-sample normalization by valid pixels to prevent cheating
        # (B,1,H,W) -> (B,)
        denom = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        loss_per_sample = per_pix.sum(dim=(1, 2, 3)) / denom

        # mean over batch
        return loss_per_sample.mean()
