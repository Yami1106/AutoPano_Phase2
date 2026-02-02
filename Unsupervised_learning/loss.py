#!/usr/bin/env python3

'''
Unsupervised Learning - Step 5.1 (FIXED)

Masked Photometric Loss
'''

import torch
import torch.nn as nn


class PhotometricLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        PA_warp: torch.Tensor,   # (B,1,H,W)
        PB: torch.Tensor,        # (B,1,H,W)
        valid_mask: torch.Tensor # (B,1,H,W), values in {0,1}
    ) -> torch.Tensor:

        if PA_warp.shape != PB.shape:
            raise ValueError(
                f"Shape mismatch: PA_warp={PA_warp.shape}, PB={PB.shape}"
            )

        if valid_mask.shape != PA_warp.shape:
            raise ValueError(
                f"Mask shape mismatch: mask={valid_mask.shape}, PA_warp={PA_warp.shape}"
            )

        # L1 difference
        diff = torch.abs(PA_warp - PB)

        # Apply mask
        masked_diff = diff * valid_mask

        # Mean over valid pixels only
        loss = masked_diff.sum() / (valid_mask.sum() + self.eps)

        return loss
