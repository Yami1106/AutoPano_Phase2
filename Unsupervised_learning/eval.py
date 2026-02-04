#!/usr/bin/env python3
"""
Unsupervised Learning - Evaluation Script (NO DataLoader)

Evaluates trained model on test/validation set.
Computes average photometric loss and corner error.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp
from Unsupervised_learning.loss import PhotometricLoss


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    """Bound h4pt predictions."""
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


def evaluate(model, dataset, device, batch_size=32, rho=32.0, bound_mode="none",
             mask_thresh=0.9, align_corners=False):
    """
    Evaluate model on dataset without DataLoader.
    
    Returns:
        avg_photo_loss: Average photometric loss
        avg_corner_error: Average corner error (if H4Pt_gt available)
    """
    model.eval()
    loss_fn = PhotometricLoss()
    
    total_photo_loss = 0.0
    total_corner_error = 0.0
    num_batches = 0
    num_samples_with_gt = 0
    
    num_samples = len(dataset)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Manually create batch
            batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
            
            # Stack into batches
            stacked = torch.stack([s["stacked"] for s in batch_samples]).to(device)
            PA = torch.stack([s["PA"] for s in batch_samples]).to(device)
            PB = torch.stack([s["PB"] for s in batch_samples]).to(device)
            CA = torch.stack([s["CA"] for s in batch_samples]).to(device)
            
            # Forward pass
            h4pt_hat = model(stacked)
            h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)
            
            H_hat = tensor_dlt(CA, h4pt_hat)
            PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)
            
            # Compute photometric loss
            ones = torch.ones_like(PA)
            ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
            valid_mask = (ones_warp > mask_thresh).float()
            
            photo_loss = loss_fn(PA_warp, PB, valid_mask)
            total_photo_loss += float(photo_loss.item())
            num_batches += 1
            
            # Compute corner error if ground truth available
            if "H4Pt_gt" in batch_samples[0]:
                H4Pt_gt = torch.stack([s["H4Pt_gt"] for s in batch_samples]).to(device)
                corner_error = torch.abs(h4pt_hat - H4Pt_gt).mean()
                total_corner_error += float(corner_error.item())
                num_samples_with_gt += 1
    
    avg_photo_loss = total_photo_loss / max(num_batches, 1)
    avg_corner_error = total_corner_error / max(num_samples_with_gt, 1) if num_samples_with_gt > 0 else -1
    
    return avg_photo_loss, avg_corner_error


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained unsupervised homography model")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--test_dir", type=str, required=True,
                       help="Path to test/val patch directory")
    
    # Model settings
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--rho", type=float, default=32.0)
    parser.add_argument("--bound_mode", default="none", choices=["none", "clamp", "tanh"])
    parser.add_argument("--mask_thresh", type=float, default=0.9)
    parser.add_argument("--align_corners", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = Homographynet(in_channels=2).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            print(f"  Train loss: {checkpoint['loss']:.6f}")
        if "val_loss" in checkpoint:
            print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading test dataset from: {args.test_dir}")
    test_dataset = UnsupervisedDataset(
        args.test_dir,
        patch_hw=(128, 128),
        return_h4pt_gt=True
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print(f"\nEvaluating...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Bound mode: {args.bound_mode}")
    print(f"  Rho: {args.rho}")
    
    avg_photo_loss, avg_corner_error = evaluate(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        rho=args.rho,
        bound_mode=args.bound_mode,
        mask_thresh=args.mask_thresh,
        align_corners=args.align_corners
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average Photometric Loss: {avg_photo_loss:.6f}")
    if avg_corner_error >= 0:
        print(f"Average Corner Error (MAE): {avg_corner_error:.6f} pixels")
    else:
        print("Corner Error: N/A (no ground truth H4Pt)")
    print("="*60)


if __name__ == "__main__":
    main()