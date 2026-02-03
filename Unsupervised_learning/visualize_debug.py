#!/usr/bin/env python3
"""
Unsupervised Learning - Visualization Script (NO DataLoader)

Visualizes predictions from trained model on test samples.
Creates side-by-side comparisons of:
- PA (original patch A)
- PB (target patch B)
- PA_warped (PA after predicted homography)
- Difference map
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    """Bound h4pt predictions."""
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


def visualize_predictions(model, dataset, device, num_samples=9, rho=32.0,
                          bound_mode="none", align_corners=False, save_path="predictions.png"):
    """
    Visualize model predictions on random samples.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Where to save visualization
    """
    model.eval()
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols * 4, figsize=(cols * 12, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            row = plot_idx // cols
            col_base = (plot_idx % cols) * 4
            
            # Get sample
            sample = dataset[sample_idx]
            
            # Prepare tensors
            stacked = sample["stacked"].unsqueeze(0).to(device)
            PA = sample["PA"].unsqueeze(0).to(device)
            PB = sample["PB"].unsqueeze(0).to(device)
            CA = sample["CA"].unsqueeze(0).to(device)
            
            # Forward pass
            h4pt_hat = model(stacked)
            h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)
            
            H_hat = tensor_dlt(CA, h4pt_hat)
            PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)
            
            # Convert to numpy for visualization
            PA_np = PA[0, 0].cpu().numpy()
            PB_np = PB[0, 0].cpu().numpy()
            PA_warp_np = PA_warp[0, 0].cpu().numpy()
            diff_np = np.abs(PA_warp_np - PB_np)
            
            # Plot PA
            ax = axes[row, col_base] if rows > 1 or cols > 1 else axes[0, col_base]
            ax.imshow(PA_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Sample {sample_idx}\nPA (Original)")
            ax.axis('off')
            
            # Plot PB
            ax = axes[row, col_base + 1] if rows > 1 or cols > 1 else axes[0, col_base + 1]
            ax.imshow(PB_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title("PB (Target)")
            ax.axis('off')
            
            # Plot PA_warped
            ax = axes[row, col_base + 2] if rows > 1 or cols > 1 else axes[0, col_base + 2]
            ax.imshow(PA_warp_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title("PA Warped")
            ax.axis('off')
            
            # Plot difference
            ax = axes[row, col_base + 3] if rows > 1 or cols > 1 else axes[0, col_base + 3]
            im = ax.imshow(diff_np, cmap='hot', vmin=0, vmax=0.5)
            ax.set_title(f"Diff (Mean: {diff_np.mean():.3f})")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Print h4pt prediction
            if sample_idx == indices[0]:
                h4pt_vals = h4pt_hat[0].cpu().numpy()
                print(f"\nSample {sample_idx} H4Pt prediction:")
                print(f"  {h4pt_vals}")
                print(f"  Mean magnitude: {np.abs(h4pt_vals).mean():.3f}")
                print(f"  Max magnitude: {np.abs(h4pt_vals).max():.3f}")
    
    # Hide unused subplots
    for plot_idx in range(num_samples, rows * cols):
        row = plot_idx // cols
        for col_offset in range(4):
            col_base = (plot_idx % cols) * 4 + col_offset
            ax = axes[row, col_base] if rows > 1 or cols > 1 else axes[0, col_base]
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close()


def visualize_single_detailed(model, dataset, device, sample_idx=0, rho=32.0,
                               bound_mode="none", align_corners=False, save_path="detailed_prediction.png"):
    """
    Create detailed visualization of a single prediction.
    Shows warped corners and homography matrix.
    """
    model.eval()
    
    # Get sample
    sample = dataset[sample_idx]
    stem = sample["stem"]
    
    # Prepare tensors
    stacked = sample["stacked"].unsqueeze(0).to(device)
    PA = sample["PA"].unsqueeze(0).to(device)
    PB = sample["PB"].unsqueeze(0).to(device)
    CA = sample["CA"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass
        h4pt_hat = model(stacked)
        h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)
        
        H_hat = tensor_dlt(CA, h4pt_hat)
        PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)
        
        # Get valid mask
        ones = torch.ones_like(PA)
        ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
        valid_mask = (ones_warp > 0.9).float()
    
    # Convert to numpy
    PA_np = PA[0, 0].cpu().numpy()
    PB_np = PB[0, 0].cpu().numpy()
    PA_warp_np = PA_warp[0, 0].cpu().numpy()
    valid_mask_np = valid_mask[0, 0].cpu().numpy()
    diff_np = np.abs(PA_warp_np - PB_np)
    
    h4pt_vals = h4pt_hat[0].cpu().numpy()
    H_mat = H_hat[0].cpu().numpy()
    CA_vals = CA[0].cpu().numpy()
    
    # Compute CB (warped corners)
    CB_vals = CA_vals + h4pt_vals.reshape(4, 2)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # PA
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(PA_np, cmap='gray', vmin=0, vmax=1)
    ax1.scatter(CA_vals[:, 0], CA_vals[:, 1], c='red', s=100, marker='o', label='CA corners')
    ax1.set_title(f"PA (Original)\nSample: {stem}")
    ax1.legend()
    ax1.axis('off')
    
    # PB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(PB_np, cmap='gray', vmin=0, vmax=1)
    ax2.scatter(CB_vals[:, 0], CB_vals[:, 1], c='blue', s=100, marker='x', label='CB corners (target)')
    ax2.set_title("PB (Target)")
    ax2.legend()
    ax2.axis('off')
    
    # PA warped
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(PA_warp_np, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("PA Warped")
    ax3.axis('off')
    
    # Difference
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(diff_np, cmap='hot', vmin=0, vmax=0.5)
    ax4.set_title(f"Absolute Difference\nMean: {diff_np.mean():.4f}")
    ax4.axis('off')
    plt.colorbar(im, ax=ax4)
    
    # Valid mask
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(valid_mask_np, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f"Valid Mask\nValid %: {valid_mask_np.mean()*100:.1f}%")
    ax5.axis('off')
    
    # Overlay
    ax6 = fig.add_subplot(gs[1, 2])
    overlay = np.stack([PB_np, PA_warp_np, PB_np], axis=-1)
    ax6.imshow(overlay)
    ax6.set_title("Overlay (PB=R/B, Warp=G)")
    ax6.axis('off')
    
    # Text info
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    info_text = f"""
Predicted H4Pt (corner offsets):
{h4pt_vals}

Mean magnitude: {np.abs(h4pt_vals).mean():.3f}
Max magnitude: {np.abs(h4pt_vals).max():.3f}

Homography Matrix (3x3):
{H_mat}

Corner Positions:
CA (original):  {CA_vals.tolist()}
CB (predicted): {CB_vals.tolist()}
    """
    
    ax7.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax7.transAxes)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved detailed visualization to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize unsupervised homography predictions")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--test_dir", type=str, required=True,
                       help="Path to test/val patch directory")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Where to save visualizations")
    
    # Visualization settings
    parser.add_argument("--num_samples", type=int, default=9,
                       help="Number of samples to visualize in grid")
    parser.add_argument("--detailed_idx", type=int, default=0,
                       help="Index for detailed single visualization")
    
    # Model settings
    parser.add_argument("--rho", type=float, default=32.0)
    parser.add_argument("--bound_mode", default="none", choices=["none", "clamp", "tanh"])
    parser.add_argument("--align_corners", action="store_true")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = Homographynet(in_channels=2).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
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
    
    # Create visualizations
    print(f"\nCreating grid visualization ({args.num_samples} samples)...")
    visualize_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        num_samples=args.num_samples,
        rho=args.rho,
        bound_mode=args.bound_mode,
        align_corners=args.align_corners,
        save_path=str(output_dir / "predictions_grid.png")
    )
    
    print(f"\nCreating detailed visualization (sample {args.detailed_idx})...")
    visualize_single_detailed(
        model=model,
        dataset=test_dataset,
        device=device,
        sample_idx=args.detailed_idx,
        rho=args.rho,
        bound_mode=args.bound_mode,
        align_corners=args.align_corners,
        save_path=str(output_dir / f"detailed_sample_{args.detailed_idx}.png")
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()