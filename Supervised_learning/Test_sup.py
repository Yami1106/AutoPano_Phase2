#!/usr/bin/env python
"""
Enhanced Test.py with improved visualizations including bounding boxes
for Ground Truth PA, Ground Truth PB, and Predicted corners.

This follows the HomographyNet supervised learning approach.
"""

import torch
import cv2
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Network.Network import HomographyModel
import seaborn as sns


def LoadModel(ModelPath, device):
    """
    Load the trained model from checkpoint.

    Args:
        ModelPath: Path to the model checkpoint
        device: Device to load model on (cuda/cpu)

    Returns:
        model: Loaded HomographyModel
    """
    hparams = {"InputSize": 128, "OutputSize": 8}
    model = HomographyModel(hparams=hparams).to(device)

    # Load checkpoint
    checkpoint = torch.load(ModelPath, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model.eval()
    return model


def LoadTestData(BasePath, mode="test"):
    """
    Load test/validation data.

    Args:
        BasePath: Base path to data
        mode: "test" or "val"

    Returns:
        stems: List of file stems
        coordinates: List of ground truth coordinates
        patch_path: Path to patches
    """
    if mode == "test":
        patch_path = os.path.join(BasePath, "patch_test")
    else:
        patch_path = os.path.join(BasePath, "patch_val")

    if not os.path.exists(patch_path):
        raise ValueError(f"Path {patch_path} does not exist!")

    # Get all stems
    stems = [
        f.name.replace("_H4Pt.npy", "") for f in Path(patch_path).glob("*_H4Pt.npy")
    ]

    # Load ground truth coordinates
    coordinates = [np.load(os.path.join(patch_path, f"{s}_H4Pt.npy")) for s in stems]

    print(f"Found {len(stems)} samples in {patch_path}")
    return stems, coordinates, patch_path


def EvaluatePixelError(pred_corners, gt_corners):
    """
    Compute mean corner error in pixels.

    Args:
        pred_corners: Predicted 4-point homography offsets [B, 8]
        gt_corners: Ground truth 4-point homography offsets [B, 8]

    Returns:
        mean_error: Mean pixel error across all corners
        corner_errors: Per-corner errors [B, 4]
    """
    # Reshape to [B, 4, 2] for corner-wise computation
    pred = pred_corners.reshape(-1, 4, 2)
    gt = gt_corners.reshape(-1, 4, 2)

    # Compute Euclidean distance for each corner
    corner_errors = torch.sqrt(((pred - gt) ** 2).sum(dim=2))  # [B, 4]

    # Mean error across all corners and batch
    mean_error = corner_errors.mean()

    return mean_error.item(), corner_errors


def draw_bounding_box(ax, corners, color, label, linestyle="-", linewidth=2, alpha=1.0):
    """
    Draw a bounding box connecting the 4 corners.

    Args:
        ax: Matplotlib axis
        corners: Array of shape (4, 2) with corner coordinates
        color: Color for the box
        label: Label for the legend
        linestyle: Line style
        linewidth: Line width
        alpha: Transparency
    """
    # Create polygon (closed loop)
    polygon = Polygon(
        corners,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )
    ax.add_patch(polygon)

    # Also draw corner points
    ax.scatter(corners[:, 0], corners[:, 1], c=color, s=50, zorder=5, alpha=alpha)

    # Number the corners
    for i, (x, y) in enumerate(corners):
        ax.text(
            x + 3,
            y + 3,
            f"{i+1}",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=color
            ),
        )


def get_pa_corners(patch_size=128, rho=32):
    """
    Get the corners of Patch A (reference patch).
    In HomographyNet, PA is centered, so corners are at the boundary.

    Args:
        patch_size: Size of the patch
        rho: Perturbation range

    Returns:
        corners: (4, 2) array of PA corner coordinates
    """
    # PA corners are at the boundary of the patch
    # Order: top-left, top-right, bottom-right, bottom-left
    half = patch_size // 2
    corners = np.array(
        [
            [half - rho, half - rho],  # Top-left
            [half + rho, half - rho],  # Top-right
            [half + rho, half + rho],  # Bottom-right
            [half - rho, half + rho],  # Bottom-left
        ],
        dtype=np.float32,
    )

    return corners


def get_pb_corners_from_h4pt(h4pt, patch_size=128):
    """
    Get PB corners from H4Pt (4-point homography representation).

    Args:
        h4pt: (8,) array of corner offsets [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
        patch_size: Size of the patch

    Returns:
        corners: (4, 2) array of PB corner coordinates
    """
    center = patch_size // 2
    h4pt_reshaped = h4pt.reshape(4, 2)
    corners = np.array([center, center]) + h4pt_reshaped
    return corners


def VisualizeExamplesEnhanced(
    model,
    stems,
    coordinates,
    patch_path,
    device,
    num_examples=5,
    save_path="Phase2/TestResults",
    rho=32,
):
    """
    Enhanced visualization with bounding boxes for PA GT, PB GT, and Predicted.

    Args:
        model: Trained model
        stems: List of file stems
        coordinates: Ground truth coordinates
        patch_path: Path to patches
        device: Device to run on
        num_examples: Number of examples to visualize
        save_path: Path to save visualizations
        rho: Perturbation range used in data generation
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()

    # Select examples: best, worst, and median
    all_errors = []

    print("\nComputing errors for example selection...")
    with torch.no_grad():
        for stem in tqdm(stems):
            # Load patches
            path_a = os.path.join(patch_path, f"{stem}_PA.png")
            path_b = os.path.join(patch_path, f"{stem}_PB.png")

            pa = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
            pb = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

            # Preprocess
            pa_t = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
            pb_t = torch.from_numpy(pb).float().unsqueeze(0) / 255.0
            input_tensor = torch.cat([pa_t, pb_t], dim=0).unsqueeze(0).to(device)

            # Get prediction
            pred = model(input_tensor)

            # Load GT
            gt_idx = stems.index(stem)
            gt_tensor = (
                torch.from_numpy(coordinates[gt_idx]).float().unsqueeze(0).to(device)
            )

            # Compute error
            error, _ = EvaluatePixelError(pred, gt_tensor)
            all_errors.append((stem, error))

    # Sort by error
    all_errors.sort(key=lambda x: x[1])

    # Select examples
    examples = [
        ("best", all_errors[0]),
        ("median", all_errors[len(all_errors) // 2]),
        ("worst", all_errors[-1]),
    ]

    # Add random examples
    if len(all_errors) > 3:
        import random

        random_samples = random.sample(
            all_errors[1:-1], min(num_examples - 3, len(all_errors) - 2)
        )
        examples.extend([("random", ex) for ex in random_samples])

    print(f"\nVisualizing {len(examples)} examples with bounding boxes...")

    for idx, (label, (stem, error)) in enumerate(examples):
        # Load patches
        path_a = os.path.join(patch_path, f"{stem}_PA.png")
        path_b = os.path.join(patch_path, f"{stem}_PB.png")

        pa = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        pb = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

        # Get prediction
        pa_t = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
        pb_t = torch.from_numpy(pb).float().unsqueeze(0) / 255.0
        input_tensor = torch.cat([pa_t, pb_t], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        # Get GT
        gt_idx = stems.index(stem)
        gt_h4pt = coordinates[gt_idx]
        pred_h4pt = pred[0].cpu().numpy()

        # Get corners
        pa_corners = get_pa_corners(patch_size=128, rho=rho)
        pb_gt_corners = get_pb_corners_from_h4pt(gt_h4pt, patch_size=128)
        pb_pred_corners = get_pb_corners_from_h4pt(pred_h4pt, patch_size=128)

        # Compute per-corner errors
        corner_errors = np.linalg.norm(pb_pred_corners - pb_gt_corners, axis=1)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        # ===== Patch A with GT bounding box =====
        axes[0, 0].imshow(pa, cmap="gray")
        draw_bounding_box(
            axes[0, 0], pa_corners, color="cyan", label="PA Ground Truth", linewidth=2.5
        )
        axes[0, 0].set_title(
            "Patch A - Ground Truth Region", fontsize=13, fontweight="bold"
        )
        axes[0, 0].legend(loc="upper right", fontsize=10)
        axes[0, 0].axis("off")

        # ===== Patch B with GT bounding box =====
        axes[0, 1].imshow(pb, cmap="gray")
        draw_bounding_box(
            axes[0, 1],
            pb_gt_corners,
            color="lime",
            label="PB Ground Truth",
            linewidth=2.5,
        )
        axes[0, 1].set_title(
            "Patch B - Ground Truth Corners", fontsize=13, fontweight="bold"
        )
        axes[0, 1].legend(loc="upper right", fontsize=10)
        axes[0, 1].axis("off")

        # ===== Patch B with Predicted bounding box =====
        axes[1, 0].imshow(pb, cmap="gray")
        draw_bounding_box(
            axes[1, 0], pb_pred_corners, color="red", label="Predicted", linewidth=2.5
        )
        axes[1, 0].set_title(
            f"Patch B - Predicted Corners\nMean Error: {error:.2f}px",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 0].legend(loc="upper right", fontsize=10)
        axes[1, 0].axis("off")

        # ===== Overlay: GT vs Predicted =====
        axes[1, 1].imshow(pb, cmap="gray")
        draw_bounding_box(
            axes[1, 1],
            pb_gt_corners,
            color="lime",
            label="Ground Truth",
            linewidth=2.5,
            alpha=0.7,
        )
        draw_bounding_box(
            axes[1, 1],
            pb_pred_corners,
            color="red",
            label="Predicted",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )

        # Draw error vectors
        for i in range(4):
            axes[1, 1].plot(
                [pb_gt_corners[i, 0], pb_pred_corners[i, 0]],
                [pb_gt_corners[i, 1], pb_pred_corners[i, 1]],
                "yellow",
                linewidth=1.5,
                alpha=0.7,
                linestyle=":",
            )
            # Add error text
            mid_x = (pb_gt_corners[i, 0] + pb_pred_corners[i, 0]) / 2
            mid_y = (pb_gt_corners[i, 1] + pb_pred_corners[i, 1]) / 2
            axes[1, 1].text(
                mid_x,
                mid_y,
                f"{corner_errors[i]:.1f}px",
                color="yellow",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

        axes[1, 1].set_title(
            "Overlay - GT vs Predicted\n(Yellow lines show errors)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 1].legend(loc="upper right", fontsize=10)
        axes[1, 1].axis("off")

        # Overall title
        plt.suptitle(
            f"Example {idx+1}: {label.upper()} - {stem}\n"
            f"Mean Error: {error:.2f}px | Corner Errors: "
            f"[{corner_errors[0]:.1f}, {corner_errors[1]:.1f}, "
            f"{corner_errors[2]:.1f}, {corner_errors[3]:.1f}]px",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            os.path.join(save_path, f"enhanced_example_{idx+1}_{label}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: enhanced_example_{idx+1}_{label}.png (Error: {error:.2f}px)")


def CreateComparisonGrid(
    model,
    stems,
    coordinates,
    patch_path,
    device,
    save_path="Phase2/TestResults",
    num_samples=9,
    rho=32,
):
    """
    Create a grid showing multiple examples side-by-side.

    Args:
        model: Trained model
        stems: List of file stems
        coordinates: Ground truth coordinates
        patch_path: Path to patches
        device: Device to run on
        save_path: Path to save visualizations
        num_samples: Number of samples to show (should be perfect square)
        rho: Perturbation range
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()

    # Compute errors for all samples
    all_errors = []
    print("\nComputing errors for grid visualization...")

    with torch.no_grad():
        for stem in tqdm(stems):
            # Load patches
            path_a = os.path.join(patch_path, f"{stem}_PA.png")
            path_b = os.path.join(patch_path, f"{stem}_PB.png")

            pa = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
            pb = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

            # Preprocess
            pa_t = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
            pb_t = torch.from_numpy(pb).float().unsqueeze(0) / 255.0
            input_tensor = torch.cat([pa_t, pb_t], dim=0).unsqueeze(0).to(device)

            # Get prediction
            pred = model(input_tensor)

            # Load GT
            gt_idx = stems.index(stem)
            gt_tensor = (
                torch.from_numpy(coordinates[gt_idx]).float().unsqueeze(0).to(device)
            )

            # Compute error
            error, _ = EvaluatePixelError(pred, gt_tensor)
            all_errors.append(
                (stem, error, pa, pb, pred[0].cpu().numpy(), coordinates[gt_idx])
            )

    # Sort by error and select diverse samples
    all_errors.sort(key=lambda x: x[1])

    # Select samples: best, worst, and evenly spaced
    if num_samples >= len(all_errors):
        selected = all_errors
    else:
        indices = np.linspace(0, len(all_errors) - 1, num_samples, dtype=int)
        selected = [all_errors[i] for i in indices]

    # Create grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size)
    )
    axes = axes.flatten()

    print(f"\nCreating comparison grid with {len(selected)} samples...")

    for idx, (stem, error, pa, pb, pred_h4pt, gt_h4pt) in enumerate(selected):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get corners
        pb_gt_corners = get_pb_corners_from_h4pt(gt_h4pt, patch_size=128)
        pb_pred_corners = get_pb_corners_from_h4pt(pred_h4pt, patch_size=128)

        # Show patch B with overlay
        ax.imshow(pb, cmap="gray")
        draw_bounding_box(
            ax, pb_gt_corners, color="lime", label="GT", linewidth=1.5, alpha=0.6
        )
        draw_bounding_box(
            ax,
            pb_pred_corners,
            color="red",
            label="Pred",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )

        ax.set_title(f"Error: {error:.2f}px", fontsize=10, fontweight="bold")
        ax.axis("off")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Comparison Grid: Ground Truth (Green) vs Predicted (Red)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, "comparison_grid.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: comparison_grid.png")


def TestModel(model, stems, coordinates, patch_path, device, batch_size=32):
    """
    Test the model on all test samples.
    """
    model.eval()

    all_pixel_errors = []
    all_corner_errors = []
    all_losses = []

    num_samples = len(stems)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"\nTesting on {num_samples} samples...")

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_stems = stems[start_idx:end_idx]

            batch_inputs = []
            batch_gts = []

            for stem in batch_stems:
                # Load patches
                path_a = os.path.join(patch_path, f"{stem}_PA.png")
                path_b = os.path.join(patch_path, f"{stem}_PB.png")

                pa = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
                pb = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

                # Preprocess
                pa_t = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
                pb_t = torch.from_numpy(pb).float().unsqueeze(0) / 255.0
                input_tensor = torch.cat([pa_t, pb_t], dim=0)

                # Load GT
                gt_idx = stems.index(stem)
                gt_tensor = torch.from_numpy(coordinates[gt_idx]).float()

                batch_inputs.append(input_tensor)
                batch_gts.append(gt_tensor)

            # Stack batch
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_gts = torch.stack(batch_gts).to(device)

            # Get predictions
            preds = model(batch_inputs)

            # Compute loss
            from Network.Network import LossFn_supervised

            loss = LossFn_supervised(preds, batch_gts)
            all_losses.append(loss.item())

            # Compute pixel errors
            mean_error, corner_errors = EvaluatePixelError(preds, batch_gts)
            all_pixel_errors.append(mean_error)
            all_corner_errors.append(corner_errors.cpu().numpy())

    # Aggregate results
    all_corner_errors = np.concatenate(all_corner_errors, axis=0)

    results = {
        "mean_pixel_error": np.mean(all_pixel_errors),
        "median_pixel_error": np.median(all_pixel_errors),
        "std_pixel_error": np.std(all_pixel_errors),
        "max_pixel_error": np.max(all_pixel_errors),
        "min_pixel_error": np.min(all_pixel_errors),
        "mean_loss": np.mean(all_losses),
        "all_pixel_errors": all_pixel_errors,
        "all_corner_errors": all_corner_errors,
        "per_corner_mean": all_corner_errors.mean(axis=0),
        "per_corner_std": all_corner_errors.std(axis=0),
    }

    return results


def VisualizeResults(results, save_path="Phase2/TestResults"):
    """
    Visualize and save test results.
    """
    os.makedirs(save_path, exist_ok=True)
    sns.set_style("whitegrid")

    # Print Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mean Pixel Error:     {results['mean_pixel_error']:.4f} px")
    print(f"Median Pixel Error:   {results['median_pixel_error']:.4f} px")
    print(f"Std Pixel Error:      {results['std_pixel_error']:.4f} px")
    print(f"Max Pixel Error:      {results['max_pixel_error']:.4f} px")
    print(f"Min Pixel Error:      {results['min_pixel_error']:.4f} px")
    print(f"Mean Loss:            {results['mean_loss']:.6f}")
    print("-" * 60)
    print("Per-Corner Mean Errors:")
    for i, error in enumerate(results["per_corner_mean"]):
        print(f"  Corner {i+1}: {error:.4f} ± {results['per_corner_std'][i]:.4f} px")
    print("=" * 60 + "\n")

    # Error Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results["all_pixel_errors"], bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(
        results["mean_pixel_error"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {results["mean_pixel_error"]:.2f}px',
    )
    plt.axvline(
        results["median_pixel_error"],
        color="g",
        linestyle="--",
        linewidth=2,
        label=f'Median: {results["median_pixel_error"]:.2f}px',
    )
    plt.xlabel("Pixel Error (px)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(
        "Distribution of Corner Prediction Errors", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "error_distribution.png"), dpi=300)
    plt.close()

    # Per-Corner Error Box Plot
    plt.figure(figsize=(10, 6))
    corner_data = [results["all_corner_errors"][:, i] for i in range(4)]
    bp = plt.boxplot(
        corner_data,
        labels=["Corner 1", "Corner 2", "Corner 3", "Corner 4"],
        patch_artist=True,
        showmeans=True,
    )
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel("Pixel Error (px)", fontsize=12)
    plt.title("Error Distribution per Corner", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "per_corner_errors.png"), dpi=300)
    plt.close()

    # Save results to text file
    with open(os.path.join(save_path, "test_results.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean Pixel Error:     {results['mean_pixel_error']:.4f} px\n")
        f.write(f"Median Pixel Error:   {results['median_pixel_error']:.4f} px\n")
        f.write(f"Std Pixel Error:      {results['std_pixel_error']:.4f} px\n")
        f.write(f"Max Pixel Error:      {results['max_pixel_error']:.4f} px\n")
        f.write(f"Min Pixel Error:      {results['min_pixel_error']:.4f} px\n")
        f.write(f"Mean Loss:            {results['mean_loss']:.6f}\n")
        f.write("-" * 60 + "\n")
        f.write("Per-Corner Mean Errors:\n")
        for i, error in enumerate(results["per_corner_mean"]):
            f.write(
                f"  Corner {i+1}: {error:.4f} ± {results['per_corner_std'][i]:.4f} px\n"
            )
        f.write("=" * 60 + "\n")


def main():
    """
    Main testing function with enhanced visualizations.
    """
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="Phase2/Data",
        help="Base path to data, Default: Phase2/Data",
    )
    Parser.add_argument(
        "--ModelPath",
        default="Phase2/Checkpoints/best_model.ckpt",
        help="Path to trained model",
    )
    Parser.add_argument(
        "--Mode",
        default="test",
        help="Test on 'test' or 'val' set",
    )
    Parser.add_argument(
        "--BatchSize",
        type=int,
        default=32,
        help="Batch size for testing",
    )
    Parser.add_argument(
        "--SavePath",
        default="Phase2/TestResults",
        help="Path to save results",
    )
    Parser.add_argument(
        "--NumExamples",
        type=int,
        default=5,
        help="Number of detailed examples to visualize",
    )
    Parser.add_argument(
        "--Rho",
        type=int,
        default=32,
        help="Perturbation range (rho) used in data generation",
    )

    Args = Parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {Args.ModelPath}...")
    model = LoadModel(Args.ModelPath, device)

    # Load test data
    print(f"\nLoading {Args.Mode} data from {Args.BasePath}...")
    stems, coordinates, patch_path = LoadTestData(Args.BasePath, mode=Args.Mode)

    # Test model
    results = TestModel(model, stems, coordinates, patch_path, device, Args.BatchSize)

    # Visualize results
    print("\nGenerating visualizations...")
    VisualizeResults(results, save_path=Args.SavePath)

    # Enhanced examples with bounding boxes
    VisualizeExamplesEnhanced(
        model,
        stems,
        coordinates,
        patch_path,
        device,
        num_examples=Args.NumExamples,
        save_path=Args.SavePath,
        rho=Args.Rho,
    )

    # Create comparison grid
    CreateComparisonGrid(
        model,
        stems,
        coordinates,
        patch_path,
        device,
        save_path=Args.SavePath,
        num_samples=9,
        rho=Args.Rho,
    )

    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print(f"Results saved to: {Args.SavePath}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
