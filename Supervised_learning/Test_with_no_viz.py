#!/usr/bin/env python
"""
RBE/CS Fall 2025: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""
#!/usr/bin/env python
import torch
import cv2
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    # Unscale to get actual pixel values (matching loss function scaling)
    # rho = 32.0
    # pred = pred * rho
    # gt = gt * rho

    # Compute Euclidean distance for each corner
    corner_errors = torch.sqrt(((pred - gt) ** 2).sum(dim=2))  # [B, 4]

    # Mean error across all corners and batch
    mean_error = corner_errors.mean()

    return mean_error.item(), corner_errors


def TestModel(model, stems, coordinates, patch_path, device, batch_size=32):
    """
    Test the model on all test samples.

    Args:
        model: Trained HomographyModel
        stems: List of file stems
        coordinates: Ground truth coordinates
        patch_path: Path to patch directory
        device: Device to run on
        batch_size: Batch size for testing

    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()

    all_pixel_errors = []
    all_corner_errors = []  # [N, 4] - per corner errors
    all_losses = []

    num_samples = len(stems)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"\nTesting on {num_samples} samples...")

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_stems = stems[start_idx:end_idx]

            # Load batch
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
    all_corner_errors = np.concatenate(all_corner_errors, axis=0)  # [N, 4]

    results = {
        "mean_pixel_error": np.mean(all_pixel_errors),
        "median_pixel_error": np.median(all_pixel_errors),
        "std_pixel_error": np.std(all_pixel_errors),
        "max_pixel_error": np.max(all_pixel_errors),
        "min_pixel_error": np.min(all_pixel_errors),
        "mean_loss": np.mean(all_losses),
        "all_pixel_errors": all_pixel_errors,
        "all_corner_errors": all_corner_errors,
        "per_corner_mean": all_corner_errors.mean(axis=0),  # [4]
        "per_corner_std": all_corner_errors.std(axis=0),  # [4]
    }

    return results


def VisualizeResults(results, save_path="Phase2/TestResults"):
    """
    Visualize and save test results.

    Args:
        results: Dictionary from TestModel
        save_path: Path to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Print Summary Statistics
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

    # 2. Error Distribution Histogram
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
    print(f"Saved: {save_path}/error_distribution.png")

    # 3. Per-Corner Error Box Plot
    plt.figure(figsize=(10, 6))
    corner_data = [results["all_corner_errors"][:, i] for i in range(4)]
    bp = plt.boxplot(
        corner_data,
        labels=["Corner 1", "Corner 2", "Corner 3", "Corner 4"],
        patch_artist=True,
        showmeans=True,
    )

    # Color the boxes
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
    print(f"Saved: {save_path}/per_corner_errors.png")

    # 4. Cumulative Error Distribution
    plt.figure(figsize=(10, 6))
    sorted_errors = np.sort(results["all_pixel_errors"])
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative, linewidth=2)
    plt.axvline(
        results["mean_pixel_error"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {results["mean_pixel_error"]:.2f}px',
    )
    plt.axhline(50, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(90, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Pixel Error (px)", fontsize=12)
    plt.ylabel("Cumulative Percentage (%)", fontsize=12)
    plt.title("Cumulative Error Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cumulative_distribution.png"), dpi=300)
    plt.close()
    print(f"Saved: {save_path}/cumulative_distribution.png")

    # 5. Save numerical results to text file
    results_file = os.path.join(save_path, "test_results.txt")
    with open(results_file, "w") as f:
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
    print(f"Saved: {results_file}")


def VisualizeExamples(
    model,
    stems,
    coordinates,
    patch_path,
    device,
    num_examples=5,
    save_path="Phase2/TestResults",
):
    """
    Visualize predictions on example images.

    Args:
        model: Trained model
        stems: List of file stems
        coordinates: Ground truth coordinates
        patch_path: Path to patches
        device: Device to run on
        num_examples: Number of examples to visualize
        save_path: Path to save visualizations
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

    print(f"\nVisualizing {len(examples)} examples...")

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
        gt = coordinates[gt_idx]

        # Unscale predictions

        # pred_np = (pred[0].cpu().numpy() * 32.0).reshape(4, 2)
        pred_np = pred[0].cpu().numpy().reshape(4, 2)

        # gt_np = (gt * 32.0).reshape(4, 2)
        gt_np = gt.reshape(4, 2)

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Patch A
        axes[0].imshow(pa, cmap="gray")
        axes[0].set_title("Patch A (Input)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Patch B with GT corners
        axes[1].imshow(pb, cmap="gray")
        # Draw GT corners (corners are relative to patch center)
        center = 64  # 128/2
        for i, (dx, dy) in enumerate(gt_np):
            x, y = center + dx, center + dy
            axes[1].plot(x, y, "go", markersize=10, label="GT" if i == 0 else "")
            axes[1].text(
                x + 3, y + 3, f"{i+1}", color="green", fontsize=10, fontweight="bold"
            )
        axes[1].set_title("Patch B with GT Corners", fontsize=12, fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].axis("off")

        # Patch B with predicted corners
        axes[2].imshow(pb, cmap="gray")
        for i, (dx, dy) in enumerate(pred_np):
            x, y = center + dx, center + dy
            axes[2].plot(
                x,
                y,
                "rx",
                markersize=10,
                markeredgewidth=2,
                label="Pred" if i == 0 else "",
            )
            axes[2].text(
                x + 3, y + 3, f"{i+1}", color="red", fontsize=10, fontweight="bold"
            )
        # Also draw GT for comparison
        for i, (dx, dy) in enumerate(gt_np):
            x, y = center + dx, center + dy
            axes[2].plot(x, y, "go", markersize=8, alpha=0.5)
        axes[2].set_title(
            f"Prediction (Error: {error:.2f}px)", fontsize=12, fontweight="bold"
        )
        axes[2].legend(loc="upper right")
        axes[2].axis("off")

        plt.suptitle(
            f"Example {idx+1}: {label.upper()} - {stem}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"example_{idx+1}_{label}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {save_path}/example_{idx+1}_{label}.png (Error: {error:.2f}px)")


def main():
    """
    Main testing function.
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
        help="Path to trained model, Default: Phase2/Checkpoints/best_model.ckpt",
    )
    Parser.add_argument(
        "--Mode",
        default="test",
        help="Test on 'test' or 'val' set, Default: test",
    )
    Parser.add_argument(
        "--BatchSize",
        type=int,
        default=32,
        help="Batch size for testing, Default: 32",
    )
    Parser.add_argument(
        "--SavePath",
        default="Phase2/TestResults",
        help="Path to save results, Default: Phase2/TestResults",
    )
    Parser.add_argument(
        "--NumExamples",
        type=int,
        default=5,
        help="Number of examples to visualize, Default: 5",
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

    # Visualize examples
    VisualizeExamples(
        model,
        stems,
        coordinates,
        patch_path,
        device,
        num_examples=Args.NumExamples,
        save_path=Args.SavePath,
    )

    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print(f"Results saved to: {Args.SavePath}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
