#!/usr/bin/env python3
"""
Phase 2 - GPU-Accelerated Synthetic Homography Data Generation

MAJOR SPEEDUP via:
1. Batch processing multiple samples per image on GPU
2. PyTorch-based warping (GPU accelerated)
3. Parallel image processing
4. Efficient tensor operations

Generates MULTIPLE patches per image (configurable via --samples_per_image).

Usage:
    python HomographyDataGen_GPU.py --gray --samples_per_image 10 --batch_size 50

Expected speedup: 10-50x faster than CPU version (depending on GPU)
"""

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from tqdm import tqdm
import torch
import torch.nn.functional as F


# =========================
# Config
# =========================
@dataclass
class DataGenConfig:
    patch_h: int = 128
    patch_w: int = 128
    rho: int = 32
    use_grayscale: bool = True
    allow_translation: bool = True
    max_translation: int = 16
    normalize: bool = True
    samples_per_image: int = 10
    batch_size: int = 50  # NEW: process this many samples at once on GPU
    seed: Optional[int] = None
    device: str = "cuda"  # "cuda" or "cpu"


# =========================
# Paths (fixed + portable)
# =========================
THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parent
PHASE2_DIR = THIS_FILE.parents[2] if len(THIS_FILE.parents) > 2 else THIS_FILE.parent
DATA_DIR = PHASE2_DIR / "Data"
OVERLAY_DIR = CODE_DIR / "overlays"


# =========================
# Utilities
# =========================
def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_zip(data_root: Path):
    """Auto-extract Train.zip / Val.zip if folders don't exist."""
    for split in ["Train", "Val"]:
        split_dir = data_root / split
        zip_path = data_root / f"{split}.zip"

        if split_dir.exists():
            continue

        if not zip_path.exists():
            raise FileNotFoundError(
                f"Expected either {split_dir} or {zip_path} to exist"
            )

        print(f"[INFO] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)

        if not split_dir.exists():
            raise RuntimeError(f"Extraction failed for {zip_path}")


def read_image(path: Path, use_grayscale: bool) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def ensure_min_size(
    img: np.ndarray, patch_h: int, patch_w: int, rho: int
) -> np.ndarray:
    """Ensures image is big enough for sampling."""
    H, W = img.shape[:2]
    minH = patch_h + 2 * rho + 1
    minW = patch_w + 2 * rho + 1

    if H >= minH and W >= minW:
        return img

    scale = max(minH / H, minW / W)
    newW = int(np.ceil(W * scale))
    newH = int(np.ceil(H * scale))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
    return resized


# =========================
# GPU-Accelerated Batch Generation
# =========================


def sample_batch_patches(
    img_tensor: torch.Tensor,  # (1, H, W) on device
    cfg: DataGenConfig,
    num_samples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Generate multiple random patches from a single image on GPU.

    Returns:
        PA_batch: (N, 1, pH, pW)
        PB_batch: (N, 1, pH, pW)
        CA_batch: (N, 4, 2)
        CB_batch: (N, 4, 2)
        HAB_batch: (N, 3, 3)
        HBA_batch: (N, 3, 3)
        xy_batch: (N, 2)
    """
    _, H, W = img_tensor.shape

    # Calculate valid sampling region
    x_min = cfg.rho
    y_min = cfg.rho
    x_max = W - cfg.patch_w - cfg.rho
    y_max = H - cfg.patch_h - cfg.rho

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Image too small even after resize")

    # Sample random top-left corners (on CPU, then move to device)
    xs = torch.randint(x_min, x_max + 1, (num_samples,), device=device)
    ys = torch.randint(y_min, y_max + 1, (num_samples,), device=device)

    # Build CA for all samples: [TL, TR, BR, BL]
    CA_batch = torch.zeros(num_samples, 4, 2, device=device, dtype=torch.float32)
    CA_batch[:, 0, 0] = xs.float()  # TL x
    CA_batch[:, 0, 1] = ys.float()  # TL y
    CA_batch[:, 1, 0] = xs.float() + cfg.patch_w  # TR x
    CA_batch[:, 1, 1] = ys.float()  # TR y
    CA_batch[:, 2, 0] = xs.float() + cfg.patch_w  # BR x
    CA_batch[:, 2, 1] = ys.float() + cfg.patch_h  # BR y
    CA_batch[:, 3, 0] = xs.float()  # BL x
    CA_batch[:, 3, 1] = ys.float() + cfg.patch_h  # BL y

    # Perturb corners to get CB
    delta = torch.empty(num_samples, 4, 2, device=device).uniform_(-cfg.rho, cfg.rho)

    if cfg.allow_translation:
        tx = torch.empty(num_samples, 1, 1, device=device).uniform_(
            -cfg.max_translation, cfg.max_translation
        )
        ty = torch.empty(num_samples, 1, 1, device=device).uniform_(
            -cfg.max_translation, cfg.max_translation
        )
        translation = torch.cat([tx, ty], dim=2)  # (N, 1, 2)
        delta = delta + translation

    CB_batch = CA_batch + delta

    # Compute homographies using batched DLT
    HAB_batch = batch_compute_homography(CA_batch, CB_batch)  # CA -> CB
    HBA_batch = torch.linalg.inv(HAB_batch)  # CB -> CA

    # Warp entire image with each homography
    img_batch = img_tensor.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # (N, 1, H, W)
    imgB_batch = batch_warp_image(img_batch, HBA_batch)  # (N, 1, H, W)

    # Extract patches from original and warped images
    PA_batch = batch_extract_patches(img_batch, xs, ys, cfg.patch_w, cfg.patch_h)
    PB_batch = batch_extract_patches(imgB_batch, xs, ys, cfg.patch_w, cfg.patch_h)

    # Store xy coordinates
    xy_batch = torch.stack([xs, ys], dim=1)  # (N, 2)

    return {
        "PA": PA_batch,
        "PB": PB_batch,
        "CA": CA_batch,
        "CB": CB_batch,
        "HAB": HAB_batch,
        "HBA": HBA_batch,
        "xy": xy_batch,
    }


def batch_compute_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute homography for a batch using DLT.

    src, dst: (B, 4, 2)
    returns: (B, 3, 3)
    """
    B = src.shape[0]
    x = src[:, :, 0]  # (B, 4)
    y = src[:, :, 1]
    u = dst[:, :, 0]
    v = dst[:, :, 1]

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    # Build A matrix for each sample
    Au = torch.stack(
        [x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1
    )  # (B, 4, 8)
    Av = torch.stack(
        [zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1
    )  # (B, 4, 8)

    A = torch.cat([Au, Av], dim=1)  # (B, 8, 8)
    b = torch.cat([u, v], dim=1).unsqueeze(-1)  # (B, 8, 1)

    # Solve Ah = b for each sample
    h = torch.linalg.solve(A, b).squeeze(-1)  # (B, 8)

    # Reshape to 3x3 matrices
    H = torch.zeros(B, 3, 3, device=src.device, dtype=src.dtype)
    H[:, 0, 0] = h[:, 0]
    H[:, 0, 1] = h[:, 1]
    H[:, 0, 2] = h[:, 2]
    H[:, 1, 0] = h[:, 3]
    H[:, 1, 1] = h[:, 4]
    H[:, 1, 2] = h[:, 5]
    H[:, 2, 0] = h[:, 6]
    H[:, 2, 1] = h[:, 7]
    H[:, 2, 2] = 1.0

    return H


def batch_warp_image(img_batch: torch.Tensor, H_batch: torch.Tensor) -> torch.Tensor:
    """
    Warp a batch of images using corresponding homographies.

    img_batch: (B, 1, H, W)
    H_batch: (B, 3, 3) - inverse homographies (target -> source)
    returns: (B, 1, H, W)
    """
    B, _, H, W = img_batch.shape
    device = img_batch.device

    # Create normalized grid for grid_sample: [-1, 1]
    y_range = torch.linspace(-1, 1, H, device=device)
    x_range = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

    # Convert normalized [-1,1] to pixel coordinates
    xx_px = (xx + 1) * (W - 1) / 2
    yy_px = (yy + 1) * (H - 1) / 2

    # Create homogeneous coordinates
    ones = torch.ones_like(xx_px)
    grid_homo = torch.stack([xx_px, yy_px, ones], dim=0)  # (3, H, W)
    grid_homo = grid_homo.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)

    # Apply homography
    grid_warped = H_batch @ grid_homo  # (B, 3, H*W)

    # Convert from homogeneous
    grid_x = grid_warped[:, 0, :] / (grid_warped[:, 2, :] + 1e-8)
    grid_y = grid_warped[:, 1, :] / (grid_warped[:, 2, :] + 1e-8)

    # Normalize back to [-1, 1] for grid_sample
    grid_x_norm = 2.0 * grid_x / (W - 1) - 1.0
    grid_y_norm = 2.0 * grid_y / (H - 1) - 1.0

    # Reshape to (B, H, W, 2)
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1).view(B, H, W, 2)

    # Warp
    warped = F.grid_sample(
        img_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return warped


def batch_extract_patches(
    img_batch: torch.Tensor,  # (B, 1, H, W)
    xs: torch.Tensor,  # (B,)
    ys: torch.Tensor,  # (B,)
    patch_w: int,
    patch_h: int,
) -> torch.Tensor:
    """
    Extract patches from batch of images at given coordinates.

    Returns: (B, 1, patch_h, patch_w)
    """
    B, _, H, W = img_batch.shape
    device = img_batch.device

    # Create grid for each patch
    y_range = torch.arange(patch_h, device=device, dtype=torch.float32)
    x_range = torch.arange(patch_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

    # Add offsets for each sample
    xx = xx.unsqueeze(0) + xs.view(-1, 1, 1)  # (B, pH, pW)
    yy = yy.unsqueeze(0) + ys.view(-1, 1, 1)  # (B, pH, pW)

    # Normalize to [-1, 1]
    xx_norm = 2.0 * xx / (W - 1) - 1.0
    yy_norm = 2.0 * yy / (H - 1) - 1.0

    grid = torch.stack([xx_norm, yy_norm], dim=-1)  # (B, pH, pW, 2)

    # Extract patches
    patches = F.grid_sample(
        img_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return patches


def stack_patches_gpu(
    PA: torch.Tensor, PB: torch.Tensor, normalize: bool
) -> torch.Tensor:
    """Stack PA and PB on GPU."""
    if normalize:
        PA = PA.float()
        PB = PB.float()
        PA = (PA - PA.amin(dim=(2, 3), keepdim=True)) / (
            PA.amax(dim=(2, 3), keepdim=True) - PA.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        PB = (PB - PB.amin(dim=(2, 3), keepdim=True)) / (
            PB.amax(dim=(2, 3), keepdim=True) - PB.amin(dim=(2, 3), keepdim=True) + 1e-8
        )

    return torch.cat([PA, PB], dim=1)  # (B, 2, H, W)


def h4pt_label_gpu(CA: torch.Tensor, CB: torch.Tensor) -> torch.Tensor:
    """Compute H4pt labels on GPU."""
    return (CB - CA).reshape(CA.shape[0], -1)  # (B, 8)


# =========================
# Saving (CPU operations, but parallelizable)
# =========================


def save_batch_outputs(
    batch_data: Dict[str, torch.Tensor],
    img_stem: str,
    start_idx: int,
    patch_dir: Path,
    overlay_dir: Path,
    save_overlays: bool = True,
):
    """Save a batch of samples to disk."""
    # Move to CPU and convert to numpy
    PA = batch_data["PA"].cpu().numpy()  # (B, 1, H, W)
    PB = batch_data["PB"].cpu().numpy()
    CA = batch_data["CA"].cpu().numpy()  # (B, 4, 2)
    CB = batch_data["CB"].cpu().numpy()
    HAB = batch_data["HAB"].cpu().numpy()  # (B, 3, 3)
    HBA = batch_data["HBA"].cpu().numpy()
    xy = batch_data["xy"].cpu().numpy()  # (B, 2)

    h4pt = h4pt_label_gpu(batch_data["CA"], batch_data["CB"]).cpu().numpy()  # (B, 8)

    B = PA.shape[0]

    for i in range(B):
        sample_name = f"{img_stem}_sample{start_idx + i}"

        # Save patches
        _save_raw_patch(PA[i, 0], patch_dir / f"{sample_name}_PA.png")
        _save_raw_patch(PB[i, 0], patch_dir / f"{sample_name}_PB.png")

        # Save labels
        np.save(str(patch_dir / f"{sample_name}_H4Pt.npy"), h4pt[i].astype(np.float32))

        # Save metadata
        np.savez_compressed(
            str(patch_dir / f"{sample_name}_meta.npz"),
            CA=CA[i].astype(np.float32),
            CB=CB[i].astype(np.float32),
            HAB=HAB[i].astype(np.float32),
            HBA=HBA[i].astype(np.float32),
            xy=xy[i].astype(np.int32),
        )


def _save_raw_patch(patch: np.ndarray, out_path: Path):
    """Save a single patch (H, W) as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch_uint8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), patch_uint8)


# =========================
# Full split processing (GPU accelerated)
# =========================


def process_split_gpu(split_name: str, cfg: DataGenConfig):
    """Process entire split using GPU acceleration."""
    split_dir = DATA_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    img_paths = sorted(split_dir.glob("*.jpg"))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .jpg found in: {split_dir}")

    # Output directories
    patch_dir = DATA_DIR / ("patch_train" if split_name == "Train" else "patch_val")
    patch_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = OVERLAY_DIR / ("train" if split_name == "Train" else "val")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    total_samples = len(img_paths) * cfg.samples_per_image

    print(f"[INFO] Processing {split_name}: {len(img_paths)} images")
    print(f"[INFO] Generating {cfg.samples_per_image} samples per image")
    print(f"[INFO] Total samples to generate: {total_samples}")
    print(f"[INFO] Batch size: {cfg.batch_size}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Saving patches to: {patch_dir}")

    with tqdm(total=len(img_paths), desc=f"{split_name}") as pbar:
        for img_path in img_paths:
            # Read and prepare image
            imgA = read_image(img_path, cfg.use_grayscale)
            imgA = ensure_min_size(imgA, cfg.patch_h, cfg.patch_w, cfg.rho)

            # Convert to tensor and move to GPU
            if imgA.ndim == 2:
                imgA_tensor = (
                    torch.from_numpy(imgA).unsqueeze(0).float().to(device) / 255.0
                )
            else:
                imgA_tensor = (
                    torch.from_numpy(imgA).permute(2, 0, 1).float().to(device) / 255.0
                )

            img_stem = img_path.stem

            # Generate samples in batches
            num_batches = (cfg.samples_per_image + cfg.batch_size - 1) // cfg.batch_size

            for batch_idx in range(num_batches):
                start_sample = batch_idx * cfg.batch_size
                end_sample = min(
                    (batch_idx + 1) * cfg.batch_size, cfg.samples_per_image
                )
                batch_size = end_sample - start_sample

                # Generate batch on GPU
                batch_data = sample_batch_patches(
                    imgA_tensor,
                    cfg,
                    batch_size,
                    device,
                )

                # Save to disk (CPU operation)
                save_batch_outputs(
                    batch_data,
                    img_stem,
                    start_sample,
                    patch_dir,
                    overlay_dir,
                    save_overlays=False,  # Skip overlays for speed
                )

            pbar.update(1)


# =========================
# CLI
# =========================


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_h", type=int, default=128)
    ap.add_argument("--patch_w", type=int, default=128)
    ap.add_argument("--rho", type=int, default=32)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_translation", action="store_true")
    ap.add_argument("--max_translation", type=int, default=16)
    ap.add_argument("--samples_per_image", type=int, default=10)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of samples to process at once on GPU (default: 50)",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()


def main():
    args = _parse_args()

    # Ensure Train/Val are extracted
    extract_zip(DATA_DIR)

    cfg = DataGenConfig(
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        rho=args.rho,
        use_grayscale=args.gray,
        allow_translation=not args.no_translation,
        max_translation=args.max_translation,
        samples_per_image=args.samples_per_image,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    _set_seed(cfg.seed)

    # Process splits
    import time

    t0 = time.time()

    process_split_gpu("Train", cfg)
    process_split_gpu("Val", cfg)

    elapsed = time.time() - t0
    total_samples = cfg.samples_per_image * 2  # rough estimate for both splits

    print(f"\n[DONE] All patches saved.")
    print(f"[TIME] Total time: {elapsed:.1f}s")
    print(f"[SPEED] ~{total_samples/elapsed:.1f} samples/sec")


if __name__ == "__main__":
    main()
