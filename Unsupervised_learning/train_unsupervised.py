#!/usr/bin/env python3
"""
Unsupervised Learning - Training Script (NO DataLoader)
FIXED VERSION - Stable, clean, no experimental features
"""

import os
import glob
import time
import random
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp
from Unsupervised_learning.loss import PhotometricLoss


# ----------------------------
# Helpers
# ----------------------------

def get_default_paths():
    this = Path(__file__).resolve()
    phase2 = this.parents[2]
    data_dir = phase2 / "Data"

    train_dir = data_dir / "patch_train"
    val_dir = data_dir / "patch_val"
    if not val_dir.exists():
        val_dir = data_dir / "patch_test"

    ckpt_dir = phase2 / "Checkpoints" / "unsup"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = phase2 / "Logs" / "unsup"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, train_dir, val_dir, ckpt_dir, logs_dir


def read_gray01_png(path: str) -> np.ndarray:
    """Read PNG as grayscale float32 in [0,1], shape (H,W)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


def extract_CA_from_meta(meta_path: str) -> np.ndarray:
    """Extract CA corners from meta.npz file."""
    meta = np.load(meta_path, allow_pickle=True)
    keys = list(meta.keys())

    candidate_keys = [
        "CA", "Ca", "cA",
        "cornersA", "CornersA", "cornerA",
        "PA_corners", "patchA_corners",
        "ptsA", "pts_a",
        "corners", "src_pts",
    ]

    for k in candidate_keys:
        if k in meta:
            CA = np.array(meta[k], dtype=np.float32).reshape(4, 2)
            return CA

    # Heuristic: any array that can reshape to (4,2)
    for k in keys:
        arr = np.array(meta[k])
        if arr.size == 8:
            try:
                return arr.astype(np.float32).reshape(4, 2)
            except Exception:
                pass

    raise KeyError(
        f"Could not find CA (4x2 corners) inside meta file: {meta_path}\n"
        f"Available keys: {keys}"
    )


def to_torch_chw(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy image to torch float tensor CHW."""
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim == 3:
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    return torch.from_numpy(x).to(device)


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    """Bound h4pt predictions."""
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


# ----------------------------
# Sample discovery and caching
# ----------------------------

def list_samples(patch_dir: Path):
    """Collect samples from a folder."""
    patch_dir = str(patch_dir)
    pa_files = sorted(glob.glob(os.path.join(patch_dir, "*_PA.png")))

    samples = []
    for pa in pa_files:
        base = pa[:-7]  # remove "_PA.png"
        pb = base + "_PB.png"
        meta = base + "_meta.npz"
        if os.path.exists(pb) and os.path.exists(meta):
            samples.append({"PA_png": pa, "PB_png": pb, "meta_npz": meta})

    return samples


def preload_samples(samples, device, max_cache=10000):
    """
    Pre-load samples into memory to avoid repeated disk I/O.
    """
    print(f"[INFO] Pre-loading up to {max_cache} samples into memory...")
    cached_samples = []
    
    for i, s in enumerate(samples[:max_cache]):
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{min(len(samples), max_cache)}...")
        
        try:
            PA = read_gray01_png(s["PA_png"])
            PB = read_gray01_png(s["PB_png"])
            CA = extract_CA_from_meta(s["meta_npz"])
            
            # Pre-convert to tensors and move to device
            PA_t = to_torch_chw(PA, device)
            PB_t = to_torch_chw(PB, device)
            stacked_t = torch.cat([PA_t, PB_t], dim=0)
            CA_t = torch.from_numpy(CA.astype(np.float32)).to(device)
            
            cached_samples.append({
                "stacked": stacked_t,
                "PA": PA_t,
                "PB": PB_t,
                "CA": CA_t,
            })
        except Exception as e:
            print(f"  Warning: Failed to load {s['PA_png']}: {e}")
            continue
    
    print(f"[INFO] Loaded {len(cached_samples)} samples into memory")
    return cached_samples


# ----------------------------
# Batch generation
# ----------------------------

def GenerateBatch(cached_samples, MiniBatchSize):
    """Generate batch from pre-loaded cached samples."""
    batch_samples = random.choices(cached_samples, k=MiniBatchSize)
    
    stacked_list = [s["stacked"] for s in batch_samples]
    pa_list = [s["PA"] for s in batch_samples]
    pb_list = [s["PB"] for s in batch_samples]
    ca_list = [s["CA"] for s in batch_samples]
    
    stacked_batch = torch.stack(stacked_list, dim=0)
    PA_batch = torch.stack(pa_list, dim=0)
    PB_batch = torch.stack(pb_list, dim=0)
    CA_batch = torch.stack(ca_list, dim=0)
    
    return stacked_batch, PA_batch, PB_batch, CA_batch


# ----------------------------
# Train / Val loops
# ----------------------------

@torch.no_grad()
def RunValidation(model, cached_samples, MiniBatchSize, NumIterations, loss_fn,
                  rho, bound_mode, mask_thresh, align_corners):
    model.eval()
    total = 0.0
    num_batches = 0

    for _ in range(NumIterations):
        stacked, PA, PB, CA = GenerateBatch(cached_samples, MiniBatchSize)

        h4pt_hat = model(stacked)
        h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)

        H_hat = tensor_dlt(CA, h4pt_hat)
        PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)

        ones = torch.ones_like(PA)
        ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
        valid_mask = (ones_warp > mask_thresh).float()

        loss = loss_fn(PA_warp, PB, valid_mask)
        total += float(loss.item())
        num_batches += 1

    return total / max(num_batches, 1)


def FindLatestModel(CheckPointPath):
    ckpts = sorted(glob.glob(os.path.join(CheckPointPath, "*model.ckpt")))
    if len(ckpts) > 0:
        return os.path.basename(ckpts[-1])
    pts = sorted(glob.glob(os.path.join(CheckPointPath, "*.pt")))
    if len(pts) > 0:
        return os.path.basename(pts[-1])
    return None


def PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples, NumValSamples, LatestFile):
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Samples " + str(NumTrainSamples))
    print("Number of Val Samples " + str(NumValSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    TrainCachedSamples,
    ValCachedSamples,
    NumEpochs,
    MiniBatchSize,
    CheckPointPath,
    LogsPath,
    LatestFile,
    Device,
    LR,
    Rho,
    BoundMode,
    MaskThresh,
    AlignCorners
):
    model = Homographynet(in_channels=2).to(Device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = PhotometricLoss()

    writer = SummaryWriter(LogsPath)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if LatestFile is not None:
        ckpt_path = os.path.join(CheckPointPath, LatestFile)
        ckpt = torch.load(ckpt_path, map_location=Device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print("Loaded checkpoint:", LatestFile, "start_epoch:", start_epoch)

    train_iters_per_epoch = max(len(TrainCachedSamples) // MiniBatchSize, 1)
    val_iters = max(len(ValCachedSamples) // MiniBatchSize, 1) if len(ValCachedSamples) > 0 else 0

    print(f"\nStarting training...")
    print(f"Iterations per epoch: {train_iters_per_epoch}")
    print(f"Validation iterations: {val_iters}\n")

    for epoch in range(start_epoch, NumEpochs):
        t0 = time.time()
        model.train()
        running = 0.0

        for it in range(train_iters_per_epoch):
            stacked, PA, PB, CA = GenerateBatch(TrainCachedSamples, MiniBatchSize)

            optimizer.zero_grad(set_to_none=True)

            h4pt_hat = model(stacked)
            h4pt_hat = bound_h4pt(h4pt_hat, Rho, BoundMode)

            H_hat = tensor_dlt(CA, h4pt_hat)
            PA_warp = apply_warp(PA, H_hat, align_corners=AlignCorners)

            ones = torch.ones_like(PA)
            ones_warp = apply_warp(ones, H_hat, align_corners=AlignCorners)
            valid_mask = (ones_warp > MaskThresh).float()

            loss = loss_fn(PA_warp, PB, valid_mask)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            global_step += 1

            writer.add_scalar("LossEveryIter/train", float(loss.item()), global_step)

        train_loss = running / max(train_iters_per_epoch, 1)

        if val_iters > 0:
            val_loss = RunValidation(
                model, ValCachedSamples, MiniBatchSize, val_iters, loss_fn,
                Rho, BoundMode, MaskThresh, AlignCorners
            )
        else:
            val_loss = float("inf")

        writer.add_scalar("LossEveryEpoch/train", train_loss, epoch)
        if val_iters > 0:
            writer.add_scalar("LossEveryEpoch/val", val_loss, epoch)
        writer.flush()

        dt = time.time() - t0

        # Clean output
        print(f"Epoch {epoch+1:03d}/{NumEpochs:03d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | time: {dt:.1f}s")

        # Save every epoch (silently)
        save_name = os.path.join(CheckPointPath, f"{epoch}model.ckpt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "val_loss": val_loss,
            },
            save_name
        )

        # Save best (silently)
        if val_loss < best_val:
            best_val = val_loss
            best_name = os.path.join(CheckPointPath, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "val_loss": val_loss,
                },
                best_name
            )
            print(f"  ✓ saved best.pt (val_loss: {val_loss:.6f})")

    writer.close()


def main():
    data_dir, train_dir, val_dir, ckpt_dir, logs_dir = get_default_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainDir", default=str(train_dir))
    parser.add_argument("--ValDir", default=str(val_dir))
    parser.add_argument("--CheckPointPath", default=str(ckpt_dir))
    parser.add_argument("--LogsPath", default=str(logs_dir))

    parser.add_argument("--NumEpochs", type=int, default=50)
    parser.add_argument("--MiniBatchSize", type=int, default=64, 
                       help="Batch size (default: 64)")
    parser.add_argument("--LoadCheckPoint", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5)")
    parser.add_argument("--rho", type=float, default=32.0)
    parser.add_argument("--bound_mode", default="none", choices=["none", "clamp", "tanh"],
                       help="Bounding mode (default: none)")
    parser.add_argument("--mask_thresh", type=float, default=0.9)
    parser.add_argument("--align_corners", action="store_true")
    parser.add_argument("--max_cache", type=int, default=10000,
                       help="Max samples to cache in memory (default: 10000)")

    args = parser.parse_args()

    train_dir = Path(args.TrainDir).expanduser().resolve()
    val_dir = Path(args.ValDir).expanduser().resolve()

    os.makedirs(args.CheckPointPath, exist_ok=True)
    os.makedirs(args.LogsPath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List all samples
    print("[INFO] Discovering samples...")
    train_samples = list_samples(train_dir)
    val_samples = list_samples(val_dir) if val_dir.exists() else []

    if len(train_samples) == 0:
        raise RuntimeError(f"No training samples found in: {train_dir}")

    latest = FindLatestModel(args.CheckPointPath) if args.LoadCheckPoint == 1 else None

    PrettyPrint(args.NumEpochs, args.MiniBatchSize, len(train_samples), len(val_samples), latest)
    print("Device:", device)
    print(f"LR: {args.lr}, bound_mode: {args.bound_mode}, mask_thresh: {args.mask_thresh}, align_corners: {args.align_corners}")
    print("TrainDir:", train_dir)
    print("ValDir:", val_dir)

    # PRE-LOAD samples into memory
    train_cached = preload_samples(train_samples, device, max_cache=args.max_cache)
    val_cached = preload_samples(val_samples, device, max_cache=min(2000, len(val_samples))) if val_samples else []

    print(f"\n[INFO] Cached {len(train_cached)} training samples")
    print(f"[INFO] Cached {len(val_cached)} validation samples")

    TrainOperation(
        TrainCachedSamples=train_cached,
        ValCachedSamples=val_cached,
        NumEpochs=args.NumEpochs,
        MiniBatchSize=args.MiniBatchSize,
        CheckPointPath=args.CheckPointPath,
        LogsPath=args.LogsPath,
        LatestFile=latest,
        Device=device,
        LR=args.lr,
        Rho=args.rho,
        BoundMode=args.bound_mode,
        MaskThresh=args.mask_thresh,
        AlignCorners=args.align_corners,
    )


if __name__ == "__main__":
    main()