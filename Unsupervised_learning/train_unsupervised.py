#!/usr/bin/env python3
"""
Unsupervised Learning - Training (NO DataLoader version)

Dataset layout:
  Phase2/Data/patch_train/
    1_PA.png
    1_PB.png
    1_meta.npz
  Phase2/Data/patch_val/
    ...

Train loop:
  stacked(PA,PB) -> HomographyNet -> H4Pt_hat -> TensorDLT -> H_hat
  -> warp(PA) -> masked photometric L1 loss vs PB
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
from Unsupervised_learning.warp import apply_warp_with_mask
from Unsupervised_learning.loss import PhotometricLoss


# ----------------------------
# Speed knobs (safe)
# ----------------------------
torch.backends.cudnn.benchmark = True


# ----------------------------
# Helpers
# ----------------------------

def get_default_paths():
    this = Path(__file__).resolve()
    phase2 = this.parents[2]  # Phase2/
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


# --- Simple RAM caches to kill disk bottleneck ---
_IMG_CACHE = {}   # path -> uint8 (H,W)
_META_CACHE = {}  # path -> float32 (4,2)


def read_gray_uint8_cached(path: str) -> np.ndarray:
    if path not in _IMG_CACHE:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        _IMG_CACHE[path] = img
    return _IMG_CACHE[path]


def read_gray01_cached(path: str) -> np.ndarray:
    return read_gray_uint8_cached(path).astype(np.float32) / 255.0


def extract_CA_from_meta(meta_path: str) -> np.ndarray:
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

    for k in keys:
        arr = np.array(meta[k])
        if arr.size == 8:
            try:
                return arr.astype(np.float32).reshape(4, 2)
            except Exception:
                pass

    raise KeyError(
        f"Could not find CA (4x2 corners) inside meta file: {meta_path}\n"
        f"Available keys: {keys}\n"
        f"Fix: store CA in meta.npz under key 'CA' (recommended), "
        f"or update extract_CA_from_meta() with your key."
    )


def extract_CA_cached(meta_path: str) -> np.ndarray:
    if meta_path not in _META_CACHE:
        _META_CACHE[meta_path] = extract_CA_from_meta(meta_path)
    return _META_CACHE[meta_path]


def to_torch_chw(x: np.ndarray, device: torch.device) -> torch.Tensor:
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim == 3:
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    return torch.from_numpy(x).to(device)


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


def normalize_patch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-patch normalization improves convergence when brightness/exposure differs.
    x: (B,1,H,W)
    """
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)


# ----------------------------
# Sample discovery
# ----------------------------

def list_samples(patch_dir: Path):
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


# ----------------------------
# Batch generation (NO DataLoader)
# ----------------------------

def GenerateBatch(samples, MiniBatchSize, device):
    stacked_list, pa_list, pb_list, ca_list = [], [], [], []

    for _ in range(MiniBatchSize):
        s = random.choice(samples)

        PA = read_gray01_cached(s["PA_png"])
        PB = read_gray01_cached(s["PB_png"])
        CA = extract_CA_cached(s["meta_npz"])

        if PA.shape != PB.shape:
            raise ValueError(f"PA and PB sizes differ: {s['PA_png']} vs {s['PB_png']}")

        PA_t = to_torch_chw(PA, device)                   # (1,H,W)
        PB_t = to_torch_chw(PB, device)                   # (1,H,W)
        stacked_t = torch.cat([PA_t, PB_t], dim=0)        # (2,H,W)

        CA_t = torch.from_numpy(CA.astype(np.float32)).to(device)  # (4,2)

        stacked_list.append(stacked_t)
        pa_list.append(PA_t)
        pb_list.append(PB_t)
        ca_list.append(CA_t)

    stacked_batch = torch.stack(stacked_list, dim=0)  # (B,2,H,W)
    PA_batch = torch.stack(pa_list, dim=0)            # (B,1,H,W)
    PB_batch = torch.stack(pb_list, dim=0)            # (B,1,H,W)
    CA_batch = torch.stack(ca_list, dim=0)            # (B,4,2)

    return stacked_batch, PA_batch, PB_batch, CA_batch


# ----------------------------
# Val loop
# ----------------------------

@torch.no_grad()
def RunValidation(model, samples, MiniBatchSize, NumIterations, loss_fn,
                  device, rho, bound_mode, align_corners, use_patch_norm):
    model.eval()
    total = 0.0

    for _ in range(NumIterations):
        stacked, PA, PB, CA = GenerateBatch(samples, MiniBatchSize, device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            h4pt_hat = model(stacked)
            h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)

            H_hat = tensor_dlt(CA, h4pt_hat)
            PA_warp, valid_mask = apply_warp_with_mask(PA, H_hat, align_corners=align_corners)

            if use_patch_norm:
                PA_warp_n = normalize_patch(PA_warp)
                PB_n = normalize_patch(PB)
                loss = loss_fn(PA_warp_n, PB_n, valid_mask)
            else:
                loss = loss_fn(PA_warp, PB, valid_mask)

        total += float(loss.item())

    return total / max(NumIterations, 1)


# ----------------------------
# Checkpoint helpers
# ----------------------------

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


# ----------------------------
# Train
# ----------------------------

def TrainOperation(
    TrainSamples,
    ValSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    LogsPath,
    LatestFile,
    Device,
    LR,
    Rho,
    BoundMode,
    AlignCorners,
    LogEvery,
    ValEvery,
    UsePatchNorm,
    WeightDecay
):
    model = Homographynet(in_channels=2).to(Device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WeightDecay)
    loss_fn = PhotometricLoss()

    writer = SummaryWriter(LogsPath)

    scaler = torch.cuda.amp.GradScaler(enabled=(Device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if LatestFile is not None:
        ckpt = torch.load(os.path.join(CheckPointPath, LatestFile), map_location=Device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print("Loaded checkpoint:", LatestFile, "start_epoch:", start_epoch)

    # You are printing "Number of Training Samples 50000" etc, but that is *logical* count.
    # Actual iteration count is computed here:
    train_iters_per_epoch = max(len(TrainSamples) // MiniBatchSize, 1)
    val_iters = max(len(ValSamples) // MiniBatchSize, 1) if len(ValSamples) > 0 else 0

    for epoch in range(start_epoch, NumEpochs):
        t0 = time.time()
        model.train()
        running = 0.0

        for it in range(train_iters_per_epoch):
            stacked, PA, PB, CA = GenerateBatch(TrainSamples, MiniBatchSize, Device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(Device.type == "cuda")):
                h4pt_hat = model(stacked)
                h4pt_hat = bound_h4pt(h4pt_hat, Rho, BoundMode)

                H_hat = tensor_dlt(CA, h4pt_hat)
                PA_warp, valid_mask = apply_warp_with_mask(PA, H_hat, align_corners=AlignCorners)

                if UsePatchNorm:
                    PA_warp_n = normalize_patch(PA_warp)
                    PB_n = normalize_patch(PB)
                    loss = loss_fn(PA_warp_n, PB_n, valid_mask)
                else:
                    loss = loss_fn(PA_warp, PB, valid_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            running += loss_val
            global_step += 1

            # TensorBoard (NO flush every iter)
            if global_step % LogEvery == 0:
                writer.add_scalar("LossEveryIter/train", loss_val, global_step)
                writer.flush()

            # Optional console heartbeat (helps you see it's not "stuck")
            if it % max(50, (train_iters_per_epoch // 10)) == 0:
                print(f"  [epoch {epoch+1:02d}] it {it:05d}/{train_iters_per_epoch:05d} loss {loss_val:.6f}")

            # checkpoint every N iters (avoid it==0)
            if SaveCheckPoint > 0 and (it % SaveCheckPoint == 0) and it != 0:
                save_name = os.path.join(CheckPointPath, f"{epoch}a{it}model.ckpt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss_val,
                    },
                    save_name
                )

        train_loss = running / max(train_iters_per_epoch, 1)

        # Validation (optionally not every epoch to save time)
        if (epoch + 1) % ValEvery == 0 and val_iters > 0:
            val_loss = RunValidation(
                model, ValSamples, MiniBatchSize, val_iters, loss_fn,
                Device, Rho, BoundMode, AlignCorners, UsePatchNorm
            )
        else:
            val_loss = float("inf") if val_iters > 0 else float("inf")

        # Epoch scalars
        writer.add_scalar("LossEveryEpoch/train", train_loss, epoch)
        if val_iters > 0 and (epoch + 1) % ValEvery == 0:
            writer.add_scalar("LossEveryEpoch/val", val_loss, epoch)
        writer.flush()

        dt = time.time() - t0
        if val_iters > 0 and (epoch + 1) % ValEvery == 0:
            print(f"Epoch {epoch+1:03d}/{NumEpochs:03d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | time: {dt:.1f}s")
        else:
            print(f"Epoch {epoch+1:03d}/{NumEpochs:03d} | train_loss: {train_loss:.6f} | val_loss: (skipped) | time: {dt:.1f}s")

        # Save every epoch (one file)
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

        # Save best
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
            print("  ✓ saved best.pt")


def main():
    data_dir, train_dir, val_dir, ckpt_dir, logs_dir = get_default_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainDir", default=str(train_dir), help="patch_train directory")
    parser.add_argument("--ValDir", default=str(val_dir), help="patch_val/patch_test directory")

    parser.add_argument("--CheckPointPath", default=str(ckpt_dir), help="Path to save checkpoints")
    parser.add_argument("--LogsPath", default=str(logs_dir), help="Path to save Tensorboard logs")

    parser.add_argument("--NumEpochs", type=int, default=10)
    parser.add_argument("--MiniBatchSize", type=int, default=32)

    # Reduce checkpoint stalls by default
    parser.add_argument("--SaveCheckPoint", type=int, default=500, help="save every N iters (0 disables)")
    parser.add_argument("--LoadCheckPoint", type=int, default=0)

    # IMPORTANT: better LR default
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    parser.add_argument("--rho", type=float, default=32.0)
    parser.add_argument("--bound_mode", default="tanh", choices=["none", "clamp", "tanh"])
    parser.add_argument("--align_corners", action="store_true")

    # Logging + val frequency
    parser.add_argument("--log_every", type=int, default=50, help="tensorboard scalar every N steps")
    parser.add_argument("--val_every", type=int, default=1, help="run val every N epochs (increase to speed up)")

    # Helps convergence
    parser.add_argument("--use_patch_norm", action="store_true", help="normalize PA_warp/PB per patch before loss")

    args = parser.parse_args()

    train_dir = Path(args.TrainDir).expanduser().resolve()
    val_dir = Path(args.ValDir).expanduser().resolve()

    os.makedirs(args.CheckPointPath, exist_ok=True)
    os.makedirs(args.LogsPath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = list_samples(train_dir)
    val_samples = list_samples(val_dir) if val_dir.exists() else []

    latest = FindLatestModel(args.CheckPointPath) if args.LoadCheckPoint == 1 else None

    PrettyPrint(args.NumEpochs, args.MiniBatchSize, len(train_samples), len(val_samples), latest)
    print("Device:", device)
    print("rho:", args.rho, "bound_mode:", args.bound_mode, "align_corners:", args.align_corners)
    print("TrainDir:", train_dir)
    print("ValDir:", val_dir)
    print("LR:", args.lr, "weight_decay:", args.weight_decay, "use_patch_norm:", args.use_patch_norm)

    if len(train_samples) == 0:
        raise RuntimeError(
            f"No training samples found in: {train_dir}\n"
            f"Expected files like: 1_PA.png, 1_PB.png, 1_meta.npz"
        )

    TrainOperation(
        TrainSamples=train_samples,
        ValSamples=val_samples,
        NumEpochs=args.NumEpochs,
        MiniBatchSize=args.MiniBatchSize,
        SaveCheckPoint=args.SaveCheckPoint,
        CheckPointPath=args.CheckPointPath if args.CheckPointPath.endswith(os.sep) else args.CheckPointPath + os.sep,
        LogsPath=args.LogsPath,
        LatestFile=latest,
        Device=device,
        LR=args.lr,
        Rho=args.rho,
        BoundMode=args.bound_mode,
        AlignCorners=args.align_corners,
        LogEvery=max(1, args.log_every),
        ValEvery=max(1, args.val_every),
        UsePatchNorm=args.use_patch_norm,
        WeightDecay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
