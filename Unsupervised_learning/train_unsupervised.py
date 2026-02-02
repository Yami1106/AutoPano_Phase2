#!/usr/bin/env python3
"""
Unsupervised Learning - Step 5b (FIXED)
Train loop for unsupervised homography estimation:
  stacked -> Homographynet -> H4Pt_hat -> TensorDLT -> H_hat -> warp(PA) -> masked L1 loss vs PB
"""

from pathlib import Path
import argparse
import time

import torch
from torch.utils.data import DataLoader

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp
from Unsupervised_learning.loss import PhotometricLoss


def get_default_paths():
    this = Path(__file__).resolve()
    # .../Phase2/Code/Unsupervised_learning/train_unsupervised.py
    phase2 = this.parents[2]  # Phase2/
    data_dir = phase2 / "Data"

    train_dir = data_dir / "patch_train"
    val_dir = data_dir / "patch_val"
    if not val_dir.exists():
        val_dir = data_dir / "patch_test"

    ckpt_dir = phase2 / "Checkpoints" / "unsup"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, train_dir, val_dir, ckpt_dir


def save_checkpoint(path: Path, model, optimizer, epoch: int, extra: dict = None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    """
    h4pt_hat: (B,8)
    mode:
      - "clamp": clamp to [-rho, rho]
      - "tanh" : rho * tanh(h4pt_hat)
      - "none" : no bounding
    """
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


@torch.no_grad()
def run_epoch_val(model, loader, loss_fn, device, rho, bound_mode, mask_thresh, align_corners=True):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        stacked = batch["stacked"].to(device)  # (B,2,H,W)
        PA = batch["PA"].to(device)            # (B,1,H,W)
        PB = batch["PB"].to(device)            # (B,1,H,W)
        CA = batch["CA"].to(device)            # (B,4,2)

        h4pt_hat = model(stacked)              # (B,8)
        h4pt_hat = bound_h4pt(h4pt_hat, rho=rho, mode=bound_mode)

        H_hat = tensor_dlt(CA, h4pt_hat)       # (B,3,3)
        PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)

        # valid mask
        ones = torch.ones_like(PA)
        ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
        valid_mask = (ones_warp > mask_thresh).float()

        loss = loss_fn(PA_warp, PB, valid_mask)
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_epoch_train(model, loader, loss_fn, optimizer, device, rho, bound_mode, mask_thresh, align_corners=True):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        stacked = batch["stacked"].to(device)  # (B,2,H,W)
        PA = batch["PA"].to(device)            # (B,1,H,W)
        PB = batch["PB"].to(device)            # (B,1,H,W)
        CA = batch["CA"].to(device)            # (B,4,2)

        optimizer.zero_grad(set_to_none=True)

        h4pt_hat = model(stacked)              # (B,8)
        h4pt_hat = bound_h4pt(h4pt_hat, rho=rho, mode=bound_mode)

        H_hat = tensor_dlt(CA, h4pt_hat)       # (B,3,3)
        PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)

        # valid mask
        ones = torch.ones_like(PA)
        ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
        valid_mask = (ones_warp > mask_thresh).float()

        loss = loss_fn(PA_warp, PB, valid_mask)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def parse_args():
    data_dir, train_dir, val_dir, ckpt_dir = get_default_paths()

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(train_dir), help="patch_train directory")
    ap.add_argument("--val_dir", default=str(val_dir), help="patch_val (or patch_test) directory")
    ap.add_argument("--ckpt_dir", default=str(ckpt_dir), help="checkpoint output directory")

    ap.add_argument("--patch_h", type=int, default=128)
    ap.add_argument("--patch_w", type=int, default=128)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.0005)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--align_corners", action="store_true", help="match warp.py default (recommended True)")
    ap.add_argument("--save_every", type=int, default=1)

    # NEW:
    ap.add_argument("--rho", type=float, default=32.0, help="max corner perturbation used in dataset generation")
    ap.add_argument("--bound_mode", default="tanh", choices=["none", "clamp", "tanh"],
                    help="how to constrain predicted H4Pt to [-rho,rho]")
    ap.add_argument("--mask_thresh", type=float, default=0.9,
                    help="valid pixel threshold for warped mask (typical 0.9)")

    return ap.parse_args()


def main():
    args = parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Datasets
    train_ds = UnsupervisedDataset(str(train_dir), patch_hw=(args.patch_h, args.patch_w), return_h4pt_gt=False)
    val_ds = UnsupervisedDataset(str(val_dir), patch_hw=(args.patch_h, args.patch_w), return_h4pt_gt=False)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Model
    model = Homographynet(in_channels=2).to(device)

    # Loss + optimizer
    # IMPORTANT: PhotometricLoss must now accept (PA_warp, PB, valid_mask)
    loss_fn = PhotometricLoss()  # <- your masked version (no reduction arg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Device:", device)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))
    print("Batch size:", args.batch_size, "Epochs:", args.epochs)
    print("rho:", args.rho, "bound_mode:", args.bound_mode, "mask_thresh:", args.mask_thresh)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch_train(
            model, train_loader, loss_fn, optimizer, device,
            rho=args.rho, bound_mode=args.bound_mode, mask_thresh=args.mask_thresh,
            align_corners=args.align_corners
        )
        val_loss = run_epoch_val(
            model, val_loader, loss_fn, device,
            rho=args.rho, bound_mode=args.bound_mode, mask_thresh=args.mask_thresh,
            align_corners=args.align_corners
        )

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={dt:.1f}s")

        # save checkpoints
        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                model, optimizer, epoch,
                extra={"train_loss": train_loss, "val_loss": val_loss}
            )

        # save best (best photometric on val)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, epoch,
                extra={"train_loss": train_loss, "val_loss": val_loss}
            )
            print("  ✓ saved best.pt")

    print("Done.")


if __name__ == "__main__":
    main()
