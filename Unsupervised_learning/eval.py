#!/usr/bin/env python3
"""
Unsupervised Learning - Evaluation Script (NO DataLoader)

Evaluates trained model on test/validation set.
Computes:
  - average photometric loss (masked L1)
  - corner L2 error statistics (pixels) if GT exists
"""

import argparse
from pathlib import Path
import time

import torch

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp
from Unsupervised_learning.loss import PhotometricLoss
from Unsupervised_learning.metrics import corner_l2_errors_px, summarize_errors


def bound_h4pt(h4pt_hat: torch.Tensor, rho: float, mode: str) -> torch.Tensor:
    if mode == "none":
        return h4pt_hat
    if mode == "clamp":
        return torch.clamp(h4pt_hat, -rho, rho)
    if mode == "tanh":
        return rho * torch.tanh(h4pt_hat)
    raise ValueError("bound_mode must be one of: none, clamp, tanh")


def stack_batch(batch_samples, device):
    stacked = torch.stack([s["stacked"] for s in batch_samples]).to(device, non_blocking=True)
    PA      = torch.stack([s["PA"]      for s in batch_samples]).to(device, non_blocking=True)
    PB      = torch.stack([s["PB"]      for s in batch_samples]).to(device, non_blocking=True)
    CA      = torch.stack([s["CA"]      for s in batch_samples]).to(device, non_blocking=True)

    # GT is optional
    h4_gt = None
    if "H4Pt_gt" in batch_samples[0]:
        # Some datasets might miss GT for some samples; guard it
        if all(("H4Pt_gt" in s) for s in batch_samples):
            h4_gt = torch.stack([s["H4Pt_gt"] for s in batch_samples]).to(device, non_blocking=True)

    return stacked, PA, PB, CA, h4_gt


def evaluate(
    model,
    dataset,
    device,
    batch_size=32,
    rho=32.0,
    bound_mode="none",
    mask_thresh=0.9,
    align_corners=False,
    use_amp=True,
    compute_mask_with_ones=True,   # keep correctness by default
):
    model.eval()
    loss_fn = PhotometricLoss().to(device)

    total_photo_loss = 0.0
    n_batches = 0

    all_corner_err = []  # will store (B,4) per batch if GT exists

    num_samples = len(dataset)

    t0 = time.time()

    # inference_mode is faster than no_grad for eval
    with torch.inference_mode():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # build python batch
            batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
            stacked, PA, PB, CA, h4_gt = stack_batch(batch_samples, device)

            # AMP helps a lot on GPU
            autocast_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast
            with autocast_ctx(enabled=(use_amp and device.type == "cuda")):
                h4pt_hat = model(stacked)
                h4pt_hat = bound_h4pt(h4pt_hat, rho, bound_mode)

                H_hat = tensor_dlt(CA, h4pt_hat)
                PA_warp = apply_warp(PA, H_hat, align_corners=align_corners)

                # valid mask
                if compute_mask_with_ones:
                    ones = torch.ones_like(PA)
                    ones_warp = apply_warp(ones, H_hat, align_corners=align_corners)
                    valid_mask = (ones_warp > mask_thresh).float()
                else:
                    # faster approximate mask (use only if you accept approximation)
                    valid_mask = (PA_warp != 0).float()

                photo_loss = loss_fn(PA_warp, PB, valid_mask)

            total_photo_loss += float(photo_loss.item())
            n_batches += 1

            # Corner error using proper metric if GT exists
            if h4_gt is not None:
                # returns (B,4)
                err = corner_l2_errors_px(CA, h4pt_hat.float(), h4_gt.float())
                all_corner_err.append(err.detach().cpu())

    avg_photo_loss = total_photo_loss / max(n_batches, 1)

    corner_stats = None
    if len(all_corner_err) > 0:
        all_err = torch.cat(all_corner_err, dim=0)  # (N,4)
        corner_stats = summarize_errors(all_err)

    dt = time.time() - t0
    return avg_photo_loss, corner_stats, dt


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained unsupervised homography model (no DataLoader)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test/val patch directory")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rho", type=float, default=32.0)
    parser.add_argument("--bound_mode", default="none", choices=["none", "clamp", "tanh"])
    parser.add_argument("--mask_thresh", type=float, default=0.9)
    parser.add_argument("--align_corners", action="store_true")

    # speed toggles
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--fast_mask", action="store_true", help="Approximate mask without warping ones (faster, less strict)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optional: speed for fixed shapes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = Homographynet(in_channels=2).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
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
    test_dir = Path(args.test_dir).expanduser().resolve()
    print(f"\nLoading dataset from: {test_dir}")
    test_dataset = UnsupervisedDataset(
        str(test_dir),
        patch_hw=(128, 128),
        return_h4pt_gt=True,
    )
    print(f"Samples: {len(test_dataset)}")

    # Evaluate
    print(f"\nEvaluating...")
    avg_photo_loss, corner_stats, dt = evaluate(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        rho=args.rho,
        bound_mode=args.bound_mode,
        mask_thresh=args.mask_thresh,
        align_corners=args.align_corners,
        use_amp=(not args.no_amp),
        compute_mask_with_ones=(not args.fast_mask),
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Average Photometric Loss: {avg_photo_loss:.6f}")

    if corner_stats is not None:
        print("Corner error statistics (pixels):")
        for k, v in corner_stats.items():
            print(f"  {k}: {v}")
    else:
        print("Corner Error: N/A (no ground truth H4Pt found)")

    print(f"Eval time: {dt:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
