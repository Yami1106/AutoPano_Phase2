#!/usr/bin/env python3
"""
Step 7: Evaluate unsupervised model using GT H4Pt (only for evaluation).

Requires:
  Data/patch_val (or patch_test) containing *_H4Pt.npy
  best checkpoint in Phase2/Checkpoints/unsup/best.pt
"""

from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.metrics import corner_l2_errors_px, summarize_errors


def get_default_paths():
    this = Path(__file__).resolve()
    phase2 = this.parents[2]
    data_dir = phase2 / "Data"
    ckpt = phase2 / "Checkpoints" / "unsup" / "best.pt"

    # prefer patch_val, fallback patch_test
    val_dir = data_dir / "patch_val"
    if not val_dir.exists():
        val_dir = data_dir / "patch_test"

    return phase2, data_dir, val_dir, ckpt


def parse_args():
    phase2, data_dir, val_dir, ckpt = get_default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", default=str(val_dir))
    ap.add_argument("--ckpt", default=str(ckpt))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch_h", type=int, default=128)
    ap.add_argument("--patch_w", type=int, default=128)
    ap.add_argument("--max_batches", type=int, default=-1, help="limit for quick eval, -1 = full")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    val_dir = Path(args.val_dir).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()

    # IMPORTANT: eval needs GT H4Pt
    ds = UnsupervisedDataset(str(val_dir), patch_hw=(args.patch_h, args.patch_w), return_h4pt_gt=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = Homographynet(in_channels=2).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_err = []
    seen = 0

    for bi, batch in enumerate(loader):
        if args.max_batches != -1 and bi >= args.max_batches:
            break

        stacked = batch["stacked"].to(device)     # (B,2,H,W)
        CA = batch["CA"].to(device)               # (B,4,2)
        h4_gt = batch.get("H4Pt_gt", None)

        if h4_gt is None:
            raise RuntimeError(
                "H4Pt_gt missing. Make sure your patch_val (or patch_test) has *_H4Pt.npy saved."
            )

        h4_gt = h4_gt.to(device)                  # (B,8)
        h4_pred = model(stacked)                  # (B,8)

        err = corner_l2_errors_px(CA, h4_pred, h4_gt)  # (B,4)
        all_err.append(err.detach().cpu())
        seen += err.shape[0]

    all_err = torch.cat(all_err, dim=0)  # (N,4)
    stats = summarize_errors(all_err)

    print("Evaluated samples:", all_err.shape[0])
    print("Corner error statistics (pixels):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
