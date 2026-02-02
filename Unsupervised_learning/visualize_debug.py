#!/usr/bin/env python3
"""
Visual sanity check for unsupervised homography network.

Saves:
  PA.png
  PB.png
  PA_warp.png
"""

from pathlib import Path
import torch
import cv2
import numpy as np

from Unsupervised_learning.dataset import UnsupervisedDataset
from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt
from Unsupervised_learning.warp import apply_warp


def save_img(tensor, path):
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def main():
    this = Path(__file__).resolve()
    phase2 = this.parents[2]
    data_dir = phase2 / "Data"
    ckpt_dir = phase2 / "Checkpoints" / "unsup"

    out_dir = phase2 / "Debug_unsup"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    ds = UnsupervisedDataset(str(data_dir / "patch_val"), patch_hw=(128, 128))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    # Model
    model = Homographynet(in_channels=2).to(device)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded best.pt")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 1000:
                break

            PA = batch["PA"].to(device)
            PB = batch["PB"].to(device)
            stacked = batch["stacked"].to(device)
            CA = batch["CA"].to(device)

            h4pt = model(stacked)
            H = tensor_dlt(CA, h4pt)
            PA_warp = apply_warp(PA, H)

            save_img(PA, out_dir / f"{i:02d}_PA.png")
            save_img(PB, out_dir / f"{i:02d}_PB.png")
            save_img(PA_warp, out_dir / f"{i:02d}_PA_warp.png")

            print(f"Saved sample {i}")

    print("Done.")


if __name__ == "__main__":
    main()
