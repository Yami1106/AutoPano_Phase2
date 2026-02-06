#!/usr/bin/env python3
"""
Phase 2 - Synthetic Homography Data Generation (PA, PB, H4Pt)

Generates (per image):
  - PA: rectangular patch from IA
  - PB: rectangular patch from IB (IA warped by HBA)
  - H4Pt label: (CB - CA) flattened -> shape (8,)
  - stacked (optional in-memory): concat(PA, PB) -> (patch_h, patch_w, 2*C)

Saves:
  (A) DATASET PATCHES into Phase2/Data/
      - Phase2/Data/patch_train/   (for Train images)
      - Phase2/Data/patch_val/     (for Val images)
      Each image produces:
        <stem>_PA.png
        <stem>_PB.png
        <stem>_H4Pt.npy   (float32, shape (8,))
        <stem>_meta.npz   (optional: CA, CB, HAB, HBA, x,y)

  (B) DEBUG OVERLAYS into the SAME folder as this .py file (Phase2/Code/DataGen/)
      - Phase2/Code/DataGen/overlays/train/
      - Phase2/Code/DataGen/overlays/val/
      Each image produces:
        <stem>_overlay_IA.png
        <stem>_overlay_IB.png
"""

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
import zipfile


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
    seed: Optional[int] = None


THIS_FILE = Path(__file__).resolve()  # Phase2/Code/DataGen/HomographyDataGen.py
CODE_DIR = THIS_FILE.parent  # Phase2/Code/DataGen
PHASE2_DIR = THIS_FILE.parents[2]  # Phase2

DATA_DIR = PHASE2_DIR / "Data"  # dataset patches go here
OVERLAY_DIR = CODE_DIR / "overlays"  # debug overlays go here


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def extract_zip(data_root: Path):
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
    """
    Makes sure image is big enough so sampling a patch of (patch_h, patch_w) with margin rho is ALWAYS possible.

    Required:
      H >= patch_h + 2*rho + 1
      W >= patch_w + 2*rho + 1
    """
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


def get_active_region(
    H: int, W: int, patch_h: int, patch_w: int, rho: int
) -> Tuple[int, int, int, int]:
    x_min = rho
    y_min = rho
    x_max = W - patch_w - rho
    y_max = H - patch_h - rho
    return x_min, y_min, x_max, y_max


def sample_patch_top_left(H: int, W: int, cfg: DataGenConfig) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = get_active_region(
        H, W, cfg.patch_h, cfg.patch_w, cfg.rho
    )
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Image too small for patch+rho even after resize. Image=({H},{W}), "
            f"patch=({cfg.patch_h},{cfg.patch_w}), rho={cfg.rho}"
        )
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    return x, y


def corners_from_top_left(x: int, y: int, patch_w: int, patch_h: int) -> np.ndarray:
    # TL, TR, BR, BL (clockwise)
    return np.array(
        [
            [x, y],  # TL
            [x + patch_w, y],  # TR
            [x + patch_w, y + patch_h],  # BR
            [x, y + patch_h],  # BL
        ],
        dtype=np.float32,
    )


def perturb_corners(CA: np.ndarray, cfg: DataGenConfig) -> np.ndarray:
    delta = np.random.uniform(-cfg.rho, cfg.rho, size=(4, 2)).astype(np.float32)
    if cfg.allow_translation:
        tx = np.random.uniform(-cfg.max_translation, cfg.max_translation)
        ty = np.random.uniform(-cfg.max_translation, cfg.max_translation)
        delta = delta + np.array([tx, ty], dtype=np.float32)
    return CA + delta


def compute_H(CA: np.ndarray, CB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    HAB = cv2.getPerspectiveTransform(CA, CB).astype(np.float32)  # CA -> CB
    HBA = np.linalg.inv(HAB).astype(np.float32)  # CB -> CA
    return HAB, HBA


def warp_image(imgA: np.ndarray, HBA: np.ndarray) -> np.ndarray:
    H, W = imgA.shape[:2]
    return cv2.warpPerspective(imgA, HBA, (W, H), flags=cv2.INTER_LINEAR)


def extract_patch(
    img: np.ndarray, x: int, y: int, patch_w: int, patch_h: int
) -> np.ndarray:
    return img[y : y + patch_h, x : x + patch_w].copy()


def stack_patches(PA: np.ndarray, PB: np.ndarray, cfg: DataGenConfig) -> np.ndarray:
    if cfg.normalize:
        PA = PA.astype(np.float32)
        PB = PB.astype(np.float32)
        PA = (PA - PA.min()) / (PA.max() - PA.min() + 1e-8)
        PB = (PB - PB.min()) / (PB.max() - PB.min() + 1e-8)

    if PA.ndim == 2:
        PA = PA[..., None]
        PB = PB[..., None]

    return np.concatenate([PA, PB], axis=2).astype(np.float32)


def h4pt_label(CA: np.ndarray, CB: np.ndarray) -> np.ndarray:
    return (CB - CA).reshape(-1).astype(np.float32)


def _draw_quad(ax, pts, color, label=None):
    pts = np.array(pts, dtype=np.float32)
    poly = np.vstack([pts, pts[0]])
    ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=3, color=color)
    ax.scatter(pts[:, 0], pts[:, 1], s=200, color=color)

    if label is not None:
        ax.text(
            float(pts[0, 0]) + 5,
            float(pts[0, 1]) - 5,
            label,
            color=color,
            fontsize=14,
            weight="bold",
        )


def _save_overlay_image(img, CA, CB, out_path: Path, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

    _draw_quad(ax, CA, color="deepskyblue", label="A")
    _draw_quad(ax, CB, color="red", label="B")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_raw_patch(patch: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if patch.ndim == 2:
        cv2.imwrite(str(out_path), patch)
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


def generate_one_pair_for_image(
    imgA: np.ndarray,
    cfg: DataGenConfig,
) -> Dict[str, np.ndarray]:

    imgA = ensure_min_size(imgA, cfg.patch_h, cfg.patch_w, cfg.rho)
    H, W = imgA.shape[:2]

    x, y = sample_patch_top_left(H, W, cfg)
    CA = corners_from_top_left(x, y, cfg.patch_w, cfg.patch_h)

    CB = perturb_corners(CA, cfg)

    HAB, HBA = compute_H(CA, CB)

    imgB = warp_image(imgA, HBA)

    # map CB into IB coords for overlay
    CB_on_IB = (
        cv2.perspectiveTransform(CB.reshape(1, 4, 2), HBA)
        .reshape(4, 2)
        .astype(np.float32)
    )

    PA = extract_patch(imgA, x, y, cfg.patch_w, cfg.patch_h)
    PB = extract_patch(imgB, x, y, cfg.patch_w, cfg.patch_h)

    stacked = stack_patches(PA, PB, cfg)
    label = h4pt_label(CA, CB)

    return {
        "stacked": stacked,
        "H4Pt": label,
        "CA": CA,
        "CB": CB,
        "CB_on_IB": CB_on_IB,
        "HAB": HAB,
        "HBA": HBA,
        "PA": PA,
        "PB": PB,
        "IA": imgA,
        "IB": imgB,
        "xy": np.array([x, y], dtype=np.int32),
    }


def save_outputs_for_image(
    out: Dict[str, np.ndarray],
    img_stem: str,
    patch_dir: Path,
    overlay_dir: Path,
):
    # Patches (dataset) -> Phase2/Data/...
    _save_raw_patch(out["PA"], patch_dir / f"{img_stem}_PA.png")
    _save_raw_patch(out["PB"], patch_dir / f"{img_stem}_PB.png")

    # Labels -> keep next to patches
    np.save(str(patch_dir / f"{img_stem}_H4Pt.npy"), out["H4Pt"])

    # Optional metadata (super useful later)
    np.savez_compressed(
        str(patch_dir / f"{img_stem}_meta.npz"),
        CA=out["CA"],
        CB=out["CB"],
        HAB=out["HAB"],
        HBA=out["HBA"],
        xy=out["xy"],
    )

    # Overlays (debug) -> Phase2/Code/DataGen/overlays/...
    _save_overlay_image(
        out["IA"],
        out["CA"],
        out["CB"],
        out_path=overlay_dir / f"{img_stem}_overlay_IA.png",
        title="IA: Patch A (blue) and Patch B (red)",
    )
    _save_overlay_image(
        out["IB"],
        out["CA"],
        out["CB_on_IB"],
        out_path=overlay_dir / f"{img_stem}_overlay_IB.png",
        title="IB (warped): Patch A (blue) and Patch B (red, warped)",
    )


def process_split(split_name: str, cfg: DataGenConfig):
    """
    Iterates through ALL images in a split and saves:
      - dataset patches + labels into Phase2/Data/
      - overlays into Phase2/Code/DataGen/overlays/
    """
    split_dir = DATA_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    img_paths = sorted(split_dir.glob("*.jpg"))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .jpg found in: {split_dir}")

    # dataset output dirs
    patch_dir = DATA_DIR / ("patch_train" if split_name == "Train" else "patch_val")
    patch_dir.mkdir(parents=True, exist_ok=True)

    # overlay output dirs (same folder as .py)
    overlay_dir = OVERLAY_DIR / ("train" if split_name == "Train" else "val")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing {split_name}: {len(img_paths)} images")
    print(f"[INFO] Saving patches to:  {patch_dir}")
    print(f"[INFO] Saving overlays to: {overlay_dir}")

    for idx, p in enumerate(img_paths):
        imgA = read_image(p, cfg.use_grayscale)

        out = generate_one_pair_for_image(imgA, cfg)

        # use original filename stem so mapping is easy
        img_stem = p.stem

        save_outputs_for_image(
            out, img_stem=img_stem, patch_dir=patch_dir, overlay_dir=overlay_dir
        )

        if (idx + 1) % 100 == 0 or (idx + 1) == len(img_paths):
            print(f"[{split_name}] {idx+1}/{len(img_paths)} done")


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
    return ap.parse_args()


def main():
    args = _parse_args()

    # Check and make sure Train/Val are extracted inside Phase2/Data/
    extract_zip(DATA_DIR)

    cfg = DataGenConfig(
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        rho=args.rho,
        use_grayscale=args.gray,
        allow_translation=not args.no_translation,
        max_translation=args.max_translation,
        seed=args.seed,
    )

    _set_seed(cfg.seed)

    # Process ALL images in Train and Val
    process_split("Train", cfg)
    process_split("Val", cfg)

    print("[DONE] All patches + overlays saved.")


if __name__ == "__main__":
    main()
