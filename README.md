<div align="center">

# HomographyNet — Deep Learning Panorama Stitching

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Replacing classical RANSAC homography estimation with learned networks — supervised and unsupervised — then combining both approaches into a hybrid stitcher.*

</div>

---

## Overview

Homography estimation (finding the geometric transform between two overlapping images) is classically done with RANSAC. This project replaces that with deep networks, implementing both a **supervised** and an **unsupervised** HomographyNet, then building a **hybrid stitcher** that combines classical feature detection with learned homography estimation.

---

## Supervised HomographyNet

| Detail | Value |
|---|---|
| Training data | 50k synthetic pairs from MS-COCO (128×128 patches, ρ=32px) |
| Architecture | 8 conv layers (64→512 channels) + 2 FC layers → 8D H4Pt output |
| Loss | MSE on H4Pt (corner displacement) |
| Optimiser | AdamW + OneCycleLR (max LR 5×10⁻³ at epoch 12) |
| **Validation MAE** | **8.7 px** |

---

## Unsupervised HomographyNet

| Detail | Value |
|---|---|
| Loss | Photometric consistency (Charbonnier) + valid pixel masking |
| Key components | TensorDLT layer + Spatial Transformer Network (differentiable warping) |
| **Validation MAE** | **17.4 px** |

TensorDLT makes the homography computation fully differentiable, allowing end-to-end backpropagation through the warping operation.

---

## Hybrid Stitcher

Combines the best of both worlds:
- Classical **Shi-Tomasi + SSD** feature matching for robust correspondences
- **DL homography** for accurate geometric estimation
- All-pairs photometric ordering for seamless multi-image stitching

> Noted: domain gap between MS-COCO training data and natural scene test images. Addressed with photometric ordering heuristics.

---

## Tech stack

`Python` · `PyTorch` · `NumPy` · `OpenCV`

---

<div align="center">
WPI Computer Vision (RBE/CS 549) · <a href="https://github.com/Yami1106">Ashish Sukumar</a>
</div>
<!-- -->
