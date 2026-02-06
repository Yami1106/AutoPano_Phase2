# import cv2
# import torch
# import numpy as np

# # Add any python libraries here


# def apply_tensordlt(pred_offsets, patch_size=128):
#     """
#     Implements the Differentiable Tensor Direct Linear Transform.

#     Args:
#         pred_offsets: Tensor [B, 8] - Predicted corner displacements
#         patch_size: Scalar - The size of the square patch (usually 128)

#     Returns:
#         H_matrix: Tensor [B, 3, 3] - The estimated 3x3 homography matrices
#     """
#     device = pred_offsets.device
#     batch_size = pred_offsets.shape[0]

#     # 1. Define source corners (xi) for a 128x128 patch
#     # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
#     x = torch.tensor(
#         [
#             [0, 0],
#             [patch_size - 1, 0],
#             [patch_size - 1, patch_size - 1],
#             [0, patch_size - 1],
#         ],
#         dtype=torch.float32,
#         device=device,
#     )

#     # Expand to batch size [B, 4, 2]
#     xi = x.unsqueeze(0).repeat(batch_size, 1, 1)
#     ui, vi = xi[:, :, 0], xi[:, :, 1]

#     # 2. Define target corners (xi') = xi + offsets
#     yi = xi + pred_offsets.view(-1, 4, 2)
#     u_prime, v_prime = yi[:, :, 0], yi[:, :, 1]

#     # 3. Construct Matrix A_hat [B, 8, 8]
#     # Based on Equation (7) and the definition of A_i in the paper
#     A_hat = torch.zeros((batch_size, 8, 8), device=device)

#     for i in range(4):
#         # Even rows of A_hat (from A_i)
#         A_hat[:, 2 * i, 0] = 0
#         A_hat[:, 2 * i, 1] = 0
#         A_hat[:, 2 * i, 2] = 0
#         A_hat[:, 2 * i, 3] = -ui[:, i]
#         A_hat[:, 2 * i, 4] = -vi[:, i]
#         A_hat[:, 2 * i, 5] = -1
#         A_hat[:, 2 * i, 6] = v_prime[:, i] * ui[:, i]
#         A_hat[:, 2 * i, 7] = v_prime[:, i] * vi[:, i]

#         # Odd rows of A_hat (from A_i)
#         A_hat[:, 2 * i + 1, 0] = ui[:, i]
#         A_hat[:, 2 * i + 1, 1] = vi[:, i]
#         A_hat[:, 2 * i + 1, 2] = 1
#         A_hat[:, 2 * i + 1, 3] = 0
#         A_hat[:, 2 * i + 1, 4] = 0
#         A_hat[:, 2 * i + 1, 5] = 0
#         A_hat[:, 2 * i + 1, 6] = -u_prime[:, i] * ui[:, i]
#         A_hat[:, 2 * i + 1, 7] = -u_prime[:, i] * vi[:, i]

#     # 4. Construct Vector b_hat [B, 8, 1]
#     # b_i = [-v'_i, u'_i]^T
#     b_hat = torch.zeros((batch_size, 8, 1), device=device)
#     for i in range(4):
#         b_hat[:, 2 * i, 0] = -v_prime[:, i]
#         b_hat[:, 2 * i + 1, 0] = u_prime[:, i]

#     # 5. Solve Ah = b using Pseudo-Inverse (Differentiable)
#     # h = A_hat.pinverse() @ b_hat
#     h_8 = torch.linalg.solve(A_hat, b_hat)  # Use solve for square matrix A_hat

#     # 6. Reconstruct 3x3 Homography Matrix
#     # Append the H33 = 1 element
#     h_9 = torch.cat(
#         [h_8.squeeze(-1), torch.ones((batch_size, 1), device=device)], dim=1
#     )
#     H_matrix = h_9.view(-1, 3, 3)

#     return H_matrix


#!/usr/bin/env python3

"""
Unsupervised Learning - Step 3

TensorDLT : convert H4pt to Homography matrix H(3x3)

input :

CA:      (B, 4, 2)  patch corners in PATCH coordinates (x,y)
h4pt:    (B, 8)     predicted offset (CB - CA) in PATCH coordinates:
                    [dx1,dy1, dx2,dy2, dx3,dy3, dx4,dy4]


Output:
H:       (B, 3, 3)  homography mapping CA -> CB_hat (patch-to-patch)

Corner order : CA = [TL, TR, BR, BL]
"""


from typing import Tuple
import torch


def build_homography(
    src: torch.Tensor, dst: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Ah = B -> linear system to solve for homography h

    (x,y) = (B,H,W)
    (u,v) = (B,H,W)

    h = [h11,h12,h13,h21,h22,h23,h31,h32]^T
    h33 = 1
    """

    B = src.shape[0]
    x = src[:, :, 0]
    y = src[:, :, 1]
    u = dst[:, :, 0]
    v = dst[:, :, 1]

    # u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
    # v = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)

    # [x y 1 0 0 0 -u x -u y] h = u
    # [0 0 0 x y 1 -v x -v y] h = v

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    Au = torch.stack(
        [x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1
    )  # (B, 4, 8)
    Av = torch.stack(
        [zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1
    )  # (B, 4, 8)

    A = torch.cat([Au, Av], dim=1)  # (B, 8, 8)

    b = torch.cat([u, v], dim=1).unsqueeze(-1)

    return A, b


def apply_tensordlt(CA: torch.Tensor, h4pt: torch.Tensor) -> torch.Tensor:

    # Check input dimensions
    if CA.ndim != 3 or CA.shape[1:] != (4, 2):
        raise ValueError(f"CA must be (B,4,2). Got {CA.shape}")
    if h4pt.ndim != 2 or h4pt.shape[1] != 8:
        raise ValueError(f"h4pt must be (B,8). Got {h4pt.shape}")

    B = CA.shape[0]
    delta = h4pt.view(B, 4, 2)
    CB = CA + delta

    A, b = build_homography(CA, CB)

    # h = torch.linalg.solve(A, b).squeeze(-1)
    h = torch.linalg.lstsq(A, b).solution.squeeze(-1)

    h11, h12, h13 = h[:, 0], h[:, 1], h[:, 2]
    h21, h22, h23 = h[:, 3], h[:, 4], h[:, 5]
    h31, h32 = h[:, 6], h[:, 7]

    one = torch.ones_like(h11)

    H = torch.stack(
        [
            torch.stack([h11, h12, h13], dim=-1),
            torch.stack([h21, h22, h23], dim=-1),
            torch.stack([h31, h32, one], dim=-1),
        ],
        dim=1,
    )

    return H


# Test if calculations are correct with h4pt = 0


# if __name__ == "__main__":
#     # If h4pt = 0, H should be approximately identity
#     B, Hh, Ww = 2, 128, 128
#     CA = torch.tensor([
#         [0.0, 0.0],
#         [Ww - 1.0, 0.0],
#         [Ww - 1.0, Hh - 1.0],
#         [0.0, Hh - 1.0],
#     ]).unsqueeze(0).repeat(B, 1, 1)  # (B,4,2)

#     h4pt = torch.zeros(B, 8)
#     H = tensor_dlt(CA, h4pt)
#     print("H shape:", H.shape)
#     print("H[0]:\n", H[0])
