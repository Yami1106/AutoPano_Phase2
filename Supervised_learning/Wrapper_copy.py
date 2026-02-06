#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import cv2
import torch
import numpy as np

# Add any python libraries here


def TensorDLT(pred_offsets, patch_size=128):
    """
    Implements the Differentiable Tensor Direct Linear Transform.

    Args:
        pred_offsets: Tensor [B, 8] - Predicted corner displacements
        patch_size: Scalar - The size of the square patch (usually 128)

    Returns:
        H_matrix: Tensor [B, 3, 3] - The estimated 3x3 homography matrices
    """
    device = pred_offsets.device
    batch_size = pred_offsets.shape[0]

    # 1. Define source corners (xi) for a 128x128 patch
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    x = torch.tensor(
        [
            [0, 0],
            [patch_size - 1, 0],
            [patch_size - 1, patch_size - 1],
            [0, patch_size - 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Expand to batch size [B, 4, 2]
    xi = x.unsqueeze(0).repeat(batch_size, 1, 1)
    ui, vi = xi[:, :, 0], xi[:, :, 1]

    # 2. Define target corners (xi') = xi + offsets
    yi = xi + pred_offsets.view(-1, 4, 2)
    u_prime, v_prime = yi[:, :, 0], yi[:, :, 1]

    # 3. Construct Matrix A_hat [B, 8, 8]
    # Based on Equation (7) and the definition of A_i in the paper
    A_hat = torch.zeros((batch_size, 8, 8), device=device)

    for i in range(4):
        # Even rows of A_hat (from A_i)
        A_hat[:, 2 * i, 0] = 0
        A_hat[:, 2 * i, 1] = 0
        A_hat[:, 2 * i, 2] = 0
        A_hat[:, 2 * i, 3] = -ui[:, i]
        A_hat[:, 2 * i, 4] = -vi[:, i]
        A_hat[:, 2 * i, 5] = -1
        A_hat[:, 2 * i, 6] = v_prime[:, i] * ui[:, i]
        A_hat[:, 2 * i, 7] = v_prime[:, i] * vi[:, i]

        # Odd rows of A_hat (from A_i)
        A_hat[:, 2 * i + 1, 0] = ui[:, i]
        A_hat[:, 2 * i + 1, 1] = vi[:, i]
        A_hat[:, 2 * i + 1, 2] = 1
        A_hat[:, 2 * i + 1, 3] = 0
        A_hat[:, 2 * i + 1, 4] = 0
        A_hat[:, 2 * i + 1, 5] = 0
        A_hat[:, 2 * i + 1, 6] = -u_prime[:, i] * ui[:, i]
        A_hat[:, 2 * i + 1, 7] = -u_prime[:, i] * vi[:, i]

    # 4. Construct Vector b_hat [B, 8, 1]
    # b_i = [-v'_i, u'_i]^T
    b_hat = torch.zeros((batch_size, 8, 1), device=device)
    for i in range(4):
        b_hat[:, 2 * i, 0] = -v_prime[:, i]
        b_hat[:, 2 * i + 1, 0] = u_prime[:, i]

    # 5. Solve Ah = b using Pseudo-Inverse (Differentiable)
    # h = A_hat.pinverse() @ b_hat
    h_8 = torch.linalg.solve(A_hat, b_hat)  # Use solve for square matrix A_hat

    # 6. Reconstruct 3x3 Homography Matrix
    # Append the H33 = 1 element
    h_9 = torch.cat(
        [h_8.squeeze(-1), torch.ones((batch_size, 1), device=device)], dim=1
    )
    H_matrix = h_9.view(-1, 3, 3)

    return H_matrix


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
