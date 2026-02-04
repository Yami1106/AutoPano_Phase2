#!/usr/bin/env python3

'''
Unsupervised Learning - Step 3

TensorDLT : convert H4pt to Homography matrix H(3x3)

input :

CA:      (B, 4, 2)  patch corners in PATCH coordinates (x,y)
h4pt:    (B, 8)     predicted offset (CB - CA) in PATCH coordinates:
                    [dx1,dy1, dx2,dy2, dx3,dy3, dx4,dy4]


Output:
H:       (B, 3, 3)  homography mapping CA -> CB_hat (patch-to-patch)

Corner order : CA = [TL, TR, BR, BL]
'''


from typing import Tuple
import torch


def build_homography(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    Ah = B -> linear system to solve for homography h

    (x,y) = (B,H,W)
    (u,v) = (B,H,W)

    h = [h11,h12,h13,h21,h22,h23,h31,h32]^T
    h33 = 1
    '''

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

    Au = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1) # (B, 4, 8)
    Av = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1) # (B, 4, 8)

    A = torch.cat([Au, Av], dim=1)  # (B, 8, 8)

    b = torch.cat([u, v], dim=1).unsqueeze(-1)

    return A, b


def tensor_dlt(CA: torch.Tensor, h4pt: torch.Tensor) -> torch.Tensor:

    # Check input dimensions
    if CA.ndim != 3 or CA.shape[1:] != (4, 2):
        raise ValueError(f"CA must be (B,4,2). Got {CA.shape}")
    if h4pt.ndim != 2 or h4pt.shape[1] != 8:
        raise ValueError(f"h4pt must be (B,8). Got {h4pt.shape}")
    

    B = CA.shape[0]
    delta = h4pt.view(B, 4, 2)
    CB = CA + delta

    A, b = build_homography(CA, CB)

    #h = torch.linalg.solve(A, b).squeeze(-1)
    h = torch.linalg.lstsq(A, b).solution.squeeze(-1)


    h11, h12, h13 = h[:, 0], h[:, 1], h[:, 2]
    h21, h22, h23 = h[:, 3], h[:, 4], h[:, 5]
    h31, h32      = h[:, 6], h[:, 7]


    one = torch.ones_like(h11)

    H = torch.stack([
        torch.stack([h11, h12, h13], dim=-1),
        torch.stack([h21, h22, h23], dim=-1),
        torch.stack([h31, h32, one], dim=-1),
    ], dim=1) 


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
    
 