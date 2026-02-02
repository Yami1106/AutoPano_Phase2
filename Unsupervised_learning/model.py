#!/usr/bin/env python3

'''
Unsupervised Learning - Step 2

Homographynet

input :

stacked patches: (B,2,H,W), channel 0: PA, channel 1: PB

Output:

H4pt_hat: (B,8) 
          [dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4]
          CB_hat - CA

ONLY our CNN model is defined here
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Homographynet(nn.Module):
    def __init__(self,in_channels: int=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #2nd block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #3rd block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #4th block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )


        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        h4pt = self.regressor(x)
        return h4pt
        

# if __name__ == "__main__":

#     # test the shape with dummy data 
#     # B, H, W = 8, 128, 128
#     # dummy = torch.randn(B, 2, H, W)
#     # net = Homographynet(in_channels=2)
#     # out = net(dummy)
#     # print("Input:", dummy.shape)
#     # print("Output:", out.shape)