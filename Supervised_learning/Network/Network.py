"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl


# import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn_supervised(delta, gt):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    # loss = F.mse_loss(delta, gt)
    # rho = 32.0
    # loss = F.mse_loss(delta / rho, gt / rho)
    loss = F.mse_loss(delta, gt)
    # loss = F.l1_loss(delta, gt)
    return loss
    # return loss


def LossFn_unsupervised(delta, img_a, patch_b, corners):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    loss = ...
    return loss


# class HomographyModel(pl.LightningModule):
class HomographyModel(nn.Module):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.hparams = hparams
        self.model = Net(
            InputSize=self.hparams["InputSize"], OutputSize=self.hparams["OutputSize"]
        )

    # def forward(self, patch_a, patch_b):
    def forward(self, x):
        # x = torch.cat((patch_a, patch_b), dim=1)
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # img_a, patch_a, patch_b, corners, gt = batch
        x, gt = batch

        # Pass the combined tensor to HomographyNet
        delta = self.model(x)

        # Compute the loss
        loss = LossFn_supervised(delta, gt)
        # loss = LossFn_unsupervised(delta, img_a, patch_b, corners)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        # img_a, patch_a, patch_b, corners, gt = batch
        x, gt = batch  # x is [Batch, 2, 128, 128]

        # Pass the combined tensor to HomographyNet
        delta = self.model(x)

        # Compute the loss
        loss = LossFn_supervised(delta, gt)
        # loss = LossFn_unsupervised(delta, img_a, patch_b, corners)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()

        #############################
        # Fill your network initialization of choice here!
        #############################

        # Helper for conv blocks
        def conv_block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

        ##########################
        # Convolution layers
        ##########################
        self.conv1 = conv_block(2, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.conv5 = conv_block(64, 128)
        self.conv6 = conv_block(128, 128)
        self.conv7 = conv_block(128, 128)
        self.conv8 = conv_block(128, 128)

        ##########################
        # Max-pooling layers
        ##########################
        self.pool_ly1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_ly2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_ly3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_ly4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU(inplace=True)

        ##########################
        # Fully-connected layers
        ##########################
        # After three max-pooling the spatial dimension is 1/8th of the Input Size
        self.fc_ly1 = nn.Linear(InputSize // 8 * InputSize // 8 * 128, 1024)
        self.fc_ly2 = nn.Linear(1024, OutputSize)  # 4 corner regression

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    # def forward(self, xa, xb):
    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        # Input: 128x128x2
        # x = torch.cat((xa, xb), dim=1)

        # Block 1
        # 128x128x64 -> 128x128x64 -> 64x64x64
        x = self.pool_ly1(self.conv2(self.conv1(x)))

        # Block 2
        # 64x64x64 -> 64x64x64 -> 32x32x128
        x = self.pool_ly2(self.conv4(self.conv3(x)))

        # Block 3
        # 32x32x128 -> 32x32x128 -> 16x16x128
        x = self.pool_ly3(self.conv6(self.conv5(x)))

        # Block 4
        # 16x16x128 -> 16x16x128 -> 16x16x128
        x = self.conv8(self.conv7(x))

        # Flattening: 32,768
        x = x.view(x.size(0), -1)

        # 16x16x128,
        x = F.relu(self.fc_ly1(self.dropout(x)))
        out = self.fc_ly2(x)

        return out
