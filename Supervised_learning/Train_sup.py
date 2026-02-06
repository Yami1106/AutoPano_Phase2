#!/usr/bin/env python

"""
RBE/CS Fall 2025: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from pathlib import Path
from Network.Network import LossFn_supervised, LossFn_unsupervised


class HomographyDataset:
    def __init__(self, patch_dir):
        """
        patch_dir: Path to Phase2/Data/patch_train or patch_val
        """
        self.patch_dir = Path(patch_dir)
        # Get all unique stems by looking for _H4Pt.npy files
        self.stems = [
            f.name.replace("_H4Pt.npy", "") for f in self.patch_dir.glob("*_H4Pt.npy")
        ]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # 1. Load Patches
        path_a = self.patch_dir / f"{stem}_PA.png"
        path_b = self.patch_dir / f"{stem}_PB.png"

        # Load as grayscale (since patches are 128x128)
        pa = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        pb = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)

        # 2. Load Label (shape: 8,)
        label_path = self.patch_dir / f"{stem}_H4Pt.npy"
        h4pt = np.load(str(label_path)).astype(np.float32)

        # 3. Preprocess for PyTorch [C, H, W]
        # Normalize to 0-1
        pa = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
        pb = torch.from_numpy(pb).float().unsqueeze(0) / 255.0

        # Concatenate into 2-channel input: [2, 128, 128]
        input_tensor = torch.cat([pa, pb], dim=0)
        gt_tensor = torch.from_numpy(h4pt)

        return input_tensor, gt_tensor


def GenerateBatch(
    BasePath,
    DirNamesTrain,
    TrainCoordinates,
    ImageSize,
    MiniBatchSize,
    device,
    mode="train",
    indices=None,
):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I1Batch = []
    CoordinatesBatch = []

    # Select the correct subdirectory
    sub_dir = "patch_train" if mode == "train" else "patch_val"
    patch_path = os.path.join(BasePath, sub_dir)

    if indices is None:
        indices = [
            random.randint(0, len(DirNamesTrain) - 1) for _ in range(MiniBatchSize)
        ]

    for idx in indices:

        stem = DirNamesTrain[idx]
        # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        # ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # I1 = np.float32(cv2.imread(RandImageName))
        # Coordinates = TrainCoordinates[RandIdx]

        # 1. Load Patches
        path_a = os.path.join(patch_path, f"{stem}_PA.png")
        path_b = os.path.join(patch_path, f"{stem}_PB.png")

        pa = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        pb = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

        ##########################################################
        # 2. Data Augmentation (Training Only)
        ##########################################################
        if mode == "train":

            random_brightness_prob = 0.8  # 20% (was >0.5 = 50%)
            random_noise_prob = 0.85  # 15% (was >0.6 = 40%)
            gaussian_blur_prob = 0.8  # 20% (was >0.5 = 50%)
            random_contrast_prob = 0.85  # 15% (was >0.6 = 40%)
            random_erasing_prob = 0.9  # 10% (was >0.5 = 50%) ← CRITICAL

            # Brightness (BOTH patches - maintains correspondence)
            if random.random() > random_brightness_prob:
                bf = random.uniform(0.85, 1.15)  # Milder: ±15% vs ±30%
                pa = np.clip(pa * bf, 0, 255).astype(np.uint8)
                pb = np.clip(pb * bf, 0, 255).astype(np.uint8)

            # Noise (independent - tests robustness)
            if random.random() > random_noise_prob:
                noise_std = random.randint(1, 4)  # Milder: 1-4 vs 1-8
                noise_a = np.random.normal(0, noise_std, pa.shape)
                noise_b = np.random.normal(0, noise_std, pb.shape)
                pa = np.clip(pa.astype(np.int16) + noise_a, 0, 255).astype(np.uint8)
                pb = np.clip(pb.astype(np.int16) + noise_b, 0, 255).astype(np.uint8)

            # Gaussian Blur (BOTH - same degradation)
            if random.random() > gaussian_blur_prob:
                kernel_size = 3  # Fixed mild blur (vs random 3/5)
                pa = cv2.GaussianBlur(pa, (kernel_size, kernel_size), 0)
                pb = cv2.GaussianBlur(pb, (kernel_size, kernel_size), 0)

            # Contrast (BOTH - maintains relative intensities)
            if random.random() > random_contrast_prob:
                alpha = random.uniform(0.9, 1.1)  # Milder: ±10% vs ±20%
                pa = np.clip(alpha * pa, 0, 255).astype(np.uint8)
                pb = np.clip(alpha * pb, 0, 255).astype(np.uint8)

            # Erasing (PB only - hardest augmentation)
            if random.random() > random_erasing_prob:
                erase_size = random.randint(15, 25)  # Smaller: 15-25 vs 20-40
                y = random.randint(0, ImageSize[0] - erase_size)
                x = random.randint(0, ImageSize[1] - erase_size)
                pb[y : y + erase_size, x : x + erase_size] = random.randint(0, 255)

        ##########################################################
        # 3. Preprocess: [1, 128, 128] normalized
        ##########################################################
        pa_t = torch.from_numpy(pa).float().unsqueeze(0) / 255.0
        pb_t = torch.from_numpy(pb).float().unsqueeze(0) / 255.0

        # 3. Concatenate to [2, 128, 128]
        input_tensor = torch.cat([pa_t, pb_t], dim=0)
        gt_tensor = torch.from_numpy(TrainCoordinates[idx]).float()

        # SCALE LABELS: Divide by max displacement (32px)
        # rho = 32.0
        # gt_tensor = torch.from_numpy(TrainCoordinates[idx]).float() / rho

        I1Batch.append(input_tensor)
        CoordinatesBatch.append(gt_tensor)

        # Append All Images and Mask
        # I1Batch.append(torch.from_numpy(I1))
        # CoordinatesBatch.append(torch.tensor(Coordinates))
        # ImageNum += 1

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)

    # return torch.stack(I1Batch), torch.stack(CoordinatesBatch)


def EvaluatePixelError(pred_corners, gt_corners):
    """
    Compute mean corner error in pixels.

    Args:
        pred_corners: Predicted 4-point homography offsets [B, 8]
        gt_corners: Ground truth 4-point homography offsets [B, 8]

    Returns:
        mean_error: Mean pixel error across all corners
        corner_errors: Per-corner errors [B, 4]
    """
    # Reshape to [B, 4, 2] for corner-wise computation
    pred = pred_corners.reshape(-1, 4, 2)
    gt = gt_corners.reshape(-1, 4, 2)

    # Since loss function scales by rho, predictions are also scaled
    # Unscale to get actual pixel values
    rho = 32.0
    # pred = pred * rho

    # Compute Euclidean distance for each corner
    corner_errors = torch.sqrt(((pred - gt) ** 2).sum(dim=2))  # [B, 4]

    # Mean error across all corners and batch
    mean_error = corner_errors.mean()

    return mean_error.item(), corner_errors


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    # model = HomographyModel()

    # Initialize the model and move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {"InputSize": ImageSize[0], "OutputSize": 8}
    model = HomographyModel(hparams=hparams).to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################

    # No LR cause OneCycleLR controls it completely
    Optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4, eps=1e-8)

    # Optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=1e-5, weight_decay=1e-2
    # )  # Trying with this optimizer setting to be able to generalize better (increasing the weight decay)

    # Optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=1e-4, weight_decay=1e-3
    # )  # Global learning rate used

    # Optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=5e-5, weight_decay=1e-3
    # )  # Train Loss: 0.0693 | Val Loss: 0.2210 | (On larger dataset: Train Loss: 0.0451 | Val Loss (Avg.): 0.1795 | Per-pixel loss (Avg.): 16.4319px)

    # # lower learning rate and increased the value of the weight decay for stronger regularization
    # Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) # Train Loss: 0.0693 | Val Loss: 0.2210

    # Optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=3e-4, weight_decay=1e-4
    # )  # weight decay was too high earlier

    # Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=1e-4, weight_decay=1e-4
    # )  # Leading overfitting

    # Epoch 49: Val Loss: 183.1978 | with 100 epochs
    # Optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) # Epoch 49: Val Loss: 234.8409
    # Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Epoch 49: Val Loss: 211.4395 (gamma: 0.1, step-size: 10, weight-decay: 1e-4, lr: 1e-4)

    # Decay the learning rate by 10% every 10 epochs
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     Optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    # )  # Global

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     Optimizer, mode="min", factor=0.5, patience=3, threshold=0.01
    # )  # Incremental testing for moving past 16px off

    # scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=20, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)

    # train_path = os.path.join(BasePath, "patch_train")
    # TrainSet = HomographyDataset(train_path)
    # TrainLoader = DataLoader(
    #     TrainSet, batch_size=MiniBatchSize, shuffle=True, num_workers=8, pin_memory=True
    # )

    # --- TRAINING DATA SETUP ---
    train_path = os.path.join(BasePath, "patch_train")
    DirNamesTrain = [
        f.name.replace("_H4Pt.npy", "") for f in Path(train_path).glob("*_H4Pt.npy")
    ]
    TrainCoordinates = [
        np.load(os.path.join(train_path, f"{s}_H4Pt.npy")) for s in DirNamesTrain
    ]

    # --- VALIDATION DATA SETUP ---
    val_path = os.path.join(BasePath, "patch_val")
    DirNamesVal = [
        f.name.replace("_H4Pt.npy", "") for f in Path(val_path).glob("*_H4Pt.npy")
    ]

    ValCoordinates = [
        np.load(os.path.join(val_path, f"{s}_H4Pt.npy")) for s in DirNamesVal
    ]

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    # if LatestFile is not None:
    #     CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
    #     # Extract only numbers from the name
    #     StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
    #     model.load_state_dict(CheckPoint["model_state_dict"])
    #     print("Loaded latest checkpoint with the name " + LatestFile + "....")

    if LatestFile is not None:
        # Construct the full path
        checkpoint_file = os.path.join(CheckPointPath, LatestFile + ".ckpt")

        # checkpoint = torch.load(checkpoint_file)  # Before PyTorch 2.6, if you're saving other data "pixel_error": avg_pixel_error,  # numpy.float64 scalar! # May contain numpy, new pytorch version(2.6+) doesn't allow that
        checkpoint = torch.load(checkpoint_file, weights_only=False)

        # 1. Restore Model Weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # 2. Restore Optimizer State (resumes AdamW buffers)
        Optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # # 3. Restore Scheduler State (resumes LR and patience count)
        # if "scheduler_state_dict" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 4. Update the Epoch counter to start where you left off
        StartEpoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Successfully resumed from Epoch {StartEpoch}")
        print(f"Current Learning Rate: {Optimizer.param_groups[0]['lr']}")

    else:
        StartEpoch = 0
        best_val_loss = float("inf")
        print("New model initialized....")

    ##########################################
    ### Dynamic Learning Rate (OneCycleLR) ###
    ##########################################

    NumStepsPerEpoch = len(TrainCoordinates) // MiniBatchSize  # e.g., 50000//128=390
    total_steps = NumEpochs * NumStepsPerEpoch

    # Using OneCycleLR for dynamic learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        Optimizer,
        max_lr=5e-3,
        total_steps=total_steps,
        pct_start=0.25,
        anneal_strategy="cos",
        div_factor=20,
        final_div_factor=1e4,
    )

    ##########################################

    # Create an index list
    all_indices = np.arange(len(DirNamesTrain))

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):

        train_losses = []
        model.train()
        np.random.shuffle(all_indices)  # Shuffle at start of every epoch

        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)

        # for PerEpochCounter in tqdm(range(NumIterationsPerEpoch), leave=False):
        for PerEpochCounter in range(NumIterationsPerEpoch):
            # for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

            batch_indices = all_indices[
                PerEpochCounter * MiniBatchSize : (PerEpochCounter + 1) * MiniBatchSize
            ]

            # Generate Batch from patch_train
            input_tensor, CoordinatesBatch = GenerateBatch(
                BasePath,
                DirNamesTrain,
                TrainCoordinates,
                ImageSize,
                MiniBatchSize,
                device,
                mode="train",
                indices=batch_indices,
            )

            # # Moving the patches and CoordinatesBatch to gpu
            # input_tensor, CoordinatesBatch = (
            #     input_tensor.to(device),
            #     CoordinatesBatch.to(device),
            # )

            Optimizer.zero_grad()
            batch = (input_tensor, CoordinatesBatch)

            PredicatedCoordinatesBatch = model.training_step(
                batch, batch_idx=PerEpochCounter
            )

            LossThisBatch = PredicatedCoordinatesBatch["loss"]

            train_losses.append(LossThisBatch.item())

            LossThisBatch.backward()
            Optimizer.step()
            scheduler.step()

            # Every 20 batches
            if PerEpochCounter % 20 == 0:
                current_lr = Optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {Epochs} Batch {PerEpochCounter}/{NumIterationsPerEpoch}, "
                    f"LR: {current_lr:.2e}, Loss: {LossThisBatch.item():.4f}"
                )

            # Log Training Loss every iteration so you can see the curve
            Writer.add_scalar(
                "Loss/Train_Batch",
                LossThisBatch.item(),
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:

                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # result = model.validation_step(Batch)

        model.eval()
        val_losses = []
        pixel_errors = []

        # Use a fixed number of samples for validation to save time
        NumValIters = max(1, int(len(DirNamesVal) / MiniBatchSize))

        with torch.no_grad():
            for _ in range(NumValIters):
                v_input, v_gt = GenerateBatch(
                    BasePath,
                    DirNamesVal,
                    ValCoordinates,
                    ImageSize,
                    MiniBatchSize,
                    device,
                    mode="val",
                )

                # Move validation batch to GPU
                input_tensor, gt = v_input.to(device), v_gt.to(device)

                # If your validation_step expects [patch_a, patch_b, gt]
                current_batch = (input_tensor, gt)

                # Compute loss
                result = model.validation_step(current_batch, batch_idx=None)
                val_losses.append(result["val_loss"])

                # Compute pixel error
                pred = model(input_tensor)
                mean_pixel_error, _ = EvaluatePixelError(pred, gt)
                pixel_errors.append(mean_pixel_error)

        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = torch.stack(val_losses).mean()
        avg_pixel_error = np.mean(pixel_errors)

        # Enhanced logging
        Writer.add_scalar("Loss/Validation_Epoch", avg_val_loss, Epochs)
        Writer.add_scalar("Metrics/Mean_Pixel_Error", avg_pixel_error, Epochs)
        Writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], Epochs)
        print(
            f"Epoch {Epochs}: Train Loss: {LossThisBatch.item():.4f} | Train Loss (Avg.): {avg_train_loss:.4f} | Val Loss (Avg.): {avg_val_loss:.4f} | Per-pixel loss (Avg.): {avg_pixel_error:.4f}px"
        )

        Writer.flush()
        # scheduler.step(avg_val_loss) ## (for other LR schedulers like StepLR, ReduceLROnPlateau, and not for the OneCycleLR)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_save_dict = {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),  # Save this for LR tracking
                "val_loss": avg_val_loss,
                "pixel_error": avg_pixel_error,
                "rho": 32.0,  # Good to save constants for future reference
            }

            torch.save(best_save_dict, CheckPointPath + "best_model.ckpt")
            print(
                f"New Best Model saved! Loss: {best_val_loss:.4f} | Error: {avg_pixel_error:.2f}px"
            )

        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        # default="/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
        default="Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="Phase2/Checkpoints",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=128,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Phase2/Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    # Extract arguments
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    # (
    #     DirNamesTrain,
    #     SaveCheckPoint,
    #     ImageSize,
    #     NumTrainSamples,
    #     TrainCoordinates,
    #     NumClasses,
    # ) = SetupAll(BasePath, CheckPointPath)

    # Create Checkpoint path if it doesn't exist
    if not os.path.exists(CheckPointPath):
        os.makedirs(CheckPointPath)

    # Bypass legacy text file reading
    train_path = os.path.join(BasePath, "patch_train")
    # Count how many H4Pt files are in the directory to get NumTrainSamples
    NumTrainSamples = len(glob.glob(os.path.join(train_path, "*_H4Pt.npy")))

    # Set default values for variables the script still expects
    SaveCheckPoint = 100
    ImageSize = [128, 128, 2]
    DirNamesTrain = None  # Not needed by DataLoader
    TrainCoordinates = None  # Not needed by DataLoader

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
