#!/usr/bin/env python3
"""
RBE/CS Spring 2026: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Testing Code

Author(s):
Ashish Sukumar (asukumar1@wpi.edu)
Rajdeep Banerjee (rbanerjee1@wpi.edu)
Worcester Polytechnic Institute
"""

import cv2
import os
import sys
import glob
import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import time

# Don't generate pyc codes
sys.dont_write_bytecode = True

# Smart import: works from both Code/ and Code/Unsupervised_learning/
try:
    # Try importing as if we're in Code/ directory
    from Unsupervised_learning.model import HomographyNet
except ModuleNotFoundError:
    # We're in Unsupervised_learning/ directory, import directly
    from model import HomographyNet


def SetupAll(BasePath):
    PatchSize = 128
    ImageSize = [PatchSize, PatchSize, 1]  # Grayscale
    
    # Get all test images
    TestImagePaths = []
    if os.path.isdir(BasePath):
        for ImageName in glob.glob(os.path.join(BasePath, '*.jpg')):
            TestImagePaths.append(ImageName)
    
    if len(TestImagePaths) == 0:
        print(f"ERROR: No test images found in {BasePath}")
        sys.exit()
    
    print(f"Found {len(TestImagePaths)} test images")
    
    return ImageSize, TestImagePaths


def ReadAndPreprocessImage(ImagePath, PatchSize=128):
    Image = cv2.imread(ImagePath)
    if Image is None:
        print(f"ERROR: Cannot read image {ImagePath}")
        return None, None
    
    ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    
    return Image, ImageGray


def ExtractCenterPatch(Image, PatchSize=128):
    H, W = Image.shape[:2]
    
    # Calculate center
    CenterY = H // 2
    CenterX = W // 2
    
    # Extract patch
    HalfPatch = PatchSize // 2
    TopLeft = (CenterX - HalfPatch, CenterY - HalfPatch)
    
    Patch = Image[
        CenterY - HalfPatch : CenterY + HalfPatch,
        CenterX - HalfPatch : CenterX + HalfPatch
    ]
    
    if Patch.shape[0] != PatchSize or Patch.shape[1] != PatchSize:
        print(f"WARNING: Image too small, resizing to {PatchSize}x{PatchSize}")
        Patch = cv2.resize(Image, (PatchSize, PatchSize))
        TopLeft = (0, 0)
    
    return Patch, TopLeft


def GenerateSyntheticPair(ImageGray, PatchSize=128, Rho=32):
    H, W = ImageGray.shape
    
    # Ensure we can extract patch after perturbation
    if H < PatchSize + 2*Rho or W < PatchSize + 2*Rho:
        # Resize image if too small
        scale = max((PatchSize + 2*Rho) / H, (PatchSize + 2*Rho) / W)
        NewH = int(H * scale) + 10
        NewW = int(W * scale) + 10
        ImageGray = cv2.resize(ImageGray, (NewW, NewH))
        H, W = ImageGray.shape
    
    # Random patch location in valid region
    y = np.random.randint(Rho, H - PatchSize - Rho)
    x = np.random.randint(Rho, W - PatchSize - Rho)
    
    # Extract patch A
    PA = ImageGray[y:y+PatchSize, x:x+PatchSize].copy()
    
    # Define corners of patch A
    CA = np.array([
        [x, y],                        # Top-left
        [x + PatchSize, y],            # Top-right
        [x + PatchSize, y + PatchSize],  # Bottom-right
        [x, y + PatchSize]             # Bottom-left
    ], dtype=np.float32)
    
    # Random perturbation for each corner
    perturbation = np.random.randint(-Rho, Rho, size=(4, 2)).astype(np.float32)
    CB = CA + perturbation
    
    # Compute H4Pt ground truth
    H4Pt_GT = (CB - CA).reshape(-1)  # Shape: (8,)
    
    # Compute homography from CA to CB
    H_AB = cv2.getPerspectiveTransform(CA, CB)
    H_BA = np.linalg.inv(H_AB)
    
    # Warp image with inverse homography
    IB = cv2.warpPerspective(ImageGray, H_BA, (W, H))
    
    # Extract patch B at same location as PA
    PB = IB[y:y+PatchSize, x:x+PatchSize].copy()
    
    return PA, PB, H4Pt_GT, CA


def ComputeEPE(H4Pt_pred, H4Pt_GT):
    # Reshape to (4, 2) for easier computation
    pred_corners = H4Pt_pred.reshape(4, 2)
    gt_corners = H4Pt_GT.reshape(4, 2)
    
    # Compute L2 distance for each corner
    distances = np.linalg.norm(pred_corners - gt_corners, axis=1)
    
    # Average across 4 corners
    EPE = np.mean(distances)
    
    return EPE


def TestOperation(ImageSize, ModelPath, TestImagePaths, NumTestsPerImage=10):
    PatchSize = ImageSize[0]
    
    # Check if CUDA is available
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {Device}")
    
    # Load model
    print(f"Loading model from {ModelPath}")
    model = HomographyNet()
    model = model.to(Device)
    
    CheckPoint = torch.load(ModelPath, map_location=Device)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Statistics
    all_epes = []
    all_times = []
    
    print(f"\nTesting on {len(TestImagePaths)} images...")
    print(f"Generating {NumTestsPerImage} synthetic pairs per image\n")
    
    # Test on each image
    for img_idx, ImagePath in enumerate(tqdm(TestImagePaths, desc="Testing Images")):
        ImageName = os.path.basename(ImagePath)
        
        # Read image
        ImageColor, ImageGray = ReadAndPreprocessImage(ImagePath, PatchSize)
        if ImageGray is None:
            continue
        
        image_epes = []
        
        # Generate multiple synthetic pairs per image
        for test_idx in range(NumTestsPerImage):
            # Generate synthetic pair with ground truth
            PA, PB, H4Pt_GT, CA = GenerateSyntheticPair(ImageGray, PatchSize, Rho=32)
            
            # Stack patches depthwise
            PA_norm = PA.astype(np.float32) / 255.0
            PB_norm = PB.astype(np.float32) / 255.0
            
            # Create input tensor [1, 2, H, W]
            Input = np.stack([PA_norm, PB_norm], axis=0)
            Input = torch.from_numpy(Input).unsqueeze(0).float().to(Device)
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                H4Pt_pred = model(Input)
                inference_time = time.time() - start_time
                
            # Convert to numpy
            H4Pt_pred = H4Pt_pred.cpu().numpy().squeeze()
            
            # Compute EPE
            epe = ComputeEPE(H4Pt_pred, H4Pt_GT)
            
            image_epes.append(epe)
            all_epes.append(epe)
            all_times.append(inference_time * 1000)  # Convert to ms
        
        # Print per-image statistics
        if (img_idx + 1) % 10 == 0 or img_idx == len(TestImagePaths) - 1:
            avg_epe = np.mean(image_epes)
            print(f"\n{ImageName}: Average EPE = {avg_epe:.2f} pixels")
    
    # Compute overall statistics
    mean_epe = np.mean(all_epes)
    std_epe = np.std(all_epes)
    median_epe = np.median(all_epes)
    min_epe = np.min(all_epes)
    max_epe = np.max(all_epes)
    
    mean_time = np.mean(all_times)
    
    # Print results
    print("\n" + "="*60)
    print("TESTING RESULTS")
    print("="*60)
    print(f"Total test samples: {len(all_epes)}")
    print(f"Images tested: {len(TestImagePaths)}")
    print(f"Tests per image: {NumTestsPerImage}")
    print("-"*60)
    print(f"EPE Statistics (pixels):")
    print(f"  Mean:   {mean_epe:.2f} ± {std_epe:.2f}")
    print(f"  Median: {median_epe:.2f}")
    print(f"  Min:    {min_epe:.2f}")
    print(f"  Max:    {max_epe:.2f}")
    print("-"*60)
    print(f"Inference time: {mean_time:.2f} ms per forward pass")
    print("="*60)
    
    # Save results to file
    ResultsPath = os.path.join(os.path.dirname(ModelPath), 'test_results.txt')
    with open(ResultsPath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("UNSUPERVISED HOMOGRAPHY TESTING RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {ModelPath}\n")
        f.write(f"Test images: {len(TestImagePaths)}\n")
        f.write(f"Total samples: {len(all_epes)}\n")
        f.write(f"Tests per image: {NumTestsPerImage}\n")
        f.write("-"*60 + "\n")
        f.write(f"EPE Statistics (pixels):\n")
        f.write(f"  Mean:   {mean_epe:.2f} ± {std_epe:.2f}\n")
        f.write(f"  Median: {median_epe:.2f}\n")
        f.write(f"  Min:    {min_epe:.2f}\n")
        f.write(f"  Max:    {max_epe:.2f}\n")
        f.write("-"*60 + "\n")
        f.write(f"Inference time: {mean_time:.2f} ms\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {ResultsPath}")
    
    return mean_epe


def main():
    """
    Main function to test homography model (supervised or unsupervised)
    """
    # Parse command line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        '--ModelType',
        dest='ModelType',
        default='unsupervised',
        choices=['supervised', 'unsupervised'],
        help='Model type: supervised or unsupervised, Default: unsupervised'
    )
    Parser.add_argument(
        '--ModelPath',
        dest='ModelPath',
        default=None,
        help='Path to trained model checkpoint. If None, uses default path based on ModelType'
    )
    Parser.add_argument(
        '--BasePath',
        dest='BasePath',
        default='../Data/Phase2',
        help='Path to test images, Default: ../Data/Phase2'
    )
    Parser.add_argument(
        '--NumTestsPerImage',
        type=int,
        default=10,
        help='Number of synthetic test pairs per image, Default: 10'
    )
    
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    BasePath = Args.BasePath
    NumTestsPerImage = Args.NumTestsPerImage

    if Args.ModelPath is None:
        if ModelType == 'unsupervised':
            ModelPath = '../Checkpoints/unsup/best.pt'
        else:
            ModelPath = '../Checkpoints/sup/best.pt'
    else:
        ModelPath = Args.ModelPath
    
    print(f"Testing {ModelType} model")
    print(f"Model path: {ModelPath}")
    
    # Setup
    ImageSize, TestImagePaths = SetupAll(BasePath)
    
    # Check if model exists
    if not os.path.isfile(ModelPath):
        print(f"ERROR: Model checkpoint does not exist at {ModelPath}")
        sys.exit()
    
    # Run testing
    TestOperation(ImageSize, ModelPath, TestImagePaths, NumTestsPerImage)


if __name__ == '__main__':
    main()