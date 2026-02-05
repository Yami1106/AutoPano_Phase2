#!/usr/bin/env python3
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Wrapper

Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute

Panorama stitching using:
- Shi-Tomasi corner detection
- Feature descriptors (ORB/SIFT)
- Custom SSD-based feature matching with ratio test
- Deep learning homography estimation (best.pt)
- Feathered blending
"""

import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from Unsupervised_learning.model import Homographynet
from Unsupervised_learning.tensordlt import tensor_dlt


# ========================
# 1. IMAGE LOADING
# ========================

def read_images(image_path: str) -> List[np.ndarray]:
    """
    Read all images from the given path.
    
    Args:
        image_path: Path to folder containing images or pattern (e.g., "images/*.jpg")
    
    Returns:
        List of images in BGR format
    """
    if Path(image_path).is_dir():
        pattern = str(Path(image_path) / "*.jpg")
        img_files = sorted(glob.glob(pattern))
        if len(img_files) == 0:
            pattern = str(Path(image_path) / "*.png")
            img_files = sorted(glob.glob(pattern))
    else:
        img_files = sorted(glob.glob(image_path))
    
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in: {image_path}")
    
    images = []
    for img_file in img_files:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        images.append(img)
        print(f"Loaded: {Path(img_file).name} - Shape: {img.shape}")
    
    return images


# ========================
# 2. SHI-TOMASI CORNER DETECTION
# ========================

def detect_shi_tomasi_corners(image: np.ndarray, max_corners: int = 1000, 
                               quality_level: float = 0.01, min_distance: int = 10) -> np.ndarray:
    """
    Detect Shi-Tomasi corners using cv2.goodFeaturesToTrack.
    
    Args:
        image: Input image (BGR)
        max_corners: Maximum number of corners to detect
        quality_level: Quality level (0-1)
        min_distance: Minimum distance between corners
    
    Returns:
        corners: Nx2 array of corner coordinates (x, y)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )
    
    if corners is None:
        return np.array([])
    
    # Reshape from (N, 1, 2) to (N, 2)
    corners = corners.reshape(-1, 2)
    return corners


# ========================
# 3. CUSTOM FEATURE DESCRIPTORS (NO CV2.SIFT/ORB)
# ========================

def compute_descriptors(image: np.ndarray, keypoints: np.ndarray, 
                        patch_size: int = 40, output_size: int = 8) -> Tuple[List, np.ndarray]:
    """
    Compute custom feature descriptors around detected keypoints.
    Based on Phase 1 implementation - extracts normalized patches.
    
    This is a custom implementation that does not use cv2.SIFT or cv2.ORB.
    Instead, it extracts Gaussian-blurred, subsampled patches and normalizes them.
    
    Args:
        image: Input image (BGR)
        keypoints: Nx2 array of keypoint coordinates
        patch_size: Size of patch to extract around each keypoint (default: 40)
        output_size: Final descriptor dimension after subsampling (default: 8)
    
    Returns:
        cv_keypoints: List of cv2.KeyPoint objects (for compatibility)
        descriptors: Nx(output_size^2) descriptor array
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    half_patch = patch_size // 2
    
    # Add padding to handle features near boundaries
    padded = cv2.copyMakeBorder(
        gray, 
        half_patch, half_patch, half_patch, half_patch, 
        cv2.BORDER_REFLECT
    )
    
    descriptors = []
    valid_keypoints = []
    
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        
        # Offset coordinates to account for padding
        x_pad, y_pad = x + half_patch, y + half_patch
        
        # Extract high-resolution patch
        patch = padded[
            y_pad - half_patch : y_pad + half_patch + 1,
            x_pad - half_patch : x_pad + half_patch + 1
        ]
        
        # Apply Gaussian blur (anti-aliasing filter)
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        
        # Subsample to reduce dimensionality (e.g., 40x40 -> 8x8)
        subsampled = cv2.resize(
            blurred_patch, 
            (output_size, output_size), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Flatten to 1D vector
        feature_vec = subsampled.reshape(-1)
        
        # Standardize: zero-mean, unit-variance (intensity invariance)
        mean = np.mean(feature_vec)
        std = np.std(feature_vec)
        feature_vec = (feature_vec - mean) / (std + 1e-7)
        
        descriptors.append(feature_vec)
        valid_keypoints.append(pt)
    
    descriptors = np.array(descriptors, dtype=np.float32)
    
    # Convert to cv2.KeyPoint objects for compatibility
    cv_keypoints = [
        cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) 
        for pt in valid_keypoints
    ]
    
    return cv_keypoints, descriptors


# ========================
# 4. CUSTOM FEATURE MATCHING (NO CV2 MATCHERS)
# ========================

def match_features(desc1: np.ndarray, desc2: np.ndarray, 
                   ratio_thresh: float = 0.75) -> List[Tuple[int, int, float]]:
    """
    Match features using custom Sum of Squared Differences (SSD) and Lowe's ratio test.
    
    This is a custom implementation based on Phase 1 code that does not use 
    cv2.BFMatcher or cv2.FlannBasedMatcher.
    
    Args:
        desc1: Descriptors from image 1 (NxD array)
        desc2: Descriptors from image 2 (MxD array)
        ratio_thresh: Lowe's ratio test threshold (default: 0.75)
    
    Returns:
        matches: List of tuples (queryIdx, trainIdx, distance)
    """
    # Flatten descriptors to ensure they are 2D arrays
    desc1 = desc1.reshape(desc1.shape[0], -1).astype(np.float32)
    desc2 = desc2.reshape(desc2.shape[0], -1).astype(np.float32)
    
    matches = []
    
    # For each descriptor in image 1, find best and second-best match in image 2
    for i in range(len(desc1)):
        # Compute Sum of Squared Differences (SSD) to all descriptors in desc2
        diff = desc1[i] - desc2
        ssd = np.sum(diff**2, axis=1)
        
        # Find two best matches (lowest SSD)
        sorted_indices = np.argsort(ssd)
        
        if len(sorted_indices) < 2:
            continue
        
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_ssd = ssd[best_idx]
        second_best_ssd = ssd[second_best_idx]
        
        # Apply Lowe's ratio test
        # Accept match only if best SSD is significantly lower than second best
        # This rejects ambiguous matches in repetitive patterns
        if best_ssd < ratio_thresh * second_best_ssd:
            matches.append((i, best_idx, float(best_ssd)))
    
    return matches


def get_matched_points(kp1: List, kp2: List, matches: List[Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matched point coordinates from keypoints and matches.
    
    Args:
        kp1: List of cv2.KeyPoint objects from image 1
        kp2: List of cv2.KeyPoint objects from image 2
        matches: List of tuples (queryIdx, trainIdx, distance)
    
    Returns:
        pts1: Nx2 array of points in image 1
        pts2: Nx2 array of corresponding points in image 2
    """
    pts1 = np.float32([kp1[m[0]].pt for m in matches])
    pts2 = np.float32([kp2[m[1]].pt for m in matches])
    return pts1, pts2


# ========================
# 5. DEEP LEARNING HOMOGRAPHY ESTIMATION
# ========================

class HomographyEstimator:
    """Homography estimator using trained deep learning model."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the estimator.
        
        Args:
            model_path: Path to best.pt checkpoint
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Homographynet(in_channels=2).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Using device: {self.device}")
    
    def estimate_homography(self, img1: np.ndarray, img2: np.ndarray, 
                           pts1: np.ndarray, pts2: np.ndarray,
                           patch_size: int = 128, num_samples: int = 10) -> np.ndarray:
        """
        Estimate homography using the deep learning model.
        Uses multiple patch pairs and averages the predictions.
        
        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            pts1: Matched points in image 1 (Nx2)
            pts2: Matched points in image 2 (Nx2)
            patch_size: Size of patches to extract
            num_samples: Number of matches to sample for averaging
        
        Returns:
            H: 3x3 homography matrix (maps img2 -> img1)
        """
        # Sample multiple matches and average their homography predictions
        num_samples = min(num_samples, len(pts1))
        
        # Randomly sample match indices
        if len(pts1) > num_samples:
            indices = np.random.choice(len(pts1), size=num_samples, replace=False)
        else:
            indices = np.arange(len(pts1))
        
        valid_homographies = []
        
        for idx in indices:
            pt1 = pts1[idx]
            pt2 = pts2[idx]
            
            # Extract patches around the matched points
            PA = self._extract_patch(img1, pt1, patch_size)
            PB = self._extract_patch(img2, pt2, patch_size)
            
            if PA is None or PB is None:
                continue
            
            # Predict homography
            H = self._predict_homography_from_patches(PA, PB, pt1, pt2, patch_size)
            
            if H is not None:
                valid_homographies.append(H)
        
        if len(valid_homographies) == 0:
            print("Warning: Could not estimate homography, using identity")
            return np.eye(3, dtype=np.float32)
        
        # Average the homographies (simple element-wise mean)
        H_avg = np.mean(valid_homographies, axis=0)
        
        # Normalize so that H[2,2] = 1
        H_avg = H_avg / H_avg[2, 2]
        
        print(f"Homography estimated from {len(valid_homographies)}/{num_samples} valid patches")
        return H_avg.astype(np.float32)
    
    def _extract_patch(self, img: np.ndarray, center: np.ndarray, size: int) -> Optional[np.ndarray]:
        """Extract a patch centered at the given point."""
        x, y = int(center[0]), int(center[1])
        half_size = size // 2
        
        x1 = x - half_size
        y1 = y - half_size
        x2 = x1 + size
        y2 = y1 + size
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            return None
        
        patch = img[y1:y2, x1:x2]
        
        # Convert to grayscale
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        return patch
    
    def _predict_homography_from_patches(self, PA: np.ndarray, PB: np.ndarray,
                                         pt1: np.ndarray, pt2: np.ndarray,
                                         patch_size: int) -> Optional[np.ndarray]:
        """Predict homography from patch pair."""
        with torch.no_grad():
            # Normalize patches
            PA_norm = PA.astype(np.float32) / 255.0
            PB_norm = PB.astype(np.float32) / 255.0
            
            # Stack patches
            stacked = np.stack([PA_norm, PB_norm], axis=0)  # (2, H, W)
            stacked_t = torch.from_numpy(stacked).unsqueeze(0).to(self.device)  # (1, 2, H, W)
            
            # Predict H4pt
            h4pt = self.model(stacked_t)  # (1, 8)
            
            # Define corner coordinates in patch coordinate system
            CA = torch.tensor([
                [0, 0],
                [patch_size - 1, 0],
                [patch_size - 1, patch_size - 1],
                [0, patch_size - 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4, 2)
            
            # Compute homography using TensorDLT
            H_patch = tensor_dlt(CA, h4pt)  # (1, 3, 3)
            H_patch = H_patch[0].cpu().numpy()  # (3, 3)
            
            # Convert patch-level homography to image-level homography
            # Shift to account for patch location
            half_size = patch_size // 2
            
            # Translation matrices
            T1 = np.array([
                [1, 0, pt1[0] - half_size],
                [0, 1, pt1[1] - half_size],
                [0, 0, 1]
            ], dtype=np.float32)
            
            T2_inv = np.array([
                [1, 0, -(pt2[0] - half_size)],
                [0, 1, -(pt2[1] - half_size)],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # H_image = T2_inv @ H_patch @ T1
            H_image = T2_inv @ H_patch @ T1
            
            return H_image


# ========================
# 6. PANORAMA STITCHING WITH FEATHERED BLENDING
# ========================

def estimate_panorama_order(images: List[np.ndarray], matcher_params: Dict) -> List[int]:
    """
    Estimate the order of images for panorama stitching.
    Uses feature matching to find adjacent images.
    
    Returns:
        order: List of indices representing stitching order
    """
    n = len(images)
    
    # Compute all pairwise matches
    match_counts = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Detect features
            corners_i = detect_shi_tomasi_corners(images[i], max_corners=matcher_params['max_corners'])
            corners_j = detect_shi_tomasi_corners(images[j], max_corners=matcher_params['max_corners'])
            
            # Compute descriptors (custom patch-based)
            kp_i, desc_i = compute_descriptors(images[i], corners_i)
            kp_j, desc_j = compute_descriptors(images[j], corners_j)
            
            # Match (custom SSD-based matching)
            matches = match_features(desc_i, desc_j, ratio_thresh=matcher_params['ratio_thresh'])
            
            match_counts[i, j] = len(matches)
            match_counts[j, i] = len(matches)
            
            print(f"Image {i} <-> Image {j}: {len(matches)} matches")
    
    # Find stitching order using greedy approach
    # Start with the image that has most total matches
    total_matches = match_counts.sum(axis=1)
    current = np.argmax(total_matches)
    order = [current]
    used = {current}
    
    while len(order) < n:
        # Find the unused image with most matches to current
        best_next = -1
        best_count = 0
        
        for i in range(n):
            if i not in used and match_counts[current, i] > best_count:
                best_count = match_counts[current, i]
                best_next = i
        
        if best_next == -1:
            # No more connected images, add remaining arbitrarily
            for i in range(n):
                if i not in used:
                    order.append(i)
                    used.add(i)
            break
        
        order.append(best_next)
        used.add(best_next)
        current = best_next
    
    print(f"Estimated stitching order: {order}")
    return order


def stitch_image_pair(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Stitch two images with feathered blending.
    
    Args:
        img1: Base image
        img2: Image to warp onto img1
        H: Homography matrix (maps img2 -> img1)
    
    Returns:
        panorama: Stitched result
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find the corners of img2 in img1's coordinate system
    corners_img2 = np.array([
        [0, 0, 1],
        [w2, 0, 1],
        [w2, h2, 1],
        [0, h2, 1]
    ], dtype=np.float32).T
    
    corners_transformed = H @ corners_img2
    corners_transformed = corners_transformed / corners_transformed[2, :]
    corners_transformed = corners_transformed[:2, :].T
    
    # Find bounding box
    all_corners = np.vstack([
        [[0, 0], [w1, 0], [w1, h1], [0, h1]],
        corners_transformed
    ])
    
    min_x, min_y = np.min(all_corners, axis=0).astype(int)
    max_x, max_y = np.max(all_corners, axis=0).astype(int)
    
    # Adjust for negative coordinates
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Warp img2
    H_adjusted = translation @ H
    out_w = max_x - min_x
    out_h = max_y - min_y
    
    img2_warped = cv2.warpPerspective(img2, H_adjusted, (out_w, out_h))
    
    # Calculate where img1 goes
    x_offset = -min_x
    y_offset = -min_y
    
    # Create masks
    gray2 = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    mask2 = (gray2 > 10).astype(np.float32)
    
    mask1 = np.zeros((out_h, out_w), dtype=np.float32)
    mask1[y_offset:y_offset + h1, x_offset:x_offset + w1] = 1.0
    
    # Find overlap
    overlap = (mask1 > 0) & (mask2 > 0)
    
    if overlap.sum() > 100:
        # Create feathered blend in overlap region
        feather_width = 30
        
        # Create distance transform from img1 boundary
        dist1 = cv2.distanceTransform((mask1 * 255).astype(np.uint8), cv2.DIST_L2, 5)
        dist1 = np.clip(dist1 / feather_width, 0, 1)
        
        # Create weight mask
        weight1 = dist1.copy()
        weight2 = 1.0 - weight1
        
        # Apply weights only in overlap region
        weight1 = weight1 * overlap
        weight2 = weight2 * overlap
        
        # Normalize weights in overlap
        total_weight = weight1 + weight2
        total_weight[total_weight == 0] = 1
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Create panorama
        panorama = img2_warped.copy()
        
        # Place img1 in non-overlap regions
        non_overlap_img1 = (mask1 > 0) & ~overlap
        img1_canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        img1_canvas[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1
        panorama[non_overlap_img1] = img1_canvas[non_overlap_img1]
        
        # Blend in overlap region
        for c in range(3):
            panorama[overlap, c] = (
                img1_canvas[overlap, c] * weight1[overlap] +
                img2_warped[overlap, c] * weight2[overlap]
            ).astype(np.uint8)
        
        print(f"  Feathered blend: {overlap.sum()} overlap pixels")
    else:
        # No overlap - simple placement
        panorama = img2_warped.copy()
        panorama[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1
        print("  No overlap - direct placement")
    
    return panorama


def create_panorama(images: List[np.ndarray], estimator: HomographyEstimator,
                   matcher_params: Dict) -> np.ndarray:
    """
    Create panorama from multiple images.
    
    Args:
        images: List of input images
        estimator: HomographyEstimator instance
        matcher_params: Parameters for feature matching
    
    Returns:
        panorama: Final stitched panorama
    """
    if len(images) == 0:
        raise ValueError("No images provided")
    
    if len(images) == 1:
        return images[0]
    
    # Estimate stitching order
    order = estimate_panorama_order(images, matcher_params)
    ordered_images = [images[i] for i in order]
    
    # Start with first image
    panorama = ordered_images[0].copy()
    
    # Stitch remaining images
    for i in range(1, len(ordered_images)):
        print(f"\nStitching image {i + 1}/{len(ordered_images)}...")
        
        img_next = ordered_images[i]
        
        # Detect corners
        corners_pano = detect_shi_tomasi_corners(panorama, max_corners=matcher_params['max_corners'])
        corners_next = detect_shi_tomasi_corners(img_next, max_corners=matcher_params['max_corners'])
        
        # Compute descriptors (custom patch-based)
        kp_pano, desc_pano = compute_descriptors(panorama, corners_pano)
        kp_next, desc_next = compute_descriptors(img_next, corners_next)
        
        # Match features (custom SSD-based matching)
        matches = match_features(desc_pano, desc_next, ratio_thresh=matcher_params['ratio_thresh'])
        
        if len(matches) < 4:
            print(f"Warning: Only {len(matches)} matches found, skipping image {i}")
            continue
        
        # Get matched points
        pts_pano, pts_next = get_matched_points(kp_pano, kp_next, matches)
        
        # Estimate homography using deep learning
        H = estimator.estimate_homography(panorama, img_next, pts_pano, pts_next)
        
        # Stitch with feathered blending
        panorama = stitch_image_pair(panorama, img_next, H)
        
        print(f"Panorama size: {panorama.shape}")
    
    return panorama


# ========================
# MAIN
# ========================

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='MyAutoPano: Phase 2 - Deep Learning Panorama Stitching')
    parser.add_argument('--ImagePath', type=str, required=True,
                       help='Path to images folder or pattern (e.g., "images/*.jpg")')
    parser.add_argument('--ModelPath', type=str, required=True,
                       help='Path to best.pt checkpoint')
    parser.add_argument('--NumFeatures', type=int, default=1000,
                       help='Maximum number of corners to detect (default: 1000)')
    parser.add_argument('--RatioThresh', type=float, default=0.75,
                       help='Lowe\'s ratio test threshold (default: 0.75)')
    parser.add_argument('--OutputPath', type=str, default='mypano.png',
                       help='Output panorama path (default: mypano.png)')
    parser.add_argument('--Device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device for model inference (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MyAutoPano: Phase 2 - Deep Learning Panorama Stitching")
    print("=" * 80)
    
    # Read images
    print("\n[1/6] Reading images...")
    images = read_images(args.ImagePath)
    print(f"Loaded {len(images)} images")
    
    # Initialize homography estimator
    print("\n[2/6] Loading deep learning model...")
    estimator = HomographyEstimator(args.ModelPath, device=args.Device)
    
    # Matcher parameters
    matcher_params = {
        'max_corners': args.NumFeatures,
        'ratio_thresh': args.RatioThresh,
    }
    
    print(f"\n[3/6] Feature detection parameters:")
    print(f"  - Max corners: {matcher_params['max_corners']}")
    print(f"  - Descriptor: Custom patch-based (40x40 -> 8x8)")
    print(f"  - Ratio threshold: {matcher_params['ratio_thresh']}")
    print(f"  - Matching: Custom SSD with Lowe's ratio test")
    
    # Create panorama
    print("\n[4/6] Creating panorama...")
    panorama = create_panorama(images, estimator, matcher_params)
    
    # Save result
    print("\n[5/6] Saving panorama...")
    cv2.imwrite(args.OutputPath, panorama)
    print(f"Saved: {args.OutputPath}")
    print(f"Panorama size: {panorama.shape}")
    
    print("\n[6/6] Creating preview...")
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title('Panorama Result (Deep Learning + Custom Matching)')
    plt.axis('off')
    plt.tight_layout()
    preview_path = args.OutputPath.replace('.png', '_preview.png')
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f"Saved preview: {preview_path}")
    
    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()