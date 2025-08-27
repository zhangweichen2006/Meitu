#!/usr/bin/env python3
"""
Debug script to check raw depth data from .npy files
"""

import numpy as np
import cv2
import os
from PIL import Image

def debug_raw_depth():
    # Check raw depth data
    npy_file = "/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/_1_小七就是小七呀_来自小红书网页版.npy"
    rgb_file = "/home/cevin/Meitu/data/test_data_img/all/_1_小七就是小七呀_来自小红书网页版.jpg"

    print(f"NPY file exists: {os.path.exists(npy_file)}")
    print(f"RGB file exists: {os.path.exists(rgb_file)}")

    if not os.path.exists(npy_file):
        # Try another file
        npy_file = "/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/Hot girl summer 🧡_1_Raychen蕾蕾_来自小红书网页版.npy"
        rgb_file = "/home/cevin/Meitu/data/test_data_img/all/Hot girl summer 🧡_1_Raychen蕾蕾_来自小红书网页版.jpg"

    print(f"NPY file exists: {os.path.exists(npy_file)}")
    print(f"RGB file exists: {os.path.exists(rgb_file)}")

    if not os.path.exists(npy_file) or not os.path.exists(rgb_file):
        print("Cannot find files!")
        return

    # Load raw depth data
    depth_raw = np.load(npy_file)
    print(f"Raw depth shape: {depth_raw.shape}")
    print(f"Raw depth min: {depth_raw.min():.6f}, max: {depth_raw.max():.6f}")
    print(f"Raw depth dtype: {depth_raw.dtype}")

    # Check for NaN or inf values
    nan_count = np.sum(np.isnan(depth_raw))
    inf_count = np.sum(np.isinf(depth_raw))
    zero_count = np.sum(depth_raw == 0)
    total_pixels = depth_raw.size

    print(f"NaN pixels: {nan_count}/{total_pixels} ({100*nan_count/total_pixels:.1f}%)")
    print(f"Inf pixels: {inf_count}/{total_pixels} ({100*inf_count/total_pixels:.1f}%)")
    print(f"Zero pixels: {zero_count}/{total_pixels} ({100*zero_count/total_pixels:.1f}%)")

    # Load RGB
    rgb_image = cv2.imread(rgb_file)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Resize RGB to match depth if needed
    if rgb_image.shape[:2] != depth_raw.shape:
        rgb_image = cv2.resize(rgb_image, (depth_raw.shape[1], depth_raw.shape[0]))
        print(f"Resized RGB to match depth: {rgb_image.shape}")

    # Check RGB background
    rgb_mask = np.any(rgb_image > 0, axis=2)
    rgb_background = np.sum(~rgb_mask)
    print(f"RGB background pixels: {rgb_background}/{total_pixels} ({100*rgb_background/total_pixels:.1f}%)")

    # Create proper masks
    depth_mask = (depth_raw > 1e-6) & (depth_raw < 10.0) & (~np.isnan(depth_raw)) & (~np.isinf(depth_raw))
    combined_mask = depth_mask & rgb_mask

    valid_pixels = np.sum(combined_mask)
    print(f"Valid pixels after masking: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")

    # Show what the masking removes
    invalid_depth = np.sum(~depth_mask)
    invalid_rgb = np.sum(~rgb_mask)
    print(f"Pixels removed by depth mask: {invalid_depth}")
    print(f"Pixels removed by RGB mask: {invalid_rgb}")

if __name__ == "__main__":
    debug_raw_depth()
