#!/usr/bin/env python3
"""
Debug script to check if background masking is working properly
"""

import cv2
import numpy as np
import os
from PIL import Image

def debug_masking():
    # Test with one sample image
    depth_file = "/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/depth_ALL/_1_小七就是小七呀_来自小红书网页版.jpg"
    rgb_file = "/home/cevin/Meitu/data/test_data_img/all/_1_小七就是小七呀_来自小红书网页版.jpg"

    print(f"Depth file exists: {os.path.exists(depth_file)}")
    print(f"RGB file exists: {os.path.exists(rgb_file)}")

    if not os.path.exists(depth_file) or not os.path.exists(rgb_file):
        print("Files don't exist, trying another sample...")
        depth_file = "/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/depth_ALL/Hot girl summer 🧡_1_Raychen蕾蕾_来自小红书网页版.jpg"
        rgb_file = "/home/cevin/Meitu/data/test_data_img/all/Hot girl summer 🧡_1_Raychen蕾蕾_来自小红书网页版.jpg"

    print(f"Depth file exists: {os.path.exists(depth_file)}")
    print(f"RGB file exists: {os.path.exists(rgb_file)}")

    if not os.path.exists(depth_file) or not os.path.exists(rgb_file):
        print("Cannot find matching files!")
        return

    # Load images
    depth_image = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.imread(rgb_file)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    print(f"Depth image shape: {depth_image.shape}")
    print(f"RGB image shape: {rgb_image.shape}")

    # Check depth values
    print(f"Depth min: {depth_image.min()}, max: {depth_image.max()}")
    print(f"Depth unique values: {np.unique(depth_image)[:10]}...")  # Show first 10 unique values

    # Check how many pixels are exactly 0 in depth
    zero_depth_pixels = np.sum(depth_image == 0)
    total_pixels = depth_image.shape[0] * depth_image.shape[1]
    print(f"Zero depth pixels: {zero_depth_pixels}/{total_pixels} ({100*zero_depth_pixels/total_pixels:.1f}%)")

    # Check RGB background pixels
    rgb_mask = np.any(rgb_image > 0, axis=2)
    rgb_background_pixels = np.sum(~rgb_mask)
    print(f"RGB background pixels (all channels = 0): {rgb_background_pixels}/{total_pixels} ({100*rgb_background_pixels/total_pixels:.1f}%)")

    # Check if depth image is actually a visualization (colorized depth)
    if len(depth_image.shape) == 2:
        print("Depth image is grayscale - this might be the issue!")
        print("The depth images from depth_ALL might be visualizations, not raw depth values.")

        # Save sample patches for inspection
        h, w = depth_image.shape
        sample_depth = depth_image[h//4:3*h//4, w//4:3*w//4]
        sample_rgb = rgb_image[h//4:3*h//4, w//4:3*w//4]

        Image.fromarray(sample_depth, mode='L').save('debug_depth_sample.png')
        Image.fromarray(sample_rgb).save('debug_rgb_sample.png')
        print("Saved debug_depth_sample.png and debug_rgb_sample.png")

        # Check if depth image has background areas that are black
        depth_background = np.sum(sample_depth == 0)
        sample_pixels = sample_depth.shape[0] * sample_depth.shape[1]
        print(f"Sample depth background: {depth_background}/{sample_pixels} ({100*depth_background/sample_pixels:.1f}%)")

if __name__ == "__main__":
    debug_masking()


