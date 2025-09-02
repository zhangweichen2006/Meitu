#!/usr/bin/env python3
"""
Test single sample point cloud generation with proper NPY usage
"""

import os
import cv2
import numpy as np
import open3d as o3d
from PIL import Image

# Import our functions from the main script
from project_pc import depth_to_point_cloud, estimate_camera_intrinsics, render_point_cloud_png

def test_single_sample():
    # Test files
    npy_file = "/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/_1_小七就是小七呀_来自小红书网页版.npy"
    rgb_file = "/home/cevin/Meitu/data/test_data_img/all/_1_小七就是小七呀_来自小红书网页版.jpg"
    output_dir = "/home/cevin/Meitu/sapiens/output/test_single_sample"

    os.makedirs(output_dir, exist_ok=True)

    print("=== Testing Single Sample Point Cloud Generation ===")
    print(f"NPY file: {npy_file}")
    print(f"RGB file: {rgb_file}")
    print(f"Output: {output_dir}")
    print()

    # Load raw depth data
    print("Loading raw depth data...")
    depth_data = np.load(npy_file)
    print(f"Depth shape: {depth_data.shape}")
    print(f"Depth range: {depth_data.min():.6f} to {depth_data.max():.6f}")
    print(f"Depth dtype: {depth_data.dtype}")

    # Load RGB image
    print("Loading RGB image...")
    rgb_image = cv2.imread(rgb_file)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"RGB shape: {rgb_image.shape}")

    # Resize RGB to match depth if needed
    if rgb_image.shape[:2] != depth_data.shape:
        rgb_image = cv2.resize(rgb_image, (depth_data.shape[1], depth_data.shape[0]))
        print(f"Resized RGB to: {rgb_image.shape}")

    # Estimate camera intrinsics
    height, width = depth_data.shape
    camera_intrinsics = estimate_camera_intrinsics(width, height)
    print(f"Camera intrinsics estimated for {width}x{height}")

    # Convert to point cloud
    print("Converting to point cloud...")
    pcd = depth_to_point_cloud(depth_data, rgb_image, camera_intrinsics, depth_scale=1.0)

    print(f"Point cloud generated with {len(pcd.points)} points")

    # Save point cloud
    ply_file = os.path.join(output_dir, "sample_point_cloud.ply")
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Saved PLY: {ply_file}")

    # Render PNG
    png_file = os.path.join(output_dir, "sample_render.png")
    success = render_point_cloud_png(ply_file, png_file, camera_intrinsics, width, height)

    if success:
        print(f"Saved PNG render: {png_file}")
    else:
        print("PNG rendering failed")

    # Show masking statistics
    rgb_mask = np.any(rgb_image > 0, axis=2)
    depth_mask = (depth_data > 1e-6) & (depth_data < 10.0) & (~np.isnan(depth_data)) & (~np.isinf(depth_data))
    combined_mask = depth_mask & rgb_mask

    total_pixels = depth_data.size
    valid_pixels = np.sum(combined_mask)
    rgb_background = np.sum(~rgb_mask)
    invalid_depth = np.sum(~depth_mask)

    print()
    print("=== Masking Statistics ===")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels after masking: {valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
    print(f"RGB background pixels: {rgb_background:,} ({100*rgb_background/total_pixels:.1f}%)")
    print(f"Invalid depth pixels: {invalid_depth:,} ({100*invalid_depth/total_pixels:.1f}%)")
    print(f"Pixels removed by masking: {total_pixels - valid_pixels:,} ({100*(total_pixels - valid_pixels)/total_pixels:.1f}%)")

    print("\n✅ Single sample test completed successfully!")

if __name__ == "__main__":
    test_single_sample()


