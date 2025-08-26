#!/usr/bin/env python3
"""
Point Cloud Generation from Depth Maps and RGB Images
Converts depth images to colored point clouds with estimated camera parameters
"""

import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import glob

def estimate_camera_intrinsics(width, height):
    """
    Estimate camera intrinsic parameters based on image dimensions
    Uses typical smartphone/camera FOV assumptions
    """
    # Typical horizontal FOV for smartphones: 60-70 degrees
    # Using 65 degrees as default
    fov_horizontal = np.radians(65)

    # Calculate focal length from FOV
    fx = width / (2 * np.tan(fov_horizontal / 2))
    fy = fx  # Assume square pixels

    # Principal point at image center
    cx = width / 2
    cy = height / 2

    # Camera intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K

def depth_to_point_cloud(depth_image, rgb_image, camera_intrinsics, depth_scale=1000.0):
    """
    Convert depth image to colored point cloud

    Args:
        depth_image: Depth map (H x W)
        rgb_image: RGB image (H x W x 3)
        camera_intrinsics: 3x3 camera intrinsic matrix
        depth_scale: Scale factor for depth values

    Returns:
        Open3D point cloud object
    """
    height, width = depth_image.shape

    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Convert depth to meters (assuming input is in mm or similar)
    depth_m = depth_image.astype(np.float32) / depth_scale

    # Remove invalid depth values
    valid_mask = (depth_m > 0) & (depth_m < 10.0)  # Keep depths between 0 and 10 meters

    # Extract valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_m[valid_mask]

    # Get camera parameters
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Convert to 3D coordinates
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    # Stack coordinates
    points_3d = np.stack([x, y, z], axis=1)

    # Get corresponding colors
    colors = rgb_image[valid_mask] / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def process_depth_to_pointcloud(depth_dir, rgb_dir, output_dir, depth_scale=1000.0):
    """
    Process all depth images in directory to point clouds

    Args:
        depth_dir: Directory containing depth images
        rgb_dir: Directory containing RGB images
        output_dir: Output directory for point clouds
        depth_scale: Scale factor for depth values
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all depth images
    depth_files = glob.glob(os.path.join(depth_dir, "*.jpg"))
    depth_files.extend(glob.glob(os.path.join(depth_dir, "*.png")))
    depth_files = sorted(depth_files)

    print(f"Found {len(depth_files)} depth images")

    for depth_file in tqdm(depth_files, desc="Converting to point clouds"):
        try:
            # Get corresponding RGB file
            base_name = os.path.basename(depth_file)
            rgb_file = os.path.join(rgb_dir, base_name)

            # Try different extensions if exact match not found
            if not os.path.exists(rgb_file):
                name_without_ext = os.path.splitext(base_name)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    rgb_candidate = os.path.join(rgb_dir, name_without_ext + ext)
                    if os.path.exists(rgb_candidate):
                        rgb_file = rgb_candidate
                        break

            if not os.path.exists(rgb_file):
                print(f"Warning: RGB file not found for {base_name}, skipping")
                continue

            # Load depth image (grayscale)
            depth_image = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
            if depth_image is None:
                print(f"Warning: Could not load depth image {depth_file}")
                continue

            # Load RGB image
            rgb_image = cv2.imread(rgb_file)
            if rgb_image is None:
                print(f"Warning: Could not load RGB image {rgb_file}")
                continue

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Resize RGB to match depth if needed
            if rgb_image.shape[:2] != depth_image.shape:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

            # Estimate camera intrinsics
            height, width = depth_image.shape
            camera_intrinsics = estimate_camera_intrinsics(width, height)

            # Convert to point cloud
            pcd = depth_to_point_cloud(depth_image, rgb_image, camera_intrinsics, depth_scale)

            # Save point cloud
            output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".ply")
            o3d.io.write_point_cloud(output_file, pcd)

            print(f"Saved point cloud: {output_file} ({len(pcd.points)} points)")

        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
            continue

def visualize_point_cloud(ply_file):
    """
    Visualize a point cloud file
    """
    pcd = o3d.io.read_point_cloud(ply_file)
    print(f"Loaded point cloud with {len(pcd.points)} points")

    # Estimate normals for better visualization
    pcd.estimate_normals()

    # Visualize
    o3d.visualization.draw_geometries([pcd],
                                    window_name="Point Cloud Viewer",
                                    width=1024,
                                    height=768)

def main():
    parser = argparse.ArgumentParser(description="Convert depth images to colored point clouds")
    parser.add_argument("--depth_dir",
                       default="/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/depth_ALL",
                       help="Directory containing depth images")
    parser.add_argument("--rgb_dir",
                       default="/home/cevin/Meitu/data/test_data_img/all",
                       help="Directory containing RGB images")
    parser.add_argument("--output_dir",
                       default="/home/cevin/Meitu/sapiens/output/point_clouds",
                       help="Output directory for point clouds")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                       help="Depth scale factor (default: 1000.0)")
    parser.add_argument("--visualize", type=str, default=None,
                       help="Visualize a specific PLY file")

    args = parser.parse_args()

    if args.visualize:
        visualize_point_cloud(args.visualize)
    else:
        print(f"Processing depth images from: {args.depth_dir}")
        print(f"Using RGB images from: {args.rgb_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Depth scale: {args.depth_scale}")

        process_depth_to_pointcloud(args.depth_dir, args.rgb_dir, args.output_dir, args.depth_scale)
        print("Point cloud generation completed!")

if __name__ == "__main__":
    main()
