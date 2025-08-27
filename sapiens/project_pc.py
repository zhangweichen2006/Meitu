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
from PIL import Image
import hashlib
from typing import Optional

def estimate_camera_intrinsics(width, height, fov_vertical_degrees: float = 65.0, fov_horizontal_degrees: float = 0.0):
    """
    Estimate camera intrinsic matrix K from image size and provided FOVs.

    - If both horizontal and vertical FOVs are provided (> 0), compute both fx and fy.
    - If only one FOV is provided (> 0), assume square pixels and set the other focal equal.
    - If neither is provided (<= 0), fall back to 65 degrees and square pixels.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_vertical_degrees: Assumed vertical field-of-view in degrees (0 to ignore).
        fov_horizontal_degrees: Assumed horizontal field-of-view in degrees (0 to ignore).

    Returns:
        3x3 intrinsic matrix K.
    """
    fx = None
    fy = None

    # Compute from provided horizontal FOV
    if float(fov_horizontal_degrees) > 0.0:
        fov_horizontal = np.radians(float(fov_horizontal_degrees))
        fx = width / (2.0 * np.tan(fov_horizontal * 0.5))

    # Compute from provided vertical FOV
    if float(fov_vertical_degrees) > 0.0:
        fov_vertical = np.radians(float(fov_vertical_degrees))
        fy = height / (2.0 * np.tan(fov_vertical * 0.5))

    # Assume square pixels to fill missing focal(s)
    if fx is None and fy is not None:
        fx = fy
    if fy is None and fx is not None:
        fy = fx

    # Fallback if neither FOV provided
    if fx is None and fy is None:
        fallback_fov = np.radians(65.0)
        fx = width / (2.0 * np.tan(fallback_fov * 0.5))
        fy = fx

    # Principal point at image center (cx along width, cy along height)
    cx = width * 0.5
    cy = height * 0.5

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K

def save_visualize_depth(depth_image: np.ndarray, save_path: Optional[str] = "depth.png") -> Optional[str]:
    """
    Create a colored visualization of a depth map and save it to disk.

    - Ignores NaN/Inf values when computing min/max
    - Normalizes valid depths to 0..255
    - Uses OpenCV's INFERNO colormap for better contrast

    Args:
        depth_image: numpy array of shape (H, W), raw depth values (float or int)
        save_path: optional explicit path to save PNG. If None, saves under
                   ./output/depth_vis_debug/depth_vis_<hash>.png

    Returns:
        The file path where the visualization was saved, or None if saving failed.
    """
    try:
        if depth_image is None or depth_image.size == 0:
            return None

        # Prepare output directory if not provided
        if save_path is None:
            out_dir = os.path.join(os.getcwd(), 'output', 'depth_vis_debug')
            os.makedirs(out_dir, exist_ok=True)
            # Hash on shape and a few stats to create a stable-ish name per map
            stats = np.array([
                depth_image.shape[0], depth_image.shape[1],
                float(np.nanmin(depth_image)) if np.isfinite(depth_image).any() else 0.0,
                float(np.nanmax(depth_image)) if np.isfinite(depth_image).any() else 0.0,
            ], dtype=np.float32).tobytes()
            short_hash = hashlib.md5(stats).hexdigest()[:8]
            save_path = os.path.join(out_dir, f'depth_vis_{short_hash}.png')

        # Mask invalids
        valid_mask = np.isfinite(depth_image)
        if not valid_mask.any():
            # If all invalid, create a black image and save
            vis = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
            cv2.imwrite(save_path, vis)
            return save_path

        valid_depth = depth_image[valid_mask]
        dmin = float(valid_depth.min())
        dmax = float(valid_depth.max())
        if dmax <= dmin:
            dmax = dmin + 1e-6

        # Normalize to 0..255 (near -> bright or invert depending preference)
        norm = (depth_image - dmin) / (dmax - dmin)
        norm[~valid_mask] = 0.0
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)

        # Apply colormap
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # Write file
        cv2.imwrite(save_path, vis)
        return save_path
    except Exception:
        return None

def save_pc(pcd):
    o3d.io.write_point_cloud("output/pc.ply", pcd)

def depth_to_point_cloud(depth_image,
                        rgb_image,
                        camera_intrinsics,
                        depth_scale: float = 1.0,
                        scale_s: float = 1.0,
                        shift_t: float = 0.0):
    """
    Convert depth image to colored point cloud

    Args:
        depth_image: Depth map (H x W)
        rgb_image: RGB image (H x W x 3)
        camera_intrinsics: 3x3 camera intrinsic matrix
        depth_scale: Unit conversion for depth values. If your abs depth is in
            meters already, use 1.0. If your abs depth is in millimeters, use 1000.0.
        scale_s: Scale to convert relative depth to absolute: abs = rel * s + t
        shift_t: Shift to convert relative depth to absolute: abs = rel * s + t

    Returns:
        Open3D point cloud object
    """
    height, width = depth_image.shape

    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Convert relative depth to absolute depth via scale and shift, then to meters
    rel_depth = depth_image.astype(np.float32)
    abs_depth = rel_depth * float(scale_s) + float(shift_t)
    depth_m = abs_depth / float(depth_scale)

    # Remove invalid depth values and background pixels
    # 1. Depth mask: remove where depth is invalid (NaN, inf, or unrealistic values)
    depth_mask = (~np.isnan(depth_m)) & (~np.isinf(depth_m))

    # 2. RGB mask: remove where RGB is black/background (all channels are 0)
    rgb_mask = np.any(rgb_image > 0, axis=2)  # True where any RGB channel > 0

    # Combine masks: valid depth AND non-black RGB
    valid_mask = depth_mask & rgb_mask

    # save_visualize_depth(valid_mask)

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

    save_pc(pcd)

    return pcd

def process_depth_to_pointcloud(depth_dir,
                               rgb_dir,
                               output_dir,
                               depth_scale: float = 1.0,
                               render_png: bool = True,
                               scale_s: float = 1.0,
                               shift_t: float = 0.0,
                               fov_horizontal_degrees: float = 65.0,
                               fov_vertical_degrees: float = 0.0):
    """
    Process all depth images in directory to point clouds

    Args:
        depth_dir: Directory containing depth images (will look for .npy files in parent dir)
        rgb_dir: Directory containing RGB images
        output_dir: Output directory for point clouds
        depth_scale: Scale factor for depth values (1.0 for raw depth data)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Look for .npy files in the parent directory (where raw depth data is stored)
    parent_dir = os.path.dirname(depth_dir)
    depth_files = glob.glob(os.path.join(parent_dir, "*.npy"))
    depth_files = sorted(depth_files)

    print(f"Found {len(depth_files)} depth images")

    for depth_file in tqdm(depth_files, desc="Converting to point clouds"):
        try:
            # Get corresponding RGB file
            base_name = os.path.basename(depth_file)
            rgb_file = os.path.join(rgb_dir, base_name).replace(".npy", ".jpg")

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

            # Load raw depth data from .npy file
            depth_image = np.load(depth_file)
            if depth_image is None or depth_image.size == 0:
                print(f"Warning: Could not load depth data {depth_file}")
                continue

            save_visualize_depth(depth_image)

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
            camera_intrinsics = estimate_camera_intrinsics(
                width,
                height,
                fov_vertical_degrees=fov_vertical_degrees,
                fov_horizontal_degrees=fov_horizontal_degrees,
            )

            # Convert to point cloud
            pcd = depth_to_point_cloud(depth_image,
                                       rgb_image,
                                       camera_intrinsics,
                                       depth_scale=depth_scale,
                                       scale_s=scale_s,
                                       shift_t=shift_t)

            # Save point cloud
            output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".ply")
            o3d.io.write_point_cloud(output_file, pcd)

            print(f"Saved point cloud: {output_file} ({len(pcd.points)} points)")

            # Render point cloud to PNG if requested
            if render_png:
                png_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + "_render.png")
                render_point_cloud_png(output_file, png_file, camera_intrinsics, width, height)

        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
            continue


def render_point_cloud_png(ply_file, output_png, camera_intrinsics, original_width, original_height):
    """
    Render a point cloud to PNG by projecting 3D points back to 2D image space
    """
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)  # 3D points in camera coordinates
        colors = np.asarray(pcd.colors)  # RGB colors [0, 1]

        print(f"Rendering {len(points)} points to PNG by projection...")

        if len(points) == 0 or len(colors) == 0:
            print("No points or colors to render!")
            return False

        H, W = original_height, original_width

        # Extract camera parameters
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        # Filter points that are in front of camera (z > 0)
        z = points[:, 2]
        front_mask = z > 1e-6

        if not front_mask.any():
            print("No points in front of camera!")
            return False

        pts_cam = points[front_mask]
        cols = colors[front_mask]
        z = z[front_mask]

        # Project 3D points to 2D image coordinates
        u = fx * (pts_cam[:, 0] / z) + cx
        v = fy * (pts_cam[:, 1] / z) + cy

        # Round to integer pixel coordinates
        x = np.round(u).astype(np.int64)
        y = np.round(v).astype(np.int64)

        # Filter points that fall within image bounds
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        if not in_bounds.any():
            print("No projected points inside image bounds!")
            return False

        x = x[in_bounds]
        y = y[in_bounds]
        z = z[in_bounds]
        cols = cols[in_bounds]

        # Convert to linear indices for depth sorting
        lin_idx = (y * W + x).astype(np.int64)

        # Sort by depth (z-buffer)
        order = np.lexsort((z, lin_idx))  # Sort by lin_idx first, then by z
        lin_sorted = lin_idx[order]

        # Keep only the closest point for each pixel (z-buffer)
        unique_idx, first_pos = np.unique(lin_sorted, return_index=True)
        sel = order[first_pos]

        # Create output image
        img_out = np.zeros((H, W, 3), dtype=np.uint8)

        # Get final pixel coordinates and colors
        final_x = lin_idx[sel] % W
        final_y = lin_idx[sel] // W
        final_cols = (cols[sel] * 255.0).clip(0, 255).astype(np.uint8)

        # Set pixel values
        img_out[final_y, final_x] = final_cols

        # Save the rendered image
        Image.fromarray(img_out).save(output_png)
        print(f"Saved rendered PNG: {output_png}")
        return True

    except Exception as e:
        print(f"Error rendering point cloud to PNG: {e}")
        return False

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
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="Unit conversion for absolute depth. If abs depth is in meters use 1.0; if in mm use 1000.0")
    parser.add_argument("--scale_s", type=float, default=1.0,
                       help="Scale s for converting relative depth to absolute: abs = rel * s + t")
    parser.add_argument("--shift_t", type=float, default=0.0,
                       help="Shift t for converting relative depth to absolute: abs = rel * s + t")
    parser.add_argument("--fov_horizontal_degrees", type=float, default=65.0,
                       help="Assumed horizontal FOV (degrees) to estimate intrinsics if unknown")
    parser.add_argument("--fov_vertical_degrees", type=float, default=0.0,
                       help="Assumed vertical FOV (degrees); if 0, ignored. If both provided, use both.")
    parser.add_argument("--render_png", action="store_true",
                       help="Render PNG images of point clouds")
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

        process_depth_to_pointcloud(
            args.depth_dir,
            args.rgb_dir,
            args.output_dir,
            depth_scale=args.depth_scale,
            render_png=args.render_png,
            scale_s=args.scale_s,
            shift_t=args.shift_t,
            fov_horizontal_degrees=args.fov_horizontal_degrees,
            fov_vertical_degrees=args.fov_vertical_degrees,
        )
        print("Point cloud generation completed!")

if __name__ == "__main__":
    main()
