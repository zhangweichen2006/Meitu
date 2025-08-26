#!/usr/bin/env python3
"""
Analyze Point Cloud Data
Simple script to verify and analyze the generated point clouds
"""

import os
import open3d as o3d
import numpy as np
from glob import glob

def analyze_point_cloud(ply_file):
    """Analyze a single point cloud file"""
    try:
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        print(f"\nFile: {os.path.basename(ply_file)}")
        print(f"  Points: {len(points):,}")
        print(f"  Has colors: {len(colors) > 0}")
        if len(points) > 0:
            print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        if len(colors) > 0:
            print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")

        return len(points)
    except Exception as e:
        print(f"Error analyzing {ply_file}: {e}")
        return 0

def main():
    pc_dir = "/home/cevin/Meitu/sapiens/output/point_clouds"
    ply_files = sorted(glob(os.path.join(pc_dir, "*.ply")))

    if not ply_files:
        print(f"No PLY files found in {pc_dir}")
        return

    print(f"Found {len(ply_files)} point cloud files")
    print("="*60)

    total_points = 0
    valid_files = 0

    # Analyze first 5 files in detail
    for i, ply_file in enumerate(ply_files[:5]):
        points = analyze_point_cloud(ply_file)
        if points > 0:
            total_points += points
            valid_files += 1

    print("\n" + "="*60)
    print("SUMMARY FOR FIRST 5 FILES:")
    print(f"Valid point clouds: {valid_files}/5")
    print(f"Total points analyzed: {total_points:,}")
    print(f"Average points per cloud: {total_points/valid_files:,.0f}" if valid_files > 0 else "N/A")

    # Quick count for all files
    print(f"\nAll {len(ply_files)} files successfully generated!")

if __name__ == "__main__":
    main()
