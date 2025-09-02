#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for DeepFashion video dataset.
This script processes video files and their corresponding parameters,
and splits the dataset into train/val/test sets.
"""

import os
import numpy as np
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare DeepFashion video dataset")
    parser.add_argument(
        "--video_dir", 
        type=str, 
        default="/apdcephfs/private_harriswen/data/deepfashion/",
        help="Base directory containing imageX folders"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./", 
        help="Directory to save the processed data"
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="DeepFashion", 
        help="Prefix for the output file names"
    )
    parser.add_argument(
        "--max_videos", 
        type=int, 
        default=20000, 
        help="Maximum number of videos to process (for creating smaller datasets)"
    )
    return parser.parse_args()


def prepare_dataset(video_dir, output_dir, prefix, max_total_videos=20000):
    """
    Prepare the DeepFashion dataset by processing videos and parameters.
    
    Args:
        video_dir: Base directory containing imageX folders
        output_dir: Directory to save processed data
        prefix: Prefix for output filenames
        max_total_videos: Maximum number of videos to process (default: 20000)
    """
    # Find all imageX subdirectories
    image_dirs = []
    for item in os.listdir(video_dir):
        if item.startswith("image") and os.path.isdir(os.path.join(video_dir, item)):
            image_dirs.append(item)
    
    image_dirs.sort()
    print(f"Found {len(image_dirs)} image directories: {image_dirs}")
    
    # Collect all video files
    all_video_files = []
    all_param_files = []
    all_dir_names = []
    # import ipdb; ipdb.set_trace()
    for image_dir in image_dirs:
        videos_path = os.path.join(video_dir, image_dir, "videos")
        params_path = os.path.join(video_dir, image_dir, "param")
        
        if not os.path.exists(videos_path):
            print(f"Warning: Videos directory not found in {image_dir}, skipping.")
            continue
        
        if not os.path.exists(params_path):
            print(f"Warning: Parameters directory not found in {image_dir}, skipping.")
            continue
        
        # Get list of video names in current directory
        param_names = os.listdir(params_path)

        # filter the files with .npy extension
        param_names = [name for name in param_names if name.endswith(".npy")]
        
        for name in param_names:
            video_path = os.path.join(videos_path, name.replace(".npy", ".mp4"))
            param_path = os.path.join(params_path, name)
            
            # Check if both video and parameter files exist
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}, skipping.")
                continue
                
            if not os.path.exists(param_path):
                print(f"Warning: Parameter file not found: {param_path}, skipping.")
                continue
                
            # Add to collection only if both files exist
            all_video_files.append(video_path)
            all_param_files.append(param_path)
            all_dir_names.append(image_dir)
    
    total_videos = len(all_video_files)
    print(f"Total valid videos found: {total_videos}")
    
    if total_videos == 0:
        print("Error: No valid video-parameter pairs found. Please check your data paths.")
        return
    
    # Limit number of videos to process
    if max_total_videos < total_videos:
        # Randomly shuffle and select first max_total_videos
        indices = list(range(total_videos))
        np.random.shuffle(indices)
        indices = indices[:max_total_videos]
        
        all_video_files = [all_video_files[i] for i in indices]
        all_param_files = [all_param_files[i] for i in indices]
        all_dir_names = [all_dir_names[i] for i in indices]
        
        print(f"Limiting to {max_total_videos} videos")
    
    # Process videos and parameters
    scenes = []
    processed_count = 0
    skipped_count = 0
    
    for video_path, param_path, dir_name in zip(all_video_files, all_param_files, all_dir_names):
        processed_count += 1
        case_name = os.path.basename(video_path)
        
        print(f"Processing {processed_count}/{len(all_video_files)}: {dir_name}/{case_name}")
        
        try:
            # Create scene dictionary
            scenes.append(dict(
                video_path=video_path,
                image_paths=None, # only fill it for the data in a images sequence instead of a video
                param_path=param_path,
                image_ref=video_path.replace(".mp4", ".jpg")
            ))
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            skipped_count += 1
    
    print(f"Total scenes collected: {len(scenes)}")
    print(f"Total scenes skipped: {skipped_count}")
    
    if len(scenes) == 0:
        print("Error: No scenes could be processed. Please check your data.")
        return
    
    # Split dataset
    total_scenes = len(scenes)
    test_scenes = scenes[-50:] if total_scenes > 50 else []
    val_scenes = scenes[-60:-50] if total_scenes > 60 else []
    train_scenes = scenes[:-60] if total_scenes > 60 else scenes
    
    # Save each split
    splits = {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
        "all": scenes
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to separate file
    for split_name, split_data in splits.items():
        if not split_data:
            continue
            
        cache_path = os.path.join(
            output_dir, 
            f"{prefix}_{split_name}_{len(split_data)}.npy"
        )
        np.save(cache_path, split_data)
        print(f"Saved {split_name} split with {len(split_data)} samples to {cache_path}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Prepare and save the dataset
    prepare_dataset(args.video_dir, args.output_dir, args.prefix, args.max_videos)
    print(f"Done processing {args.video_dir} dataset")
