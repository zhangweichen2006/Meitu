#!/bin/bash

# Ensure UTF-8 so non-ASCII filenames (e.g., Chinese) are preserved
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Activate the vipicksModels conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vipicksModels

# Process the video file
# CUDA_VISIBLE_DEVICES=1 python example.py --data_path examples/skating.mp4 --save_path examples/skating.ply


# Process individual images from test data
input_dir="/home/cevin/Meitu/data/iphone"
mask_dir="/home/cevin/Meitu/data/iphone_pha"
output_dir="/home/cevin/Meitu/Pi3/output/iphone_out"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Enable case-insensitive globbing and avoid literal patterns when no matches
shopt -s nullglob nocaseglob

# Process each image file (HEIC/HEIF, any case)
for img_file in "$input_dir"/*.heic "$input_dir"/*.heif "$input_dir"/*.png; do
    if [ -f "$img_file" ]; then
        # Extract filename without extension (robust to extension case/type)
        filename=$(basename "$img_file")
        filename="${filename%.*}"

        # Look for corresponding mask file in matted_pha directory
        mask_img_file="$mask_dir/${filename}.png"

        # Set output paths using original filename (preserve non-ASCII)
        ply_output="$output_dir/${filename}.ply"

        echo "Processing: $img_file -> $filename"

        # Run Pi3 on the individual image (quote mask path if present)
        if [ -f "$mask_img_file" ]; then
            echo "Found mask: $mask_img_file"
            if CUDA_VISIBLE_DEVICES=1 python example.py --data_path "$img_file" --mask_path "$mask_img_file" --save_path "$ply_output" --individual_image; then
                echo "Successfully processed: $filename"
            else
                echo "Failed to process: $filename (possibly corrupted image)"
            fi
        else
            echo "No mask found for: $filename"
            if CUDA_VISIBLE_DEVICES=1 python example.py --data_path "$img_file" --save_path "$ply_output" --individual_image; then
                echo "Successfully processed: $filename"
            else
                echo "Failed to process: $filename (possibly corrupted image)"
            fi
        fi
    fi
done

echo "Finished processing all images."
