#!/bin/bash

# Activate the vipicksModels conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vipicksModels

# Process the video file
# CUDA_VISIBLE_DEVICES=1 python example.py --data_path examples/skating.mp4 --save_path examples/skating.ply

# Process individual images from test data
input_dir="/home/cevin/Meitu/data/test_data_img/all"
mask_dir="/home/cevin/Meitu/data/test_data_img/matted_pha"
output_dir="/home/cevin/Meitu/Pi3/output/test_data_img"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Process each image file
for img_file in "$input_dir"/*.jpg; do
    if [ -f "$img_file" ]; then
        # Extract filename without extension
        filename=$(basename "$img_file" .jpg)

        # Look for corresponding mask file in matted_pha directory
        mask_img_file="$mask_dir/${filename}.png"

        # Create safe filename by replacing problematic characters
        safe_filename=$(echo "$filename" | sed 's/[^a-zA-Z0-9._-]/_/g')

        # Set output paths
        ply_output="$output_dir/${safe_filename}.ply"

        echo "Processing: $img_file -> $safe_filename"

        # Check if mask file exists and add mask parameter if it does
        if [ -f "$mask_img_file" ]; then
            echo "Found mask: $mask_img_file"
            mask_param="--mask_path $mask_img_file"
        else
            echo "No mask found for: $filename"
            mask_param=""
        fi

        # Run Pi3 on the individual image
        if CUDA_VISIBLE_DEVICES=1 python example.py --data_path "$img_file" $mask_param --save_path "$ply_output" --individual_image; then
            echo "Successfully processed: $safe_filename"
        else
            echo "Failed to process: $safe_filename (possibly corrupted image)"
        fi
    fi
done

echo "Finished processing all images."
