#!/bin/bash

# Data processing script for multiple datasets
# This script processes all specified datasets and saves the results to output directories

# Define the list of dataset paths
DATASET_PATHS=(
    "/PATH/TO/deepfashion"
    "/PATH/TO/flux_batch1_5000"
    "/PATH/TO/flux_batch2"
    # Add more dataset paths here as needed
)

# Output base directory for processed cache files
OUTPUT_BASE_DIR="./processed_data"

# Maximum videos to process per dataset (set to a smaller number for testing)
# if you want to process all videos, set MAX_VIDEOS to a very large number
MAX_VIDEOS=200

# Process each dataset
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    # Extract dataset name from path (use the last directory name as prefix)
    DATASET_NAME=$(basename "$DATASET_PATH")
    
    # Create output directory for this dataset
    OUTPUT_DIR="${OUTPUT_BASE_DIR}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "===== Processing ${DATASET_NAME} Dataset ====="
    echo "Source: ${DATASET_PATH}"
    echo "Destination: ${OUTPUT_DIR}"
    
    # Run the processing script
    python data_processing/prepare_cache.py \
        --video_dir "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --prefix "${DATASET_NAME}" \
        --max_videos "${MAX_VIDEOS}"
    
    # Check if processing was successful
    if [ $? -ne 0 ]; then
        echo "Error processing ${DATASET_NAME} dataset"
        echo "Continuing with next dataset..."
    else
        echo "Successfully processed ${DATASET_NAME} dataset"
    fi
    
    echo "----------------------------------------"
done

echo "===== All datasets processing completed ====="
echo "Results saved to: ${OUTPUT_BASE_DIR}"

# List all processed datasets
echo "Processed datasets:"
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    DATASET_NAME=$(basename "$DATASET_PATH")
    echo "- ${DATASET_NAME}: ${OUTPUT_BASE_DIR}/${DATASET_NAME}"
done 