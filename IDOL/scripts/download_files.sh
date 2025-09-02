#!/bin/bash

# Create necessary directories
mkdir -p work_dirs/
mkdir -p work_dirs/ckpt

# Download files from HuggingFace
echo "Downloading model files..."
wget https://huggingface.co/yiyuzhuang/IDOL/resolve/main/model.ckpt -O work_dirs/ckpt/model.ckpt
wget https://huggingface.co/yiyuzhuang/IDOL/resolve/main/sapiens_1b_epoch_173_torchscript.pt2 -O work_dirs/ckpt/sapiens_1b_epoch_173_torchscript.pt2
wget https://huggingface.co/yiyuzhuang/IDOL/resolve/main/cache_sub2.zip -O work_dirs/cache_sub2.zip

# Unzip cache file
echo "Extracting cache files..."
unzip -o work_dirs/cache_sub2.zip -d work_dirs/
rm work_dirs/cache_sub2.zip  # Remove zip file after extraction

echo "Download and extraction completed!"


