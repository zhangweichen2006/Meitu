#!/bin/bash

# Complete environment setup process

# Step 0: Ensure you create a Conda environment
# and Activate the environment
# conda activate idol
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate idol

# Step 1: Install Pytorch with CUDA (unversioned)
pip install torch torchvision torchaudio

# Step 2: Use pip to install additional dependencies (unversioned)
pip_packages=(
    "absl-py"
    "accelerate"
    "addict"
    "albumentations"
    "bitsandbytes"
    "deepspeed"
    "diffusers"
    "einops"
    "fastapi"
    "gradio"
    "matplotlib"
    "numpy"
    "opencv-python"
    "pandas"
    "pillow"
    "scikit-image"
    "scipy"
    "timm"
    "transformers"
    "pytorch-lightning"
    "omegaconf"
    "av"
    "webdataset"
    "omegaconf"
    "rembg"
    "onnxruntime"
    "tensorboard"
)

# Install pip packages in bulk
for package in "${pip_packages[@]}"
do
    pip install "$package"
done


# Create submodule directory if it doesn't exist
mkdir -p submodule
cd submodule

# Step 3: Install PyTorch3D
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

# Step 4: Install Simple-KNN
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
cd simple-knn
pip install -e .
cd ..

# Step 5: Install Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/submodules/diff-gaussian-rasterization

# from repo root of diff-gaussian-rasterization (CUDA12.8)
FILE=cuda_rasterizer/rasterizer_impl.h
grep -q '<cstdint>' "$FILE" || sudo sed -i '1i #include <cstdint>\n#include <cstddef>' "$FILE"

python setup.py develop
cd ../../..

# Step 6: Install Sapiens
git clone https://github.com/facebookresearch/sapiens
cd sapiens/engine
pip install -e .
cd ../pretrain
pip install -e .
cd ../../..

# Step 7: Install deformation module
python setup.py develop

echo "idol environment setup completed!"

pip install onnxruntime-gpu