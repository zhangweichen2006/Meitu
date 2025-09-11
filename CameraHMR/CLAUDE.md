# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

CameraHMR is a human mesh reconstruction system that estimates 3D human body poses from images using SMPL models. It combines camera intrinsics estimation, human detection, mesh regression, and optional pose refinement through CamSMPLify optimization.

## Core Architecture

### Main Components
- **Detectron2**: Person detection in images/videos
- **FLNet**: Camera intrinsics estimation (field of view → focal length)
- **CameraHMR**: SMPL parameter and weak-perspective camera regression
- **DenseKP**: Dense 2D keypoint prediction for refinement
- **CamSMPLify**: Optimization-based pose/shape/camera refinement
- **SMPL/SMPLX**: 3D human body models via `smplx` library
- **PyRender**: Mesh rendering and visualization

### Key Files
- `demo.py`: Main entry point for inference
- `mesh_estimator.py`: Core HumanMeshEstimator class orchestrating the pipeline
- `train_keypoints.py`: Training script for dense keypoint model
- `eval.py`: Evaluation on test datasets
- `CamSMPLify/optimize.py`: Offline optimization workflow
- `CamSMPLify/cam_smplify.py`: Online refinement module

## Development Commands

### Running Inference
```bash
# Process images
python demo.py --image_folder <path_to_images> --output_folder <output_path>

# Process video
python demo.py --video <path_to_video> --output_folder <output_path>

# With CamSMPLify refinement
python demo.py --image_folder <path> --output_folder <path> --use_smplify

# Export initialization params for offline optimization
python demo.py --image_folder <path> --output_folder <path> --export_init_npz <npz_path>
```

### CamSMPLify Offline Optimization
```bash
python CamSMPLify/optimize.py --vis False --save_vis_dir <output_dir> --input <init.npz> --output_dir <params_dir>
```

### Training
```bash
# Train dense keypoint model
python train_keypoints.py data=train experiment=camerahmr exp_name=<run_name>
```

### Data Setup
```bash
# Download test labels and SMPL models
bash scripts/fetch_test_labels.sh

# Download demo data
bash scripts/fetch_demo_data.sh

# Download training data
bash scripts/fetch_bedlam_training_data.sh
bash scripts/fetch_4dhumans_training_labels.sh
```

## Environment Requirements

- Python 3.10
- PyTorch 2.8+
- CUDA-enabled GPU recommended
- OpenGL environment variables for headless rendering:
  - `PYOPENGL_PLATFORM=egl`
  - `EGL_PLATFORM=surfaceless`

## Processing Pipeline

1. **Detection**: Detectron2 finds person bounding boxes
2. **Intrinsics**: FLNet estimates camera focal length from full image  
3. **Regression**: CameraHMR predicts SMPL params from crops + intrinsics
4. **Mesh Generation**: SMPL forward pass generates vertices and joints
5. **Optional Refinement**: CamSMPLify optimizes using dense 2D correspondences
6. **Rendering**: PyRender overlays meshes on original images

## Key Technical Details

- Weak-perspective camera converted to full-image translation via `convert_to_full_img_cam()`
- SMPL parameters: `global_orient (1,3,3)`, `body_pose (23,3,3)`, `betas (10)`
- Camera representation: weak-perspective `(s, tx, ty)` → full translation `cam_t`
- Batch processing up to 32 persons per forward pass
- Dense keypoints from DenseKP model for refinement
- Rotation representations: network uses rotation matrices, optimization uses axis-angle

## Output Formats

- Rendered images with mesh overlays
- Optional `.obj` mesh files (`--save_smpl_obj`)
- Camera/SMPL parameters in JSON (`--output_cam`)
- NPZ files for offline optimization (`--export_init_npz`)