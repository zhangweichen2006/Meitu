# 🌟 HuGe100K Dataset Documentation

## 📊 Dataset Overview
HuGe100K is a large-scale multi-view human dataset featuring diverse attributes, high-fidelity appearances, and well-aligned SMPL-X models.

## 📁 File Format and Structure

The dataset is organized with the following structure:

```
HuGe100K/
├── flux_batch1/
│   ├── images[0...9]/            #  different batch of images
│   │   ├── videos/               # Folder for .mp4 and .jpg files
│   │   │   ├── Algeria_female_average_high fashion_50~60 years old_844.jpg
│   │   │   ├── Algeria_female_average_high fashion_50~60 years old_844.mp4
│   │   │   └── ... (more .jpg and .mp4)
│   │   └── param/               # Folder for parameter files (.npy)
│   │       ├── Algeria_female_average_high fashion_50~60 years old_844.npy
│   │       └── ... (more .npy files)
├── flux_batch2/
│   └── ... (similar structure with images[0...9])
├── flux_batch3/
│   └── ... (similar structure with images[0...9])
├── flux_batch4/
│   └── ... (similar structure with images[0...9])
├── flux_batch5/
│   └── ... (similar structure with images[0...9])
├── flux_batch6/
│   └── ... (similar structure with images[0...9])
├── flux_batch7/
│   └── ... (similar structure with images[0...9])
├── flux_batch8/
│   └── ... (similar structure with images[0...9])
├── flux_batch9/
│   └── ... (similar structure with images[0...9])
└── deepfashion/
    └── ... (similar structure with images[0...9])
```

Where:
- Each `images[X]` folder contains:
  - `videos/`: Reference images and generatedvideo files
  - `param/`: Camera and body pose parameters
- **flux_batch1 through flux_batch7**: Contains subjects in A-pose
- **flux_batch8 and flux_batch9**: Contains subjects in diverse poses 
- **deepfashion**: Contains subjects in A-pose (derived from the DeepFashion dataset)

### File Naming Convention
Files follow the naming pattern: `Area_Gender_BodyType_Clothing_Age_ID.extension`

For example:
- `Algeria_female_average_high fashion_50~60 years old_844.jpg`: Reference image of an Algerian female with average build in high fashion clothing
- `Algeria_female_average_high fashion_50~60 years old_844.npy`: Parameter file for the same subject

### 📸 Sample Visualization

<div style="display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: nowrap; width: 100%;">
  <img src="sample/videos/Kenya_female_fit_streetwear_50~60 years old_1501.jpg" alt="Kenya Female Fit Streetwear Image" style="max-width: 45%; width: 45%; height: auto;">
  <span style="font-weight: bold;"> =MVChamp=> </span>
  <!-- <video autoplay loop muted playsinline style="max-width: 45%; width: 45%; height: auto;">
    <source src="sample/videos/Kenya_female_fit_streetwear_50~60 years old_1501.gif" type="video/mp4">
    Your browser does not support the video tag.
  </video> -->
    <img src="sample/videos/Kenya_female_fit_streetwear_50~60 years old_1501.gif" alt="Kenya Female Fit Streetwear Image" style="max-width: 45%; width: 45%; height: auto;">
</div>



## 📈 Dataset Statistics

- **Total Subjects**: 100,000+
- **Views per Subject**: Multiple viewpoints covering 360° in 24 views
- **Pose Types**: A-pose and diverse poses

## 🔍 Visualizing the Dataset

For visualization and data parsing examples, please refer to our provided script:
`visualize_samples.py`. This script demonstrates how to:

- Load the SMPL-X parameters from `.npy` files
- Render the 3D human model from multiple camera views
- Compare rendered results with the original video frames

Requirements for visualization:
- SMPL-X model (download from [official website](https://smpl-x.is.tue.mpg.de/))
- Python packages: `pyrender`, `trimesh`, `smplx`, `numpy`, `torch`

Example usage:
```bash
python visualize_samples.py
```

The script will generate:
- `rendered_results.mp4`: Rendered views of the 3D model
- `aligned_results.mp4`: Blended visualization of rendered model with original frames

## 📋 Usage Guidelines

1. **Research Purposes Only**: This dataset is intended for academic and research purposes.
2. **Citation Required**: If you use this dataset in your research, please cite our paper.
3. **No Commercial Use**: Commercial use is permitted only with explicit permission from us at yiyu.zhuang@smail.nju.edu.cn.
4. **DeepFashion Derivatives**: See License and Attribution section below for special requirements.

## ⚖️ License and Attribution (DeepFashion)

This dataset includes images derived from the **DeepFashion** dataset, originally provided by MMLAB at The Chinese University of Hong Kong. The use of DeepFashion images in this dataset has been explicitly authorized by the original authors solely for the purpose of creating and distributing this dataset. **Users must not further reproduce, distribute, sell, or commercially exploit any images or derived data originating from DeepFashion.** For any subsequent or separate use of the DeepFashion data, users must directly obtain authorization from MMLAB and comply with the original [DeepFashion License](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). 