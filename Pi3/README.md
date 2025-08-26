# $\pi^3$ Evaluation

## Overview

- [x] Monocular Depth Estimation
- [x] Video Depth Estimation
- [x] Relative Camera Pose Estimation
- [x] Multi-view Reconstruction (Point Map Estimation)

The root config file of all evaluations is `configs/eval.yaml`, however you don't need to edit it

- All main hyperparameters you need are in `configs/evaluation/xxxxx.yaml`
- Sometimes you may want to change the dataset config in `configs/data/xxxxx.yaml`, or the model config in `configs/model/xxxxx.yaml`

## 📣 Updates
* **[July 29, 2025]** 📈 Evaluation code is released! See `evaluation` branch for details.
* **[July 16, 2025]** 🚀 Hugging Face Demo and inference code are released!

Please put all evaluation datasets under `data` folder, or you can change the config in `configs/data/xxxxx.yaml`.

For data preprocessing:

- **Depth Estimation**: We follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare Sintel, Bonn, KITTI and NYU-v2.
- **Camera Pose Estimation**
    - Angular: We follow [VGGT](https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/README.md) to prepare Co3Dv2, and we afford [our script](datasets/preprocess/prepare_re10k.sh) for RealEstate10k preprocessing.
    - Distance: We follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare Sintel, TUM-dynamics and ScanNetv2.
- **Point Map Estimation**: We follow [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare 7-Scenes, Neural-NRGBD and DTU. We afford [our script](datasets/preprocess/prepare_eth3d.sh) for ETH3D preprocessing.

> We provide reference-only preprocessing scripts under `datasets/preprocess`. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

## 1. Monocular Depth Estimation

See [monodepth/README.md](monodepth/README.md) for more details.

```bash
python monodepth/infer.py
python monodepth/eval.py
```

## 2. Video Depth Estimation

configs in `configs/evaluation/videodepth.yaml`, see [videodepth/README.md](videodepth/README.md) for more details.

```bash
python videodepth/infer.py
python videodepth/eval.py
```

## 3. Relative Camera Pose Estimation

configs in `configs/evaluation/relpose-angular.yaml`, see [relpose/README.md](relpose/README.md) for more details.

### 3.1 Angular Metrics

```bash
# python relpose/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python relpose/eval_angle.py
```

### 3.2 Distance Metrics

```bash
python relpose/eval_dist.py
```

## 4. Multi-view Reconstruction (Point Map Estimation)

See [mv_recon/README.md](mv_recon/README.md) for more details.

```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
```

## Acknowledgement

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MoGe](https://github.com/microsoft/MoGe)
- [VGGT](https://github.com/facebookresearch/vggt)


## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning},
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347},
}
```


## License
For academic use, this project is licensed under the 2-clause BSD License. See the [LICENSE](./LICENSE) file for details. For commercial use, please contact the authors.
