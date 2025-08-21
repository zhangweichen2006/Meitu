---
license: other
license_name: license
license_link: LICENSE
---

# SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation

This work is the extended version of [SMPLer-X](https://arxiv.org/abs/2309.17448). This new codebase is designed for easy installation and flexible development, enabling seamless integration of new methods with the pretrained SMPLest-X model.

# Video test

# Image test
PYTHONPATH="/home/cevin/Meitu/SMPLest-X:$PYTHONPATH" python /home/cevin/Meitu/SMPLest-X/main/inference.py \
  --num_gpus 1 \
  --ckpt_name smplest_x_h \
  --image_dir /home/cevin/Meitu/SMPLest-X/demo/raw/tests \
  --output_dir /home/cevin/Meitu/SMPLest-X/demo/output_frames/tests \
  --multi_person

## Citation
```text
@article{yin2025smplest,
  title={SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation},
  author={Yin, Wanqi and Cai, Zhongang and Wang, Ruisi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Mei, Haiyi and Wang, Yanjun and Pang, Hui En and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yamashita, Atsushi and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2501.09782},
  year={2025}
}
```
