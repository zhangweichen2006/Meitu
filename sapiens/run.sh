#!/bin/bash
# source /home/cevin/miniconda3/etc/profile.d/conda.sh
# conda activate sapiens
cd seg


# depth without masks
python demo/demo_depth_vis.py \
  configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
  pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
  --input input/test_data_img/all \
  --output_root output/depth_2b_unmasked

# depth with alpha masks
# python demo/demo_depth_vis.py \
#   configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
#   pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
#   --input /home/cevin/Meitu/data/test_data_img/all \
#   --alpha_dir /home/cevin/Meitu/data/test_data_img/matted_pha \
#   --output_root output/depth_2b_alpha

# python demo/demo_seg_vis.py \
#   configs/sapiens_seg/goliath/sapiens_1b_goliath-1024x768.py \
#   ../pretrain/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth \
#   --input /home/cevin/Meitu/data/test_data_img/all \
#   --output_root ../output/seg_masks

# # # depth with masks
# python demo/demo_depth_vis_mask.py \
#   configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
#   pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
#   --input /home/cevin/Meitu/data/test_data_img/all \
#   --seg_dir output/seg_masks \
#   --output_root output/depth_2b_masked_4col

# normal with masks
python demo/demo_normal_vis.py \
  configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
  pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
  --input /home/cevin/Meitu/data/test_data_img/all \
  --seg_dir output/seg_masks \
  --output_root output/normal_2b


# all with integrated segmentation pipeline (5-column: original, seg_overlay, depth, normal_from_depth, normal_direct)
python demo/demo_all_vis_mask.py \
  configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
  configs/sapiens_normal/normal_render_people/sapiens_2b_normal_render_people-1024x768.py \
  configs/sapiens_seg/goliath/sapiens_1b_goliath-1024x768.py \
  pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
  pretrain/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70.pth \
  pretrain/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth \
  --input /home/cevin/Meitu/data/test_data_img/all \
  --seg_dir output/seg_masks \
  --output_root output/all \
  --skip_seg

cd ..

python project_pc.py --render_png --rgb_dir ../data/test_data_img/all --depth_dir output/all --output_dir output/point_clouds_FINAL_MASKED