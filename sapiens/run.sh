#!/bin/bash
# source /home/cevin/miniconda3/etc/profile.d/conda.sh
# conda activate sapiens
cd seg

INPUT=/data/RECON/EMDB/P9/76_outdoor_sitting/images
OUTPUT_ROOT=/home/cevin/Meitu/sapiens/output/EMDB/P9/76_outdoor_sitting

# create output directory if it doesn't exist
mkdir -p $OUTPUT_ROOT

# depth without masks
# python demo/demo_depth_vis.py \
#   configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
#   pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
#   --input $INPUT \
#   --output_root $OUTPUT_ROOT/depth_2b_unmasked

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
#   --input $INPUT \
#   --output_root $OUTPUT_ROOT/seg_masks

# # # depth with masks
# python demo/demo_depth_vis_mask.py \
#   configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
#   pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
#   --input /home/cevin/Meitu/data/test_data_img/all \
#   --seg_dir output/seg_masks \
#   --output_root output/depth_2b_masked_4col

# normal with masks
# python demo/demo_normal_vis.py \
#   configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
#   pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
#   --input $INPUT \
#   --seg_dir $OUTPUT_ROOT/seg_masks \
#   --output_root $OUTPUT_ROOT/normal_2b


# all with integrated segmentation pipeline (5-column: original, seg_overlay, depth, normal_from_depth, normal_direct)
python demo/demo_all_vis_mask.py \
  configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
  configs/sapiens_normal/normal_render_people/sapiens_2b_normal_render_people-1024x768.py \
  configs/sapiens_seg/goliath/sapiens_1b_goliath-1024x768.py \
  ../pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
  ../pretrain/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70.pth \
  ../pretrain/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth \
  --input $INPUT \
  --seg_dir $OUTPUT_ROOT/seg_masks \
  --output_root ${OUTPUT_ROOT}/all \
  --skip_seg

cd ..

# python project_pc.py --render_png --rgb_dir $INPUT --depth_dir ${OUTPUT_ROOT}/all --output_dir ${OUTPUT_ROOT}/point_clouds_FINAL_MASKED