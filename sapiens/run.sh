cd seg

# python demo/demo_seg_vis.py \
#   configs/sapiens_seg/goliath/sapiens_1b_goliath-1024x768.py \
#   ../pretrain/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth \
#   --input /home/cevin/Meitu/data/test_data_img/all \
#   --output_root ../output/seg_masks


# depth with masks
python demo/demo_depth_vis.py \
  configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py \
  /home/cevin/Meitu/sapiens/pretrain/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth \
  --input /home/cevin/Meitu/sapiens/input/test_data_img/all \
  --seg_dir /home/cevin/Meitu/sapiens/output/seg_masks \
  --output_root /home/cevin/Meitu/sapiens/output/depth_2b_masked
