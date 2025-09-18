CUDA_VISIBLE_DEVICES=1 python lite/demo/vis_normal.py \
    /home/cevin/Meitu/CameraHMR/SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input /home/cevin/Meitu/CameraHMR/data/training-images/ \
    --output_root /home/cevin/Meitu/CameraHMR/data/traintest-images/ \
    --batch_size 1
