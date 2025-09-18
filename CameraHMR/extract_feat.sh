CUDA_VISIBLE_DEVICES=0 python lite/demo/vis_normal.py \
    /home/cevin/Meitu/CameraHMR/SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input /home/cevin/Meitu/CameraHMR/data/training-images/ \
    --output_root /home/cevin/Meitu/CameraHMR/data/traintest-sapiens-normals/ \
    --batch_size 1

CUDA_VISIBLE_DEVICES=3 python lite/demo/vis_normal_crop_pad.py \
    /home/cevin/Meitu/CameraHMR/SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input /home/cevin/Meitu/CameraHMR/data/training-images/ \
    --output_root /home/cevin/Meitu/CameraHMR/data/traintest-sapiens-normals-pad/ \
    --batch_size 1