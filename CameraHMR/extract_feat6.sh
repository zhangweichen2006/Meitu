CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images \
    --output_root data/training-images-sapiens-normals-DirectResize \
    --preprocess resize \
    --batch_size 1
