CUDA_VISIBLE_DEVICES=1 python -u SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/test-images/EMDB \
    --output_root data/test-images-sapiens-normals/EMDB \
    --batch_size 1