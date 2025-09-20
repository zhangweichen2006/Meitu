CUDA_VISIBLE_DEVICES=4 python -u SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/test-images/3DPW/ \
    --output_root data/test-images-sapiens-normals/3DPW/ \
    --batch_size 1