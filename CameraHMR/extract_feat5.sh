CUDA_VISIBLE_DEVICES=5 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/Human36M \
    --output_root data/training-images-sapiens-normals/Human36M \
    --batch_size 1

# 20221019_1_250_highbmihand_closeup_suburb_c_6fps