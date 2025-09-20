CUDA_VISIBLE_DEVICES=0 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221011_1_250_batch01hand_closeup_suburb_d_6fps \
    --output_root data/training-images-sapiens-normals/20221011_1_250_batch01hand_closeup_suburb_d_6fps \
    --batch_size 1 \
    --swapHW \
    --preprocess crop_pad \
    --redo