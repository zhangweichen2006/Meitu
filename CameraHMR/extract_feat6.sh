CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221011_1_250_batch01hand_closeup_suburb_a_6fps \
    --output_root data/training-images-sapiens-normals-DirectResize-local/20221011_1_250_batch01hand_closeup_suburb_a_6fps \
    --output_imgmatch_root data/training-images-sapiens-normals-DirectResize-imgmatch-local/20221011_1_250_batch01hand_closeup_suburb_a_6fps \
    --preprocess resize \
    --swapHW \
    --redo \
    --batch_size 1
