CUDA_VISIBLE_DEVICES=5 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/20221011_1_250_batch01hand_closeup_suburb_c_6fps \
    --output_root data/training-images-sapiens-seg/20221011_1_250_batch01hand_closeup_suburb_c_6fps \
    --batch_size 1 \
    --swapHW