CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/test-images \
    --output_root data/test-images-sapiens-seg-pad \
    --preprocess crop_pad \
    --batch_size 1