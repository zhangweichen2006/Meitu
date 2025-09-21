CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images2/ \
    --output_root data/training-images-sapiens-seg-OrgPadCrop-local/ \
    --batch_size 1 \
    --preprocess crop_pad \
    --swapHW

# 20221019_1_250_highbmihand_closeup_suburb_c_6fps