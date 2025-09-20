CUDA_VISIBLE_DEVICES=5 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/images \
    --output_root data/training-images-sapiens-seg-OrgPadCrop-local/images \
    --batch_size 1