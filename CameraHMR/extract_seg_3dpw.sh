CUDA_VISIBLE_DEVICES=5 python -u SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/test-images/3DPW/ \
    --output_root data/test-images-sapiens-seg/3DPW/ \
    --batch_size 1


# ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg/Human36M /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg/Human36M