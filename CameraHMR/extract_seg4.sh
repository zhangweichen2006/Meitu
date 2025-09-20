CUDA_VISIBLE_DEVICES=4 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/MPII-pose \
    --output_root data/training-images-sapiens-seg/MPII-pose \
    --batch_size 1

CUDA_VISIBLE_DEVICES=4 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/mpi-inf-train-pruned \
    --output_root data/training-images-sapiens-seg/mpi-inf-train-pruned \
    --batch_size 1
# 20221019_1_250_highbmihand_closeup_suburb_b_6fps