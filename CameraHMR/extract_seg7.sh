CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/20221011_1_250_batch01hand_closeup_suburb_a_6fps \
    --output_root data/training-images-sapiens-seg-pad/20221011_1_250_batch01hand_closeup_suburb_a_6fps \
    --batch_size 1 \
    --preprocess crop_pad \
    --swapHW

CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps \
    --output_root data/training-images-sapiens-seg-pad/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps \
    --batch_size 1 \
    --preprocess crop_pad \
    --swapHW


CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/20221019_1_250_highbmihand_closeup_suburb_b_6fps \
    --output_root data/training-images-sapiens-seg-pad/20221019_1_250_highbmihand_closeup_suburb_b_6fps \
    --batch_size 1 \
    --preprocess crop_pad \
    --swapHW


CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_seg.py \
    SapiensLite/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    --input data/training-images/20221019_1_250_highbmihand_closeup_suburb_c_6fps \
    --output_root data/training-images-sapiens-seg-pad/20221019_1_250_highbmihand_closeup_suburb_c_6fps \
    --batch_size 1 \
    --preprocess crop_pad \
    --swapHW