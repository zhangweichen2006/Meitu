# CUDA_VISIBLE_DEVICES=5 python SapiensLite/demo/vis_normal.py \
#     SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
#     --input data/training-images/20221019_1_250_highbmihand_closeup_suburb_c_6fps \
#     --output_root data/training-images-sapiens-normals-OrgPadCrop/20221019_1_250_highbmihand_closeup_suburb_c_6fps \
#     --batch_size 1 \
#     --swapHW

CUDA_VISIBLE_DEVICES=5 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/Human36M \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/Human36M \
    --batch_size 1
# 20221019_1_250_highbmihand_closeup_suburb_c_6fps