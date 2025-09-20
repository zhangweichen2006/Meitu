CUDA_VISIBLE_DEVICES=0 python -u SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/test-images/EMDB \
    --output_root data/test-images-sapiens-normals/EMDB \
    --batch_size 1

# mkdir -p /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop

# ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop