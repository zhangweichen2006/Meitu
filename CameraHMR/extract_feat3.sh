CUDA_VISIBLE_DEVICES=3 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/MPII-pose \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/MPII-pose \
    --batch_size 1

CUDA_VISIBLE_DEVICES=3 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/mpi-inf-train-pruned \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/mpi-inf-train-pruned \
    --batch_size 1

# ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-normals-OrgPadCrop /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-normals-OrgPadCrop


# /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-normals-OrgPadCrop