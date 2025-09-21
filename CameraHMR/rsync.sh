rsync -av --remove-source-files --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-normals-OrgPadCrop-local/COCO /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-normals-OrgPadCrop/COCO

# /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg

# rsync -av --remove-source-files --info=progress2 /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-DirectResize

# rsync -av --remove-source-files --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop-local /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop

VER=test-images-sapiens-normals-DirectResize
ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/$VER /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/$VER

# ls /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-pad

