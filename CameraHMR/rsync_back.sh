FOLDER=training-images-sapiens-normals-OrgPadCrop
SUBFOLDER=*

rsync -av --ignore-existing --info=progress2 /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/$FOLDER/$SUBFOLDER /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/$FOLDER-local/

# /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg

# rsync -av --info=progress2 /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-DirectResize

# rsync -av --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop-local /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop

# ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/$SUBFOLDER /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/$SUBFOLDER

# # ls /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-pad
# rsync -av --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop-local /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop
