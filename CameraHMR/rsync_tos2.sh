set -euo pipefail

FOLDER=test-images-sapiens-seg-DirectResize
SUBFOLDER=*
VEPFS_FOLDER=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/${FOLDER}-local/${SUBFOLDER}
TOS_FOLDER=/picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/$FOLDER/
VEPFS_HYPERLINK=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/${FOLDER}

# check folder exists
echo "checking folder ${VEPFS_FOLDER}"
if ! compgen -G "$VEPFS_FOLDER" > /dev/null; then
    echo "folder does not exist: $VEPFS_FOLDER"
    exit 1
fi

# Bail if linkname ends with a slash (would imply a dir)
case "$VEPFS_HYPERLINK" in
  */) echo "Link path must not end with '/': $VEPFS_HYPERLINK" >&2; exit 1 ;;
esac

mkdir -p $TOS_FOLDER
echo "done mkdir"

# check if hyperlink exists and if VEPFS_HYPERLINK is not a folder
if [ ! -L "$VEPFS_HYPERLINK" ] && [ ! -d "$VEPFS_HYPERLINK" ]; then
    echo "hyperlink do not exists: $VEPFS_HYPERLINK"
    # create hyperlink
    ln -s $TOS_FOLDER $VEPFS_HYPERLINK
    echo "done create hyperlink"
else
    # if VEPFS_HYPERLINK is a folder, exit
    if [ ! -L "$VEPFS_HYPERLINK" ]; then
        echo "VEPFS_HYPERLINK is a folder or not exists: $VEPFS_HYPERLINK"
        exit 1
    fi
fi

rsync -av --remove-source-files --info=progress2 $VEPFS_FOLDER $TOS_FOLDER
echo "done rsync"

# /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg

# rsync -av --info=progress2 /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-DirectResize

# rsync -av --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop-local /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/training-images-sapiens-seg-OrgPadCrop

# ln -s /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/$SUBFOLDER /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/$SUBFOLDER

# # ls /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-pad
# rsync -av --info=progress2 /picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop-local /picassox/tos-mtlab-train-base/human-body/weichen.zhang/data/CameraHMR/data/test-images-sapiens-normals-OrgPadCrop
