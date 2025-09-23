set -euo pipefail
SCRIPT="$(basename "$0")"
TAG="$SCRIPT[$(hostname -s):$$]"

SRC_FOLDER=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-normals-DirectResize/mpiimages/mpii-train/MPII-pose/*
TGT_FOLDER=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-sapiens-normals-DirectResize/mpiimages/mpii-train
# VEPFS_HYPERLINK=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/${FOLDER}

# check folder exists
echo "checking folder ${SRC_FOLDER}"
if ! compgen -G "$SRC_FOLDER" > /dev/null; then
    echo "folder does not exist: $SRC_FOLDER"
    exit 1
fi

# Bail if linkname ends with a slash (would imply a dir)
# case "$VEPFS_HYPERLINK" in
#   */) echo "Link path must not end with '/': $VEPFS_HYPERLINK" >&2; exit 1 ;;
# esac

mkdir -p $TGT_FOLDER
echo "done mkdir"

# check if hyperlink exists and if VEPFS_HYPERLINK is not a folder
# if [ ! -L "$VEPFS_HYPERLINK" ] && [ ! -d "$VEPFS_HYPERLINK" ]; then
#     echo "hyperlink do not exists: $VEPFS_HYPERLINK"
#     # create hyperlink
#     ln -s $TGT_FOLDER $VEPFS_HYPERLINK
#     echo "done create hyperlink"
# else
#     # if VEPFS_HYPERLINK is a folder, exit
#     if [ ! -L "$VEPFS_HYPERLINK" ]; then
#         echo "VEPFS_HYPERLINK is a folder or not exists: $VEPFS_HYPERLINK"
#         exit 1
#     fi
# fi

rsync -av --remove-source-files --info=progress2 $SRC_FOLDER $TGT_FOLDER
echo "done rsync"

# remove src folder if empty
# Assume: set -euo pipefail
if [[ -d "$SRC_FOLDER" ]]; then
  if rmdir -- "$SRC_FOLDER" 2>/dev/null; then
    echo "removed empty dir: $SRC_FOLDER"
  else
    echo "not empty, kept: $SRC_FOLDER"
  fi
fi
