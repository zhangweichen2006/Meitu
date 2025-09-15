SRC="/picassox/tos-mtlab-train-base/human-body/llp1/datasets/paramatric_model_datasets/bedlam_dataset/bedlam_data_30fps/training_images"
DEST="$PWD/data/training-images"
mkdir -p "$DEST"
find "$SRC" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' d; do
  ln -sfn "$d" "$DEST/$(basename "$d")"
done