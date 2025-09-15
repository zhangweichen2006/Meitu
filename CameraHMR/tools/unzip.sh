#!/usr/bin/env bash
set -euo pipefail

# Source directory containing .tar.gz archives
SOURCE_DIR="/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/test-images/EMDB"
# "/picassox/tos-mtlab-train-base/human-body/llp1/datasets/paramatric_model_datasets/bedlam_dataset/bedlam_data_30fps/bedlam_png"

# Destination root (inside this repository)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/test-images/EMDB"
# "${SCRIPT_DIR}/data/training-images/30fps"

# DEST_ROOT="/picassox/tos-mtlab-train-base/human-body/llp1/datasets/paramatric_model_datasets/bedlam_dataset/bedlam_data_30fps/bedlam_png"

if ! command -v tar >/dev/null 2>&1; then
  echo "Error: tar is required but not found in PATH" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"
echo "Extracting: ${SOURCE_DIR} -> ${DEST_ROOT}"
# shopt -s nullglob
ext=".zip"
# ".tar"
archives=("${SOURCE_DIR}"/*"${ext}") # NOTE: changed from .tar.gz to .tar

if [ ${#archives[@]} -eq 0 ]; then
  echo "No .zip files found in ${SOURCE_DIR}"
  exit 0
fi

# if [ ${#archives[@]} -eq 0 ]; then
#   echo "No .tar.gz files found in ${SOURCE_DIR}"
#   exit 0
# fi

for archive_path in "${archives[@]}"; do
  archive_name="$(basename "${archive_path}" "${ext}")"
  # 1. use zipped filename as folder name
  # dest_dir="${DEST_ROOT}/${archive_name}"
  # mkdir -p "${dest_dir}"
  # 2. get rid of zipped filename as folder name, just unzip content to dest_dir
  dest_dir="$(dirname "${archive_path}")"
  echo "Extracting: ${archive_path} -> ${dest_dir}"
  unzip "${archive_path}" -d "${dest_dir}"
  # unzip "${archive_path}" -d "${dest_dir}"
done


# for archive_path in "${archives[@]}"; do
#   archive_name="$(basename "${archive_path}" "${ext}")"
#   dest_dir="${DEST_ROOT}/${archive_name}"
#   mkdir -p "${dest_dir}"
#   echo "Extracting: ${archive_path} -> ${dest_dir}"
#   tar -xvf "${archive_path}" -C "${dest_dir}"
# done

# echo "All archives extracted to ${DEST_ROOT}"


