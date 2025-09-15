#!/usr/bin/env bash
set -euo pipefail

# Source directory containing .tar.gz archives
SOURCE_DIR="/picassox/tos-mtlab-train-base/human-body/llp1/datasets/paramatric_model_datasets/bedlam_dataset/bedlam_data_6fps/bedlam_ann"

# Destination root (inside this repository)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="${SCRIPT_DIR}/data/training-labels/bedlam-labels"

if ! command -v tar >/dev/null 2>&1; then
  echo "Error: tar is required but not found in PATH" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

shopt -s nullglob
archives=("${SOURCE_DIR}"/*.tar.gz)

if [ ${#archives[@]} -eq 0 ]; then
  echo "No .tar.gz files found in ${SOURCE_DIR}"
  exit 0
fi

for archive_path in "${archives[@]}"; do
  archive_name="$(basename "${archive_path}" .tar.gz)"
  dest_dir="${DEST_ROOT}/${archive_name}"
  mkdir -p "${dest_dir}"
  echo "Extracting: ${archive_path} -> ${dest_dir}"
  tar -xzf "${archive_path}" -C "${dest_dir}"
done

echo "All archives extracted to ${DEST_ROOT}"


