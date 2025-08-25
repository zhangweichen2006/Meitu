#!/bin/bash
FOLDER_ID="1CORowHi56hsfwpFOaH_SS3r-5s501CIt"
LOCAL_DIR="/data"

# Sync only the target shared folder
rclone sync gdrive: "$LOCAL_DIR" \
  --drive-shared-with-me \
  --drive-root-folder-id="$FOLDER_ID" \
  --progress \
  --create-empty-src-dirs

rclone sync gdrive:/dna_rendering_release_data /data/RECON/DNA-Rendering/

rclone sync gdrive:/mobile_stage /data/RECON/Mobile_Stage