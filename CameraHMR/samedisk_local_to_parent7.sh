set -euo pipefail
FOLDER=training-images-sapiens-seg-OrgPadCrop
SUBFOLDER=$FOLDER-local
SRC_FOLDER=data/$FOLDER/$SUBFOLDER
DST_FOLDER=data/$FOLDER
# find . -type f -print0 | while IFS= read -r -d '' f; do
#   tgt="../$f"
#   mkdir -p "$(dirname "$tgt")"
#   if [ ! -e "$tgt" ]; then
#     mv -n -- "$f" "$tgt"        # -n = no clobber
#     echo "moved: $f"
#   else
#     echo "SKIP exists: $tgt"
#   fi
# done
# cd ..
rsync -av --remove-source-files --info=progress2 $SRC_FOLDER/* $DST_FOLDER
# Optionally remove now-empty dirs
find $SRC_FOLDER -type d -empty -delete
echo "done"