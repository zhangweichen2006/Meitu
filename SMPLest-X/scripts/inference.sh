#!/usr/bin/env sh
set -eu
CKPT_NAME=$1
FILE_NAME=$2
FPS=${3:-30}

# Robust paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAME="${FILE_NAME%.*}"
EXT="${FILE_NAME##*.}"

IMG_PATH="$REPO_ROOT/demo/input_frames/$NAME"
OUTPUT_PATH="$REPO_ROOT/demo/output_frames/$NAME"

mkdir -p "$IMG_PATH"
mkdir -p "$OUTPUT_PATH"

# convert video to frames
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        ffmpeg -y -i "$REPO_ROOT/demo/$FILE_NAME" -f image2 -vf fps=${FPS}/1 -q:v 2 "${IMG_PATH}/%06d.jpg"
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp "$REPO_ROOT/demo/$FILE_NAME" "$IMG_PATH/000001.$EXT"
        ;;
    *)
        echo "Unknown file type."
        exit 1
        ;;
esac

END_COUNT=$(find "$IMG_PATH" -type f | wc -l)

# Ensure we actually extracted or collected input frames
if [ "$END_COUNT" -eq 0 ]; then
    echo "No input frames were found in $IMG_PATH. Aborting without cleanup so you can inspect the workspace."
    exit 1
fi

# inference with smplest_x
# forward any extra args (e.g., --multi_person) to python entrypoint
EXTRA_ARGS="${@:4}"
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python "$REPO_ROOT/main/inference.py" \
    --num_gpus 1 \
    --file_name "$NAME" \
    --ckpt_name "$CKPT_NAME" \
    --end "$END_COUNT" \
    ${HUMAN_MODEL_PATH:+--human_model_path "$HUMAN_MODEL_PATH"} \
    $EXTRA_ARGS


# convert frames to video
case "$EXT" in
    mp4|avi|mov|mkv|flv|wmv|webm|mpeg|mpg)
        # Verify output frames exist before attempting to encode a video
        if ! ls "$OUTPUT_PATH"/*.jpg >/dev/null 2>&1; then
            echo "No output frames were generated in $OUTPUT_PATH. Aborting without cleanup so you can inspect the workspace."
            exit 1
        fi
        ffmpeg -y -framerate ${FPS} -i "${OUTPUT_PATH}/%06d.jpg" -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 18 "$REPO_ROOT/demo/result_${NAME}.mp4"
        ;;
    jpg|jpeg|png|bmp|gif|tiff|tif|webp|svg)
        cp "$OUTPUT_PATH/000001.$EXT" "$REPO_ROOT/demo/result_$FILE_NAME"
        ;;
    *)
        exit 1
        ;;
esac

# Cleanup only after successful completion of all steps
rm -rf "$REPO_ROOT/demo/input_frames"
rm -rf "$REPO_ROOT/demo/output_frames"

