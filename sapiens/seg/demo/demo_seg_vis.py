# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
torchvision.disable_beta_transforms_warning()

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)

        # Handle truncated images by re-encoding via PIL if needed
        temp_path = None
        effective_path = image_path
        result = None

        # Try inference with original path first, then fallback to PIL if it fails
        try:
            result = inference_model(model, image_path)
        except Exception:
            # Image failed inference, try PIL cleanup
            try:
                print(f"Cleaning truncated image: {image_name}")
                with Image.open(image_path) as pil_im:
                    pil_im = pil_im.convert('RGB')
                    # Convert to numpy array and handle NaN/invalid pixels
                    img_array = np.array(pil_im)
                    # Replace any NaN or invalid values with 0
                    img_array = np.nan_to_num(img_array, nan=0, posinf=255, neginf=0)
                    # Ensure valid uint8 range
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    # Convert back to PIL and save
                    cleaned_pil = Image.fromarray(img_array)
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    cleaned_pil.save(tmp.name, format='PNG')
                    temp_path = tmp.name
                    effective_path = temp_path
                    print(f"Successfully cleaned truncated image: {image_name}")

                    # Try inference again with cleaned image
                    result = inference_model(model, effective_path)
            except Exception as e:
                print(f"Skipping unreadable image: {image_path} ({e})")
                continue

        # If we still don't have a result, skip
        if result is None:
            print(f"Skipping {image_path}: no result obtained")
            continue

        output_file = os.path.join(args.output_root, os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('.png', '.npy'))
        output_seg_file = os.path.join(args.output_root, os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('.png', '_seg.npy'))

        # Load image for visualization (use temp path if available)
        image = cv2.imread(effective_path if temp_path is not None else image_path)

        pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy() ## H x W. seg ids.
        mask = (pred_sem_seg > 0)
        np.save(output_file, mask)
        np.save(output_seg_file, pred_sem_seg)

        # show the results
        vis_image = show_result_pyplot(
            model,
            effective_path,
            result,
            title=args.title,
            opacity=args.opacity,
            draw_gt=False,
            show=False,
            out_file=None)

        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        output_file = os.path.join(args.output_root, os.path.basename(image_path))
        vis_image = np.concatenate([image, vis_image], axis=1)
        cv2.imwrite(output_file, vis_image)

        # Clean up temporary file if created
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == '__main__':
    main()
