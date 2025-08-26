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
from matplotlib import pyplot as plt
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
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument('--alpha_dir', '--alpha-dir', default=None, help='Path to alpha matte dir (PNG/JPG)')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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
                    if image_name.endswith('.jpg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    seg_dir = args.seg_dir
    alpha_dir = args.alpha_dir
    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        temp_path = None
        if image is None:
            try:
                with Image.open(image_path) as pil_im:
                    pil_im = pil_im.convert('RGB')
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    pil_im.save(tmp.name, format='PNG')
                    temp_path = tmp.name
            except Exception as e:
                print(f"Skipping unreadable image (PIL failed): {image_path} ({e})")
                continue

        # Optionally apply alpha/seg mask BEFORE inference
        masked_tmp = None
        if alpha_dir is not None or seg_dir is not None:
            try:
                mask_pre = None
                if alpha_dir is not None:
                    base = os.path.splitext(image_name)[0]
                    cand = [os.path.join(alpha_dir, base + ext) for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']]
                    alpha_path = next((p for p in cand if os.path.exists(p)), None)
                    if alpha_path is not None:
                        with Image.open(alpha_path) as aimg:
                            aimg = aimg.convert('L')
                            mask_pre = np.array(aimg) > 0
                if mask_pre is None and seg_dir is not None:
                    mask_path_pre = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
                    mask_pre = np.load(mask_path_pre)
                if image is None and temp_path is not None:
                    image = cv2.imread(temp_path)
                if mask_pre is None:
                    raise FileNotFoundError('No alpha/seg mask found')
                if mask_pre.shape[:2] != image.shape[:2]:
                    mask_pre = cv2.resize(mask_pre.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                masked = image.copy()
                masked[~mask_pre] = 0
                tmpm = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                cv2.imwrite(tmpm.name, masked)
                masked_tmp = tmpm.name
            except Exception:
                masked_tmp = None

        effective_path = masked_tmp if masked_tmp is not None else (temp_path if temp_path is not None else image_path)

        try:
            result = inference_model(model, effective_path)
        except Exception as e:
            if masked_tmp is not None:
                try:
                    os.remove(masked_tmp)
                except Exception:
                    pass
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            print(f"Skipping {image_path}: inference load failed ({e})")
            continue
        result = result.pred_depth_map.data.cpu().numpy()

        normal_map = result.transpose(1, 2, 0) ### (H, W, C)

        if seg_dir is not None:
            mask_path = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
            mask = np.load(mask_path)
        else:
            mask = np.ones_like(normal_map)

        ##----------------------------------------
        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small e
        save_path = os.path.join(args.output_root, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        np.save(save_path, normal_map_normalized)

        normal_map_normalized[mask == 0] = -1
        normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        normal_map = normal_map[:, :, ::-1]

        ## cleanup temp file if created
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        ## cleanup temp file if created
        if masked_tmp is not None:
            try:
                os.remove(masked_tmp)
            except Exception:
                pass
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        ##----------------------------------------------------
        output_file = os.path.join(args.output_root, os.path.basename(image_path))

        vis_image = np.concatenate([image, normal_map], axis=1)
        cv2.imwrite(output_file, vis_image)

if __name__ == '__main__':
    main()
