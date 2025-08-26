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
    parser.add_argument('depth_config', help='Depth config file')
    parser.add_argument('normal_config', help='Normal config file')
    parser.add_argument('seg_config', help='Segmentation config file')
    parser.add_argument('depth_checkpoint', help='Depth checkpoint file')
    parser.add_argument('normal_checkpoint', help='Normal checkpoint file')
    parser.add_argument('seg_checkpoint', help='Segmentation checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--flip', action='store_true', help='Flag to indicate if left right flipping')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

        # build the depth model from a config file and a checkpoint file
    depth_model = init_model(args.depth_config, args.depth_checkpoint, device=args.device)
    if args.device == 'cpu':
        depth_model = revert_sync_batchnorm(depth_model)

    # build the normal model from a config file and a checkpoint file
    normal_model = init_model(args.normal_config, args.normal_checkpoint, device=args.device)
    if args.device == 'cpu':
        normal_model = revert_sync_batchnorm(normal_model)

    # build the segmentation model from a config file and a checkpoint file
    seg_model = init_model(args.seg_config, args.seg_checkpoint, device=args.device)
    if args.device == 'cpu':
        seg_model = revert_sync_batchnorm(seg_model)

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
        os.makedirs(args.output_root, exist_ok=True)

    # Create separate folders for individual outputs
    depth_output_dir = os.path.join(args.output_root, 'depth_ALL')
    normal_output_dir = os.path.join(args.output_root, 'normal_ALL')
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(normal_output_dir, exist_ok=True)

    seg_dir = args.seg_dir
    flip = args.flip

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path) ## has to be bgr image

        # Handle truncated images by re-encoding via PIL if needed
        temp_path = None
        if image is None:
            try:
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
                    image = cv2.imread(temp_path)
                    if image is None:
                        print(f"Skipping unreadable image even after PIL conversion and cleaning: {image_path}")
                        continue
                    print(f"Successfully cleaned truncated image: {image_name}")
            except Exception as e:
                print(f"Skipping unreadable image: {image_path} ({e})")
                continue

        effective_path = temp_path if temp_path is not None else image_path

        # Check if mask exists, generate if not
        if seg_dir is None:
            # If no seg_dir provided, create one in output_root
            seg_dir = os.path.join(args.output_root, 'seg_masks')
            os.makedirs(seg_dir, exist_ok=True)

        mask_path = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        seg_vis_path = os.path.join(seg_dir, os.path.basename(image_path))

        # Generate segmentation if mask doesn't exist
        if not os.path.exists(mask_path):
            print(f"Generating segmentation mask for {image_name}...")
            try:
                seg_result = inference_model(seg_model, effective_path)

                # Save segmentation mask (.npy)
                seg_mask = seg_result.pred_sem_seg.data.cpu().numpy()[0] > 0  # Convert to boolean mask
                np.save(mask_path, seg_mask)

                # Save segmentation visualization (.jpg)
                show_result_pyplot(
                    seg_model,
                    effective_path,
                    seg_result,
                    title='segmentation',
                    opacity=args.opacity,
                    draw_gt=False,
                    show=False,
                    save_dir=seg_dir,
                    out_file=seg_vis_path
                )
                print(f"Generated mask: {mask_path}")

            except Exception as e:
                if temp_path is not None:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                print(f"Skipping {image_name}: segmentation failed ({e})")
                continue

        # Run depth inference with fallback for truncated images
        depth_result = None
        try:
            depth_result = inference_model(depth_model, effective_path)
        except Exception:
            # Try PIL cleanup for truncated images
            if temp_path is None:  # Haven't tried PIL cleanup yet
                try:
                    print(f"Cleaning truncated image for depth: {image_name}")
                    with Image.open(image_path) as pil_im:
                        pil_im = pil_im.convert('RGB')
                        img_array = np.array(pil_im)
                        img_array = np.nan_to_num(img_array, nan=0, posinf=255, neginf=0)
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                        cleaned_pil = Image.fromarray(img_array)
                        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        cleaned_pil.save(tmp.name, format='PNG')
                        temp_path = tmp.name
                        effective_path = temp_path
                        image = cv2.imread(temp_path)  # Update image for later use
                        depth_result = inference_model(depth_model, effective_path)
                        print(f"Successfully processed truncated image for depth: {image_name}")
                except Exception as e:
                    if temp_path is not None:
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    print(f"Skipping {image_path}: depth inference failed ({e})")
                    continue
            else:
                if temp_path is not None:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                print(f"Skipping {image_path}: depth inference failed even with cleaned image")
                continue

        if depth_result is None:
            continue
        depth_result = depth_result.pred_depth_map.data.cpu().numpy()
        depth_map = depth_result[0] ## H x W

        # Run normal inference with same effective_path
        try:
            normal_result = inference_model(normal_model, effective_path)
        except Exception as e:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            print(f"Skipping {image_path}: normal inference failed ({e})")
            continue
        normal_result = normal_result.pred_depth_map.data.cpu().numpy()
        normal_map_direct = normal_result.transpose(1, 2, 0) ### (H, W, C)

        if flip == True:
            image_flipped = cv2.flip(image, 1)
            # Save flipped image to a temporary file and run inference with path
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp_flip:
                cv2.imwrite(tmp_flip.name, image_flipped)
                try:
                    depth_result_flipped = inference_model(depth_model, tmp_flip.name)
                    normal_result_flipped = inference_model(normal_model, tmp_flip.name)
                except Exception:
                    depth_result_flipped = None
                    normal_result_flipped = None
            if depth_result_flipped is not None:
                depth_result_flipped = depth_result_flipped.pred_depth_map.data.cpu().numpy()
                depth_map_flipped = depth_result_flipped[0]
                depth_map_flipped = cv2.flip(depth_map_flipped, 1) ## H x W, flip back
                depth_map = (depth_map + depth_map_flipped) / 2 ## H x W, average
            if normal_result_flipped is not None:
                normal_result_flipped = normal_result_flipped.pred_depth_map.data.cpu().numpy()
                normal_map_flipped = normal_result_flipped.transpose(1, 2, 0)
                normal_map_flipped = cv2.flip(normal_map_flipped, 1) ## H x W x C, flip back
                normal_map_direct = (normal_map_direct + normal_map_flipped) / 2 ## H x W x C, average

        mask = np.load(mask_path)

        ##-----------save depth_map to disk---------------------
        save_path = os.path.join(args.output_root, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        np.save(save_path, depth_map)
        depth_map[~mask] = np.nan

        ##----------------------------------------
        depth_foreground = depth_map[mask] ## value in range [0, 1]
        processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

        if len(depth_foreground) > 0:
            min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
            depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## for visualization, foreground is 1 (white), background is 0 (black)
            depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

            print('{}, min_depth:{}, max_depth:{}'.format(image_name, min_val, max_val))

            depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
            depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
            processed_depth[mask] = depth_colored_foreground

        ##---------get surface normal from depth map---------------
        depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
        depth_normalized[mask > 0] = 1 - ((depth_foreground - min_val) / (max_val - min_val))

        kernel_size = 7 # ffhq
        grad_x = cv2.Sobel(depth_normalized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(depth_normalized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=kernel_size)
        z = np.full(grad_x.shape, -1)
        normals = np.dstack((-grad_x, -grad_y, z))

        # Normalize the normals
        normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        normals_normalized = normals / (normals_mag + 1e-5)  # Add a small epsilon to avoid division by zero

        # Convert normals to a 0-255 scale for visualization
        normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

        ## RGB to BGR for cv2
        normal_from_depth = normal_from_depth[:, :, ::-1]

        ##--------process direct normal map from normal model--------
        # Normalize the direct normal map
        normal_map_norm = np.linalg.norm(normal_map_direct, axis=-1, keepdims=True)
        normal_map_normalized = normal_map_direct / (normal_map_norm + 1e-5)

        # Apply mask to direct normal map
        normal_map_normalized[mask == 0] = -1
        normal_map_direct_vis = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        normal_map_direct_vis = normal_map_direct_vis[:, :, ::-1]  # RGB to BGR for cv2

        ##----------------------------------------------------
        output_file = os.path.join(args.output_root, os.path.basename(image_path))

        # Save individual depth and normal images (only the processed results, not original)
        depth_individual_file = os.path.join(depth_output_dir, os.path.basename(image_path))
        normal_individual_file = os.path.join(normal_output_dir, os.path.basename(image_path))

        # Save only the processed depth image
        cv2.imwrite(depth_individual_file, processed_depth)

        # Save only the processed normal image
        cv2.imwrite(normal_individual_file, normal_map_direct_vis)

        # Get segmentation visualization for display (second column)
        seg_vis_path = os.path.join(seg_dir, os.path.basename(image_path))
        if os.path.exists(seg_vis_path):
            seg_vis_img = cv2.imread(seg_vis_path)
            if seg_vis_img is not None:
                # The seg_vis image is 2-column (original + overlay), extract the right half (overlay part)
                h, w = seg_vis_img.shape[:2]
                seg_overlay = seg_vis_img[:, w//2:]  # Right half is the segmentation overlay
                # Resize to match original image size
                if seg_overlay.shape[:2] != image.shape[:2]:
                    seg_overlay = cv2.resize(seg_overlay, (image.shape[1], image.shape[0]))
            else:
                seg_overlay = np.zeros_like(image)  # Fallback to black image
        else:
            seg_overlay = np.zeros_like(image)  # Fallback to black image

        # Create the 5-column output: original, seg_overlay, depth, normal_from_depth, normal_direct
        vis_image = np.concatenate([image, seg_overlay, processed_depth, normal_from_depth, normal_map_direct_vis], axis=1)
        cv2.imwrite(output_file, vis_image)

        # Clean up temporary file if created
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == '__main__':
    main()
