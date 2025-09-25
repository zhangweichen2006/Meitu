# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import tempfile
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from .adhoc_image_dataset import AdhocImageDataset
from tqdm import tqdm

from .worker_pool import WorkerPool
from .revert_utils import revert_npy

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    # Warm up the model with a dummy input.
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)

def _center_embed_or_crop(res_chw, out_h, out_w):
    c, h, w = res_chw.shape
    if out_h == h and out_w == w:
        return res_chw
    if out_h >= h:
        top = (out_h - h) // 2
        bottom = top + h
        temp = torch.zeros((c, out_h, out_w), dtype=res_chw.dtype, device=res_chw.device)
        if out_w >= w:
            left = (out_w - w) // 2
            right = left + w
            temp[:, top:bottom, left:right] = res_chw
            return temp
        else:
            left = (w - out_w) // 2
            right = left + out_w
            temp[:, top:bottom, :] = res_chw[:, :, left:right]
            return temp
    else:
        top = (h - out_h) // 2
        bottom = top + out_h
        if out_w >= w:
            left = (out_w - w) // 2
            right = left + w
            temp = torch.zeros((c, out_h, out_w), dtype=res_chw.dtype, device=res_chw.device)
            temp[:, :, left:right] = res_chw[:, top:bottom, :]
            return temp
        else:
            left = (w - out_w) // 2
            right = left + out_w
            return res_chw[:, top:bottom, left:right]

def revert_npy_and_img(orig_image, proc_npy_path, output_imgmatch_path, seg_dir=None, swapHW=False, mode="resize"):
    """Invert processed-space normals back to original image resolution and save.

    Args:
        orig_image: Original image (H, W, 3) in BGR.
        proc_npy_path: Path to processed-space normals .npy (H_proc, W_proc, 3).
        output_imgmatch_path: Destination path for visualization (image). The .npy
            will be saved next to it with the same stem.
        seg_dir: Optional directory containing per-image masks (.npy) to apply.
        swapHW: If the preprocessing swapped H/W before the model.
        mode: One of ["resize", "crop_pad", "crop_resize", "pad_resize", "zoom_to_3Dpt"].
    """
    if not os.path.exists(proc_npy_path):
        raise FileNotFoundError(f"Processed npy not found: {proc_npy_path}")

    # Use shared revert util to get HxWx3 normals aligned to original image
    normal_map = revert_npy(proc_npy_path, orig_image, swapHW=swapHW, mode=mode)
    # normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map #/ (normal_map_norm + 1e-5)

    # Optional mask
    if seg_dir is not None:
        mask_path = os.path.join(
            seg_dir,
            os.path.basename(output_imgmatch_path)
            .replace(".png", ".npy")
            .replace(".jpg", ".npy")
            .replace(".jpeg", ".npy"),
        )
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            mask = np.ones_like(normal_map)
    else:
        mask = np.ones_like(normal_map)

    if mask.shape[:2] != normal_map.shape[:2]:
        if mask.ndim == 2:
            mask_resized = cv2.resize(mask.astype(np.uint8), (normal_map.shape[1], normal_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.repeat(mask_resized[..., None], 3, axis=2)
        else:
            mask = cv2.resize(mask.astype(np.uint8), (normal_map.shape[1], normal_map.shape[0]), interpolation=cv2.INTER_NEAREST)

    normal_map_normalized[mask == 0] = -1

    # Save corrected outputs to output_imgmatch_path
    output_imgmatch_npy = os.path.splitext(output_imgmatch_path)[0] + ".npy"
    np.save(output_imgmatch_npy, normal_map_normalized)
    normal_map_vis = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    normal_map_vis = normal_map_vis[:, :, ::-1]
    vis_image = np.concatenate([orig_image, normal_map_vis], axis=1)
    cv2.imwrite(output_imgmatch_path, vis_image)

    return output_imgmatch_path, output_imgmatch_npy

def img_save_and_viz(orig_image, proc_image, result, output_path, output_imgmatch_path, seg_dir, swapHW=False, mode="resize"):
    # 1) Save processed-space outputs to out_name (output_path)
    proc_npy_path = (
        output_path.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")
    )
    normal_result = F.interpolate(result.unsqueeze(0), size=proc_image.shape[:2], mode="bilinear", align_corners=True).squeeze(0)
    normal_proc = normal_result.detach().cpu().numpy().transpose(1, 2, 0)
    normal_proc_norm = np.linalg.norm(normal_proc, axis=-1, keepdims=True)
    normal_proc = normal_proc / (normal_proc_norm + 1e-5)
    np.save(proc_npy_path, normal_proc)

    normal_proc_vis = ((normal_proc + 1) / 2 * 255).astype(np.uint8)
    normal_proc_vis = normal_proc_vis[:, :, ::-1]
    vis_image_preview = np.concatenate([proc_image, normal_proc_vis], axis=1)
    cv2.imwrite(output_path, vis_image_preview)

    # 2) Revert and save to out_imgmatch_name using shared function
    revert_npy_and_img(
        orig_image,
        proc_npy_path,
        output_imgmatch_path,
        seg_dir=seg_dir,
        swapHW=swapHW,
        mode=mode,
    )

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def get_preprocess_args(preprocess):
    if preprocess == "crop_resize":
        return True, True, False
    elif preprocess == "crop_pad":
        return True, False, False
    elif preprocess == "pad_resize":
        return False, False, False
    elif preprocess == "zoom_to_3Dpt":
        return True, False, True
    elif preprocess == "resize":
        return False, True, False
    else:
        return False, False, False

def get_preprocess_adhoc_dataset(image_names, shape_hw, mean, std, cropping, resize, zoom_to_3Dpt, out_names, out_imgmatch_names):
    return AdhocImageDataset(image_names, shape_hw, mean, std, cropping, resize, zoom_to_3Dpt, out_names, out_imgmatch_names)

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
    parser.add_argument(
        "--output_imgmatch_root", "--output-imgmatch-root", default=None, help="Path to output imgmatch dir"
    )
    parser.add_argument("--seg_dir", default=None, help="Path to seg dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=32,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument(
        "--swapHW", action="store_true", default=False, help="swap height and width"
    )
    parser.add_argument(
        "--redo", action="store_true", default=False, help="redo all"
    )
    parser.add_argument(
        "--preprocess",
        choices=["resize", "crop_pad", "crop_resize", "pad_resize", "zoom_to_3Dpt"],
        default="crop_pad",
        help="Preprocess strategy: resize (no crop), crop_pad, pad_resize or crop_resize",
    )
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")


    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.to(args.device)

    input = args.input
    image_names = []
    out_names = []
    out_imgmatch_names = []

    # Check if the input is a directory or a text file
    for root, dirs, files in os.walk(input):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                outfile = os.path.join(root, file).replace(input, args.output_root)
                if not os.path.exists(outfile) or args.redo:
                    image_names.append(os.path.join(root, file))

                    out_names.append(os.path.join(root, file).replace(input, args.output_root))
                    os.makedirs(os.path.dirname(os.path.join(root, file).replace(input, args.output_root)), exist_ok=True)
                outfile_imgmatch = outfile.replace(args.output_root, args.output_imgmatch_root)
                if not os.path.exists(outfile_imgmatch) or args.redo:
                    out_imgmatch_names.append(outfile_imgmatch)
                    os.makedirs(os.path.dirname(outfile_imgmatch), exist_ok=True)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    # check training-images-sapiens-normals and merge to traintest-sapiens-normals
    # for root, dirs, files in os.walk("data/training-images-sapiens-normals"):
    #     for file in files:
    #         if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
    #             src_file = os.path.join(root, file).replace(".png", ".npy")
    #             tgt_file = os.path.join(root.replace("data/training-images-sapiens-normals", "data/traintest-sapiens-normals"), file)
    #             tgt_file = tgt_file.replace(".png", ".npy")
    #             os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
    #             os.system(f"mv {src_file} {tgt_file}")

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size

    cropping, resize, zoom_to_3Dpt = get_preprocess_args(args.preprocess)
    inference_dataset = get_preprocess_adhoc_dataset(image_names, (input_shape[1], input_shape[2]), mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5], cropping=cropping, resize=resize, zoom_to_3Dpt=zoom_to_3Dpt, out_names=out_names, out_imgmatch_names=out_imgmatch_names)

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )
    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    for batch_idx, (batch_image_name, batch_out_name, batch_out_imgmatch_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        cropped_images_np = []
        mean = np.array([123.5, 116.5, 103.5], dtype=np.float32)
        std = np.array([58.5, 57.0, 57.5], dtype=np.float32)
        for t in batch_imgs:  # t: [3, H, W] RGB, normalized
            arr = t.detach().cpu().float().numpy().transpose(1, 2, 0)  # HWC RGB
            arr = arr * std + mean  # de-normalize
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = arr[:, :, ::-1]  # RGB -> BGR for OpenCV
            cropped_images_np.append(arr)

        # For crop_resize (resize=True), do not pad to batch size
        if args.preprocess != "crop_resize":
            batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype)

        args_list = [
            (
                o,
                i,
                r,
                out_name,
                out_imgmatch_name,
                args.seg_dir,
                args.swapHW,
                args.preprocess
            )
            for o, i, r, out_name, out_imgmatch_name in zip(
                batch_orig_imgs,
                cropped_images_np[:valid_images_len],
                result[:valid_images_len],
                batch_out_name,
                batch_out_imgmatch_name
            )
        ]
        print("inferencing:", args_list[0][3])
        img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
