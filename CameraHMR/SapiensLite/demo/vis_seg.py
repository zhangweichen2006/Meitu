# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm

from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=model.dtype).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=model.dtype
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


def img_save_and_viz(
    image, result, output_path, classes, palette, title=None, opacity=0.5, threshold=0.3,
):
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )
    output_seg_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", "_seg.npy")
    )

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()

    mask = pred_sem_seg > 0
    np.save(output_file, mask)
    np.save(output_seg_file, pred_sem_seg)

    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    vis_image = np.concatenate([image, vis_image], axis=1)
    cv2.imwrite(output_path, vis_image)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
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
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument("--title", default="result", help="The image identifier.")
    parser.add_argument(
        "--swapHW", action="store_true", default=False, help="swap height and width"
    )
    parser.add_argument(
        "--redo", action="store_true", default=False, help="redo all"
    )
    parser.add_argument(
        "--preprocess",
        choices=["resize", "crop_pad", "crop_resize", "pad_resize"],
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

    # Build image list and corresponding output paths mirroring directory structure
    for root, dirs, files in os.walk(input):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                full_in = os.path.join(root, file)
                full_out = full_in.replace(args.input, args.output_root)
                if not os.path.exists(full_out) or args.redo:
                    image_names.append(full_in)
                    out_names.append(full_out)
                    os.makedirs(os.path.dirname(full_out), exist_ok=True)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size

    if args.preprocess == "crop_resize":
        inference_dataset = AdhocImageDataset(
            image_names,
            (input_shape[1], input_shape[2]),
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
            cropping=True,
            resize=True,
            out_names=out_names,
            swapHW=args.swapHW,
        )
    elif args.preprocess == "crop_pad":
        inference_dataset = AdhocImageDataset(
            image_names,
            (input_shape[1], input_shape[2]),
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
            cropping=True,
            out_names=out_names,
            swapHW=args.swapHW,
        )
    elif args.preprocess == "pad_resize":
        inference_dataset = AdhocImageDataset(
            image_names,
            (input_shape[1], input_shape[2]),
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
            cropping=False,
            resize=False,
            out_names=out_names,
            swapHW=args.swapHW,
        )
    # resize
    else:
        inference_dataset = AdhocImageDataset(
            image_names,
            (input_shape[1], input_shape[2]),
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
            cropping=False,
            resize=True,
            out_names=out_names,
            swapHW=args.swapHW,
        )
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
    for batch_idx, (batch_image_name, batch_out_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        # Convert cropped, normalized tensors back to uint8 BGR images for saving
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

        if args.preprocess != "crop_resize":
            batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype)

        args_list = [
            (
                i,
                r,
                out_name,
                GOLIATH_CLASSES,
                GOLIATH_PALETTE,
                args.title,
                args.opacity,
            )
            for i, r, out_name in zip(
                cropped_images_np[:valid_images_len],
                result[:valid_images_len],
                batch_out_name,
            )
        ]
        img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
