import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _center_embed_or_crop(res_chw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    channels, height, width = res_chw.shape
    if out_h == height and out_w == width:
        return res_chw
    if out_h >= height:
        top = (out_h - height) // 2
        bottom = top + height
        canvas = torch.zeros((channels, out_h, out_w), dtype=res_chw.dtype, device=res_chw.device)
        if out_w >= width:
            left = (out_w - width) // 2
            right = left + width
            canvas[:, top:bottom, left:right] = res_chw
            return canvas
        else:
            left = (width - out_w) // 2
            right = left + out_w
            canvas[:, top:bottom, :] = res_chw[:, :, left:right]
            return canvas
    else:
        top = (height - out_h) // 2
        bottom = top + out_h
        if out_w >= width:
            left = (out_w - width) // 2
            right = left + width
            canvas = torch.zeros((channels, out_h, out_w), dtype=res_chw.dtype, device=res_chw.device)
            canvas[:, :, left:right] = res_chw[:, top:bottom, :]
            return canvas
        else:
            left = (width - out_w) // 2
            right = left + out_w
            return res_chw[:, top:bottom, left:right]


def _invert_cropping_portrait(
    res_chw: torch.Tensor, pre_h: int, pre_w: int, tgt_h: int, tgt_w: int, do_resize: bool
) -> torch.Tensor:
    h_ratio = tgt_h / pre_h
    w_ratio = tgt_w / pre_w
    resize_ratio = None
    if h_ratio > (4.0 / 3.0) or w_ratio > (4.0 / 3.0):
        resize_ratio = min(h_ratio, w_ratio)
    elif h_ratio < 0.75 or w_ratio < 0.75:
        resize_ratio = max(h_ratio, w_ratio)
    elif do_resize:
        resize_ratio = min(h_ratio, w_ratio)

    if resize_ratio is not None:
        new_h = int(pre_h * resize_ratio)
        new_w = int(pre_w * resize_ratio)
    else:
        new_h = pre_h
        new_w = pre_w

    inv_crop = _center_embed_or_crop(res_chw, new_h, new_w)
    if resize_ratio is not None and (new_h != pre_h or new_w != pre_w):
        return F.interpolate(inv_crop.unsqueeze(0), size=(pre_h, pre_w), mode="bilinear", align_corners=False).squeeze(0)
    return inv_crop


def _invert_cropping_landscape(
    res_chw: torch.Tensor, pre_h: int, pre_w: int, tgt_h: int, tgt_w: int, do_resize: bool, is_zoom: bool
) -> torch.Tensor:
    w_ratio = tgt_w / pre_w
    resized = False
    if do_resize or (w_ratio > (4.0 / 3.0)) or ((w_ratio < 0.75) and (not is_zoom)):
        new_w = tgt_w
        new_h = int(pre_h * w_ratio)
        resized = True
    else:
        new_w = pre_w
        new_h = pre_h

    undo_width = _center_embed_or_crop(res_chw, tgt_h, new_w)
    undo_vert = _center_embed_or_crop(undo_width, new_h, new_w)

    if resized and (new_h != pre_h or new_w != pre_w):
        return F.interpolate(undo_vert.unsqueeze(0), size=(pre_h, pre_w), mode="bilinear", align_corners=False).squeeze(0)
    return undo_vert


def _invert_zoom(res_chw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    half_h, half_w = target_h // 2, target_w // 2
    down = F.interpolate(res_chw.unsqueeze(0), size=(half_h, half_w), mode="bilinear", align_corners=False).squeeze(0)
    canvas = torch.zeros_like(res_chw)
    y0 = (target_h - half_h) // 2
    x0 = (target_w - half_w) // 2
    canvas[:, y0:y0 + half_h, x0:x0 + half_w] = down
    return canvas


def revert_npy(
    proc_npy_path: str,
    orig_image: np.ndarray,
    *,
    swapHW: bool = False,
    mode: str = "resize",
) -> np.ndarray:
    """Load a processed-space .npy and invert preprocessing to original image size.

    Returns an array shaped (H_orig, W_orig[, C]).
    """
    if type(proc_npy_path) == torch.Tensor:
        proc_arr = proc_npy_path.detach().cpu().numpy()
    elif type(proc_npy_path) == np.ndarray:
        proc_arr = proc_npy_path
    elif type(proc_npy_path) == str:

        if not os.path.exists(proc_npy_path):
            raise FileNotFoundError(f"Processed npy not found: {proc_npy_path}")
        proc_arr = np.load(proc_npy_path)
    else:
        raise ValueError(f"Unsupported type for revert {type(proc_npy_path)}: {proc_npy_path}")
    if proc_arr.ndim == 2:
        channels_first = torch.from_numpy(proc_arr).unsqueeze(0).float()
    elif proc_arr.ndim == 3 and proc_arr.shape[2] in (1, 3):
        channels_first = torch.from_numpy(proc_arr).permute(2, 0, 1).float()
    elif proc_arr.ndim == 3 and proc_arr.shape[0] in (1, 3):
        channels_first = torch.from_numpy(proc_arr).permute(1, 2, 0).float()
    else:
        raise ValueError(f"Unsupported array shape for revert: {proc_arr.shape}")

    proc_h, proc_w = proc_arr.shape[:2]

    if mode == "zoom_to_3Dpt":
        channels_first = _invert_zoom(channels_first, proc_h, proc_w)

    if type(orig_image) == str:
        orig_image = cv2.imread(orig_image)
        orig_image = orig_image[:, :, ::-1]
    elif type(orig_image) == np.ndarray:
        orig_image = orig_image
        if orig_image.ndim == 3 and orig_image.shape[0] == 3:
            orig_image = orig_image.permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported type for revert {type(orig_image)}: {orig_image}")
    orig_h, orig_w = orig_image.shape[:2]
    # Rotate first to final orientation to avoid large gaps with align_corners=False
    if swapHW:
        channels_first = torch.rot90(channels_first, k=1, dims=(1, 2))
    pre_h, pre_w = (orig_h, orig_w)
    tgt_h, tgt_w = proc_h, proc_w

    if mode in ["resize", "pad_resize"]:
        inv_pre = F.interpolate(channels_first.unsqueeze(0), size=(pre_h, pre_w), mode="bilinear", align_corners=False).squeeze(0)
    else:
        do_resize_flag = (mode == "crop_resize")
        is_zoom = (mode == "zoom_to_3Dpt")
        if pre_h > pre_w:
            inv_pre = _invert_cropping_portrait(channels_first, pre_h, pre_w, tgt_h, tgt_w, do_resize_flag)
        else:
            inv_pre = _invert_cropping_landscape(channels_first, pre_h, pre_w, tgt_h, tgt_w, do_resize_flag, is_zoom)

    # No rotation here; handled before resizing

    if inv_pre.shape[0] == 1:
        return inv_pre.squeeze(0).float().cpu().numpy()
    return inv_pre.float().cpu().numpy().transpose(1, 2, 0)


