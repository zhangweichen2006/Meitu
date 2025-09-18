import os
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count


def _load_sapiens_model(checkpoint_path: str, use_torchscript: bool):
    if use_torchscript:
        return torch.jit.load(checkpoint_path)
    return torch.export.load(checkpoint_path).module()


def fake_pad_images_to_batchsize(imgs: torch.Tensor, target_bs: int) -> torch.Tensor:
    if imgs.shape[0] == target_bs:
        return imgs
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, target_bs - imgs.shape[0]), value=0)


@torch.no_grad()
def inference_model(model: nn.Module, imgs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    outputs = model(imgs.to(dtype=dtype, device=device))
    # Normalize return to list of per-image [C,H,W] tensors (as in vis_normal)
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 4:
            # [B, C, H, W] -> list([C,H,W])
            results = [t.detach().cpu() for t in outputs]
        else:
            results = [outputs.detach().cpu()]
    elif isinstance(outputs, (list, tuple)):
        results = [t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in outputs]
    else:
        raise TypeError(f"Unexpected model output type: {type(outputs)}")
    return results

class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape_hw=None, mean=None, std=None):
        self.image_list = image_list
        if shape_hw:
            assert len(shape_hw) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape_hw = shape_hw
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

    def __len__(self):
        return len(self.image_list)

    def _preprocess(self, img):
        if self.shape_hw:
            img = cv2.resize(img, (self.shape_hw[1], self.shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img

    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        orig_img = cv2.imread(orig_img_dir)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        return orig_img_dir, orig_img, img


class SapiensNormalWrapper:

    def __init__(
        self,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cuda:0",
        use_torchscript: Optional[bool] = None,
        fp16: bool = False,
        input_size_hw: Tuple[int, int] = None,
        compile_model: bool = False,
        batch_size: int = 32,
    ) -> None:
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)

        if use_torchscript is None:
            use_torchscript = "_torchscript" in os.path.basename(checkpoint_path)

        self.device = torch.device(device)
        self.use_torchscript = use_torchscript
        self.input_size_hw = input_size_hw
        self.batch_size = batch_size

        # dtype per vis_normal
        if self.use_torchscript:
            self.run_dtype = torch.float32
        else:
            self.run_dtype = torch.float16 if fp16 else torch.bfloat16

        model = _load_sapiens_model(checkpoint_path, use_torchscript)
        if not self.use_torchscript:
            model = model.to(self.run_dtype)
            if compile_model:
                try:
                    model = torch.compile(model, mode="max-autotune", fullgraph=True)
                except Exception:
                    pass
        self.model = model.to(self.device).eval()

    def build_dataloader(self, image_paths: List[str]) -> DataLoader:
        dataset = AdhocImageDataset(
            image_paths=image_paths,
            shape_hw=self.input_size_hw,
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(min(self.batch_size, cpu_count()), 1),
            pin_memory=True,
        )
        return loader

    @torch.no_grad()
    def infer_paths(self, image_paths: List[str]) -> List[torch.Tensor]:
        loader = self.build_dataloader(image_paths)
        all_results: List[torch.Tensor] = []
        for _, _, batch_imgs in loader:
            valid_images_len = batch_imgs.shape[0]
            padded_imgs = fake_pad_images_to_batchsize(batch_imgs, self.batch_size)
            result = inference_model(self.model, padded_imgs, device=self.device, dtype=self.run_dtype)
            # Trim to valid batch length and extend
            for r in result[:valid_images_len]:
                all_results.append(r)
        return all_results
