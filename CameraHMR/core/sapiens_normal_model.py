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


class AdhocImageDataset(Dataset):

    def __init__(self, image_paths: List[str], shape_hw: Tuple[int, int], mean: List[float], std: List[float]):
        super().__init__()
        self.image_paths = image_paths
        self.shape_hw = shape_hw  # (H, W)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.shape_hw
        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_chw = img_resized.transpose(2, 0, 1).astype(np.float32)
        # Normalize in 0..255 domain using provided mean/std
        img_norm = (img_chw - self.mean) / self.std
        return os.path.basename(path), img_resized, torch.from_numpy(img_norm)


class SapiensNormalWrapper:

    def __init__(
        self,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cuda:0",
        use_torchscript: Optional[bool] = None,
        fp16: bool = False,
        input_size_hw: Tuple[int, int] = (1440, 1080),
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
