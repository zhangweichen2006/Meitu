import os
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


def _load_sapiens_model(checkpoint_path: str, use_torchscript: bool):
    if use_torchscript:
        return torch.jit.load(checkpoint_path)
    # torch.export (ExportedProgram)
    return torch.export.load(checkpoint_path).module()


class SapiensNormalModel(pl.LightningModule):

    def __init__(
        self,
        checkpoint_path: str,
        use_torchscript: Optional[bool] = None,
        use_fp16: bool = False,
        input_size_hw: Optional[Tuple[int, int]] = (1024, 768),  # (H, W)
        compile_model: bool = False,
    ) -> None:
        super().__init__()

        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)

        if use_torchscript is None:
            use_torchscript = "_torchscript" in os.path.basename(checkpoint_path)

        self.use_torchscript = use_torchscript
        self.input_size_hw = input_size_hw

        # dtype policy follows SapiensLite demo: bf16 by default (or fp16 if requested)
        if self.use_torchscript:
            self.model_dtype = torch.float32
        else:
            self.model_dtype = torch.float16 if use_fp16 else torch.bfloat16

        model = _load_sapiens_model(checkpoint_path, use_torchscript)

        if not self.use_torchscript:
            model = model.to(self.model_dtype)
            if compile_model:
                try:
                    model = torch.compile(model, mode="max-autotune", fullgraph=True)
                except Exception:
                    pass
        self.model = model

        # SapiensLite demo mean/std are in 0..255 scale
        mean_255 = torch.tensor([123.5, 116.5, 103.5]).view(1, 3, 1, 1)
        std_255 = torch.tensor([58.5, 57.0, 57.5]).view(1, 3, 1, 1)
        self.register_buffer("_mean_255", mean_255, persistent=False)
        self.register_buffer("_std_255", std_255, persistent=False)

    @torch.no_grad()
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: expects key 'img' with shape [B, 3, H, W], RGB, float in [0,1] or [0,255].

        Returns:
            Float tensor of surface normals with shape [B, 3, H, W], values ~ in [-1, 1].
        """
        images: torch.Tensor = batch["img"]
        b, c, h, w = images.shape

        images = images.contiguous()

        # Scale to [0, 255] if inputs look like [0,1]
        with torch.no_grad():
            if images.max() <= 2.0:
                images = images * 255.0

        # Optional resize to the model's expected (H, W); Sapiens normal export is typically shape-fixed
        if self.input_size_hw is not None:
            target_h, target_w = self.input_size_hw
            if (h, w) != (target_h, target_w):
                images = F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Normalize using SapiensLite convention (0..255 scale)
        mean_255 = self._mean_255.to(device=images.device, dtype=images.dtype)
        std_255 = self._std_255.to(device=images.device, dtype=images.dtype)
        images = (images - mean_255) / std_255

        # Cast dtype as required by the backend
        run_dtype = self.model_dtype
        images = images.to(run_dtype)

        # Run model
        outputs = self.model(images)

        # Handle different possible output structures from export
        normals = self._coerce_to_bchw(outputs)

        # L2-normalize vector per pixel
        normals = F.normalize(normals, dim=1, eps=1e-5)

        # Resize back to original input size if we changed it
        if self.input_size_hw is not None and (h, w) != self.input_size_hw:
            normals = F.interpolate(normals, size=(h, w), mode="bilinear", align_corners=False)

        # Ensure float32 for downstream consumers
        return normals.to(dtype=torch.float32)

    def _coerce_to_bchw(self, outputs: Union[torch.Tensor, List[torch.Tensor], tuple]) -> torch.Tensor:
        """Coerce various return types into a [B, 3, H, W] tensor.
        The SapiensLite demo iterates over a list of per-image tensors; support that case too.
        """
        if isinstance(outputs, torch.Tensor):
            # Expect shape [B, 3, H, W] already
            return outputs
        if isinstance(outputs, (list, tuple)):
            # Could be list of per-image [3, H, W]
            if len(outputs) == 0:
                raise RuntimeError("Sapiens normal model returned empty outputs list")
            if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 3:
                return torch.stack(outputs, dim=0)
            # Or tuple/list with a single tensor payload
            if len(outputs) == 1 and isinstance(outputs[0], torch.Tensor):
                return outputs[0]
        if isinstance(outputs, dict):
            # Heuristic: try common keys
            for key in ("normal", "normals", "output", "out"):
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    return outputs[key]
        raise TypeError(f"Unsupported output type from Sapiens model: {type(outputs)}")

    # No device orchestration here: Lightning handles device/precision/ddp.
