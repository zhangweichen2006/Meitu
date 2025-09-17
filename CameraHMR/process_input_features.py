"Part of the code has been taken from "
"4DHumans: https://github.com/shubham-goel/4D-Humans"
from typing import Optional, Tuple
import os
from pathlib import Path

# Resolve project root from current file location instead of using git/pyproject
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("SAPIENS_NORMAL_CKPT", "SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2")
os.environ.setdefault("VGGT_CKPT", "../VGGT/vggt.ckpt")
os.environ.setdefault("PI3_CKPT", "../Pi3/model.safetensors")

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.sapiens_normal_model import SapiensNormalModel
from core.pi3_decoder_model import Pi3Model
from core.vggt_decoder_model import VGGTModel
from core.utils.pylogger import get_pylogger
from core.utils.camera_ray_utils import calc_plucker_embeds
from core.utils.misc import task_wrapper, log_hyperparameters
from core.utils.torch_compat import torch as _torch_compat  # registers safe globals on import
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs (kept for reproducibility)
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # DataModule with validation splits
    datamodule = DataModule(cfg, dataset_cfg)
    datamodule.setup(stage='validate')

    # Resolve devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # Resolve checkpoints (env var takes precedence)
    sapiens_ckpt = os.environ.get('SAPIENS_NORMAL_CKPT', getattr(cfg, 'SAPIENS_NORMAL_CKPT', None))
    vggt_ckpt = os.environ.get('VGGT_CKPT', getattr(cfg, 'VGGT_CKPT', None))
    pi3_ckpt = os.environ.get('PI3_CKPT', getattr(cfg, 'PI3_CKPT', None))
    if sapiens_ckpt is None:
        raise RuntimeError('SAPIENS_NORMAL_CKPT not provided (env or cfg).')
    if vggt_ckpt is None:
        raise RuntimeError('VGGT_CKPT not provided (env or cfg).')
    if pi3_ckpt is None:
        raise RuntimeError('PI3_CKPT not provided (env or cfg).')

    # Build models for inference
    sapiens_normal_model = SapiensNormalModel(
        checkpoint_path=sapiens_ckpt,
        use_torchscript=None,
        use_fp16=False,
        input_size_hw=(1024, 768),
        compile_model=False,
    ).eval().to(device)
    vggt_model = VGGTModel(vggt_ckpt, device)
    pi3_model = Pi3Model(pi3_ckpt, device)

    # Helper to unnormalize to [0,1]
    mean = torch.tensor(cfg.MODEL.IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(cfg.MODEL.IMAGE_STD).view(1, 3, 1, 1)

    def unnormalize_to_unit(img: torch.Tensor) -> torch.Tensor:
        # img: [B,3,H,W] normalized by mean/std
        nonlocal mean, std
        m = mean.to(device=img.device, dtype=img.dtype)
        s = std.to(device=img.device, dtype=img.dtype)
        out = img * s + m
        return out.clamp(0.0, 1.0)

    # Evaluate over validation loaders
    val_loaders = datamodule.val_dataloader()
    results = []
    with torch.no_grad():
        for loader_idx, loader in enumerate(val_loaders):
            log.info(f"Evaluating loader {loader_idx+1}/{len(val_loaders)} with {len(loader.dataset)} samples")
            for batch in loader:
                # Inputs
                imgs_model = batch['img'].to(device, non_blocking=True)
                imgs_full = batch['img_full_resized'].to(device, non_blocking=True)
                imgs_sapiens = unnormalize_to_unit(imgs_full)

                # Ground truths (move lazily if used)
                gts = {
                    'keypoints_2d': batch.get('keypoints_2d'),
                    'keypoints_3d': batch.get('keypoints_3d'),
                    'vertices': batch.get('vertices'),
                    'smpl_params': batch.get('smpl_params'),
                    'cam_int': batch.get('cam_int'),
                }

                # Inference
                normals_pred = sapiens_normal_model({'img': imgs_sapiens})
                vggt_pred = vggt_model.forward(imgs_model)
                pi3_pred = pi3_model.forward(imgs_model)

                # Minimal bookkeeping (shapes)
                results.append({
                    'normals_shape': tuple(normals_pred.shape),
                    'vggt_shape': tuple(vggt_pred.shape) if isinstance(vggt_pred, torch.Tensor) else str(type(vggt_pred)),
                    'pi3_shape': tuple(pi3_pred.shape) if isinstance(pi3_pred, torch.Tensor) else str(type(pi3_pred)),
                })

    log.info(f"Evaluation completed. Num batches: {len(results)}")
    return {'results': results}, {}


@hydra.main(version_base="1.2", config_path=str((PROJECT_ROOT/"core"/"configs_hydra").resolve()), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    evaluate(cfg)


if __name__ == "__main__":
    main()
