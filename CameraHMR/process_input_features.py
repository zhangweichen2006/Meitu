"Part of the code has been taken from "
"4DHumans: https://github.com/shubham-goel/4D-Humans"
from typing import Optional, Tuple
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.sapiens_normal_model import SapiensNormalWrapper
from core.pi3_decoder_model import Pi3Model
from core.vggt_decoder_model import VGGTModel
# from core.mapanything_model import MapAnythingModel
from core.utils.pylogger import get_pylogger
from core.utils.misc import task_wrapper, log_hyperparameters
from core.utils.torch_compat import torch as _torch_compat  # registers safe globals on import
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)


# Resolve project root from current file location instead of using git/pyproject
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("SAPIENS_NORMAL_CKPT", "SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2")
os.environ.setdefault("VGGT_CKPT", "../VGGT/vggt_1B_commercial.pt")
os.environ.setdefault("PI3_CKPT", "../Pi3/model.safetensors")
os.environ.setdefault("MAPANYTHING_CKPT", "../MapAnything/MapAnything_V1_3.pt")


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def evaluate(cfg: DictConfig, **kwargs) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs (kept for reproducibility)
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # DataModule; prepare validation/test splits for SAPIENS (no train augmentations)
    sapiens_datamodule = DataModule(cfg, dataset_cfg)
    sapiens_datamodule.setup(stage='train_test', is_train=False, mean=cfg.pretrained_models.sapiens.mean, std=cfg.pretrained_models.sapiens.std, cropsize=cfg.pretrained_models.sapiens.input_crop_size_hw)

    # Resolve devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # Resolve checkpoints (env var takes precedence)
    sapiens_ckpt = cfg.paths.get('sapiens_normal_ckpt', os.environ.get('SAPIENS_NORMAL_CKPT', None))
    vggt_ckpt = cfg.paths.get('vggt_ckpt', os.environ.get('VGGT_CKPT', None))
    pi3_ckpt = cfg.paths.get('pi3_ckpt', os.environ.get('PI3_CKPT', None))
    if sapiens_ckpt is None:
        raise RuntimeError('SAPIENS_NORMAL_CKPT not provided (env or cfg).')
    if vggt_ckpt is None:
        raise RuntimeError('VGGT_CKPT not provided (env or cfg).')
    if pi3_ckpt is None:
        raise RuntimeError('PI3_CKPT not provided (env or cfg).')

    # Build models for inference
    sapiens_normal_model = SapiensNormalWrapper(
        checkpoint_path=sapiens_ckpt,
        use_torchscript=cfg.pretrained_models.sapiens.use_torchscript,
        fp16=cfg.pretrained_models.sapiens.fp16,
        input_size_hw=cfg.pretrained_models.sapiens.input_crop_size_hw,
        compile_model=cfg.pretrained_models.sapiens.compile_model,
    ).model
    # vggt_model = VGGTModel(vggt_ckpt, device)
    # pi3_model = Pi3Model(pi3_ckpt, device)

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

    # Evaluate over test dataloaders (same as val, without train-time augmentation)
    test_loaders = sapiens_datamodule.train_test_dataloader()
    results = []
    with torch.no_grad():
        for loader_idx, loader in enumerate(test_loaders):
            log.info(f"Evaluating test loader {loader_idx+1}/{len(test_loaders)} with {len(loader.dataset)} samples")
            printed_keys = False
            for batch in loader:
                if not printed_keys:
                    print(list(batch.keys()))
                    printed_keys = True
                # Inputs
                imgs_model = batch['img'].to(device, non_blocking=True)
                # Ground truths (move lazily if used)
                # print(batch['smpl_normals'].shape)

                # Inference
                normals_list = sapiens_normal_model.forward(imgs_model)
                # vggt_pred = vggt_model.forward(imgs_model)
                # pi3_pred = pi3_model.forward(imgs_model)

                # Minimal bookkeeping (shapes)
                results.append({
                    'normals_shapes': [tuple(t.shape) for t in normals_list],
                    # 'vggt_shape': tuple(vggt_pred.shape) if isinstance(vggt_pred, torch.Tensor) else str(type(vggt_pred)),
                    # 'pi3_shape': tuple(pi3_pred.shape) if isinstance(pi3_pred, torch.Tensor) else str(type(pi3_pred)),
                })

                # save to local

    log.info(f"Evaluation completed. Num batches: {len(results)}")
    return {'results': results}, {}


@hydra.main(version_base="1.2", config_path=str((PROJECT_ROOT/"core"/"configs_hydra").resolve()), config_name="traintest.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # evaluate sapiens normal model
    evaluate(cfg)


if __name__ == "__main__":
    main()
