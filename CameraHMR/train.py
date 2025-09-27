"Part of the code has been taken from "
"4DHumans: https://github.com/shubham-goel/4D-Humans"
from typing import Optional, Tuple
import os
from pathlib import Path

# Resolve project root from current file location instead of using git/pyproject
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.camerahmr_trainer import CameraHMR
from core.utils.pylogger import get_pylogger
from core.utils.misc import task_wrapper, log_hyperparameters
from pytorch_lightning.strategies import DDPStrategy
from core.utils.torch_compat import torch as _torch_compat  # registers safe globals on import
from core.configs import DATASET_FOLDERS
from mesh_estimator import HumanMeshEstimator
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("SAPIENS_NORMAL_CKPT", "SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2")
os.environ.setdefault("VGGT_CKPT", "../VGGT/vggt_1B_commercial.pt")
os.environ.setdefault("PI3_CKPT", "../Pi3/model.safetensors")
os.environ.setdefault("CAMERAHMR_CKPT", "data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt")

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    preprocess_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datamodule = DataModule(cfg, dataset_cfg)

    # Setup model
    model = CameraHMR(cfg)

    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor='val_loss'

    )
    rich_callback = pl.callbacks.TQDMProgressBar(refresh_rate=100)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        rich_callback
    ]

    trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            num_nodes=cfg.trainer.num_nodes,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            val_check_interval=cfg.trainer.val_check_interval,
            check_val_every_n_epoch=(cfg.trainer.check_val_every_n_epoch if hasattr(cfg.trainer, 'check_val_every_n_epoch') else 1),
            limit_val_batches=(cfg.trainer.limit_val_batches if hasattr(cfg.trainer, 'limit_val_batches') else 1.0),
            num_sanity_val_steps=(cfg.trainer.num_sanity_val_steps if hasattr(cfg.trainer, 'num_sanity_val_steps') else 0),
            precision=cfg.trainer.precision,
            max_steps=cfg.trainer.max_steps,
            logger=loggers,
            callbacks=callbacks,
            strategy=(cfg.trainer.strategy if hasattr(cfg.trainer, 'strategy') else DDPStrategy(find_unused_parameters=True)),
            sync_batchnorm=(cfg.trainer.sync_batchnorm if hasattr(cfg.trainer, 'sync_batchnorm') else False),
        )
    print("trainer.enable_validation", trainer.enable_validation)
    print(trainer)

    # Attach estimator_test(trainer, ...) method to run HumanMeshEstimator on TEST_DATASETS without val dataloader
    def _attach_estimator_test(_trainer: pl.Trainer, _model: CameraHMR, _cfg: DictConfig):
        class _TrainerModelWrapper:
            def __init__(self, trainer_ref: CameraHMR):
                self.trainer_ref = trainer_ref
                self.cfg = trainer_ref.cfg
            def eval(self):
                return self
            def __call__(self, batch):
                with torch.no_grad():
                    outputs, fl_h = self.trainer_ref.forward_step(batch, train=False)
                if self.cfg.MODEL.SMPL_HEAD.TYPE == 'transformer_decoder_gendered':
                    smpls = [o['pred_smpl_params'] for o in outputs]
                    cams = [o['pred_cam'] for o in outputs]
                    return smpls, cams, fl_h
                else:
                    out0 = outputs[0]
                    return out0['pred_smpl_params'], out0['pred_cam'], fl_h

        def _estimator_test(_model_unused=None, ckpt_path=None):
            wrapper = _TrainerModelWrapper(_model)
            estimator = HumanMeshEstimator(model=wrapper, cam_model=_model.cam_model)
            folders = []
            test_spec = getattr(_cfg.DATASETS, 'TEST_DATASETS', '') or ''
            for ds in [d for d in test_spec.split('_') if d]:
                img_dir = DATASET_FOLDERS.get(ds)
                if img_dir and os.path.isdir(img_dir):
                    folders.append(img_dir)
            # Fallback: skip if nothing configured
            if not folders:
                return
            epoch_idx = int(getattr(_trainer, 'current_epoch', 0) or 0)
            out_root = _cfg.paths.output_dir
            out_dir = os.path.join(out_root, f"ep_{epoch_idx}")
            os.makedirs(out_dir, exist_ok=True)
            for f in folders:
                estimator.run_on_images(f, out_dir, None)

        setattr(_trainer, 'estimator_test', _estimator_test)

    _attach_estimator_test(trainer, model, cfg)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)


    # Respect resume flag / explicit ckpt_path. Starting fresh avoids strict missing-keys errors when architecture changed.
    ckpt_path = None
    if getattr(cfg, 'ckpt_path', None):
        ckpt_path = cfg.ckpt_path
    elif getattr(cfg.GENERAL, 'RESUME', False):
        ckpt_path = 'last'

    # Optional: run a full validation pass before training starts (epoch 0 renders)
    if getattr(cfg.trainer, 'run_initial_validation', True):
        # move entire LightningModule to the selected device and switch to eval
        model = model.to(preprocess_device)
        model.set_model_mode('eval')
        if getattr(cfg.trainer, 'run_estimator_after_epoch', False):
            trainer.estimator_test(model, ckpt_path=ckpt_path)
        else:
            trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # back to train mode before fitting
    model.train()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=str((PROJECT_ROOT/"core"/"configs_hydra").resolve()), config_name="traintest.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
