from typing import Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
import os
from ..configs import DATASET_FOLDERS, DATASET_FILES
from .dataset_train import DatasetTrain
from .dataset_val import DatasetVal
from .dataset_test import DatasetTest
from .dataset_traintest import DatasetTrainTest
from .dataset_wai import DatasetWAI

class DataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:

        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None, **kwargs):
        if stage == 'train_test':
            self.train_test_dataset = self.train_test_dataset_prepare(**kwargs)
        else:
            # kwargs['is_train'] = True
            # kwargs['version'] = 'traintest'
            # kwargs['mean'] = self.cfg.MODEL.IMAGE_MEAN
            # kwargs['std'] = self.cfg.MODEL.IMAGE_STD
            # kwargs['cropsize'] = self.cfg.MODEL.IMAGE_SIZE
            self.train_dataset = self.train_dataset_prepare()
            # Use test dataset(s) for validation stage (render-only flows)
            if self.cfg.DATASETS.VAL_DATASETS:
                self.val_dataset = self.test_dataset_prepare()
            if self.cfg.DATASETS.TEST_DATASETS and not bool(getattr(self.cfg.trainer, 'run_estimator_after_epoch', True)):
                self.test_dataset = self.test_dataset_prepare()

    def train_dataset_prepare(self):
        if self.cfg.DATASETS.TRAIN_DATASETS:
            dataset_names = self.cfg.DATASETS.TRAIN_DATASETS.split('_')
            # Filter out datasets whose image folder or label file does not exist
            valid_datasets = []
            for ds in dataset_names:
                if ds.startswith('wai:'):
                    valid_datasets.append(ds)
                else:
                    img_dir = DATASET_FOLDERS.get(ds)
                    lbl_file = DATASET_FILES.get(ds) if isinstance(DATASET_FILES, dict) else None
                    if img_dir and os.path.isdir(img_dir) and lbl_file and os.path.isfile(lbl_file):
                        valid_datasets.append(ds)
                    else:
                        print(f"Train Dataset {ds} does not exist")
            dataset_list = [
                (DatasetWAI(self.cfg, ds, version='train', is_train=True) if ds.startswith('wai:') else DatasetTrain(self.cfg, ds, version='train'))
                for ds in valid_datasets
            ]
            train_ds = torch.utils.data.ConcatDataset(dataset_list)
            return train_ds
        else:
            return None

    def val_dataset_prepare(self):
        # Prefer explicit VAL_DATASETS; if empty, fall back to TEST_DATASETS for render-only validation
        val_spec = (self.cfg.DATASETS.VAL_DATASETS or '').strip()
        use_test_as_val = (len(val_spec) == 0)
        dataset_names = []
        if not use_test_as_val:
            dataset_names = [ds for ds in val_spec.split('_') if ds]
        else:
            test_spec = (getattr(self.cfg.DATASETS, 'TEST_DATASETS', '') or '').strip()
            dataset_names = [ds for ds in test_spec.split('_') if ds]

        valid_datasets = []
        for ds in dataset_names:
            if ds.startswith('wai:'):
                valid_datasets.append(ds)
            else:
                img_dir = DATASET_FOLDERS.get(ds)
                if img_dir and os.path.isdir(img_dir):
                    valid_datasets.append(ds)
                else:
                    print(f"Val/Test Dataset {ds} does not exist: {img_dir}")

        # Build dataset objects
        dataset_list = []
        for ds in valid_datasets:
            if ds.startswith('wai:'):
                dataset_list.append(DatasetWAI(self.cfg, ds, version='test', is_train=False))
            else:
                if use_test_as_val:
                    # Use DatasetTest (image-only, optional normals) when VAL is empty and TEST is provided
                    dataset_list.append(DatasetTest(self.cfg, ds, version='test', is_train=False))
                else:
                    dataset_list.append(DatasetVal(self.cfg, ds, version='test'))
        return dataset_list

    def test_dataset_prepare(self):
        dataset_names = self.cfg.DATASETS.TEST_DATASETS.split('_')
        dataset_list = [DatasetTest(self.cfg, ds, version='test', is_train=False) for ds in dataset_names]
        return dataset_list

    def train_test_dataset_prepare(self, **kwargs):
        if kwargs['is_train']:
            dataset_names = self.cfg.DATASETS.TRAIN_DATASETS.split('_')
        else:
            dataset_names = self.cfg.DATASETS.VAL_DATASETS.split('_')
        valid_datasets = []
        for ds in dataset_names:
            if ds.startswith('wai:'):
                valid_datasets.append(ds)
            else:
                img_dir = DATASET_FOLDERS.get(ds)
                lbl_map = DATASET_FILES['traintest'] if isinstance(DATASET_FILES, list) or isinstance(DATASET_FILES, dict) else DATASET_FILES
                lbl_file = lbl_map.get(ds) if isinstance(lbl_map, dict) else None
                if img_dir and os.path.isdir(img_dir) and lbl_file and os.path.isfile(lbl_file):
                    valid_datasets.append(ds)
                else:
                    print(f"Train (Train Test) Dataset {ds} does not exist")
        dataset_list = [
            (DatasetWAI(self.cfg, ds, **kwargs) if ds.startswith('wai:') else DatasetTrainTest(self.cfg, ds, **kwargs))
            for ds in valid_datasets
        ]
        return dataset_list

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True, num_workers=self.cfg.GENERAL.NUM_WORKERS, prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
        return {'img': train_dataloader}

    def val_dataloader(self):
        val_dataloaders = []
        for val_ds in self.val_dataset:
            val_dataloaders.append(torch.utils.data.DataLoader(val_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return val_dataloaders

    def test_dataloader(self):
        test_dataloaders = []
        for test_ds in self.test_dataset:
            test_dataloaders.append(torch.utils.data.DataLoader(test_ds, 1, drop_last=False, num_workers=1))
        return test_dataloaders

    def train_test_dataloader(self):
        train_test_dataloaders = []
        for train_test_ds in self.train_test_dataset:
            train_test_dataloaders.append(torch.utils.data.DataLoader(train_test_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=False, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return train_test_dataloaders
