from typing import Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
import os
from ..configs import DATASET_FOLDERS, DATASET_FILES
from .dataset_train import DatasetTrain
from .dataset_val import DatasetVal
from .dataset_traintest import DatasetTrainTest

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
            self.train_dataset = self.train_dataset_prepare()
            self.val_dataset = self.val_dataset_prepare()

    def train_dataset_prepare(self):
        if self.cfg.DATASETS.TRAIN_DATASETS:
            dataset_names = self.cfg.DATASETS.TRAIN_DATASETS.split('_')
            # Filter out datasets whose image folder or label file does not exist
            valid_datasets = []
            for ds in dataset_names:
                img_dir = DATASET_FOLDERS.get(ds)
                lbl_map = DATASET_FILES['train'] if isinstance(DATASET_FILES, list) or isinstance(DATASET_FILES, dict) else DATASET_FILES
                lbl_file = lbl_map.get(ds) if isinstance(lbl_map, dict) else None
                if img_dir and os.path.isdir(img_dir) and lbl_file and os.path.isfile(lbl_file):
                    valid_datasets.append(ds)
                else:
                    print(f"Train Dataset {ds} does not exist")
            dataset_list = [DatasetTrain(self.cfg, ds, version='train') for ds in valid_datasets]
            train_ds = torch.utils.data.ConcatDataset(dataset_list)
            return train_ds
        else:
            return None

    def val_dataset_prepare(self):
        dataset_names = self.cfg.DATASETS.VAL_DATASETS.split('_')
        valid_datasets = []
        for ds in dataset_names:
            img_dir = DATASET_FOLDERS.get(ds)
            lbl_map = DATASET_FILES['test'] if isinstance(DATASET_FILES, list) or isinstance(DATASET_FILES, dict) else DATASET_FILES
            lbl_file = lbl_map.get(ds) if isinstance(lbl_map, dict) else None
            if img_dir and os.path.isdir(img_dir) and lbl_file and os.path.isfile(lbl_file):
                valid_datasets.append(ds)
            else:
                print(f"Val Dataset {ds} does not exist")
        dataset_list = [DatasetVal(self.cfg, ds, version='test') for ds in valid_datasets]
        return dataset_list

    def train_test_dataset_prepare(self, **kwargs):
        dataset_names = self.cfg.DATASETS.TRAIN_DATASETS.split('_')
        valid_datasets = []
        for ds in dataset_names:
            img_dir = DATASET_FOLDERS.get(ds)
            lbl_map = DATASET_FILES['traintest'] if isinstance(DATASET_FILES, list) or isinstance(DATASET_FILES, dict) else DATASET_FILES
            lbl_file = lbl_map.get(ds) if isinstance(lbl_map, dict) else None
            if img_dir and os.path.isdir(img_dir) and lbl_file and os.path.isfile(lbl_file):
                valid_datasets.append(ds)
            else:
                print(f"Train (Train Test) Dataset {ds} does not exist")
        dataset_list = [DatasetTrainTest(self.cfg, ds, version='traintest', **kwargs) for ds in valid_datasets]
        return dataset_list

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True, num_workers=self.cfg.GENERAL.NUM_WORKERS, prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
        return {'img': train_dataloader}

    def val_dataloader(self):
        val_dataloaders = []
        for val_ds in self.val_dataset:
            val_dataloaders.append(torch.utils.data.DataLoader(val_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return val_dataloaders

    def train_test_dataloader(self):
        train_test_dataloaders = []
        for train_test_ds in self.train_test_dataset:
            train_test_dataloaders.append(torch.utils.data.DataLoader(train_test_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=False, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return train_test_dataloaders
