
import os, sys
import json


import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

file_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.join(file_dir, '..', '..')
sys.path.append(project_root)  
from lib.utils.train_util import instantiate_from_config
from torch.utils.data import DataLoader 

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test

    def setup(self, stage):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def train_dataloader(self):
        sampler = DistributedSampler(self.datasets['train']) if torch.distributed.is_initialized() else None
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.datasets['validation']) if torch.distributed.is_initialized() else None
        return DataLoader(
            self.datasets['validation'],
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
