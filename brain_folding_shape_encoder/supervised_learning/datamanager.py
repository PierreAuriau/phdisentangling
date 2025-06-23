# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import numpy as np
import pandas as pd
import logging
from typing import List, Union

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# project imports
from dataset import UKBDataset, ClinicalDataset
from data_augmentation import Cutout, Shift, Blur, ToTensor


class DataManager(object):

    def __init__(self, dataset: str, label: str = None, smooth_preproc: bool = False,
                 two_views: bool = False, data_augmentation: str = None):
        
        self.logger = logging.getLogger("datamanager")
        self.two_views = two_views
        self.label = label

        if data_augmentation == "cutout":
            tr = transforms.Compose([Cutout(patch_size=0.4, random_size=True,
                                            localization="on_data", min_size=0.1,
                                            img_size=(128, 160, 128)),
                                            ToTensor()])
        elif data_augmentation == "shift":
            tr = transforms.Compose([Shift(nb_voxels=1, random=True),
                                     Cutout(patch_size=0.4, random_size=True,
                                            localization="on_data", min_size=0.1,
                                            img_size=(128, 160, 128)),
                                            ToTensor()])
        elif data_augmentation == "blur":
            tr = transforms.Compose([Blur(sigma=1.0),
                                     Cutout(patch_size=0.4, random_size=True,
                                            localization="on_data", min_size=0.1),
                                     ToTensor()])
        elif data_augmentation == "all":
            tr = transforms.Compose([transforms.RandomApply([Blur(sigma=1.0),
                                                             Shift(nb_voxels=1, random=True)],
                                                            p=0.5),
                                     Cutout(patch_size=0.4, random_size=True,
                                            localization="on_data", min_size=0.1),
                                     ToTensor()])
        else:
            tr = ToTensor()

        if label == "sex":
            target_mapping = {"H": 0, "F": 1}
        elif label == "diagnosis":
            target_mapping = {"control": 0, 
                              "asd": 1,
                              "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                              "scz": 1}
        else:
            target_mapping = None
        
        self.dataset = dict()
        self.dataset["train"] = ClinicalDataset(split="train", label=label,
                                                dataset=dataset, transforms=tr,
                                                smooth_preproc=smooth_preproc,
                                                target_mapping=target_mapping)
        self.dataset["validation"] = ClinicalDataset(split="validation", label=label,
                                                     dataset=dataset, transforms=tr,
                                                     smooth_preproc=smooth_preproc,
                                                     target_mapping=target_mapping)
        self.dataset["test_intra"] = ClinicalDataset(split="test_intra", label=label,
                                                     dataset=dataset, transforms=tr,
                                                     smooth_preproc=smooth_preproc,
                                                     target_mapping=target_mapping)
        self.dataset["test"] = ClinicalDataset(split="test", label=label,
                                               dataset=dataset, transforms=tr,
                                               smooth_preproc=smooth_preproc,
                                               target_mapping=target_mapping)

    def get_dataloader(self, split, batch_size, **kwargs):
        dataset = self.dataset[split]
        drop_last = True if len(dataset) % batch_size == 1 else False
        if drop_last:
            self.logger.warning(f"The last subject of the {split} set will not be feed into the model ! "
                                f"Change the batch size ({batch_size}) to keep all subjects ({len(dataset)})")
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, **kwargs)
        loader.split = split
        return loader
        
    def __str__(self):
        return "DataManager"
