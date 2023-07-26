import numpy as np
import pandas as pd
import copy
import torch
import logging
from collections import namedtuple

from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

from datasets.open_bhb import OpenBHB, SubOpenBHB, Callable, List
from datasets.bhb_10k import BHB
from datasets.temporary import TemporaryDataset
from datasets.clinical_multisites import SCZDataset, BipolarDataset, ASDDataset, SubSCZDataset, \
    SubBipolarDataset, SubASDDataset
from dl_training.self_supervision.sim_clr import SimCLROpenBHB, SimCLRSubOpenBHB
from dl_training.transforms import Padding, Crop, Normalize, Standardize, Binarize, GaussianConvolution, RandomRotation
from dl_training.preprocessing.combat import CombatModel

from dl_training.datamanager import ClinicalDataManager, BHBDataManager


SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class TemporaryDataManager(ClinicalDataManager):

    def __init__(self, root: str, preproc: str, db: str, labels: List[str] = None, sampler: str = "random",
                 batch_size: int = 1, number_of_folds: int = None, N_train_max: int = None, residualize: str = None,
                 mask=None, device: str = "cuda", input_transforms: List[str] = None, **dataloader_kwargs):

        assert db in ["scz", "bipolar", "asd"], "Unknown db: %s" % db
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        assert residualize in [None, "linear", "combat"], "Unkown residulalizer %s" % residualize

        self.logger = logging.getLogger("SMLvsDL")
        self.dataset = dict()
        self.labels = labels or []
        self.residualize = residualize
        self.mask = mask
        self.number_of_folds = number_of_folds
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

        # ComBat residualization (on site) attributes
        self.discrete_vars = ["sex", "diagnosis"]
        self.continuous_vars = ["age"]
        # Linear Adj. (on site) residualization attributes
        self.formula_res, self.formula_full = "site + age + sex", "site + age + sex + diagnosis"

        self.logger.debug("TemporaryDataset")

        self.transforms = input_transforms
        if N_train_max == 1:
            N_train_max = None
        self.logger.debug(f"N_train_max : {N_train_max}")
        dataset_cls = None
        if db == "scz":
            dataset_cls = SCZDataset if N_train_max is None else SubSCZDataset
        elif db == "bipolar":
            dataset_cls = BipolarDataset if N_train_max is None else SubBipolarDataset
        elif db == "asd":
            dataset_cls = ASDDataset if N_train_max is None else SubASDDataset

        self.logger.debug(f"Dataset CLS : {dataset_cls}")
        input_transforms = self.get_input_transforms(preproc=preproc, transforms=self.transforms, train=True)
        self.logger.debug(f"Train Input Transforms : {input_transforms}")
        if N_train_max is None:
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for _ in range(self.number_of_folds)]
        else:
            self.dataset["train"] = [dataset_cls(root, N_train_max=N_train_max, fold=f,
                                                 nb_folds=self.number_of_folds, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for f in range(self.number_of_folds)]
        input_transforms = self.get_input_transforms(preproc=preproc, transforms=self.transforms, train=False)
        self.logger.debug(f"Test Input Transforms : {input_transforms}")
        self.dataset["validation"] = [dataset_cls(root, preproc=preproc, split="val",
                                                  transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_folds)]
        self.dataset["test"] = dataset_cls(root, preproc=preproc, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, split="test_intra",
                                                 transforms=input_transforms, target=labels)

    @staticmethod
    def get_input_transforms(preproc, model="base", transforms=None, train=False):
        if preproc in ["vbm", "quasi_raw"]:
            # Input size 121 x 145 x 121
            input_transforms = Compose(
                [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()])
            if model == "SimCLR":
                from dl_training.self_supervision.sim_clr import DA_Module
                input_transforms.transforms.append(DA_Module())
        elif preproc in ["skeleton"]:
            # Input size 128 x 152 x 128
            input_transforms = Compose(
                [Padding([1, 128, 152, 128], mode='constant'), Binarize(one_values=[30, 35, 60, 100, 110, 120])])
            if transforms is not None:
                order = 0
                for tr in transforms:
                    if tr == "gaussian_convolution":
                        input_transforms.transforms.append(GaussianConvolution(size=5, sigma=1.0))
                        order = 1
                    elif tr == "normalisation":
                        input_transforms.transforms.append(Normalize())
                    elif tr == "random_gaussian_convolution":
                        if train:
                            input_transforms.transforms.append(
                                GaussianConvolution(size=5, sigma=(0.9, 1.1), random=True))
                        else:
                            input_transforms.transforms.append(GaussianConvolution(size=5, sigma=1.0))
                        order = 1
                    elif tr == "random_rotation":
                        if train:
                            input_transforms.transforms.append(
                                RandomRotation(angles=5, axes=[(0, 1), (0, 2), (1, 2)], reshape=False,
                                               mode='constant', cval=0.0, order=order))
                    elif tr == "random_rotation_probability":
                        if train:
                            input_transforms.transforms.append(
                                RandomRotation(angles=5, axes=[(0, 1), (0, 2), (1, 2)], reshape=False,
                                               probability=0.8, with_channels=True,
                                               mode='constant', cval=0.0, order=order))
                    else:
                        raise ValueError(f"Unknown input transform : {tr}")
        else:
            raise ValueError("Unknown preproc: %s" % preproc)
        return input_transforms
