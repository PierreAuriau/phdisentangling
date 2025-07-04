# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import transforms

from kornia.augmentation import ColorJitter, RandomGrayscale, RandomResizedCrop, \
    RandomHorizontalFlip, RandomErasing, RandomAffine



class SimCLRDataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        # color distortion function
        """
        s = 0.5
        jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        random_jitter = transforms.RandomApply([jitter], p=0.8)
        random_greyscale = RandomGrayscale(p=0.2)
        color_distort = nn.Sequential(random_jitter, random_greyscale)
        """
        self.transforms = nn.Sequential(
            RandomResizedCrop((256, 256), scale=(0.2, 1), p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomErasing(scale=(0.1, 0.33), p=0.5),
            RandomAffine(degrees=1, translate=(0.1, 0.1), p=0.5)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)
        return x_out