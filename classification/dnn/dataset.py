#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:04:33 2022

@source : https://github.com/Duplums/bhb10k-dl-benchmark/tree/main/data
"""

from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from skimage import transform

from transformer import Transformer


class MRIDataset(Dataset):

    def __init__(self, config, args, training=False, validation=False, test=False, **kwargs):
        super().__init__(**kwargs)
        assert training + validation + test == 1

        self.transforms = Transformer(with_channel=True)
        self.config = config
        self.args = args
        # Crop+Pad images to have fixed dimension (1, 128, 128, 128)
        self.transforms.register(crop, probability=1.0, shape=(1, 121, 128, 121), with_channel=True)
        self.transforms.register(padding, probability=1.0, shape=(1, 128, 128, 128), with_channel=True)
        self.transforms.register(normalize, probability=1.0, with_channel=True)
        # if (not validation) and (not test):
        #     self.add_data_augmentations(self.transforms, args.da)
        if training:
            self.data = np.load(args.train_data_path)
            self.labels = pd.read_csv(args.train_label_path)
        elif validation:
            self.data = np.load(args.val_data_path)
            self.labels = pd.read_csv(args.val_label_path)
        elif test:
            self.data = np.load(args.test_data_path)
            self.labels = pd.read_csv(args.test_label_path)
    
    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)
        return (list_x, list_y)

    def __getitem__(self, idx):
        x = self.transforms(self.data[idx])
        labels = self.labels[self.args.labels].values[idx]

        return (x, labels)

    def __len__(self):
        return len(self.data)
    
def normalize(arr, mean=0.0, std=1.0, eps=1e-8):
    return std * (arr - np.mean(arr))/(np.std(arr) + eps) + mean
    
def padding(arr, shape, **kwargs):
    """Fill an array to fit the desired shape.
    :param
    arr: np.array
        an input array.
    **kwargs: params to give to np.pad (value to fill, etc.)
    :return
    fill_arr: np.array
        the padded array.
    """
    orig_shape = arr.shape
    padding = []
    for orig_i, final_i in zip(orig_shape, shape):
        shape_i = final_i - orig_i
        half_shape_i = shape_i // 2
        if shape_i % 2 == 0:
            padding.append([half_shape_i, half_shape_i])
        else:
            padding.append([half_shape_i, half_shape_i + 1])
    for cnt in range(len(arr.shape) - len(padding)):
        padding.append([0, 0])
    fill_arr = np.pad(arr, padding, **kwargs)
    return fill_arr


def crop(arr, shape, crop_type="center", resize=False, keep_dim=False):
    """Crop the given n-dimensional array either at a random location or centered
    :param
            shape: tuple or list of int
                The shape of the patch to crop
            crop_type: 'center' or 'random'
                Wheter the crop will be centered or at a random location
            resize: bool, default False
                If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
            keep_dim: bool, default False
                if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
    """
    assert isinstance(arr, np.ndarray)
    assert type(shape) == int or len(shape) == len(arr.shape), "Shape of array {} does not match {}". \
        format(arr.shape, shape)

    img_shape = np.array(arr.shape)
    if type(shape) == int:
        size = [shape for _ in range(len(shape))]
    else:
        size = np.copy(shape)
    indexes = []
    for ndim in range(len(img_shape)):
        if size[ndim] > img_shape[ndim] or size[ndim] < 0:
            size[ndim] = img_shape[ndim]
        if crop_type == "center":
            delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
        elif crop_type == "random":
            delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
        indexes.append(slice(delta_before, delta_before + size[ndim]))
    if resize:
        # resize the image to the input shape
        return transform.resize(arr[tuple(indexes)], img_shape, preserve_range=True)

    if keep_dim:
        mask = np.zeros(img_shape, dtype=np.bool)
        mask[tuple(indexes)] = True
        arr_copy = arr.copy()
        arr_copy[~mask] = 0
        return arr_copy

    return arr[tuple(indexes)]
    
# _____________________________________________________________________________________________________________________________ #

# _____________________________________________________________________________________________________________________________ #

