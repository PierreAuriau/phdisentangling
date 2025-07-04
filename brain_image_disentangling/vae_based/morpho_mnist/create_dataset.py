# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import multiprocessing

sys.path.append("/home/pa267054/git/Morpho-MNIST")
from morphomnist import io, morpho, perturb, measure

# Single Processing
"""
## Data Loading
images = io.load_idx("/home/pa267054/data/MorphoMNIST/train-images-idx3-ubyte.gz")
labels = io.load_idx("/home/pa267054/data/MorphoMNIST/train-labels-idx1-ubyte.gz")

metadata = defaultdict(list)
images_1 = []
images_2 = []
for i, (img, label) in enumerate(zip(images, labels)):
    morphology = morpho.ImageMorphology(img, scale=4, label=label)
    img_1, centre = perturb.Fracture(thickness=4, num_frac=1, prune=2, return_skeleton=True, bias_sample=True)(morphology, return_centres=True)
    rand_num  = np.round(np.random.random() / + 0.7, 2)
    if rand_num < 1:
        perturbation = perturb.Thinning(amount=rand_num)
    else:
        perturbation = perturb.Thickening(amount=rand_num)
    img_2 = perturbation(morphology)

    img_1 = img_1.astype(int)
    img_2 = img_2.astype(int)
    images_1.append(np.pad(img_1, 8))
    images_2.append(np.pad(img_2, 8))
    metadata["label"].append(labels[i])
    metadata["fracture_x"].append(centre[0][0])
    metadata["fracture_y"].append(centre[0][1])
    metadata["swelling_amount"].append(rand_num)
    
    area, length, thickness, slant, width, height = measure.measure_image(img, scale=4, verbose=False)
    metadata["area"].append(area)
    metadata["length"].append(length)
    metadata["thickness"].append(thickness)
    metadata["slant"].append(slant)
    metadata["width"].append(width)
    metadata["height"].append(height)

    area, length, thickness, slant, width, height = measure.measure_image(img_1, scale=1, verbose=False)
    metadata["area_1"].append(area)
    metadata["length_1"].append(length)
    metadata["thickness_1"].append(thickness)
    metadata["slant_1"].append(slant)
    metadata["width_1"].append(width)
    metadata["height_1"].append(height)

    area, length, thickness, slant, width, height = measure.measure_image(img_2, scale=1, verbose=False)
    metadata["area_2"].append(area)
    metadata["length_2"].append(length)
    metadata["thickness_2"].append(thickness)
    metadata["slant_2"].append(slant)
    metadata["width_2"].append(width)
    metadata["height_2"].append(height)
    
metadata = pd.DataFrame(metadata)
images_1 = np.asarray(images_1).astype(int)
images_2 = np.asarray(images_2).astype(int)
print(metadata.head())

metadata.to_csv("/home/pa267054/data/MorphoMNIST/train-morpho.csv", sep=",", index=False)
np.save("/home/pa267054/data/MorphoMNIST/train-images-1.npy", images_1)
np.save("/home/pa267054/data/MorphoMNIST/train-images-2.npy", images_2)
"""

# Multiprocessing

## Data Loading
images = io.load_idx("/home/pa267054/data/MorphoMNIST/train-images-idx3-ubyte.gz")
labels = io.load_idx("/home/pa267054/data/MorphoMNIST/train-labels-idx1-ubyte.gz")


def process_img(i):
    img = images[i]
    label = labels[i]
    morphology = morpho.ImageMorphology(img, scale=4, label=label)

    if np.random.random() < 0.5:
        img_skel, centre = perturb.Fracture(thickness=4, num_frac=1, prune=2, return_skeleton=True, bias_sample=False)(morphology, return_centres=True)
        fracture = True
    else:
        img_skel = morphology.skeleton
        centre = [[None, None]]
        fracture = False
    rand_num  = np.round(np.random.random() / 2 + 0.8, 2)
    if rand_num < 1:
        perturbation = perturb.Thinning(amount=rand_num)
    else:
        perturbation = perturb.Thickening(amount=rand_num)
    img_img = perturbation(morphology)

    img_skel = img_skel.astype(int)
    img_img = img_img.astype(int)
    
    metadata = {}
    metadata["label"] = labels[i]
    metadata["fracture"].append(fracture)
    metadata["fracture_x"] = centre[0][0]
    metadata["fracture_y"] = centre[0][1]
    metadata["thickening_amount"] = rand_num
    
    area, length, thickness, slant, width, height = measure.measure_image(img, scale=4, verbose=False)
    metadata["area"] = area
    metadata["length"] = length
    metadata["thickness"] = thickness
    metadata["slant"] = slant
    metadata["width"] = width
    metadata["height"] = height

    area, length, thickness, slant, width, height = measure.measure_image(img_skel, scale=1, verbose=False)
    metadata["area_skel"] = area
    metadata["length_skel"] = length
    metadata["thickness_skel"] = thickness
    metadata["slant_skel"] = slant
    metadata["width_skel"] = width
    metadata["height_skel"] = height

    area, length, thickness, slant, width, height = measure.measure_image(img_img, scale=1, verbose=False)
    metadata["area_img"] = area
    metadata["length_img"] = length
    metadata["thickness_img"] = thickness
    metadata["slant_img"] = slant
    metadata["width_img"] = width
    metadata["height_img"] = height
    
    img_skel = np.pad(img_skel, 8)
    img_img = np.pad(img_img, 8)
    return img_skel, img_img, metadata


with multiprocessing.Pool() as pool:
    n = len(images)
    gen = pool.imap(process_img, [i for i in range(n)])
    gen = tqdm(gen, total=n, unit='img', ascii=True)
    images_skel, images_img, metadata = [], [], []
    for img_skel, img_img, data in gen:
        images_skel.append(img_skel)
        images_img.append(img_img)
        metadata.append(data)

images_skel = np.asarray(images_skel)
images_img = np.asarray(images_img)
metadata = pd.DataFrame(metadata)

print("Shape", images_skel.shape, images_img.shape, len(metadata))
print("Dtype", images_skel.dtype, images_img.dtype)
print("Order", np.all(metadata["label"].values == labels[:n]))

metadata.to_csv("/home/pa267054/data/MorphoMNIST/dataset-2/train-morpho.csv", sep=",", index=False)
np.save("/home/pa267054/data/MorphoMNIST/dataset-2/train-skeletons.npy", images_skel)
np.save("/home/pa267054/data/MorphoMNIST/dataset-2/train-images.npy", images_img)