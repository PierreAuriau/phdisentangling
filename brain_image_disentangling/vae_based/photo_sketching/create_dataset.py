# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# Folders where images have been downloaded
root = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/photo_sketching"
img_dir = os.path.join(root, "image")
sktch_dir = os.path.join(root, "sketch-rendered", "sketch-rendered", "width-5")
list_dir = os.path.join(root, 'list', 'list')

# Parameters
fine_size = 128 # final size of images and sketches
nb_sketches_per_image = 5

# Load and resize images of each data split
# Match each image with its 5 sketches
# Dump them into numpy arrays
for split in ("train", "val", "test"):
    # Load list of files
    list_file = os.path.join(list_dir, f'{split}.txt')
    with open(list_file) as f:
        content = f.readlines()
    list_img = sorted([x.strip() for x in content])
    N = len(list_img)
    print(f"Number of images in {split}: {N}")
    
    arr_pht = np.zeros((N*nb_sketches_per_image, fine_size, fine_size, 3))
    arr_sktch = np.zeros((N*nb_sketches_per_image, fine_size, fine_size))
    for i in tqdm(range(0, N), desc=split):
        # Load and resize image
        filename = list_img[i]
        pathA = os.path.join(img_dir, filename + '.jpg')
        pht = Image.open(pathA)
        pht = pht.resize((fine_size, fine_size), Image.BICUBIC)
        # Load and resize sketches
        for j in range(nb_sketches_per_image):
            pathB = os.path.join(sktch_dir, '%s_%02d.png' % (filename, j+1))
            sktch = Image.open(pathB)
            sktch = sktch.resize((fine_size, fine_size), Image.BICUBIC)

            index = i*nb_sketches_per_image + j
            arr_pht[index, :] = pht
            arr_sktch[index, :] = sktch

    # save arrays
    arr_pht = arr_pht.astype(int)
    arr_sktch = arr_sktch.astype(int)    
    np.save(os.path.join(root, f"{split}_photos.npy"), arr_pht)
    np.save(os.path.join(root, f"{split}_sketches.npy"), arr_sktch)
    # deallocate memory
    del arr_sktch, arr_pht
