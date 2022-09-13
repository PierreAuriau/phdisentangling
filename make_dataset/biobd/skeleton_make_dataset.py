#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to a create a numpy array with all skeletons 
of the BIOBD dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

#Module import

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy

# path to neurospin : to delete ##
if os.path.exists('/neurospin'):
    prefixe = '/'
else:
    prefixe = '/home/pa267054'
##################################

# Directories
study = "biobd"

study_dir = os.path.join(prefixe, "neurospin", "psy_sbox", study)

output_dir = os.path.join(prefixe, "neurospin", "psy_sbox", "analyses", "202205_predict_neurodev", "data")

voxel_size = 1.5
resampled_skeleton_dir = os.path.join(output_dir, study, "skeleton", "full", str(voxel_size) +"mm")

### Creation of skeleton array ###

# Parameters
regex = "L/Lresampled_full_skeleton_sub-*_ses-*_run-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, regex)

qc = {"cat12vbm": os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv")}

output_path = os.path.join(output_dir, study, "skeleton")

side = "full"

#check = {"shape": (128, 152, 128), 
#         "voxel_size": (1.5, 1.5, 1.5)}

check = {"voxel_size": [1.5, 1.5, 1.5]} 

phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')

# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                 phenotype=phenotype, 
                 dataset_name=study, 
                 output_path=output_path, 
                 qc=qc, 
                 sep='\t', 
                 id_type=str,
                 check=check, 
                 side=side)

    