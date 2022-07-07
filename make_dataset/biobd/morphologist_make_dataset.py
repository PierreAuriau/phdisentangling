#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:32:22 2022

@author: pa267054
"""

import os
from make_dataset_utils import skeleton_nii2npy
import pandas as pd

study = 'biobd'

if os.path.exists('/home/pa267054/neurospin'):
    pre = '/home/pa267054'
else:
    pre = '/'
study_dir = os.path.join(pre, 'neurospin', 'psy_sbox', study)


morpho_dir = os.path.join(study_dir, 'derivatives', 'morphologist-2021')
sbj_dir = os.path.join(morpho_dir, 'subjects')

Lregex = 'sub-*/t1mri/ses-*/default_analysis/segmentation/Lskeleton_sub-*.nii.gz'
Rregex = 'sub-*/t1mri/ses-*/default_analysis/segmentation/Rskeleton_sub-*.nii.gz'

output_path = os.path.join(pre, 'neurospin/psy_sbox/analyses/202205_predict_neurodev', study)
qc_file = None

# Filename completion
nii_path_l = os.path.join(study_dir, Lregex)
nii_path_r = os.path.join(study_dir, Rregex)
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
dataset_name = study
qc = os.path.join(study_dir, qc_file)

# Array creation
skeleton_nii2npy(nii_path_l, phenotype, dataset_name, output_path, qc=qc, sep='\t', id_type=str,
                 check = None, side='L')