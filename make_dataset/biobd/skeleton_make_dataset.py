#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 18 15:01:27 2022

@author: Pierre
"""

#Module import

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy

# path to neurospin
if os.path.exists('/neurospin'):
    prefixe = '/'
else:
    prefixe = '/home/pa267054'
    

# Parameters
study = "biobd"

data_dir = os.path.join(prefixe, 'neurospin', 'psy_sbox', 'analyses', '202205_predict_neurodev', 'data', study, 'skeleton')
regex = "raw/1.5mm/*/*resampled_skeleton_sub-*.nii.gz"
qc_file = {"vbm": "derivatives/cat12-12.6_vbm_qc/qc.tsv"}

output_path = os.path.join(prefixe, 'neurospin/psy_sbox/analyses/202205_predict_neurodev', 'data', study, 'skeleton')

 
# Filename completion
study_dir = os.path.join(prefixe, "neurospin", "psy_sbox", study)
nii_path = os.path.join(data_dir, regex)
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
dataset_name = study
qc = {}
for k, file in qc_file.items():
    qc[k] = os.path.join(study_dir, file)

# Array creation
skeleton_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=qc, sep='\t', id_type=str,
                 check=dict(), side="both")

