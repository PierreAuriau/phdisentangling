#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:01:27 2022

@author: pa267054
"""

#Module import

import os
import os.path as op
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import quasi_raw_nii2npy

# path to neurospin
if op.exists('/neurospin'):
    prefixe = '/'
else:
    prefixe = '/home/pa267054'
    

# Parameters
study = "schizconnect-vip-prague"

regex = "derivatives/quasi-raw/sub-*/ses*/anat/*preproc-linear*.nii.gz"
qc_file = "derivatives/cat12-12.6_vbm_qc/qc.tsv"

output_path = op.join(prefixe, 'neurospin/dico/pauriau/data', study)


# Filename completion
study_dir = op.join(prefixe, "neurospin/psy_sbox", study)
nii_path = op.join(study_dir, regex)
phenotype_filename = op.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
dataset_name = study
qc = op.join(study_dir, qc_file)

# Array creation
quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=qc, sep='\t', id_type=str,
                 check = dict(shape=(182, 218, 182), zooms=(1, 1, 1)), resampling=1.5)

