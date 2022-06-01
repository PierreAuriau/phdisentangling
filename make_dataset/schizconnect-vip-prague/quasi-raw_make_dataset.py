#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:01:27 2022

@author: pa267054

Study : Schizconnect-vip-prague
"""

#Module import

import os
import os.path as op
import sys

import numpy as np
import pandas as pd

sys.path.append(op.abspath(os.path.dirname(sys.argv[0])))
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

output_path = op.join(prefixe, '/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/schizconnect-vip-prague/', study)


# Filename completion
study_dir = op.join(prefixe, "neurospin/psy_sbox", study)
nii_path = op.join(study_dir, regex)
phenotype_filename = op.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
dataset_name = study
qc_filename = op.join(study_dir, qc_file)
qc = pd.read_csv(qc_filename, sep='\t')
qc['session'] = qc['session'].replace(1, 'v1')
phenotype['session'] = phenotype['session'].fillna('v1')

# Array creation
quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=qc, sep='\t', id_type=str,
                 check = dict(shape=(182, 218, 182), zooms=(1, 1, 1)), resampling=1.5)

