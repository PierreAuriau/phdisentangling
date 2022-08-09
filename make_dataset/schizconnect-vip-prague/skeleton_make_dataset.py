#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:41:10 2022

@author: pa267054
"""
import os, sys
# Make dataset
sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy
import pandas as pd

######## TO DELETE ########
if os.path.exists('/home/pa267054/neurospin'):
    prefixe = '/home/pa267054'
else:
    prefixe = '/'
###########################

study = 'schizconnect-vip-prague'

# Directories
study_dir = os.path.join(prefixe, 'neurospin', 'psy_sbox', study)

output_dir = os.path.join(prefixe, "neurospin", "psy_sbox", "analyses", "202205_predict_neurodev", "data")

# Directory wher resample skeleton files are
voxel_size = 1.5
resampled_skeleton_dir = os.path.join(output_dir, study, "skeleton", str(voxel_size)+"mm")



### Creation of skeleton array ###

# Parameters
regex = "raw/1.5mm/*/*resampled_skeleton_sub-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, regex)
qc = {"vbm": os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")}
output_path = os.path.join(output_dir, "root", "morphologist")
side = "both"
check = {"shape": (128, 152, 128), 
         "zooms": (1.5, 1.5, 1.5)}
 
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