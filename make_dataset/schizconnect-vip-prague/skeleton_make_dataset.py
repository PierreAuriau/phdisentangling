#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:41:10 2022

@author: Pierre Auriau
"""
import os, sys
# Make dataset
sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy
import pandas as pd


study = 'schizconnect-vip-prague'

# Directories
neurospin = "/neurospin"

study_dir = os.path.join(neurospin, 'psy_sbox', study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data")

# Directory where resample skeleton files are
voxel_size = 1.5
#resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size)+"mm")
resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size)+"mm", "wo_ventricles")



### Creation of skeleton array ###

# Parameters
side = "F"

#regex = "?/?resampled_full_skeleton_sub-*_ses-*.nii.gz"
regex = f"{side}/{side}resampled_skeleton_sub-*_ses-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, regex)

#output_path = os.path.join(output_dir, "skeletons", study)
output_path = os.path.join(output_dir, "skeletons", study, "wo_ventricles")

qc_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")
qc_file = pd.read_csv(qc_filename, sep='\t')
qc_file["session"] = qc_file["session"].replace(1, "v1")

qc = {"vbm": qc_file}
check = {"shape": (128, 152, 128), 
         "zooms": (1.5, 1.5, 1.5)}

phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype["session"] = phenotype["session"].fillna("v1")

assert phenotype["study"].isnull().values.any()
assert phenotype["site"].isnull().values.any()

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
