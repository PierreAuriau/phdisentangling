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

import json

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
resampled_skeleton_dir = os.path.join(output_dir, study, "skeleton", "interim", str(voxel_size) +"mm")

### Creation of skeleton array ###

# Parameters
regex = "L/Lresampled_full_skeleton_sub-*_ses-*_run-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, regex)

qc_file = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")

map_subject_run = json.load(open("mapping_subject_run.json", "r"))

for sbj in map_subject_run.keys():
    run = map_subject_run[sbj]["run"]   
    qc_file.loc[qc_file["participant_id"] == int(sbj), ["run"]] = run

qc = {"cat12vbm": qc_file}

output_path = os.path.join(output_dir, study, "skeleton")

side = "F"

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5)}


phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')

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

    