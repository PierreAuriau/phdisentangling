#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to a create a numpy array with all skeletons 
of the CANDI dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

#Module import

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy


# Directories
study = "candi"

neurospin = "/neurospin"

study_dir = os.path.join(neurospin, "psy_sbox", study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data")

voxel_size = 1.5
#resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm")
resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm", "wo_ventricles")

### Creation of skeleton array ###

# Parameters
#regex = "F/Fresampled_full_skeleton_sub-*_ses-*.nii.gz"
regex = "F/Fresampled_skeleton_sub-*_ses-*.nii.gz"

nii_path = os.path.join(resampled_skeleton_dir, regex)

qc = {"cat12vbm": os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")}

#output_path = os.path.join(output_dir, "skeletons", study)
output_path = os.path.join(output_dir, "skeletons", study, "wo_ventricles")

side = "F"

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5)}

phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')

diagnosis = []
participant_id = []
for sbj in phenotype["Subject"].tolist():
    if sbj.startswith("HC"):
        diagnosis.append("control")
    elif sbj.startswith("SS"):
        diagnosis.append("schizophrenia")
    elif sbj.startswith("BP"):
        diagnosis.append("bipolar disorder")
    else:
        raise ValueError
    participant_id.append(sbj.replace("_", ""))

phenotype = phenotype.rename(columns={"M/F": "sex"})
phenotype = phenotype.replace({"sex": {"F" : 0, "M": 1}})
phenotype["tiv"] = 0
phenotype["age"]= 0
phenotype["diagnosis"] = diagnosis
phenotype["participant_id"] = participant_id

phenotype["study"] = study.upper()

assert phenotype["study"].isnull().values.any()

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

    