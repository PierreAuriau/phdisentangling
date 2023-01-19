#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to a create a numpy array with all skeletons 
of the CNP dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

#Module import

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy

prefixe = "/"

# Directories
study = "cnp"

study_dir = os.path.join(prefixe, "neurospin", "psy_sbox", study)

output_dir = os.path.join(prefixe, "neurospin", "psy_sbox", "analyses", "202205_predict_neurodev", "data")

voxel_size = 1.5
#resampled_skeleton_dir = os.path.join(output_dir, study, "skeleton", str(voxel_size) +"mm")
resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm", "wo_ventricles")

### Creation of skeleton array ###

# Parameters
#regex = "F/Fresampled_full_skeleton_sub-*_ses-*.nii.gz"
regex = "F/Fresampled_skeleton_sub-*_ses-*.nii.gz"

nii_path = os.path.join(resampled_skeleton_dir, regex)

qc_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")
qc = {"cat12vbm": qc_filename}

#output_path = os.path.join(output_dir, "skeletons", study)
output_path = os.path.join(output_dir, "skeletons", study, "wo_ventricles")

side = "F"

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5)}


phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
participant_id = phenotype["participant_id"].tolist()
phenotype["participant_id"] = [i.replace("sub-", "") for i in participant_id]

phenotype = phenotype.rename(columns={"gender": "sex"})
phenotype = phenotype.replace(to_replace="M", value=1)
phenotype = phenotype.replace({"sex": {"F" : 0, "M": 1}})
phenotype["tiv"] = 0

phenotype["study"] = study.upper()
phenotype["site"] = study.upper()
assert phenotype["study"].isnull().values.any()
assert phenotype["site"].isnull().values.any()

phenotype["diagnosis"] = phenotype["diagnosis"].apply(lambda s: s.lower())
phenotype["diagnosis"] = phenotype["diagnosis"].replace({"schz": "schizophrenia"})

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

    