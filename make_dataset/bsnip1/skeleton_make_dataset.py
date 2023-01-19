#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to a create a numpy array with all skeletons 
of the BSNIP1 dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

#Module import

import os
import sys
import re

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy

prefixe = "/"

# Directories
study = "bsnip1"

study_dir = os.path.join(prefixe, "neurospin", "psy_sbox", study)

output_dir = os.path.join(prefixe, "neurospin", "psy_sbox", "analyses", "202205_predict_neurodev", "data")

side = "F"
voxel_size = 1.5
wo_ventricles = True

if wo_ventricles:
    output_path = os.path.join(output_dir, "skeletons", study, "wo_ventricles")
    resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm", "wo_ventricles")
else:
    output_path = os.path.join(output_dir, "skeletons", study)
    resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm")

### Creation of skeleton array ###

# Parameters
regex = "F/Fresampled_skeleton_sub-*_ses-*_acq-*_run-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, regex)

qc_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc_2018", "qc.tsv")
qc_file = pd.read_csv(qc_filename, sep="\t")

# A TESTER #
qc_file["acq"] = None
with open(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_2018", "list_t1w.txt"), "r") as f:
          selected_files = f.readlines()
for file in selected_files:
    sbj = re.search("sub-([^_/]+)", file)
    ses = re.search("ses-([^_/]+)", file)
    acq = re.search("acq-([^_/]+)", file)
    assert np.all(qc_file.loc[qc_file["participant_id"] == sbj[1], ["session"]] == ses[1]),  \
        print(sbj[1], ses[1], "\n", qc_file.loc[qc_file["participant_id"] == sbj[1], ["session"]])
    if len(acq[1]) > 3:
        qc_file.loc[qc_file["participant_id"] == sbj[1], ["acq"]] = acq[1].replace(".", "")
    else:
        qc_file.loc[qc_file["participant_id"] == sbj[1], ["acq"]] = acq[1].replace(".", "0")

qc_file["session"] = qc_file["session"].replace("V1", "1")

assert not qc_file["acq"].isna().any()
qc = {"cat12vbm": qc_file}

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5)}


phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype["tiv"] = 0

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

    