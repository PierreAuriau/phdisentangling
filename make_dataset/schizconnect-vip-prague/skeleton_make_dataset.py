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
import numpy as np

study = 'schizconnect-vip-prague'

# Directories
neurospin = "/neurospin"

study_dir = os.path.join(neurospin, 'psy_sbox', study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", 
                          "data", "skeletons", study)

# Directory where resample skeleton files are
voxel_size = 1.5
resampled_skeleton_dir = os.path.join(output_dir, str(voxel_size)+"mm", "wo_ventricles")

### Creation of skeleton array ###

# Parameters
side = "F"
skeleton_size = True
stored_data = False

#regex = "?/?resampled_full_skeleton_sub-*_ses-*.nii.gz"
regex = f"{side}resampled_skeleton_sub-*_ses-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)

output_path = os.path.join(output_dir, "arrays", "with_skeleton_size")

qc_vbm_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")
qc_vbm = pd.read_csv(qc_vbm_filename, sep='\t')
qc_vbm["session"] = qc_vbm["session"].replace(1, "v1")

qc_skel_filename = os.path.join(output_dir, "metadata", f"{study}_morphologist_qc.tsv")
qc_skel = pd.read_csv(qc_skel_filename, sep='\t')
qc_skel["session"] = qc_skel["session"].replace(1, "v1")

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype["session"] = phenotype["session"].fillna("v1")
phenotype = phenotype.rename(columns={"TIV": "tiv"})

assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["tiv"].notnull().values.all(), "tiv column in phenotype has nan values"

# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                  phenotype=phenotype, 
                  dataset_name=study, 
                  output_path=output_path, 
                  qc=qc, 
                  sep='\t', 
                  id_type=str,
                  check=check, 
                  side=side,
                  skeleton_size=skeleton_size,
                  stored_data=stored_data)
