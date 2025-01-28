#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the BIOBD study


"""

#Module import
import logging
import os
import json
import pandas as pd
import numpy as np

# Make dataset
from makedataset.logs import setup_logging
from makedataset.summary import ID_TYPES
from makedataset.nii2npy import skeleton_nii2npy
from makedataset.metadata import standardize_df

study = "biobd"

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")


# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study)
path_to_data = os.path.join(neurospin, "dico", "data", "deep_folding", "current", "datasets")
resampled_skeleton_dir = os.path.join(path_to_data, study, "skeletons", "without_ventricle", f"1.5mm")
resampled_skeleton_filename = "resampled_skeleton"
side = "F"

# Output directory where to put all generated files
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "2023_pauriau_diag_pred_from_sulci", "data")
raw_dir = os.path.join(output_dir, "raw")
interim_dir = os.path.join(output_dir, "interim")
processed_dir = os.path.join(output_dir, "processed")


# MAKE SKELETON ARRAY

# Parameters
regex = f"{side}{resampled_skeleton_filename}_*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

# Phenotype
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype["sex"] = phenotype["sex"].apply(lambda s: {0.0: "M", 0: "M", 1: "F", 1.0: "F"}[s])
phenotype = standardize_df(phenotype, id_types=ID_TYPES)
assert phenotype["study"].notnull().values.all(), logger.error("study column in phenotype has nan values")
assert phenotype["site"].notnull().values.all(), logger.error("site column in phenotype has nan values")
assert phenotype["tiv"].notnull().values.all(), logger.error("tiv column in phenotype has nan values")

# Quality checks

# VBM
qc_vbm = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv"), sep="\t")
qc_vbm = standardize_df(qc_vbm, id_types=ID_TYPES)
# Skeleton
qc_skel = pd.read_csv(os.path.join(path_to_data, study, "qc.tsv"), sep="\t", dtype=ID_TYPES)

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                 phenotype=phenotype, 
                 dataset_name=study, 
                 output_path=raw_dir,  
                 qc=qc, 
                 sep=',',
                 check=check,
                 data_type="float32",
                 id_types=ID_TYPES,
                 side=side,
                 skeleton_size=True,
                 stored_data=True)

# Binarization and padding
pd_files = "%s_t1mri_skeleton_participants.csv"
npy_files = "%s_t1mri_skeleton_data32.npy"
df = pd.read_csv(os.path.join(raw_dir, pd_files % study), dtype=ID_TYPES)
arr = np.load(os.path.join(raw_dir, npy_files % study))

# pad array from (128, 152, 128) --> (128, 160, 128)
img_shape = (128, 160, 128)
shape = arr.shape[2:]
pad_width = [(0,0), (0,0)]
for s, ns in zip(shape, img_shape):
    pad_width.append(((ns-s) //2, (ns-s)//2))
arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
print(f"New shape array: {arr.shape}")

# binarize skeletons
arr = (arr > 0).astype(np.float32)
print(f"New values of array : {np.unique(arr)}")

# save array
np.save(os.path.join(processed_dir, npy_files % study), arr)
print("New array save in processed directory")
# copy df
if not os.path.exists(os.path.join(processed_dir, pd_files % study)):
    os.symlink(os.path.join(raw_dir, pd_files % study), os.path.join(processed_dir, pd_files % study))
    print("Participant dataframe saved in processed directory")
