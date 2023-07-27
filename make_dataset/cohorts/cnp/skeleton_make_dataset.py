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

sys.path.append(os.path.abspath('../..'))
from make_dataset_utils import skeleton_nii2npy

neurospin = "/neurospin"

# Directories
study = "cnp"

study_dir = os.path.join(neurospin, "psy_sbox", study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons")

voxel_size = 1.5

resampled_skeleton_dir = os.path.join(output_dir, study, str(voxel_size) +"mm", "wo_ventricles")

### Creation of skeleton array ###

# Parameters
side = "F"
skeleton_size = True
stored_data = False
#regex = "F/Fresampled_full_skeleton_sub-*_ses-*.nii.gz"
regex = f"{side}resampled_skeleton_sub-*_ses-*.nii.gz"

nii_path = os.path.join(resampled_skeleton_dir, side, regex)

output_path = os.path.join(output_dir, study, "arrays", "with_skeleton_size")

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}


# QUALITY CHECKS
qc_vbm = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
qc_skel = pd.read_csv(os.path.join(output_dir, study, "metadata", f"{study}_morphologist_qc.tsv"), sep="\t")

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# PHENOTYPE
vbm_participants = pd.read_csv(os.path.join(study_dir, f"{study.upper()}_t1mri_mwp1_participants.csv"), sep='\t')
phenotype = vbm_participants[["participant_id", "session", "diagnosis", "age", "sex", "TIV", "site", "study"]]
phenotype = phenotype.rename(columns={"TIV": "tiv"})

"""
vbm_participants = pd.read_csv(os.path.join(neurospin, "psy_sbox", "analyses", 
                                            "201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all",
                                            "data", "cat12vbm",
                                            f"{study}_t1mri_mwp1_participants.csv"))
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype["participant_id"] = phenotype["participant_id"].apply(lambda i: i.replace("sub-", ""))
phenotype = phenotype.rename(columns={"gender": "sex"})
phenotype = phenotype.replace({"sex": {"F" : 1, "M": 0}})
phenotype["diagnosis"] = phenotype["diagnosis"].apply(lambda s: s.lower())
phenotype["diagnosis"] = phenotype["diagnosis"].replace({"schz": "schizophrenia"})
phenotype["study"] = study.upper()
phenotype["site"] = study.upper()

id_type = str
phenotype["particpant_id"] = phenotype.astype(id_type)
vbm_participants["particpant_id"] = vbm_participants["participant_id"].astype(id_type)
vbm_participants = vbm_participants.set_index("participant_id")
vbm_participants = vbm_participants.reindex(index=phenotype["participant_id"])
vbm_participants = vbm_participants.reset_index()
phenotype["tiv"] = vbm_participants["tiv"].values
"""

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

    