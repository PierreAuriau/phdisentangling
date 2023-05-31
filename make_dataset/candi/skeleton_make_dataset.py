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

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons")

voxel_size = 1.5
#resampled_skeleton_dir = os.path.join(output_dir, "skeletons", study, str(voxel_size) +"mm")
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

# QUALTIY CHECK
qc_vbm = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
qc_skel = pd.read_csv(os.path.join(output_dir, study, "metadata", f"{study}_morphologist_qc.tsv"), sep="\t")

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# PHENOTYPE
vbm_participants = pd.read_csv(os.path.join(study_dir,
                                            f"{study.upper()}_t1mri_mwp1_participants.csv"), sep="\t")
print(vbm_participants.columns)
phenotype = vbm_participants[["participant_id", "diagnosis", "age", "sex", "TIV", "site", "study"]].copy()
phenotype = phenotype.rename(columns={"TIV": "tiv"})

"""
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
vbm_participants = pd.read_csv(os.path.join(neurospin, "psy_sbox", "analyses", 
                                            "201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all",
                                            "data", "cat12vbm",
                                            f"{study}_t1mri_mwp1_participants.csv"))

subjects = {"participant_id": [], "diagnosis": [], "age": [], "tiv": []}
for i, sbj in enumerate(phenotype["Subject"].tolist()):
    if sbj.startswith("HC"):
        subjects["diagnosis"].append("control")
    elif sbj.startswith("SS"):
        subjects["diagnosis"].append("schizophrenia")
    elif sbj.startswith("BPDwPsy"):
        subjects["diagnosis"].append("bipolar disorder with psychosis")
    elif sbj.startswith("BPDwoPsy"):
        subjects["diagnosis"].append("bipolar disorder without psychosis")
    else:
        raise ValueError(f"Diagnosis unknown for the subject {sbj}")
    subjects["participant_id"].append(sbj.replace("_", ""))
    sbj_info = vbm_participants.loc[vbm_participants["participant_id"] == subjects["participant_id"][i], \
                                    ["age", "tiv"]]
    if len(sbj_info) == 1:
        subjects["age"].append(sbj_info["age"].iloc[0])
        subjects["tiv"].append(sbj_info["tiv"].iloc[0])
    elif len(sbj_info) == 0:
        subjects["age"].append(None)
        subjects["tiv"].append(None)
    else:
        if len(np.unique(sbj_info["age"].values)) == 1:
            subjects["age"].append(sbj_info["age"].iloc[0])
        else:
            raise ValueError(f"The subject {sbj} has different values for age")
        if len(np.unique(sbj_info["tiv"].values)) == 1:
            subjects["tiv"].append(sbj_info["tiv"].iloc[0])
        else:
            raise ValueError(f"The subject {sbj} has different values for TIV")
    
phenotype = phenotype.rename(columns={"M/F": "sex"})
phenotype = phenotype.replace({"sex": {"F" : 1, "M": 0}})
phenotype["tiv"] = subjects["tiv"]
phenotype["age"]= subjects["age"]
phenotype["diagnosis"] = subjects["diagnosis"]
phenotype["participant_id"] = subjects["participant_id"]
phenotype["study"] = study.upper()
phenotype["site"] = study.upper()
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

    