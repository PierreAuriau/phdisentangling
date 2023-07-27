#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to a create a numpy array with all skeletons 
of the BSNIP1 dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

# Module import

import os
import sys
import re

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))
from make_dataset_utils import skeleton_nii2npy

neurospin = "/neurospin"

# Directories
study = "bsnip1"

study_dir = os.path.join(neurospin, "psy_sbox", study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", 
                          "data", "skeletons", study)

side = "F"
voxel_size = 1.5

output_path = os.path.join(output_dir, "arrays", "with_skeleton_size")
resampled_skeleton_dir = os.path.join(output_dir, str(voxel_size) +"mm", "wo_ventricles")

### Creation of skeleton array ###

# Parameters
skeleton_size = True
stored_data = False

# Nii_path
regex = f"{side}resampled_skeleton_sub-*_ses-*_acq-*_run-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)

# QUALITY CHECKS
qc_vbm_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")
qc_vbm = pd.read_csv(qc_vbm_filename, sep="\t")

qc_vbm["acq"] = None
with open(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm", "list_t1w.txt"), "r") as f:
          selected_files = f.readlines()
for file in selected_files:
    sbj = re.search("sub-([^_/]+)", file)
    ses = re.search("ses-([^_/]+)", file)
    acq = re.search("acq-([^_/]+)", file)
    assert np.all(qc_vbm.loc[qc_vbm["participant_id"] == sbj[1], ["session"]] == ses[1]),  \
        print(sbj[1], ses[1], "\n", qc_vbm.loc[qc_vbm["participant_id"] == sbj[1], ["session"]])
    if len(acq[1]) > 3:
        qc_vbm.loc[qc_vbm["participant_id"] == sbj[1], "acq"] = acq[1].replace(".", "")
    else:
        qc_vbm.loc[qc_vbm["participant_id"] == sbj[1], "acq"] = acq[1].replace(".", "0")

qc_vbm["session"] = qc_vbm["session"].replace("V1", 1)



qc_skel_filename = os.path.join(output_dir, "metadata", f"{study}_morphologist_qc.tsv")
qc_skel = pd.read_csv(qc_skel_filename, sep="\t")

mapping = np.vectorize(lambda p: re.search("acq-([^_/]+)", p)[1])
qc_skel["acq"] = mapping(qc_skel["ni_path"])

assert qc_vbm["acq"].notnull().all(), print("acq column of vbm qc file has nan values")
assert qc_skel["acq"].notnull().all(), print("acq column of skeleton qc file has nan values")


qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "acq", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)
qc.loc[qc['session'].isin(['v1', 'V1']), 'session'] = 1

# Check
check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

# PHENOTYPE
participants_filename = os.path.join(study_dir, 'participants.tsv')
participants = pd.read_csv(participants_filename, sep='\t')

vbm_roi_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_roi", "cat12-12.6_vbm_roi.tsv")
vbm_roi = pd.read_csv(vbm_roi_filename, sep="\t")
phenotype = pd.merge(participants, vbm_roi, how="left", on="participant_id", validate="1:m")
phenotype = phenotype.rename(columns={"TIV": "tiv"})
phenotype["tiv"] = phenotype["tiv"].fillna(0)
phenotype.loc[phenotype['session'].isin(['v1', 'V1']), 'session'] = 1

"""
vbm_participants = pd.read_csv(os.path.join(neurospin, "psy_sbox", "analyses", 
                                            "201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all",
                                            "data", "cat12vbm",
                                            f"{study}_t1mri_mwp1_participants.csv"))
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
