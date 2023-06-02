#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to create a numpy array with all skeletons 
of the ABIDE1 dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

#Module import

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))
from make_dataset_utils import skeleton_nii2npy

neurospin = "/neurospin"

# Study
study = "abide1"

study_dir = os.path.join(neurospin, "psy_sbox", study)

# Output
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", 
                          "data", "skeletons", study)
output_path = os.path.join(output_dir, "arrays", "qc_morphologist")

# Skeleton directory
voxel_size = 1.5
resampled_skeleton_dir = os.path.join(output_dir, f"{voxel_size}mm")

# Nii path
side = "F"
regex = f"{side}resampled_skeleton_sub-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)

"""
# Summaries
from make_skeleton_summary import make_morphologist_summary, make_deep_folding_summary, merge_skeleton_summaries
morphologist_df = make_morphologist_summary(morpho_dir=os.path.join(study_dir, "derivatives", "morphologist-2021", "subjects"), 
                                            dataset_name=study,
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            labelling_session="default_session_auto",
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

df_directories = {"skeleton": os.path.join(output_dir, "raw"),
                  "transform": os.path.join(output_dir, "transforms"),
                  "skeleton_without_ventricle": os.path.join(output_dir, "without_ventricle"),
                  "resampled_skeleton": os.path.join(output_dir, "1.5mm")
                  }
deep_folding_df = make_deep_folding_summary(deep_folding_directories=df_directories, 
                                            side=side, 
                                            dataset_name=study, 
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

df = merge_skeleton_summaries(morphologist_df, deep_folding_df, 
                              dataset_name=study, 
                              path_to_save=os.path.join(output_dir, "metadata"))
"""

# Parameters
check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])
        }
skeleton_size = True
stored_data = False

# Quality checks
qc_cat12vbm = pd.read_csv(os.path.join(study_dir, "derivatives", 
                                       "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
qc_skeleton = pd.read_csv(os.path.join(output_dir, "metadata", 
                                          f"{study}_skeleton_qc.tsv"), sep="\t")

qc = pd.merge(qc_cat12vbm, qc_skeleton, how="outer", on=["participant_id", "session"],
              validate="1:1", suffixes=("_cat12vbm", "_skeleton"))
qc["run"] = qc["run"].fillna(1)
qc["run"] = qc["run"].astype(int)
qc["qc_cat12vbm"] = qc["qc_cat12vbm"].fillna(0)
qc["qc"] = qc[["qc_cat12vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# Phenotype
participants_filename = os.path.join(study_dir, 'participants.tsv')
participants = pd.read_csv(participants_filename, sep='\t')

participants_diagnosis_filename = os.path.join(study_dir, "phenotype", 'participants_diagnosis.tsv')
participants_diagnosis = pd.read_csv(participants_diagnosis_filename, sep="\t")

participants_tiv_filename = os.path.join(study_dir, "phenotype", "participants_ROIS.tsv")
participants_tiv = pd.read_csv(participants_tiv_filename, sep="\t")

participants_merged = pd.merge(participants, participants_diagnosis, how="left", validate="1:1")

phenotype = pd.merge(participants_merged, participants_tiv, how="left", validate="1:1")
phenotype = phenotype.rename(columns={"TIV": "tiv"})
phenotype["tiv"] = phenotype["tiv"].fillna(0)

"""
# Add TIV
id_type = str
phenotype["particpant_id"] = phenotype["participant_id"].astype(id_type)
phenotype_tiv["particpant_id"] = phenotype_tiv["participant_id"].astype(id_type)
phenotype_tiv = phenotype_tiv.set_index("participant_id")
phenotype_tiv = phenotype_tiv.reindex(index=phenotype["participant_id"])
phenotype_tiv = phenotype_tiv.reset_index()
phenotype["tiv"] = phenotype_tiv["tiv"].values
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
