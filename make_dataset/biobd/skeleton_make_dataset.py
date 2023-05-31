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

neurospin = "/neurospin"

# Directories
study = "biobd"

study_dir = os.path.join(neurospin, "psy_sbox", study)

output_dir = os.path.join(neurospin, "psy_sbox", "analyses", 
                          "202205_predict_neurodev", 
                          "data", "skeletons", study)

voxel_size = 1.5
side = "F"

resampled_skeleton_dir = os.path.join(output_dir, f"{voxel_size}mm")
### Summary ###
"""
from make_skeleton_summary import make_morphologist_summary, make_deep_folding_summary, merge_skeleton_summaries
morphologist_df = make_morphologist_summary(morpho_dir=os.path.join(study_dir, "derivatives", "morphologist-2021", "subjects"), 
                                            dataset_name=study,
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            labelling_session="deepcnn_session_auto",
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

df_directories = {"skeleton": os.path.join(output_dir, "raw"),
                  "transform": os.path.join(output_dir, "transforms"),
                  "skeleton_without_ventricle": os.path.join(output_dir, "without_ventricle")
                  #,
                  #"resampled_skeleton": os.path.join(output_dir, "1.5mm")
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
### Creation of skeleton array ###

# Parameters
stored_data=False
skeleton_size=False
regex = f"{side}resampled_skeleton_*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)
output_path = os.path.join(output_dir, "arrays", "with_tiv")

vbm_qc = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
#vbm_qc = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv"), sep="\t")

map_subject_run = json.load(open(os.path.join(output_dir, "metadata", "mapping_all_subjects_run.json"), "r"))
cnt = 0
for sbj in map_subject_run.keys():
    run = map_subject_run[sbj]["run"]   
    #print("RUN :", "previous", vbm_qc.loc[vbm_qc["participant_id"] == int(sbj), "run"].values, "now", run)
    vbm_qc.loc[vbm_qc["participant_id"] == int(sbj), "run"] = run
    if run == 1:
        cnt += 1
vbm_qc["run"] = vbm_qc["run"].astype(int)
skel_qc = pd.read_csv(os.path.join(output_dir, "metadata", "qc_morphologist.tsv"), sep="\t")

qc = pd.merge(vbm_qc, skel_qc, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)
#qc["session"] = qc["session"].fillna(1)
#qc["session"] = qc["session"].astype(int)

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])
        }

phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')

assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["TIV"].notnull().values.all(), "tiv column in phenotype has nan values"

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
