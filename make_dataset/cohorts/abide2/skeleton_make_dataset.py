#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a numpy array with all skeletons 
of the ABIDE2 dataset

Skeletons need to be pre-processed with the morphologist_make_dataset.py script

"""

import os
import pandas as pd
import numpy as np
# Make dataset
from makedataset.nii2npy import skeleton_nii2npy


study = 'abide2'

# Directories
neurospin = "/neurospin"

study_dir = os.path.join(neurospin, 'psy_sbox', study)

# Outputs
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", 
                          "data", "skeletons", study)
output_path = os.path.join(output_dir, "arrays", "morphologist_qc")

# Directory where resample skeleton files are
voxel_size = 1.5
resampled_skeleton_dir = os.path.join(output_dir, f"{voxel_size}mm")

# Nii path
side = "F"
regex = f"{side}resampled_skeleton_sub-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)

# Summaries
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
skeleton_size = True
stored_data = False
check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])
        }

# Quality Checks
vbm_qc_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv")
vbm_qc_file = pd.read_csv(vbm_qc_filename, sep='\t')
vbm_qc_file = vbm_qc_file[vbm_qc_file["run"] != "average"]
vbm_qc_file["run"] = vbm_qc_file["run"].astype(int)

skel_qc_filename = os.path.join(output_dir, "metadata", f"{study}_skeleton_qc.tsv")
skel_qc_file = pd.read_csv(skel_qc_filename, sep='\t')

qc = pd.merge(vbm_qc_file, skel_qc_file, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# Phenotype
participants_filename = os.path.join(study_dir, 'participants.tsv')
participants = pd.read_csv(participants_filename, sep='\t')
participants["study"] = participants["study"].fillna("ABIDE2")

roi_filename = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_roi", "cat12-12.6_vbm_roi.tsv")
roi = pd.read_csv(roi_filename, sep="\t")
roi["TIV"] = roi["TIV"].fillna(0)

phenotype = pd.merge(participants, roi, how="left", on="participant_id", validate="1:m")
phenotype = phenotype.rename(columns={"TIV": "tiv"})
phenotype["tiv"] = phenotype["tiv"].fillna(0)
phenotype["session"] = phenotype["session"].fillna(1)
phenotype["session"] = phenotype["session"].astype(int)

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
