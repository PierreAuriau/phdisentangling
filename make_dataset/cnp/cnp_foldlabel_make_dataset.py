#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the CNP study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

"""

import os
import subprocess
import json
import pandas as pd
import numpy as np
# deep_folding
from deep_folding.brainvisa import generate_foldlabels
from deep_folding.brainvisa import generate_ICBM2009c_transforms 
from deep_folding.brainvisa import remove_ventricle
from deep_folding.brainvisa import resample_files 
# Make dataset
from make_dataset.make_skeleton_summary import make_morphologist_summary, \
                                  make_deep_folding_summary, \
                                  merge_skeleton_summaries
from make_dataset.make_dataset_utils import skeleton_nii2npy


study = 'cnp'

# Study directories
neurospin = '/neurospin'
study_dir = os.path.join(neurospin, 'psy_sbox', study)
morpho_dir = os.path.join(study_dir, "derivatives", "morphologist-2021")

# Output directory where to put all generated files
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "foldlabels", study)

# Path towards the deep_folding scripts
deep_folding_dir = os.path.join(neurospin, "dico", "pauriau", "git", "deep_folding", "brainvisa")

# Parameters
voxel_size = "1.5" #resampled voxel size
junction = "thin" # "wide" or "thin"
side = "F" # "F", "L" or "R"
bids = True

labelling_session = "deepcnn_session_auto"
foldlabel_filename = "foldlabel"
without_ventricle_foldlabel_filename = "foldlabel_without_ventricle"
resampled_foldlabel_filename = "resampled_foldlabel"

# For debugging, set parallel=False, number_subjects=1 and verbosity=True
parallel = True
number_subjects = "all" # all subjects
verbosity = False

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, "subjects")

# Output directory where to put raw foldlabel
foldlabel_dir = os.path.join(output_dir, "raw")

# Relative path to graph for hcp dataset
path_to_graph = "t1mri/*/default_analysis/folds/3.1"

# Output directory where to put transform files
transform_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "transforms")

# Output directory where to put foldlabel files without ventricles
without_ventricle_foldlabel_dir = os.path.join(output_dir, "without_ventricle", "raw")

# Output directory where to put resample foldlabel files
resampled_foldlabel_dir = os.path.join(output_dir, "without_ventricle", f"{voxel_size}mm")




# Morphologist summary
path_to_morpho_df =os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "metadata", f"{study}_morphologist_summary.tsv")
if os.path.exists(path_to_morpho_df):
    morphologist_df = pd.read_csv(path_to_morpho_df, sep="\t")
else:
    morphologist_df = make_morphologist_summary(morpho_dir=src_dir, 
                                                dataset_name=study,
                                                path_to_save=os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "metadata"),
                                                labelling_session=labelling_session,
                                                id_regex="sub-([^_/\.]+)",
                                                ses_regex="ses-([^_/\.]+)",
                                                acq_regex="acq-([^_/\.]+)",
                                                run_regex="run-([^_/\.]+)")

# DEEP FOLDING
"""
# Generate transform files
argv = ["--src_dir", src_dir, 
        "--output_dir", transform_dir, 
        "--path_to_graph", path_to_graph,
        "--side", side,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")

generate_ICBM2009c_transforms.main(argv)

# Generate foldlabels from graphs
argv = ["--src_dir", src_dir, 
        "--output_dir", foldlabel_dir, 
        "--path_to_graph", path_to_graph,
        "--side", side,
        "--junction", junction,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")
    
generate_foldlabels.main(argv)

# Remove ventricles from the foldlabels
argv = ["--src_dir", foldlabel_dir, 
        "--output_dir", without_ventricle_foldlabel_dir, 
        "--path_to_graph", path_to_graph,
        "--morpho_dir", src_dir,
        "--side", side,
        "--labelling_session", labelling_session,
        "--src_filename", foldlabel_filename,
        "--output_filename", without_ventricle_foldlabel_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity") 
    
remove_ventricle.main(argv)

# Resample foldlabels in the ICBM2099c template
argv = ["--src_dir", without_ventricle_foldlabel_dir, 
        "--output_dir", resampled_foldlabel_dir, 
        "--side", side,
        "--input_type", "foldlabel",
        "--transform_dir", transform_dir,
        "--out_voxel_size", voxel_size,
        "--src_filename", without_ventricle_foldlabel_filename,
        "--output_filename", resampled_foldlabel_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if verbosity:
    argv.append("--verbosity")

resample_files.main(argv)



# deep_folding summary
skeleton_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "raw")
without_ventricle_skeleton_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "raw", "wo_ventricles")
resampled_skeleton_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "1.5mm", "wo_ventricles")
df_directories = {"transform": transform_dir,
                  "skeleton": skeleton_dir,
                  "skeleton_without_ventricle":  without_ventricle_skeleton_dir,
                  "resampled_skeleton": resampled_skeleton_dir,
                  "foldlabel": foldlabel_dir,
                  "foldlabel_without_ventricle": without_ventricle_foldlabel_dir,
                  "resampled_foldlabel": resampled_foldlabel_dir
                  }
deep_folding_df = make_deep_folding_summary(deep_folding_directories=df_directories, 
                                            side=side, 
                                            dataset_name=study, 
                                            path_to_save=os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "metadata"),
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")
morphologist_df["participant_id"] = morphologist_df["participant_id"].astype(str)
deep_folding_df["participant_id"] = deep_folding_df["participant_id"].astype(str)
deep_folding_df["session"] = deep_folding_df["session"].astype(int)
df = merge_skeleton_summaries(morphologist_df, deep_folding_df, 
                              dataset_name=study, 
                              path_to_save=os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "metadata"))
"""
# MAKE ARRAYS

# Parameters
stored_data=False
skeleton_size=True
regex = f"{side}resampled_foldlabel_*.nii.gz"
nii_path = os.path.join(resampled_foldlabel_dir, side, regex)
output_path = os.path.join(output_dir, "arrays", "with_skeleton_size")

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])
        }

# PHENOTYPE
vbm_participants = pd.read_csv(os.path.join(study_dir, f"{study.upper()}_t1mri_mwp1_participants.csv"), sep='\t')
phenotype = vbm_participants[["participant_id", "session", "diagnosis", "age", "sex", "TIV", "site", "study"]]
phenotype = phenotype.rename(columns={"TIV": "tiv"})
assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["tiv"].notnull().values.all(), "tiv column in phenotype has nan values"

# QUALITY CHECKS
qc_vbm = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
qc_skel = pd.read_csv(os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeletons", study, "metadata", f"{study}_morphologist_qc.tsv"), sep="\t")

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                  phenotype=phenotype, 
                  dataset_name=study, 
                  output_path=output_path, 
                  side=side,
                  qc=qc, 
                  sep=',',
                  id_types={"participant_id": str, "session": int, "acq": int, "run":int},
                  check=check, 
                  data_type="float32",
                  preproc="foldlabel",
                  skeleton_size=skeleton_size,
                  stored_data=stored_data)

