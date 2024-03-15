#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to a create a numpy array with all skeletons 
of the BSNIP1 dataset.
"""

#Module import
import logging
import os
import re
import pandas as pd
import numpy as np
# deep_folding
from deep_folding.brainvisa import generate_skeletons
from deep_folding.brainvisa import generate_ICBM2009c_transforms
from deep_folding.brainvisa import remove_ventricle
from deep_folding.brainvisa import resample_files
# Make dataset
from makedataset.logs import setup_logging
from makedataset.summary import make_morphologist_summary, \
                                  make_deep_folding_summary, \
                                  merge_skeleton_summaries, \
                                  ID_TYPES
from makedataset.nii2npy import skeleton_nii2npy
from makedataset.metadata import standardize_df

study = "bsnip1"

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")


# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study)
morpho_dir = os.path.join(study_dir, "derivatives", "morphologist-2021")
path_to_data = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data")

# Output directory where to put all generated files
output_dir = os.path.join(path_to_data, "skeletons", study)

# Parameters
voxel_size = "1.5"
junction = "thin" # "wide" or "thin"
side = "F" # "F", "L" or "R"
do_skel = "True" # apply skeletonisation
bids = True

# Filenames
labelling_session = "deepcnn_session_auto"
skeleton_filename = "skeleton_generated"
without_ventricle_skeleton_filename = "skeleton_without_ventricle"
resampled_skeleton_filename = "resampled_skeleton"

# For debugging, set parallel=False, number_subjects=1 and verbosity=True
parallel = True
number_subjects = "all" # all subjects
verbosity = True

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, "subjects")

# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, "raw")

# Relative path to graph for hcp dataset
path_to_graph = "t1mri/*/default_analysis/folds/3.1"

# Output directory where to put transform files
transform_dir = os.path.join(output_dir, "transforms")

# Output directory where to put skeleton files without ventricles
without_ventricle_skeleton_dir = os.path.join(output_dir, "without_ventricle", "raw")

# Output directory where to put resample skeleton files
resampled_skeleton_dir = os.path.join(output_dir, "without_ventricle", f"{voxel_size}mm")


# Morphologist summary
path_to_morpho_df = os.path.join(output_dir, "metadata", f"{study}_morphologist_summary.tsv")
if os.path.exists(path_to_morpho_df):
    morphologist_df = pd.read_csv(path_to_morpho_df, sep="\t", dtype=ID_TYPES)
else:
    morphologist_df = make_morphologist_summary(morpho_dir=src_dir,
                                                dataset_name=study,
                                                path_to_save=os.path.join(output_dir, "metadata"),
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
    argv.append("--verbose")

generate_ICBM2009c_transforms.main(argv)

# Generate skeletons from graphs
argv = ["--src_dir", src_dir,
        "--output_dir", skeleton_dir,
        "--path_to_graph", path_to_graph,
        "--side", side,
        "--junction", junction,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbose")

generate_skeletons.main(argv)

# Remove ventricles from the skeletons
argv = ["--src_dir", skeleton_dir,
        "--output_dir", without_ventricle_skeleton_dir,
        "--path_to_graph", path_to_graph,
        "--morpho_dir", src_dir,
        "--side", side,
        "--labelling_session", labelling_session,
        "--src_filename", skeleton_filename,
        "--output_filename", without_ventricle_skeleton_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbose")

remove_ventricle.main(argv)

# Resample skeletons in the ICBM2099c template
argv = ["--src_dir", without_ventricle_skeleton_dir,
        "--output_dir", resampled_skeleton_dir,
        "--side", side,
        "--input_type", "skeleton",
        "--transform_dir", transform_dir,
        "--out_voxel_size", voxel_size,
        "--do_skel", do_skel,
        "--src_filename", without_ventricle_skeleton_filename,
        "--output_filename", resampled_skeleton_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if verbosity:
    argv.append("--verbose")

resample_files.main(argv)

# deep_folding summary

df_directories = {"skeleton": skeleton_dir,
                  "transform": transform_dir,
                  "skeleton_without_ventricle": without_ventricle_skeleton_dir,
                  "resampled_skeleton": resampled_skeleton_dir
                  }
deep_folding_df = make_deep_folding_summary(deep_folding_directories=df_directories,
                                            side=side,
                                            dataset_name=study,
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

skel_df = merge_skeleton_summaries(morphologist_df, deep_folding_df,
                                   dataset_name=study,
                                   path_to_save=os.path.join(output_dir, "metadata"))

morphologist_qc = pd.read_csv(os.path.join(output_dir, "metadata", f"{study}_morphologist_qc.tsv"), sep="\t", dtype=ID_TYPES)

with open(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm", "list_t1w.txt"), "r") as f:
          selected_files = f.readlines()
for file in selected_files:
    sbj = re.search("sub-([^_/]+)", file)
    ses = re.search("ses-([^_/]+)", file)
    acq = re.search("acq-([^_/]+)", file)
    assert ses[1] == "V1", print("Wrong session")
    img = (morphologist_qc["participant_id"] == sbj[1])
    
    if len(acq[1]) > 3:
        morphologist_qc.loc[img, "acq"] = acq[1].replace(".", "")
    else:
        morphologist_qc.loc[img, "acq"] = acq[1].replace(".", "0")

morphologist_qc["ni_path"] = morphologist_qc.apply(lambda row: f"/neurospin/psy_sbox/bsnip1/derivatives/morphologist-2021/subjects/sub-{row['participant_id']}/t1mri/ses-{row['session']}_acq-{row['acq']}_run-{row['run']}/sub-{row['participant_id']}.nii.gz", axis=1)

for file in morphologist_qc["ni_path"].values:
    assert os.path.exists(file), print("Wrong filename")

morphologist_qc = morphologist_qc[['participant_id', 'session', 'acq', 'run', 'ni_path', 'qc', 'comments']]
morphologist_qc.to_csv(os.path.join(output_dir, "metadata", f"{study}_morphologist_qc.tsv"), index=False, sep="\t")

sbj2remove = morphologist_qc[morphologist_qc["qc"] == 0]

for index, sbj in sbj2remove.iterrows() :
    skel_df.loc[(skel_df["participant_id"] == sbj["participant_id"])
                 & (skel_df["session"] == sbj["session"])
                 & (skel_df["acq"] == sbj["acq"])
                 & (skel_df["run"] == sbj["run"]), ["qc", "comment"]] = morphologist_qc[["qc", "comments"]]

path2save = os.path.join(output_dir, "metadata", f"{study}_skeleton_qc.tsv")
skel_df.to_csv(path2save, sep="\t", index=False)

logger.info(f"Skeleton QC saved at : {path2save}")
"""

# MAKE SKELETON ARRAY

# Parameters
side = "F"
skeleton_size = True
stored_data = True
output_path = os.path.join(output_dir, "arrays", "stored_data")

# Nii_path
regex = f"{side}resampled_skeleton_sub-*_ses-*_acq-*_run-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)
# Check
check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

# Quality Checks
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
qc_vbm = standardize_df(qc_vbm, id_types=ID_TYPES)

qc_skel_filename = os.path.join(output_dir, "metadata", f"{study}_skeleton_qc.tsv")
qc_skel = pd.read_csv(qc_skel_filename, sep="\t", dtype=ID_TYPES)
assert qc_vbm["acq"].notnull().all(), logger.error("acq column of vbm qc file has nan values")
assert qc_skel["acq"].notnull().all(), logger.error("acq column of skeleton qc file has nan values")
qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session", "acq", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

# phenotype
# TODO : change filename into participants.tsv when updated
participants_filename = os.path.join(study_dir, '20231108_participants.tsv')
participants_df = pd.read_csv(participants_filename, sep='\t')
participants_df["participant_id"] = participants_df["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"
participants_df["study"] = participants_df["study"].apply(lambda s: s + "1") # Change BSNIP into BSNIP1
phenotype = standardize_df(participants_df, id_types=ID_TYPES)

assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["diagnosis"].notnull().values.all(), "diagnosis column in phenotype has nan values"


# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                 phenotype=phenotype, 
                 dataset_name=study, 
                 output_path=output_path, 
                 qc=qc, 
                 sep=',',
                 check=check,
                 data_type="float32",
                 id_types=ID_TYPES,
                 side=side,
                 skeleton_size=skeleton_size,
                 stored_data=stored_data)
