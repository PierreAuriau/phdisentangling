#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the AUSZ study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

Past study: <https://github.com/neurospin/scripts/blob/master/2016_AUSZ/2017/sept_2017/VBM/00_create_population.py>
"""

#Module import
import logging
import os
import glob
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

study = 'ausz'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")

# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study.upper())
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
verbosity = False

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, "subjects2")
logger.warning(f"Subject directory is subjects2 and will be changed to subjects in future.")

# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, "raw")

# Relative path to graph for hcp dataset
path_to_graph = "t1mri/*/default_analysis_without_bc/folds/3.1"
logger.warning(f"Path to graph : {path_to_graph}")

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
    logger.info("Loading previous morphologist summay")
else:
    morphologist_df = make_morphologist_summary(morpho_dir=src_dir,
                                                dataset_name=study,
                                                path_to_save=os.path.join(output_dir, "metadata"),
                                                labelling_session=labelling_session,
                                                analysis="default_analysis_without_bc",
                                                id_regex="sub-([^_/\.]+)",
                                                ses_regex="ses-([^_/\.]+)",
                                                acq_regex="acq-([^_/\.]+)",
                                                run_regex="run-([^_/\.]+)")

"""
# DEEP FOLDING
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

sbj2remove = morphologist_qc[morphologist_qc["qc"] == 0]

for index, sbj in sbj2remove.iterrows() :
    skel_df.loc[(skel_df["participant_id"] == sbj["participant_id"])
                        & (skel_df["session"] == sbj["session"]), ["qc", "comment"]] = morphologist_qc[["qc", "comments"]]

path2save = os.path.join(output_dir, "metadata", f"{study}_skeleton_qc.tsv")
skel_df.to_csv(path2save, sep="\t", index=False)

logger.info(f"Skeleton QC saved at : {path2save}")
"""
# MAKE SKELETON ARRAY

# Parameters
side = "F"
skeleton_size = True
stored_data = True
regex = f"{side}resampled_skeleton_sub-*_ses-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)
output_path = os.path.join(output_dir, "arrays", "without_nss_scores_from_tableau")

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

# Quality Checks
qc_vbm_filename = os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv")
qc_vbm = pd.read_csv(qc_vbm_filename, sep='\t')
qc_vbm = standardize_df(qc_vbm, id_types=ID_TYPES)
qc_vbm = qc_vbm.drop("run", axis=1) # remove run
qc_skel_filename = os.path.join(output_dir, "metadata", f"{study}_skeleton_qc.tsv")
qc_skel = pd.read_csv(qc_skel_filename, sep='\t', dtype=ID_TYPES)

qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session"], validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

print(qc.head())

# Phenotype
#path_to_participants = os.path.join(study_dir, "participants.tsv")
path_to_participants = os.path.join(study_dir, "participants_v-20240715.tsv")
logger.warning(f"You are using {path_to_participants} as participants dataframe.")
# remove sbj without Study Subject ID --> df3.StudySubjectID
"""
path_2 = "/neurospin/psy_sbox/AUSZ/sourcedata/AUSZ_2023_clinical_data_from_EmmaKrebs/data_nss_dev_filtre2.csv"
# Remove Age column (only Nan)
# Transform Sexe to str (with 0: h, 1: f) --> df3.Sex
# numerosujet --> df3.PersonID ?
# StudySubjectID --> df1.Study Subject ID
"""

path_to_score_nss = "/neurospin/psy_sbox/AUSZ/sourcedata/AUSZ_2022_clinical_data_from_AntonIftimovici/DataAUSZviaSPSS_Gilles_21102021.csv"
# Remove sbj without PersonID
# Transform PersonID into int --> df2.numerosujet ?

# Participant dataframe
participant_df = pd.read_csv(path_to_participants, sep="\t")
logger.info(f"participant_df: {len(participant_df)} subjects")
participant_df = participant_df[participant_df["Study Subject ID"].notnull()]
logger.info(f"participant_df: remove subjects without Study Subject ID -> {len(participant_df)} subjects left")

""" Oldies
ni_path = glob.glob("/neurospin/psy_sbox/AUSZ/rawdata/sub-*/ses-*/anat/sub-*_ses-*_T1w.nii.gz")
sbj_with_mri = [re.search("sub-([^_/.]+)", f).group(0) for f in ni_path]
participant_df = participant_df[participant_df["participant_id"].isin(sbj_with_mri)]
logger.info(f"participant_df: remove subjects without MRI -> {len(participant_df)} subjects left")
"""
# Dataframe with NSS scores
df_score_nss = pd.read_csv(path_to_score_nss, sep=",")
logger.info(f"df_score_nss: {len(df_score_nss)} subjects")
df_score_nss = df_score_nss[df_score_nss["StudySubjectID"].notnull()]
logger.info(f"df_score_nss: remove subjects without StudySubjectID -> {len(df_score_nss)} subjects left.")
# Filling NSS scores column 
# NB: not confident in NSS score from Tableau as it differs from those of DataAUSZviaSPSS_Gilles
"""
logger.info(f"{df_score_nss['NSS'].isnull().sum()} participants do not have NSS score.")
tableau = pd.read_excel(os.path.join(study_dir, "phenotype", "Tableau_IRM_AUSZ_MASC_NSS_BPRS_DTD_.xlsx"), 
                        sheet_name=0, engine="openpyxl")
df_merged = pd.merge(df_score_nss[['StudySubjectID', 'Sex', 'Age', 'NSS']],
                     tableau[['Ausz', 'Sexe', 'Âge', ' NSS', 'IRM.1']],
                     left_on='StudySubjectID', right_on='Ausz', 
                     how="inner", validate="1:1")
for _, row in df_merged[df_merged["NSS"].isnull()].iterrows():
    if pd.notnull(row[" NSS"]):
        if row["Sex"].capitalize() != row["Sexe"]: 
            logger.warning("Inconsistensy in the two dataframes in sex colum:", row["sex"], row["Sexe"])
        if row["Age"] != row["Âge"]:
            logger.warning(f"Inconsistency in the two dataframes in the age column: {row['Age']}, {row['Âge']}")
    df_score_nss.loc[df_score_nss["StudySubjectID"] == row["StudySubjectID"], "NSS"] = row[" NSS"]
logger.info(f"Filling 'NSS' column: {df_score_nss['NSS'].isnull().sum()} participants left without NSS score.")
"""
""" Oldies
# Add NSS scores from : /neurospin/psy_sbox/AUSZ/phenotype/Tableau_IRM_AUSZ_MASC_NSS_BPRS_DTD_.xlsx
for sbj, score in zip(["AZ-FF-01-061", "AZ-RH-01-105", "AZ-IA-01-116", "AZ-RC-01-132"],
                      [14.5, 5.0, 4.0, 4.5]):
    df_score_nss.loc[df_score_nss["StudySubjectID"] == sbj, "NSS"] = score
"""
df_score_nss = df_score_nss[df_score_nss["NSS"].notnull()]
logger.info(f"df_score_nss: remove subjects without NSS -> {len(df_score_nss)} subjects left")


df_merged = pd.merge(participant_df, df_score_nss, left_on="Study Subject ID", right_on="StudySubjectID", validate="1:1")
logger.info(f"df_merged: {len(df_merged)} subjects left.") 

"""
logger.info(f'Get subjects with MRI and not NSS : \
            {participant_df.loc[~(participant_df["Study Subject ID"].isin(df_merged["Study Subject ID"])), ["participant_id", "Study Subject ID"]]}')
logger.info(f'Get subjects with NSS and not MRI : \
            {df_score_nss.loc[~(df_score_nss["StudySubjectID"].isin(df_merged["StudySubjectID"])), ["StudySubjectID"]]}')
"""
""" Oldies
participants_filename = os.path.join(study_dir, '20231108_participants.tsv')
participants_df = pd.read_csv(participants_filename, sep='\t')
participants_df["participant_id"] = participants_df["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"
participants_df = participants_df.drop("session", axis=1) # remove session column
phenotype = standardize_df(participants_df, id_types=ID_TYPES)
"""
# add site column
df_merged["site"] = "AUSZ"
# remove "sub-"
df_merged["participant_id"] = df_merged["participant_id"].str.extract("sub-([a-zA-Z0-9]+)")[0]

# Filling age, sex and diagnosis columns
logger.info(f"{df_merged['age'].isnull().sum()} participants do not have age.")
df_merged.loc[df_merged["age"].isnull(), "age"] = df_merged.loc[df_merged["age"].isnull(), "Age"]
logger.info(f"Merging age and Age columns: {df_merged['age'].isnull().sum()} participants do not have age.")
logger.info(f"{df_merged['sex'].isnull().sum()} participants do not have sex.")
df_merged.loc[df_merged["sex"].isnull(), "sex"] = df_merged.loc[df_merged["sex"].isnull(), "Sex"]
df_merged["sex"] = df_merged["sex"].replace({"m": "M", "f": "F"})
logger.info(f"Merging sex and Sex columns: {df_merged['sex'].isnull().sum()} participants do not have sex.")
logger.info(f"{df_merged['diagnosis'].isnull().sum()} participants do not have diagnosis.")
# Group 3 ==> control
df_merged.loc[df_merged["participant_id"] == "SR160602", "diagnosis"] = "control"

phenotype = standardize_df(df_merged, id_types=ID_TYPES)
print(phenotype.head())
print(len(phenotype))

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
                  data_type="float32",
                  id_types=ID_TYPES,
                  check=check, 
                  side=side,
                  skeleton_size=skeleton_size,
                  stored_data=stored_data)
