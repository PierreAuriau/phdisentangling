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

# Make dataset
from makedataset.logs import setup_logging
from makedataset.summary import ID_TYPES
from makedataset.nii2npy import freesurfer_concatnpy
from makedataset.metadata import standardize_df

study = 'ausz'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_freesurfer_dataset")

# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study.upper())
fs_dir = os.path.join(study_dir, "derivatives", "freesurfer")
path_to_data = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data")

# Output directory where to put all generated files
output_dir = os.path.join(path_to_data, "freesurfer", study)


# MAKE FREESURFER ARRAY

# Parameters
regex = "ses-1/sub-*/xhemi-textures.npy"
nii_path = os.path.join(fs_dir, regex)
output_path = os.path.join(output_dir, "arrays", "without_nss_scores_from_tableau")

check = {"shape": (4, 327684), 
         "channels": ('thickness', 'curv', 'area', 'sulc')}

# Quality Checks
qc_vbm_filename = os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv")
qc_vbm = pd.read_csv(qc_vbm_filename, sep='\t')
qc_vbm = standardize_df(qc_vbm, id_types=ID_TYPES)
qc_vbm = qc_vbm.drop("run", axis=1) # remove run

# Phenotype
path_to_participants = "/neurospin/psy_sbox/AUSZ/participants_v-20240715.tsv"
logger.warning("Using the participants_v-20240715.tsv dataframe")

path_to_score_nss = "/neurospin/psy_sbox/AUSZ/sourcedata/AUSZ_2022_clinical_data_from_AntonIftimovici/DataAUSZviaSPSS_Gilles_21102021.csv"

path_to_clinic = "/neurospin/psy_sbox/AUSZ/phenotype/Tableau_IRM_AUSZ_MASC_NSS_BPRS_DTD_.xlsx"

# Participant dataframe
participant_df = pd.read_csv(path_to_participants, sep="\t")
logger.info(f"participant_df: {len(participant_df)} subjects")
participant_df = participant_df[participant_df["Study Subject ID"].notnull()]
logger.info(f"participant_df: remove subjects without StudySubjectID -> {len(participant_df)} subjects left")

# Dataframe with NSS scores
df_score_nss = pd.read_csv(path_to_score_nss, sep=",")
logger.info(f"df_score_nss: {len(df_score_nss)} subjects")
df_score_nss = df_score_nss[df_score_nss["StudySubjectID"].notnull()]
logger.info(f"df_score_nss: remove subjects without StudySubjectID -> {len(df_score_nss)} subjects left.")
df_score_nss = df_score_nss[df_score_nss['NSS'].notnull()]
logger.info(f"--> Removing subjects without NSS score : {len(df_score_nss)} participants left.")

df_merged = pd.merge(participant_df, df_score_nss, left_on="Study Subject ID", right_on="StudySubjectID", validate="1:1")
logger.info(f"df_merged: {len(df_merged)} subjects left")
df_merged["site"] = "AUSZ"
df_merged["participant_id"] = df_merged["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"

# Filling age and sex columns
logger.info(f"{df_merged['age'].isnull().sum()} participants do not have age.")
df_merged.loc[df_merged["age"].isnull(), "age"] = df_merged.loc[df_merged["age"].isnull(), "Age"]
logger.info(f"Merging age and Age columns: {df_merged['age'].isnull().sum()} participants do not have age.")
logger.info(f"{df_merged['sex'].isnull().sum()} participants do not have sex.")
df_merged.loc[df_merged["sex"].isnull(), "sex"] = df_merged.loc[df_merged["sex"].isnull(), "Sex"]
logger.info(f"Merging sex and Sex columns: {df_merged['sex'].isnull().sum()} participants do not have sex.")
logger.info(f"{df_merged['diagnosis'].isnull().sum()} participants do not have diagnosis.")
# Group 3 ==> control
df_merged.loc[df_merged["participant_id"] == "SR160602", "diagnosis"] = "control"
logger.info(f"Filling diagnosis column : {df_merged['diagnosis'].isnull().sum()} participants left without diagnosis.")

phenotype = standardize_df(df_merged, id_types=ID_TYPES)
print(phenotype.head())
logger.info(f"Number of subjects left : {len(phenotype)}")
assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["diagnosis"].notnull().values.all(), "diagnosis column in phenotype has nan values"

# Array creation
freesurfer_concatnpy(npy_path=nii_path, 
                     phenotype=phenotype, 
                     dataset_name=study, 
                     output_path=output_path, 
                     qc=qc_vbm, 
                     sep=',',
                     dtype=np.float32,
                     id_types=ID_TYPES,
                     check=check)
