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
from makedataset.nii2npy import cat12_nii2npy
from makedataset.metadata import standardize_df

study = 'ausz'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_cat12_dataset")

# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study.upper())
cat12_dir = os.path.join(study_dir, "derivatives", "cat12-12.7_vbm")
path_to_data = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data")

# Output directory where to put all generated files
output_dir = os.path.join(path_to_data, "cat12vbm", study)


# MAKE CAT12VBM ARRAY

# Parameters
regex = "sub-*/ses-*/anat/mri/mwp1*.nii"
nii_path = os.path.join(cat12_dir, regex)
output_path = os.path.join(output_dir, "arrays", "without_nss_scores_from_tableau")

check = {"shape": (121, 145, 121), 
         "voxel_size": (1.5, 1.5, 1.5)}

# Quality Checks
qc_vbm_filename = os.path.join(study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv")
qc_vbm = pd.read_csv(qc_vbm_filename, sep='\t')
qc_vbm = standardize_df(qc_vbm, id_types=ID_TYPES)
qc_vbm = qc_vbm.drop("run", axis=1) # remove run

# Phenotype
path_to_participants = "/neurospin/psy_sbox/AUSZ/participants_v-20240715.tsv"
logger.warning("Using the participants_v-20240715.tsv dataframe")
# remove sbj without Study Subject ID --> df3.StudySubjectID
path_2 = "/neurospin/psy_sbox/AUSZ/sourcedata/AUSZ_2023_clinical_data_from_EmmaKrebs/data_nss_dev_filtre2.csv"
# Remove Age column (only Nan)
# Transform Sexe to str (with 0: h, 1: f) --> df3.Sex
# numerosujet --> df3.PersonID ?
# StudySubjectID --> df1.Study Subject ID

path_to_score_nss = "/neurospin/psy_sbox/AUSZ/sourcedata/AUSZ_2022_clinical_data_from_AntonIftimovici/DataAUSZviaSPSS_Gilles_21102021.csv"
# Remove sbj without PersonID
# Transform PersonID into int --> df2.numerosujet ?

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
for sbj, score in zip(["AZ-FF-01-061", "AZ-RH-01-105", "AZ-IA-01-116", "AZ-RC-01-132"],
                      [14.5, 5.0, 4.0, 4.5]):
    df_score_nss.loc[df_score_nss["StudySubjectID"] == sbj, "NSS"] = score
logger.info(f"{df_score_nss['NSS'].isnull().sum()} particpants do not have NSS score.")
"""
df_score_nss = df_score_nss[df_score_nss['NSS'].notnull()]
logger.info(f"--> Removing subjects without NSS score : {len(df_score_nss)} participants left.")


df_merged = pd.merge(participant_df, df_score_nss, left_on="Study Subject ID", right_on="StudySubjectID", validate="1:1")
logger.info(f"df_merged: {len(df_merged)} subjects left")
""" Oldies
participants_filename = os.path.join(study_dir, '20231108_participants.tsv')
participants_df = pd.read_csv(participants_filename, sep='\t')
participants_df["participant_id"] = participants_df["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"
participants_df = participants_df.drop("session", axis=1) # remove session column
"""
df_merged["site"] = "AUSZ"
df_merged["participant_id"] = df_merged["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"

# Add TIV column
df_merged["tiv"] = None
for sbj in df_merged["participant_id"]:
    try:
        path2report = f"/neurospin/psy_sbox/AUSZ/derivatives/cat12-12.7_vbm/sub-{sbj}/ses-1/anat/report/cat_usub-{sbj}_ses-1_T1w.xml"
        if not os.path.exists(path2report):
            raise FileNotFoundError(f"file not found {path2report}")
        df = pd.read_xml(path2report)
        vol_tiv = df["vol_TIV"].iloc[7]
        df_merged.loc[df_merged["participant_id"] == sbj, "tiv"] = vol_tiv
    except BaseException as e:
        logger.error(f"Unable to find the TIV of sub-{sbj}: {e}")
logger.info(f"df_score_nss: {df_merged['tiv'].isnull().sum()} participants do not have tiv.")
df_merged = df_merged[df_merged["tiv"].notnull()]
logger.info(f"--> Removing subjects without tiv : {len(df_merged)} participants left.")

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
cat12_nii2npy(nii_path=nii_path, 
                  phenotype=phenotype, 
                  dataset_name=study, 
                  output_path=output_path, 
                  qc=qc_vbm, 
                  sep=',',
                  data_type="float32",
                  id_types=ID_TYPES,
                  check=check)

