#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
import numpy as np

# Make dataset
from makedataset.logs import setup_logging
from makedataset.nii2npy import quasi_raw_nii2npy

study = 'hcp'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")

# Study directories
neurospin = '/neurospin'
study_dir = os.path.join(neurospin, 'psy_sbox', study)
morpho_dir = os.path.join(neurospin, 'dico', 'data', 'bv_databases', 'human', 'not_labeled', study)

# MAKE ARRAYS

# Parameters
nii_path = "/neurospin/psy_sbox/hcp/derivatives/quasi-raw/sub-*/ses-*/anat/sub-*_ses-*_preproc-linear_run-*_T1w.nii.gz"
output_path = "/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/quasiraw/hcp"

check = {"shape": (128, 145, 128), 
        "voxel_size": (1.5, 1.5, 1.5)
        }

# Phenotype
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
assert phenotype["study"].notnull().values.all(), logger.error("study column in phenotype has nan values")
assert phenotype["site"].notnull().values.all(), logger.error("site column in phenotype has nan values")
assert phenotype["tiv"].notnull().values.all(), logger.error("tiv column in phenotype has nan values")

# Quality checks
vbm_qc = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
path_to_qc = "/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/hcp"
skel_qc = pd.read_csv(os.path.join(path_to_qc, "metadata", f"{study}_skeleton_summary.tsv"), sep="\t")

qc = pd.merge(vbm_qc, skel_qc, how="outer", on=["participant_id"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)
qc = qc.drop(columns=["session", "run"])

# Array creation
quasi_raw_nii2npy(nii_path, phenotype, dataset_name=study, output_path=output_path, 
                  qc=qc, sep=',',
                  id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                  check={},
                  data_type="float32", 
                  id_regex="sub-([^_/\.]+)", session_regex='ses-([^_/\.]+)',
                  acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)',)
