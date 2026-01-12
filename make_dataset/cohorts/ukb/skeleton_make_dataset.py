#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the UK Biobank study :
1) create a numpy array for each subject from nifti images
2) binarize and pad each image

"""

#Module import
import logging
import os
import re
import sys
import glob
import pandas as pd
import numpy as np

from soma import aims, aimsalgo
from multiprocessing import Pool

# Make dataset
from makedataset.logs import setup_logging
from makedataset.summary import ID_TYPES
from makedataset.nii2npy import skeleton_nii2npy
from makedataset.metadata import standardize_df
from makedataset.metadata import load_qc, standardize_df, make_participants_df, merge_ni_df
from makedataset.output import output_skeleton

study = 'ukbiobank'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")

# Directories
neurospin = "/neurospin"
path_to_data = os.path.join(neurospin, "dico", "data", "deep_folding", "current", "datasets", "UkBioBank")
skeleton_dir = os.path.join(path_to_data, "skeletons")
voxel_size = 1.5
# resampled_skeleton_dir = os.path.join(skeleton_dir, f"{voxel_size}mm")
resampled_skeleton_dir = os.path.join(skeleton_dir, f"{voxel_size}mm", "without_ventricle")
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_pauriau_predict_neurodev", "data", "skeletons", study)

# MAKE SKELETON ARRAY

# Parameters
side = "F"
# regex = f"{side}resampled_skeleton_sub-*.nii.gz"
regex = f"{side}resampled_skeleton_without_ventricle_sub-*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)
# output_path = os.path.join(output_dir, "arrays", "without_ventricle")
output_path = os.path.join(neurospin, "psy_sbox", "analyses", "2024_pauriau_global_vs_local", "data", "global", study)
os.makedirs(output_path, exist_ok=True)

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}

# Quality Checks
qc = pd.read_csv(os.path.join(path_to_data, "qc_post.tsv"), sep="\t")
qc["participant_id"] = qc["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"
qc = standardize_df(qc)
logger.info(f"QC: nb sbj: {len(qc)} who pass the QC: {qc['qc'].sum()}")

# Phenotype
participants_filename = os.path.join(path_to_data, 'participants.csv')
participants_df = pd.read_csv(participants_filename, sep=",")
participants_df["site"] = participants_df["ImagingName"]
participants_df_sex_age = pd.read_csv(os.path.join(path_to_data, "participants_sex_age.csv"))
participants_df = pd.merge(participants_df, participants_df_sex_age, on=["participant_id", "Sex", "Age"], how="right", validate="1:1")
participants_df["participant_id"] = participants_df["participant_id"].apply(lambda s: re.search("sub-([a-zA-Z0-9]+)", s)[1]) #remove "sub-"
participants_df = pd.merge(participants_df, qc["participant_id"], on="participant_id", how="right", validate="1:1")
participants_df["study"] = "UKBiobank"
participants_df = standardize_df(participants_df, id_types=ID_TYPES)
logger.info(f"Participants df: nb sbj {len(participants_df)}")

# Array creation

# Loading QC
if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, ID_TYPES)

logger.info("# 1) Read all file names")
ni_filenames = glob.glob(nii_path)
logger.info(f'{len(ni_filenames)} ni files have been found')
ni_participants_df = make_participants_df(ni_filenames,
                                          id_regex='sub-([^/_\.]+)',
                                          session_regex='ses-([^_/\.]+)',
                                          acq_regex='acq-([^_/\.]+)',
                                          run_regex='run-([^_/\.]+)')
ni_participants_df = standardize_df(ni_participants_df, ID_TYPES)

logger.info("# 2) Merge nii's participant_id with participants dataframe")
logger.info(f"{len(participants_df)} subjects have a phenotype.")
ni_participants_df, _ = merge_ni_df(ni_participants_df, participants_df, qc=qc,
                                tiv_columns=[], participant_columns=[])
logger.info('--> Remaining samples: {} / {}'.format(len(ni_participants_df), len(ni_filenames)))

logger.info(f"# 3) Load {len(ni_participants_df)} nifti images and dump it in numpy array.")

def load_image_with_aims(filepath, check, output_path):
        # Load image and check header
        img = aims.read(filepath)
        voxel_size = np.array(img.header()["voxel_size"])[:3]
        shape = np.array(img.header()["volume_dimension"])[:3]
        transformation = np.array(img.header()["transformations"][-1])
        storage = np.array(img.header()["storage_to_memory"])
        assert np.all(voxel_size == check["voxel_size"]), \
                f"Wrong voxel size {voxel_size}"
        assert np.all(shape == check['shape']), \
                        f"Wrong image shape: {shape}"
        assert np.all(transformation == check['transformation']), \
                print(f"Wrong transformation: {transformation}")
        assert np.all(storage == check['storage']), \
                print(f"Wrong image storage transformation: {storage}")

        # Apply storage to memory transformation
        storage2memory = aims.AffineTransformation3d(img.header()["storage_to_memory"])
        translation = np.array([img.header()["storage_to_memory"][i] for i in range(3, 12, 4)])
        storage2memory.setTranslation(translation *voxel_size)
        memory2storage = aims.AffineTransformation3d(storage2memory).inverse()
        resampler = aims.ResamplerFactory_S16().getResampler(0) # Nearest-neghbours resampler
        resampler.setDefaultValue(0) # set background to 0
        resampler.setRef(img) # volume to resample
        stored_img = resampler.doit(memory2storage, *shape, voxel_size)

        # Transform into array
        arr = np.array(stored_img)
        arr = np.squeeze(arr) # remove time dimension
        arr[arr > 0] = 1 # binarize the skeleton
        arr = np.pad(arr, pad_width=((0,0), (4,4), (0,0))) # padd to (128, 160, 128)
        arr = arr.astype(np.float32) # change data type

        # assert np.all(arr.shape == (128, 160, 128)), "Wrong image shape"

        sbj = re.search("sub-([^_/.]+)", filepath).group(0)
        output_file = os.path.join(output_path, f"skeleton_{sbj}.npy")
        np.save(output_file, arr)

with Pool() as pool:
        list_args = [(filepath, check, output_path) for filepath in ni_participants_df["ni_path"]]
        pool.starmap(load_image_with_aims, list_args)


logger.info("# 4) Save the new participants dataframe")
ni_participants_df = ni_participants_df.drop("run", axis=1)
ni_participants_df = ni_participants_df.drop("session", axis=1)
ni_participants_df["participant_id"] = ni_participants_df["participant_id"].astype(str)

# Add path to array
for filepath in glob.glob(os.path.join(output_path, "*.npy")):
        sbj = re.search("sub-([^_/.]+)", filepath).group(1)
        ni_participants_df.loc[ni_participants_df["participant_id"] == sbj, "arr_path"] = filepath
ni_participants_df = standardize_df(ni_participants_df, ID_TYPES)

if os.path.exists(
        output_skeleton(study, output_path, modality="t1mri", mri_preproc="skeleton", dtype="participants",
                        ext="csv", side="")):
        answer = input(f"There is already a participants dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
                logger.warning("The nii array and the participant dataframe have not been saved.")
                sys.exist()
ni_participants_df.to_csv(output_skeleton(study, output_path, modality="t1mri", mri_preproc="skeleton", dtype="participants",
                          ext="csv", side=""), index=False, sep=",")
