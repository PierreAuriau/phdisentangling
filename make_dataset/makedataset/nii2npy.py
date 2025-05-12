# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions to create an array with all nii files and the associated participant dataframe.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import re
import glob

from makedataset.metadata import load_qc, standardize_df, make_participants_df, merge_ni_df
from makedataset.image import load_images, load_images_with_aims
from makedataset.output import output_cat12, output_quasi_raw, output_skeleton, output_dti, output_freesurfer
from makedataset.utils import global_scaling


def skeleton_nii2npy(nii_path, phenotype, dataset_name, output_path, side, qc=None,
                     id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                     check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),
                     data_type="float32", sep='\t', skeleton_size=False, stored_data=False,
                     id_regex='sub-([^/_\.]+)', session_regex='ses-([^_/\.]+)',
                     acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)', preproc="skeleton",
                     tiv_columns=[], participant_columns=[]):
        
    logger = logging.getLogger("skeleton_nii2npy")

    # Loading QC
    if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, id_types)
    """
    # Loading phenotype dataframe
    keys_required = {'participant_id', 'age', 'sex', 'diagnosis', 'site', 'study'}
    phenotype = standardize_df(phenotype, id_types)
    assert keys_required.issubset(phenotype.columns), \
        logger.error(f"Missing keys {set(keys_required) - set(phenotype.columns)} "
                     f"in phenotype Dataframe")
    # Remove participants with missing keys_required
    null_or_nan_mask = phenotype[keys_required].isna().any(axis=1)
    if null_or_nan_mask.sum() > 0:
        logger.warning(f'{null_or_nan_mask.sum()} participant_id will not be considered because of '
                       f'missing required values:\n{list(phenotype[null_or_nan_mask].participant_id.values)}')
    """
    null_or_nan_mask = np.array([False for _ in range(len(phenotype))])
    participants_df = phenotype[~null_or_nan_mask]
    
    # Loading nii files
    if side == "F":
        side = ""
    ni_filenames = glob.glob(nii_path)
    assert len(ni_filenames) > 0, \
        logger.error(f"No ni files have been found, wrong ni_path : {nii_path}")

    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    logger.info(f"# {dataset_name}")

    logger.info("# 1) Read all file names")
    logger.info(f'{len(ni_filenames)} ni files have been found')
    ni_participants_df = make_participants_df(ni_filenames,
                                              id_regex=id_regex,
                                              session_regex=session_regex,
                                              acq_regex=acq_regex,
                                              run_regex=run_regex)
    ni_participants_df = standardize_df(ni_participants_df, id_types)
    logger.info("# 2) Merge nii's participant_id with participants dataframe")
    logger.info(f"{len(participants_df)} subjects have a phenotype.")
    ni_participants_df, _ = merge_ni_df(ni_participants_df, participants_df, qc=qc,
                                        tiv_columns=tiv_columns, participant_columns=participant_columns)
    logger.info('--> Remaining samples: {} / {}'.format(len(ni_participants_df), len(ni_filenames)))

    logger.info(f"# 3) Load {len(ni_participants_df)} images")
    try:
        dtype = np.dtype(data_type)
    except TypeError:
        raise ValueError(f"Unknown value of data_type : {data_type}")
    ni_arr = load_images_with_aims(ni_participants_df, check=check, dtype=dtype, stored_data=stored_data)
    logger.info(f'--> {len(ni_participants_df)} img loaded')

    if skeleton_size:
        logger.info("# 3 bis) Compute skeleton size")
        skeleton_sizes = np.count_nonzero(ni_arr, axis=(1, 2, 3, 4))
        ni_participants_df["skeleton_size"] = skeleton_sizes

    logger.info("# 4) Save the new participants dataframe")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Output directory created : {output_path}")
    ni_participants_df = standardize_df(ni_participants_df, id_types)
    try:
        dico_ext = {"\t": "tsv", ",": "csv"}
        ext = dico_ext[sep]
    except KeyError:
        raise ValueError(f"Unknown value of separator : {sep}, \
                           accepted values are : {list(dico_ext.keys())}")

    if os.path.exists(
            output_skeleton(dataset_name, output_path, modality="t1mri", mri_preproc=preproc, dtype="participants",
                            ext=ext, side=side)):
        answer = input(f"There is already a participants dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array and the participant dataframe have not been saved.")
            return 0
    ni_participants_df.to_csv(
        output_skeleton(dataset_name, output_path, modality="t1mri", mri_preproc=preproc, dtype="participants",
                        ext=ext, side=side), index=False, sep=sep)

    logger.info("# 5) Save the raw npy file (with shape {})".format(ni_arr.shape))
    if os.path.exists(output_skeleton(dataset_name, output_path, modality="t1mri", mri_preproc=preproc,
                                      dtype=f"data{re.search('[0-9]+', data_type).group()}", ext="npy", side=side)):
        answer = input(f"There is already an array of nii files at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array has not been saved.")
            return 0
    np.save(output_skeleton(dataset_name, output_path, modality="t1mri", mri_preproc=preproc,
                            dtype=f"data{re.search('[0-9]+', data_type).group()}", ext="npy", side=side),
            ni_arr)

    # Deallocate the memory
    del ni_arr

def cat12_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t',
                  id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                  check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),
                  data_type="float32", 
                  id_regex='sub-([^/_\.]+)', session_regex='ses-([^_/\.]+)',
                  acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)',
                  tiv_columns=[], participants_columns=[]):
    
    logger = logging.getLogger("cat12_nii2npy")

    # Loading QC
    if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, id_types)

    # Loading phenotype dataframe
    keys_required = {'participant_id', 'age', 'sex', 'diagnosis', 'site', 'study'}
    phenotype = standardize_df(phenotype, id_types)
    assert keys_required.issubset(phenotype.columns), \
        logger.error(f"Missing keys {set(keys_required) - set(phenotype.columns)} "
                     f"in phenotype Dataframe")
    
    # Remove participants with missing keys_required
    null_or_nan_mask = phenotype[keys_required].isna().any(axis=1)
    if null_or_nan_mask.sum() > 0:
        logger.warning(f'{null_or_nan_mask.sum()} participant_id will not be considered because of '
                       f'missing required values:\n{list(phenotype[null_or_nan_mask].participant_id.values)}')
    participants_df = phenotype[~null_or_nan_mask]

    """
    # Save 3 files:
    participants_filename = output_cat12(dataset_name, output_path, mri_preproc='participants', ext='tsv')
    rois_filename = output_cat12(dataset_name, output_path, mri_preproc='rois', scaling="gs", ext='tsv')
    vbm_filename = output_cat12(dataset_name, output_path, mri_preproc='mwp1', scaling="gs", ext='npy')"""

    #  Read nii files
    ni_filenames = glob.glob(nii_path)
    assert len(ni_filenames) > 0, \
        logger.error(f"No ni files have been found, wrong ni_path : {nii_path}")

    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    logger.info(f"# {dataset_name}")

    logger.info("# 1) Read all file names")
    logger.info(f'{len(ni_filenames)} ni files have been found')

    ni_participants_df = make_participants_df(ni_filenames,
                                              id_regex=id_regex,
                                              session_regex=session_regex,
                                              acq_regex=acq_regex,
                                              run_regex=run_regex)
    ni_participants_df = standardize_df(ni_participants_df, id_types)

    logger.info("# 2) Merge nii's participant_id with participants dataframe")
    logger.info(f"{len(participants_df)} subjects have a phenotype.")
    ni_participants_df, ni_rois_df = merge_ni_df(ni_participants_df, participants_df, qc=qc,
                                        tiv_columns=tiv_columns, participant_columns=participants_columns)
    logger.info('--> Remaining samples: {} / {}'.format(len(ni_participants_df), len(ni_filenames)))
    logger.info('--> Remaining samples: {} / {}'.format(len(ni_rois_df), len(ni_filenames)))

    logger.info(f"# 3) Load {len(ni_participants_df)} images")
    try:
        dtype = np.dtype(data_type)
    except TypeError:
        raise ValueError(f"Unknown value of data_type : {data_type}")
    ni_arr = load_images(ni_participants_df, check=check, dtype=dtype)
    logger.info(f'--> {len(ni_participants_df)} img loaded')
    assert ni_arr.shape[0] == ni_participants_df.shape[0] == ni_rois_df.shape[0], "Unexpected nb of participants"

    logger.info("# 3bis) Global scaling of arrays and ROIs to adjust for TIV ")
    assert np.all(ni_rois_df["tiv"] == ni_participants_df["tiv"]), "rois['tiv'] !=  participants['tiv']"

    ni_arr = global_scaling(ni_arr, axis0_values=ni_participants_df["tiv"].values, target=1500)
    ni_arr = ni_arr.astype(dtype)

    # imgs_arr = global_scaling(imgs_arr, axis0_values=rois['TIV'].values, target=1500)
    gscaling = 1500 / ni_participants_df['tiv']
    ni_rois_df.loc[:, 'tiv':] = ni_rois_df.loc[:, 'tiv':].multiply(gscaling, axis="index")

    logger.info("# 4) Save the new participants dataframe")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Output directory created : {output_path}")
    ni_participants_df = standardize_df(ni_participants_df, id_types)
    ni_rois_df = standardize_df(ni_rois_df, id_types)
    try:
        dico_ext = {"\t": "tsv", ",": "csv"}
        ext = dico_ext[sep]
    except KeyError:
        raise ValueError(f"Unknown value of separator : {sep}, \
                         accepted values are : {list(dico_ext.keys())}")

    if os.path.exists(
            output_cat12(dataset_name, output_path, mri_preproc='participants', ext=ext)):
        answer = input(f"There is already a participants dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array and the participant dataframe have not been saved.")
            return 0
    ni_participants_df.to_csv(
        output_cat12(dataset_name, output_path, mri_preproc='participants', ext=ext), index=False, sep=sep)
    
    logger.info("# 4 bis) Save the rois dataframe")
    if os.path.exists(
            output_cat12(dataset_name, output_path, mri_preproc='rois', scaling="gs", ext=ext)):
        answer = input(f"There is already a rois dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array and the participant dataframe have not been saved.")
            return 0
    ni_rois_df.to_csv(
        output_cat12(dataset_name, output_path, mri_preproc='rois', scaling="gs", ext=ext), index=False, sep=sep)

    logger.info("# 5) Save the raw npy file (with shape {})".format(ni_arr.shape))
    if os.path.exists(output_cat12(dataset_name, output_path, mri_preproc='mwp1', scaling="gs", ext='npy')):
        answer = input(f"There is already an array of nii files at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array has not been saved.")
            return 0
    np.save(output_cat12(dataset_name, output_path, mri_preproc='mwp1', scaling="gs", ext='npy'),
            ni_arr)

    # Deallocate the memory
    del ni_arr

def freesurfer_concatnpy(npy_path, phenotype, dataset_name, output_path, qc=None, sep='\t',
                         id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                         check=dict(shape=(4, 327684), channels=("thickness", "curv", "area", "sulc")),
                         dtype=np.float32, 
                         id_regex='sub-([^/_\.]+)', session_regex='ses-([^_/\.]+)',
                         acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)',
                         tiv_columns=[], participants_columns=[]):
    
    logger = logging.getLogger("freesurfer")

    # Loading QC
    if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, id_types)

    # Loading phenotype dataframe
    keys_required = {'participant_id', 'age', 'sex', 'diagnosis', 'site', 'study'}
    phenotype = standardize_df(phenotype, id_types)
    assert keys_required.issubset(phenotype.columns), \
        logger.error(f"Missing keys {set(keys_required) - set(phenotype.columns)} "
                     f"in phenotype Dataframe")
    
    # Remove participants with missing keys_required
    null_or_nan_mask = phenotype[keys_required].isna().any(axis=1)
    if null_or_nan_mask.sum() > 0:
        logger.warning(f'{null_or_nan_mask.sum()} participant_id will not be considered because of '
                       f'missing required values:\n{list(phenotype[null_or_nan_mask].participant_id.values)}')
    participants_df = phenotype[~null_or_nan_mask]

    #  Read nii files
    npy_filenames = glob.glob(npy_path)
    assert len(npy_filenames) > 0, \
        logger.error(f"No ni files have been found, wrong npy_path : {npy_path}")

    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    logger.info(f"# {dataset_name}")

    logger.info("# 1) Read all npy file names")
    logger.info(f'{len(npy_filenames)} ni files have been found')

    ni_participants_df = make_participants_df(npy_filenames,
                                              id_regex=id_regex,
                                              session_regex=session_regex,
                                              acq_regex=acq_regex,
                                              run_regex=run_regex)
    ni_participants_df = standardize_df(ni_participants_df, id_types)

    logger.info("# 2) Merge nii's participant_id with participants dataframe")
    logger.info(f"{len(participants_df)} subjects have a phenotype.")
    ni_participants_df, _ = merge_ni_df(ni_participants_df, participants_df, qc=qc,
                                        tiv_columns=tiv_columns, participant_columns=participants_columns)
    logger.info('--> Remaining samples: {} / {}'.format(len(ni_participants_df), len(npy_filenames)))

    logger.info(f"# 3) Load {len(ni_participants_df)} images")
    
    with open(os.path.join(os.path.dirname(ni_participants_df['ni_path'].iloc[0]), "channels.txt"), "r") as f:
            channels = [l.rstrip() for l in f.readlines()]
    for path in ni_participants_df["ni_path"]:
        with open(os.path.join(os.path.dirname(path), "channels.txt"), "r") as f:
            ch = [l.rstrip() for l in f.readlines()]
            assert ch == channels, f"Not all the numpy array channels are the same"
    textures = list(dict.fromkeys([re.search("texture-([a-z]+)", ch)[1] for ch in channels])) # remove duplicates
    assert textures == list(check["channels"]), f"Wrong numpy array channels ({textures} / {check['channels']})"
    arr = np.concatenate([np.load(filename) for filename in ni_participants_df["ni_path"]], axis=0)
    # Concatenate right and left hemispheres
    ni_arr = np.zeros_like(arr).reshape(-1, arr.shape[1]//2, arr.shape[2]*2)
    for i, txt in enumerate(textures):
        ni_arr[:, i] = np.concatenate((arr[:, channels.index(f"hemi-lh_texture-{txt}")], arr[:, channels.index(f"hemi-rh_texture-{txt}")]), axis=1)
    channels = [f"hemi-lrh_texture-{ch}" for ch in textures]
    assert ni_arr[0].shape == check["shape"], f"Wrong shapes of freesurfer numpy files ({ni_arr[0].shape} / {check['shape']})"
    logger.info(f'--> {len(ni_participants_df)} array loaded')
    assert ni_arr.shape[0] == ni_participants_df.shape[0], "Unexpected nb of participants"
    ni_arr = ni_arr.astype(dtype)

    logger.info("# 4) Save the new participants dataframe")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Output directory created : {output_path}")
    ni_participants_df = standardize_df(ni_participants_df, id_types)
    try:
        dico_ext = {"\t": "tsv", ",": "csv"}
        ext = dico_ext[sep]
    except KeyError:
        raise ValueError(f"Unknown value of separator : {sep}, \
                         accepted values are : {list(dico_ext.keys())}")

    if os.path.exists(
            output_freesurfer(dataset_name, output_path, mri_preproc='participants', ext=ext)):
        answer = input(f"There is already a participants dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array and the participant dataframe have not been saved.")
            return 0
    ni_participants_df.to_csv(
        output_freesurfer(dataset_name, output_path, mri_preproc='participants', ext=ext), index=False, sep=sep)

    logger.info("# 5) Save the raw npy file (with shape {})".format(ni_arr.shape))
    if os.path.exists(output_freesurfer(dataset_name, output_path, modality="freesurfer", mri_preproc=textures, ext="npy")):
        answer = input(f"There is already an array of nii files at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            logger.warning("The nii array has not been saved.")
            return 0
    np.save(output_freesurfer(dataset_name, output_path, modality="freesurfer", mri_preproc=textures, ext="npy"),
            ni_arr)
    with open(os.path.join(output_path, "channels.txt"), "w") as f:
        f.write("\n".join(channels))

    # Deallocate the memory
    del ni_arr, arr

def quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t',
                      id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                      check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),
                      data_type="float32", 
                      id_regex='sub-([^/_\.]+)', session_regex='ses-([^_/\.]+)',
                      acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)',
                      tiv_columns=[], participants_columns=[]):
    
    logger = logging.getLogger("quasiraw_nii2npy")

    # Loading QC
    if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, id_types)

    # Loading phenotype dataframe
    keys_required = {'participant_id', 'age', 'sex', 'diagnosis', 'site', 'study'}
    phenotype = standardize_df(phenotype, id_types)
    assert keys_required.issubset(phenotype.columns), \
        logger.error(f"Missing keys {set(keys_required) - set(phenotype.columns)} "
                     f"in phenotype Dataframe")
    
    # Remove participants with missing keys_required
    null_or_nan_mask = phenotype[keys_required].isna().any(axis=1)
    if null_or_nan_mask.sum() > 0:
        logger.warning(f'{null_or_nan_mask.sum()} participant_id will not be considered because of '
                       f'missing required values:\n{list(phenotype[null_or_nan_mask].participant_id.values)}')
    participants_df = phenotype[~null_or_nan_mask]

    #  Read nii files
    ni_filenames = glob.glob(nii_path)
    assert len(ni_filenames) > 0, \
        logger.error(f"No ni files have been found, wrong ni_path : {nii_path}")

    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    logger.info(f"# {dataset_name}")

    logger.info("# 1) Read all file names")
    logger.info(f'{len(ni_filenames)} ni files have been found')

    ni_participants_df = make_participants_df(ni_filenames,
                                              id_regex=id_regex,
                                              session_regex=session_regex,
                                              acq_regex=acq_regex,
                                              run_regex=run_regex)
    ni_participants_df = standardize_df(ni_participants_df, id_types)

    logger.info("# 2) Merge nii's participant_id with participants dataframe")
    logger.info(f"{len(participants_df)} subjects have a phenotype.")
    ni_participants_df, _ = merge_ni_df(ni_participants_df, participants_df, qc=qc,
                                        tiv_columns=tiv_columns, participant_columns=participants_columns)
    logger.info('--> Remaining samples: {} / {}'.format(len(ni_participants_df), len(ni_filenames)))

    logger.info(f"# 3) Load {len(ni_participants_df)} images")
    try:
        dtype = np.dtype(data_type)
    except TypeError:
        raise ValueError(f"Unknown value of data_type : {data_type}")
    
    ni_arr = load_images(ni_participants_df, check=check, dtype=dtype)
    logger.info(f'--> {len(ni_participants_df)} img loaded')
    print('--> {} img loaded'.format(len(ni_participants_df)))
    
    print("# 4) Save the new participants.tsv")
    ni_participants_df.to_csv(output_quasi_raw(dataset_name, output_path, type="participants", ext="tsv"),
                              index=False, sep=sep)
    print("# 5) Save the raw npy file (with shape {})".format(ni_arr.shape))
    np.save(output_quasi_raw(dataset_name, output_path, type="data32", ext="npy"), ni_arr)

    ######################################################################################################################
    # Deallocate the memory
    del ni_arr

"""
def dti_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t', id_type=str,
                resampling=None, id_regex=None, session_regex=None, run_regex=None,
                check=dict(shape=(182, 218, 182), zooms=(1, 1, 1))):
    ########################################################################################################################

    qc = pd.read_csv(qc, sep=sep) if qc is not None else None

    keys_required = ['participant_id', 'age', 'sex', 'diagnosis', 'study']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype,
                                                                                   set(keys_required) - set(
                                                                                       phenotype.columns))
    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    # Rm participants with missing keys_required
    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name, flush=True)
    print("# 1) Read all filenames", flush=True)
    ni_filenames = glob.glob(nii_path)

    print("# 2) Extract participant_id and merge files with given dataframe", flush=True)

    ni_participants_df = make_participants_df(ni_filenames, id_regex)
    ni_participants_df, ni_rois_df = merge_ni_df(ni_participants_df, participants_df,
                                                 qc=qc, id_type=id_type, session_regex=session_regex,
                                                 run_regex=run_regex)

    print('--> (All) Remaining samples: {} / {}'.format(len(ni_participants_df), len(participants_df)))
    print('--> (ROI) Remaining samples: {} / {}'.format(len(ni_rois_df), len(participants_df)))

    print("# 3) Load images and create numpy array", flush=True)
    ni_arr = load_images(ni_participants_df, check=check, resampling=resampling, dtype=np.float32)

    print('--> {} img loaded'.format(len(ni_arr)))

    print("## Save the new participants.tsv")
    ni_participants_df.to_csv(output_dti(dataset_name, output_path, type="participants", ext="tsv"), index=False,
                              sep=sep)
    ni_rois_df.to_csv(output_dti(dataset_name, output_path, type="roi", ext="tsv"), index=False, sep=sep)
    print("## Save the raw npy file (with shape {})".format(ni_arr.shape))
    np.save(output_dti(dataset_name, output_path, type="data32", ext="npy"), ni_arr)

    ######################################################################################################################
    # Deallocate the memory
    del ni_arr
"""