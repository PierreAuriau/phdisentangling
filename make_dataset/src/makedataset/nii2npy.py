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
from makedataset


def skeleton_nii2npy(nii_path, phenotype, dataset_name, output_path, side, qc=None,
                     id_types={"participant_id": str, "session": int, "acq": int, "run": int},
                     check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),
                     data_type="float32", sep='\t', skeleton_size=False, stored_data=False,
                     id_regex='sub-([^/_\.]+)', session_regex='ses-([^_/\.]+)',
                     acq_regex='acq-([^_/\.]+)', run_regex='run-([^_/\.]+)', preproc="skeleton",
                     tiv_columns=[], participants_columns=[]):
    # FIXME : put standardize_df in merge_df ?
    # FIXME : make foldlabel_nii2npy to have the same participant_id order ?
    if qc is not None:
        qc = load_qc(qc)
        qc = standardize_df(qc, id_types)

    phenotype = standardize_df(phenotype, id_types)

    # keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']
    keys_required = ['participant_id', 'age', 'sex', 'diagnosis', 'site', 'study']

    assert keys_required in phenotype.columns, \
        print(f"Missing keys {set(keys_required) - set(phenotype.columns)} "
              f"in phenotype Dataframe")

    # Remove participants with missing keys_required
    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= phenotype[key].isnull() | phenotype[key].isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    if side == "F":
        side = ""
    NI_filenames = glob.glob(nii_path)
    assert len(NI_filenames) > 0, \
        "No NI files have been found, wrong ni_path : {}".format(nii_path)

    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("#", dataset_name)

    print("# 1) Read all file names")
    print(' {} NI files have been found'.format(str(len(NI_filenames))))
    NI_participants_df = make_participants_df(NI_filenames,
                                              id_regex=id_regex,
                                              session_regex=session_regex,
                                              acq_regex=acq_regex,
                                              run_regex=run_regex)
    NI_participants_df = standardize_df(NI_participants_df, id_types)
    print("# 2) Merge nii's participant_id with participants.tsv")
    NI_participants_df, _ = merge_ni_df(NI_participants_df, participants_df, qc=qc,
                                        tiv_columns=tiv_columns, participants_columns=participants_columns)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))

    print(f"# 3) Load {len(NI_participants_df)} images", flush=True)
    try:
        dtype = np.dtype(data_type)
    except TypeError:
        raise ValueError(f"Unknown value of data_type : {data_type}")
    NI_arr = load_images_with_aims(NI_participants_df, check=check, dtype=dtype, stored_data=stored_data)
    print('--> {} img loaded'.format(len(NI_participants_df)))

    if skeleton_size:
        print("# 3 bis) Compute skeleton size", flush=True)
        skeleton_sizes = np.count_nonzero(NI_arr, axis=(1, 2, 3, 4))
        NI_participants_df["skeleton_size"] = skeleton_sizes

    print("# 4) Save the new participants dataframe")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Output directory created : {output_path}")
    NI_participants_df = standardize_df(NI_participants_df, id_types)
    try:
        dico_ext = {"\t": "tsv", ",": "csv"}
        ext = dico_ext[sep]
    except KeyError:
        raise ValueError(f"Unknown value of separator : {sep}, \
                         accepted values are : {list(dico_ext.keys())}")

    if os.path.exists(
            OUTPUT_SKELETON(dataset_name, output_path, modality="t1mri", mri_preproc=preproc, type="participants",
                            ext=ext, side=side)):
        answer = input(f"There is already a participants dataframe at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            print("Cancelled")
            return 0
    NI_participants_df.to_csv(
        OUTPUT_SKELETON(dataset_name, output_path, modality="t1mri", mri_preproc=preproc, type="participants",
                        ext=ext, side=side), index=False, sep=sep)

    print("# 5) Save the raw npy file (with shape {})".format(NI_arr.shape))
    if os.path.exists(OUTPUT_SKELETON(dataset_name, output_path, modality="t1mri", mri_preproc=preproc,
                                      type=f"data{re.search('[0-9]+', data_type).group()}", ext="npy", side=side)):
        answer = input(f"There is already an array of nii files at {output_path}. Do you want to replace it ? (y/n)")
        if answer != "y":
            print("Cancelled")
            return 0
    np.save(OUTPUT_SKELETON(dataset_name, output_path, modality="t1mri", mri_preproc=preproc,
                            type=f"data{re.search('[0-9]+', data_type).group()}", ext="npy", side=side),
            NI_arr)

    # Deallocate the memory
    del NI_arr

def quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t', id_type=str,
                      check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), resampling=None):
    ########################################################################################################################
    ## Add resampling argument
    ## old
    # qc = pd.read_csv(qc, sep=sep) if qc is not None else None

    ## New ##
    if isinstance(qc, str):
        qc = pd.read_csv(qc, sep=sep)
    ## New ##

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis', 'study']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype.columns,
                                                                                   set(keys_required) - set(
                                                                                       phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.tsv
    # assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"

    # Rm participants with missing keys_required
    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]
    ########################################################################################################################
    #  Neuroimaging niftii and TIV
    #  mwp1 files
    #  excpected image dimensions
    NI_filenames = glob.glob(nii_path)
    ########################################################################################################################
    #  Load images, intersect with pop and do preprocessing and dump 5d npy
    print("###########################################################################################################")
    print("#", dataset_name)

    print("# 1) Read all file names")
    NI_participants_df = make_participants_df(NI_filenames)
    print("# 2) Merge nii's participant_id with participants.tsv")
    NI_participants_df, Ni_rois_df = merge_ni_df(NI_participants_df, participants_df,
                                                 qc=qc, id_type=id_type)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))
    print('--> Remaining samples: {} / {}'.format(len(Ni_rois_df), len(participants_df)))

    print("# 3) Load %i images" % len(NI_participants_df), flush=True)
    ### Old :
    ### NI_arr = load_images(NI_filenames, check=check)

    ## New ##
    NI_arr = load_images(NI_participants_df, check=check, resampling=resampling)
    ## New ##

    print('--> {} img loaded'.format(len(NI_participants_df)))
    print("# 4) Save the new participants.tsv")
    NI_participants_df.to_csv(OUTPUT_QUASI_RAW(dataset_name, output_path, type="participants", ext="tsv"),
                              index=False, sep=sep)
    Ni_rois_df.to_csv(OUTPUT_QUASI_RAW(dataset_name, output_path, type="roi", ext="tsv"),
                      index=False, sep=sep)
    print("# 5) Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_QUASI_RAW(dataset_name, output_path, type="data64", ext="npy"), NI_arr)
    np.save(OUTPUT_QUASI_RAW(dataset_name, output_path, type="data64", ext="npy"), NI_arr)

    ######################################################################################################################
    # Deallocate the memory
    del NI_arr


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
    NI_filenames = glob.glob(nii_path)

    print("# 2) Extract participant_id and merge files with given dataframe", flush=True)

    NI_participants_df = make_participants_df(NI_filenames, id_regex)
    NI_participants_df, Ni_rois_df = merge_ni_df(NI_participants_df, participants_df,
                                                 qc=qc, id_type=id_type, session_regex=session_regex,
                                                 run_regex=run_regex)

    print('--> (All) Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))
    print('--> (ROI) Remaining samples: {} / {}'.format(len(Ni_rois_df), len(participants_df)))

    print("# 3) Load images and create numpy array", flush=True)
    NI_arr = load_images(NI_participants_df, check=check, resampling=resampling, dtype=np.float32)

    print('--> {} img loaded'.format(len(NI_arr)))

    print("## Save the new participants.tsv")
    NI_participants_df.to_csv(OUTPUT_DTI(dataset_name, output_path, type="participants", ext="tsv"), index=False,
                              sep=sep)
    Ni_rois_df.to_csv(OUTPUT_DTI(dataset_name, output_path, type="roi", ext="tsv"), index=False, sep=sep)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_DTI(dataset_name, output_path, type="data32", ext="npy"), NI_arr)

    ######################################################################################################################
    # Deallocate the memory
    del NI_arr


def cat12_nii2npy(nii_path, phenotype, dataset, output_path, qc=None, sep='\t', id_type=str,
                  check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), tiv_columns=[], participants_columns=[]):
    # Save 3 files:
    participants_filename = OUTPUT_CAT12(dataset, output_path, mri_preproc='participants', ext='tsv')
    rois_filename = OUTPUT_CAT12(dataset, output_path, mri_preproc='rois', scaling="gs", ext='tsv')
    vbm_filename = OUTPUT_CAT12(dataset, output_path, mri_preproc='mwp1', scaling="gs", ext='npy')

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    ###########################################################################
    # Select participants with non missing required keys

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in phenotype df that are required to compute the npy array: {}".format(
            set(keys_required) - set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.tsv
    # assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"
    null_or_nan_mask = np.zeros(len(phenotype)).astype(bool)
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]

    ###########################################################################
    #  Read nii files

    NI_filenames = glob.glob(nii_path)
    # NI_arr, NI_participants_df, ref_img = img_to_array(NI_filenames, expected=check)
    NI_participants_df = make_participants_df(NI_filenames)
    print('--> {} images found'.format(len(NI_participants_df)))

    ###########################################################################
    # Merge nii's participant_id with participants.tsv

    NI_participants_df, Ni_rois_df = merge_ni_df(NI_participants_df, participants_df,
                                                 qc=qc, id_type=id_type,
                                                 tiv_columns=tiv_columns,
                                                 participants_columns=participants_columns)
    print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))
    print('--> Remaining samples: {} / {}'.format(len(Ni_rois_df), len(participants_df)))

    print("Loading %i images" % len(NI_participants_df), flush=True)
    NI_arr = load_images(NI_participants_df, check=check)
    assert NI_arr.shape[0] == NI_participants_df.shape[0] == Ni_rois_df.shape[0], "Unexpected nb of participants"

    ###########################################################################
    print("# 3) Global scaling of arrays and ROIs to adjust for TIV ")
    assert np.all(Ni_rois_df.tiv == NI_participants_df.tiv), "rois.tiv !=  participants.tiv"

    NI_arr = global_scaling(NI_arr, axis0_values=NI_participants_df.tiv.values, target=1500)

    # imgs_arr = global_scaling(imgs_arr, axis0_values=rois['TIV'].values, target=1500)
    gscaling = 1500 / NI_participants_df['tiv']
    Ni_rois_df.loc[:, 'tiv':] = Ni_rois_df.loc[:, 'tiv':].multiply(gscaling, axis="index")

    ###########################################################################
    print("## Save array, rois and participants files")
    NI_participants_df.to_csv(participants_filename, index=False, sep=sep)
    Ni_rois_df.to_csv(rois_filename, index=False, sep=sep)
    np.save(vbm_filename, NI_arr)

    return participants_filename, rois_filename, vbm_filename



"""

"""










