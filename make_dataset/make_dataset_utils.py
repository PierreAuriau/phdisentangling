#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benoit.dufumier


"""

# TODO: Libraries pylearn-mulm, brainomics needed for these functions. Do we hard copy them here ?
import os, sys
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import matplotlib
matplotlib.use('Agg')
import glob

import nibabel
import re
from collections import OrderedDict

participant_re = re.compile("sub-([^_/]+)")
session_re = re.compile("ses-([^_/]+)/")
run_re = re.compile("run-([a-zA-Z0-9]+)")

"""
Format:

<study>_<software>_<output>[-<options>][_resolution]

study := cat12vbm | quasiraw
output := mwp1 | rois
options := gs: global scaling
resolution := 1.5mm | 1mm

Examples:

bsnip1_cat12vbm_mwp1-gs_1.5mm.npy
bsnip1_cat12vbm_rois-gs.tsv
bsnip1_cat12vbm_participants.tsv
"""

# TODO Julie/Benoit: modify OUTPUT_CAT12 and OUTPUT_QUASI_RAW to match format
# TODO Julie/Benoit: split int participants.tsv, rois and vbm
# TODO Edouard Add 10 line of age prediction

def OUTPUT_CAT12(dataset, output_path, modality='cat12vbm', mri_preproc='mwp1', scaling=None, ext=None):
    """
    Example
    -------
    output_path = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
    dataset = 'localizer'
    modality='cat12vbm'
    mri_preproc='mwp1'
    scaling='gs'
    OUTPUT_CAT12(dataset, output_path, mri_preproc='mwp1', scaling="gs", ext='npy')
    OUTPUT_CAT12(dataset, output_path, mri_preproc='rois', scaling="gs", ext='tsv')
    OUTPUT_CAT12(dataset, output_path, mri_preproc='participants', ext='tsv')
    """
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "-" + scaling) + "." + ext)



def OUTPUT_QUASI_RAW(dataset, output_path, modality='cat12vbm', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)

def OUTPUT_DTI(dataset, output_path, modality='dwi', mri_preproc='tbss', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)

def OUTPUT_SKELETON(dataset, output_path, modality='morphologist', mri_preproc='skeleton', type=None, ext=None, side=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality + "_" + ("" if side is None else side) + mri_preproc +
                  ("" if type is None else "_" + type) + "." + ext)

def merge_ni_df(NI_participants_df, participants_df, qc=None, participant_id="participant_id",
                id_type=str, session_regex=None, run_regex=None, tiv_columns=[], participants_columns=[]):
    """
    Select participants of NI_arr and NI_participants_df participants that are also in participants_df

    Parameters
    ----------
    NI_participants_df: DataFrame, with at least 2 columns: participant_id, "ni_path"
    participants_df: DataFrame, with 2 at least 1 columns participant_id
    qc: DataFrame, with at 2 columns participant_id and qc in [0, 1].
    participant_id: column that identify participant_id
    id_type: the type of participant_id and session, eventually, that should be used for every DataFrame
    session_regex: regex to extract session from ni_path
    run_regex: regex to extract run from ni_path
    Returns
    -------
     NI_participants_df (DataFrame) participants that are also in participants_df


    >>> import numpy as np
    >>> import pandas as pd
    >>> NI_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12-12.6_vbm/sub-ICAAR017/ses-V1/anat/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12-12.6_vbm/sub-ICAAR033/ses-V1/anat/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12-12.6_vbm/sub-STARTRA160489/ses-V1/anat/mri/mwp1sub-STARTRA160489_ses-V1_T1w.nii']
    >>> NI_arr, NI_participants_df, ref_img = load_images(NI_filenames, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> NI_arr.shape
    (3, 1, 121, 145, 121)
    >>> NI_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> other_df=pd.DataFrame(dict(participant_id=['ICAAR017', 'STARTRA160489']))
    >>> NI_arr2, NI_participants_df2 = merge_ni_df(NI_arr, NI_participants_df, other_df)
    >>> NI_arr2.shape
    (2, 1, 121, 145, 121)
    >>> NI_participants_df2
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    >>> np.all(NI_arr[[0, 2], ::] == NI_arr2)
    True
    """

    # 1) Extracts the session + run if available in participants_df/qc from <ni_path> in NI_participants_df
    unique_key_pheno = [participant_id]
    unique_key_qc = [participant_id]
    NI_participants_df.participant_id = NI_participants_df.participant_id.astype(id_type)
    participants_df.participant_id = participants_df.participant_id.astype(id_type)
    if 'session' in participants_df or (qc is not None and 'session' in qc):
        NI_participants_df['session'] = NI_participants_df.ni_path.str.extract(session_regex or 'ses-([^_/]+)/')[0].astype(id_type)
        if 'session' in participants_df:
            participants_df.session = participants_df.session.astype(id_type)
            unique_key_pheno.append('session')
        if qc is not None and 'session' in qc:
            qc.session = qc.session.astype(id_type)
            unique_key_qc.append('session')
    if 'run' in participants_df or (qc is not None and 'run' in qc):
        NI_participants_df['run'] = NI_participants_df.ni_path.str.extract(run_regex or 'run-([^_/]+)\_.*nii')[0].fillna(1).astype(str)
        if 'run' in participants_df:
            unique_key_pheno.append('run')
            participants_df.run = participants_df.run.astype(str)
        if qc is not None and 'run' in qc:
            unique_key_qc.append('run')
            qc.run = qc.run.astype(str)
    # 2) Keeps only the matching (participant_id, session, run) from both NI_participants_df and participants_df by
    #    preserving the order of NI_participants_df
    # !! Very import to have a clean index (to retrieve the order after the merge)
    # Create an "index" column
    NI_participants_df = NI_participants_df.reset_index(drop=True).reset_index() # stores a clean index from 0..len(df)
    NI_participants_merged = pd.merge(NI_participants_df, participants_df, on=unique_key_pheno,
                                      how='inner', validate='m:1')

    print('--> {} {} have missing phenotype'.format(len(NI_participants_df)-len(NI_participants_merged),
          unique_key_pheno))
    print('--> {} {} have missing nifti image'.format(len(participants_df)-len(NI_participants_merged),
          unique_key_pheno))

    # 3) If QC is available, filters out the (participant_id, session, run) who did not pass the QC
    if qc is not None:
        assert np.all(qc.qc.eq(0) | qc.qc.eq(1)), 'Unexpected value in qc.tsv'
        qc = qc.reset_index(drop=True) # removes an old index
        qc_val = qc.qc.values
        if np.all(qc_val==0):
            raise ValueError('No participant passed the QC !')
        elif np.all(qc_val==1):
            pass
        else:
            # Modified this part, indeed, the old code assumes that all subject
            # after idx_first_occurence should be removed, why ?
            # idx_first_occurence = len(qc_val) - (qc_val[::-1] != 1).argmax()
            # assert np.all(qc.iloc[idx_first_occurence:].qc == 1)
            # keep = qc.iloc[idx_first_occurence:][unique_key_qc]
            # New code simply select qc['qc'] == 1
            keep = qc[qc['qc'] == 1][unique_key_qc]
            init_len = len(NI_participants_merged)
            keep.participant_id = keep.participant_id.astype(str)

            assert NI_participants_merged.participant_id.dtype==keep.participant_id.dtype
            assert NI_participants_merged.session.dtype==keep.session.dtype
            assert NI_participants_merged.run.dtype==keep.run.dtype

            # Very important to have 1:1 correspondance between the QC and the NI_participant_array
            NI_participants_merged = pd.merge(NI_participants_merged, keep, on=unique_key_qc,
                                              how='inner', validate='1:1')
            print('--> {} {} did not pass the QC'.format(init_len - len(NI_participants_merged), unique_key_qc))

    # if merge_ni_path and 'ni_path' in participants_df:
    #     # Keep only the matching session and acquisition nb according to <participants_df>
    #     sub_sess_to_keep = NI_participants_merged['ni_path_y'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
    #     sub_sess = NI_participants_merged['ni_path_x'].str.extract(r".*/.*sub-(\w+)_ses-(\w+)_.*")
    #     # Some participants have only one acq, in which case it is not mentioned
    #     acq_to_keep = NI_participants_merged['ni_path_y'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')
    #     acq = NI_participants_merged['ni_path_x'].str.extract(r"(acq-[a-zA-Z0-9\-\.]+)").fillna('')

    #     assert not (sub_sess.isnull().values.any() or sub_sess_to_keep.isnull().values.any()), \
    #         "Extraction of session_id or participant_id failed"

    #     keep_unique_participant_ids = sub_sess_to_keep.eq(sub_sess).all(1).values.flatten() & \
    #                                   acq_to_keep.eq(acq).values.flatten()

    #     NI_participants_merged = NI_participants_merged[keep_unique_participant_ids]
    #     NI_participants_merged.drop(columns=['ni_path_y'], inplace=True)
    #     NI_participants_merged.rename(columns={'ni_path_x': 'ni_path'}, inplace=True)


    unique_key = unique_key_qc if set(unique_key_qc) >= set(unique_key_pheno) else unique_key_pheno
    assert len(NI_participants_merged.groupby(unique_key)) == len(NI_participants_merged), \
        '{} similar pairs {} found'.format(len(NI_participants_merged)-len(NI_participants_merged.groupby(unique_key)),
                                           unique_key)

    # split rois and participants
    if 'session' not in NI_participants_merged:
        NI_participants_merged['session'] = np.nan
    if 'run' not in NI_participants_merged:
        NI_participants_merged['run'] = np.nan

    NI_participants = NI_participants_merged.drop(list(tiv_columns), axis=1)
    NI_participants = NI_participants.drop("index", axis=1)
    NI_participants["tiv"] = NI_participants_merged["tiv"]
    NI_rois = NI_participants_merged.drop(list(participants_columns), axis=1)
    NI_rois = NI_rois.drop("index", axis=1)


    # NI_participants_merged.drop('index')
    return (NI_participants, NI_rois)


def quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), resampling=None):
    ########################################################################################################################
    ## Add resampling argument
    ## old
    #qc = pd.read_csv(qc, sep=sep) if qc is not None else None
    
    ## New ##
    if isinstance(qc, str):
        qc = pd.read_csv(qc, sep=sep)
    ## New ##
    
    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype,
                                                                                   set(keys_required)-set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.tsv
    #assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"

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

    print("# 3) Load %i images"%len(NI_participants_df), flush=True)
    ### Old :
    ### NI_arr = load_images(NI_filenames, check=check)
    
    ## New ##
    NI_arr = load_images(NI_participants_df,check=check, resampling=resampling)
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
                                                                                   set(keys_required)-set(phenotype.columns))
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
    NI_participants_df.to_csv(OUTPUT_DTI(dataset_name, output_path, type="participants", ext="tsv"), index=False, sep=sep)
    Ni_rois_df.to_csv(OUTPUT_DTI(dataset_name, output_path, type="roi", ext="tsv"), index=False, sep=sep)
    print("## Save the raw npy file (with shape {})".format(NI_arr.shape))
    np.save(OUTPUT_DTI(dataset_name, output_path, type="data32", ext="npy"), NI_arr)

    ######################################################################################################################
    # Deallocate the memory
    del NI_arr

def cat12_nii2npy(nii_path, phenotype, dataset, output_path, qc=None, sep='\t', id_type=str,
            check = dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)),tiv_columns=[], participants_columns=[]):

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
            set(keys_required)-set(phenotype.columns))

    ## TODO: change this condition according to session and run in phenotype.tsv
    #assert len(set(phenotype.participant_id)) == len(phenotype), "Unexpected number of participant_id"
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
    #NI_arr, NI_participants_df, ref_img = img_to_array(NI_filenames, expected=check)
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

    print("Loading %i images"%len(NI_participants_df), flush=True)
    NI_arr = load_images(NI_participants_df, check=check)
    assert NI_arr.shape[0] == NI_participants_df.shape[0] == Ni_rois_df.shape[0], "Unexpected nb of participants"

    ###########################################################################
    print("# 3) Global scaling of arrays and ROIs to adjust for TIV ")
    assert np.all(Ni_rois_df.tiv == NI_participants_df.tiv),"rois.tiv !=  participants.tiv"

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

def skeleton_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=None, sep='\t', id_type=str,
            check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)), side=None):

    if qc is not None:
        qc = load_qc(qc, sep=sep)

    if 'TIV' in phenotype:
        phenotype.rename(columns={'TIV': 'tiv'}, inplace=True)

    keys_required = ['participant_id', 'age', 'sex', 'tiv', 'diagnosis']

    assert set(keys_required) <= set(phenotype.columns), \
        "Missing keys in {} that are required to compute the npy array: {}".format(phenotype,
                                                                                   set(keys_required)-set(phenotype.columns))

    #Remove participants with missing keys_required
    null_or_nan_mask = [False for _ in range(len(phenotype))]
    for key in keys_required:
        null_or_nan_mask |= getattr(phenotype, key).isnull() | getattr(phenotype, key).isna()
    if null_or_nan_mask.sum() > 0:
        print('Warning: {} participant_id will not be considered because of missing required values:\n{}'. \
              format(null_or_nan_mask.sum(), list(phenotype[null_or_nan_mask].participant_id.values)))

    participants_df = phenotype[~null_or_nan_mask]
    
    ## A TESTER ##
    if side is None:
        if re.search('/L', nii_path):
            side = "L"
        elif re.search('/R', nii_path):
            side = "R"
        else:
            side = "both"
        print('Side is set automatically to {}'.format(side))
        
    if side == "both":
        print("###########################################################################################################")
        print("#", dataset_name)
        print("# 1) Read all file names")
        NI_filenames = glob.glob(nii_path)
        NI_filenames_l = [f for f in NI_filenames if re.search('/L', f)]
        NI_filenames_r = [f for f in NI_filenames if re.search('/R', f)]
        assert len(NI_filenames_l) + len(NI_filenames_r) == len(NI_filenames)
        assert len(NI_filenames_l) == len(NI_filenames_r)
        NI_participants_df_l = make_participants_df(NI_filenames_l, id_regex='_sub-([^/_]+)_')
        NI_participants_df_r = make_participants_df(NI_filenames_r, id_regex='_sub-([^/_]+)_')
        print(' {} NI files have been found'.format(str(len(NI_filenames_l))))

        print("# 2) Merge nii's participant_id with participants.tsv")
        print('Side L')
        NI_participants_df_l, _ = merge_ni_df(NI_participants_df_l, participants_df,
                                                     qc=qc, id_type=id_type, session_regex='ses-([^_/]+)', run_regex='run-([^_/\.]+)')
        print('--> Remaining samples: {} / {}'.format(len(NI_participants_df_l), len(participants_df)))

        print('Side R')
        NI_participants_df_r, _ = merge_ni_df(NI_participants_df_r, participants_df,
                                                     qc=qc, id_type=id_type, session_regex='ses-([^_/]+)', run_regex='run-([^_/\.]+)')
        print('--> Remaining samples: {} / {}'.format(len(NI_participants_df_r), len(participants_df)))
        assert len(NI_participants_df_r) == len(NI_participants_df_l)

        # Check order of the two dataframes according to keys_to_check
        print('Reordering the two dataframes...')
        keys_to_check = ['participant_id', 'session', 'run']
        key_type = {}
        for k in keys_to_check:
            if k not in NI_participants_df_l.columns:
                keys_to_check.remove(k)
        for k in keys_to_check:
            key_type[k] = NI_participants_df_l[k].dtype
            cat_order = CategoricalDtype(NI_participants_df_l[k].unique().tolist(), ordered=True)
            NI_participants_df_l[k] = NI_participants_df_l[k].astype(cat_order)
            NI_participants_df_r[k] = NI_participants_df_r[k].astype(cat_order)
        NI_participants_df_l.sort_values(keys_to_check, inplace=True, ignore_index=True)
        NI_participants_df_r.sort_values(keys_to_check, inplace=True, ignore_index=True)
        for k in keys_to_check:
            NI_participants_df_l[k].astype(key_type[k])
            NI_participants_df_r[k].astype(key_type[k])
        assert np.all(NI_participants_df_r[keys_to_check] == NI_participants_df_l[keys_to_check])
        ## A TESTER ##
        join_on = NI_participants_df_l.columns.tolist()
        join_on.remove('ni_path')
        NI_participants_df = pd.merge(NI_participants_df_l, NI_participants_df_r, how="inner", on=join_on, validate="1:1", suffixes=("_left", "_right"))
        NI_participants_df.drop(columns=['ni_path_right'])
        index = NI_participants_df.columns.tolist().index('ni_path_left')
        NI_participants_df.insert(loc=index+1, column='ni_path_right', value=NI_participants_df['ni_path_right'])

        print("# 3) Load %i images"%len(NI_participants_df_l), flush=True)
        NI_arr_l = load_images(NI_participants_df_l, check=check, resampling=None)
        NI_arr_r = load_images(NI_participants_df_r, check=check, resampling=None)
        NI_arr = NI_arr_l + NI_arr_r
        print('--> {} img loaded'.format(np.shape(NI_arr)[0]))
        
        print("# 4) Save the new participants.tsv")
        NI_participants_df_l.to_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side='L'),
                                  index=False, sep=sep)
        NI_participants_df_r.to_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side='R'),
                                  index=False, sep=sep)
        NI_participants_df.to_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side=None),
                                  index=False, sep=sep)

        print("Sanity Check")
        df_l = pd.read_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side='L'), sep=sep)
        df_r = pd.read_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side='R'), sep=sep)
        assert np.all(df_l[keys_to_check] == df_r[keys_to_check])

        print("# 5) Save the raw npy file (with shape {})".format(NI_arr.shape))
        np.save(OUTPUT_SKELETON(dataset_name, output_path, type="data64", ext="npy", side='L'), NI_arr_l)
        np.save(OUTPUT_SKELETON(dataset_name, output_path, type="data64", ext="npy", side='R'), NI_arr_r)
        np.save(OUTPUT_SKELETON(dataset_name, output_path, type="data64", ext="npy"), NI_arr)
        del NI_arr, NI_arr_l, NI_arr_r
        
    else:
        NI_filenames = glob.glob(nii_path)
        print(' {} NI files have been found'.format(str(len(NI_filenames))))
        
        #  Load images, intersect with pop and do preprocessing and dump 5d npy
        print("###########################################################################################################")
        print("#", dataset_name)
    
        print("# 1) Read all file names")
        NI_participants_df = make_participants_df(NI_filenames, id_regex='_sub-([^/_]+)_')
        print(len(NI_participants_df))
        NI_participants_df.to_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side=side),
                                  index=False, sep=sep)
        print("# 2) Merge nii's participant_id with participants.tsv")
        NI_participants_df, _ = merge_ni_df(NI_participants_df, participants_df,
                                                     qc=qc, id_type=id_type, session_regex='ses-([^_/]+)', run_regex='run-([^_/\.]+)')
        print('--> Remaining samples: {} / {}'.format(len(NI_participants_df), len(participants_df)))
    
        print("# 3) Load %i images"%len(NI_participants_df), flush=True)
    
        NI_arr = load_images(NI_participants_df, check=check, resampling=None)
        
        print('--> {} img loaded'.format(len(NI_participants_df)))
        
        print("# 4) Save the new participants.tsv")
        NI_participants_df.to_csv(OUTPUT_SKELETON(dataset_name, output_path, type="participants", ext="tsv", side=side),
                                  index=False, sep=sep)
        
        print("# 5) Save the raw npy file (with shape {})".format(NI_arr.shape))
        np.save(OUTPUT_SKELETON(dataset_name, output_path, type="data64", ext="npy", side=side), NI_arr)
    
        ######################################################################################################################
        # Deallocate the memory
        del NI_arr

def load_qc(qc_file, sep='\t'):
    """
    Améliorations : Ajouter unique_key_qc à la place de participant_id
                    Ajouter assertion sur la présence des colonnes qc et participant_id
                    Ajouter la modification du type de la colonne qc en bool/int
    
    Functions which loads and merges qc_file depending the type of qc_file variable.

    Parameters
    ----------
    qc_file : DataFrame, str, list or dict
        Quality Check Dataframe or path to Quality Check Dataframe. All Dataframes need a 'participant_id' and 'qc' columns
    sep : str, optional
        The separator to load the files (in case of qc_file is composed of paths). The default is '\t'.

    Returns
    -------
    qc : DataFrame
        Quality Check Dataframe with a column participant_id, a column for each qc_file (the names are qc_int for list or qc_key for dict)
        and qc column which is the multiplication of all qc columns.

    """
    qc = None
    if isinstance(qc_file, pd.DataFrame):
        qc = qc_file
    
    elif isinstance(qc_file, str):
        qc = pd.read_csv(qc_file, sep=sep)
    
    elif isinstance(qc_file, list):
        for n, file in enumerate(qc_file):
            if isinstance(file, str):
                df_qc = pd.read_csv(file, sep=sep)
            elif isinstance(file, pd.DataFrame):
                df_qc = file
            else:
                raise ValueError
            if qc is None:
                qc = df_qc.copy()
                qc.reset_index(drop=True, inplace=True)
                qc.rename(columns={'qc': 'qc_' + str(n)}, inplace=True)
            else:
                qc = pd.merge(qc, df_qc, on=['participant_id'], how='inner', validate='1:1')
                qc.rename(columns={'qc': 'qc_' + str(n)}, inplace=True)
        qc['qc'] = qc[['qc_' + str(i) for i in range(len(qc_file))]].prod(axis=1)

    elif isinstance(qc_file, dict):
        for key, file in qc_file.items():
            if isinstance(file, str):
                df_qc = pd.read_csv(file, sep=sep)
            elif isinstance(file, pd.DataFrame):
                df_qc = file
            else:
                raise ValueError
            if qc is None:
                qc = df_qc
                qc.reset_index(drop=True, inplace=True)
                qc.rename(columns={'qc': 'qc_' + key}, inplace=True)
            else:
                qc = pd.merge(qc, df_qc, on=['participant_id'], how='inner', validate='1:1')
                qc.rename(columns={'qc': 'qc_' + key}, inplace=True)
        qc['qc'] = qc[['qc_' + k for k in qc_file.keys()]].prod(axis=1)
    else:
        raise ValueError('qc must be a Dataframe, a path to a DataFrame, a list of paths to DataFrames or a dict of paths to DataFrames')
    
    assert 'participant_id' in qc and 'qc' in qc
    return qc


def global_scaling(NI_arr, axis0_values=None, target=1500):
    """
    Apply a global proportional scaling, such that axis0_values * gscaling == target
    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    axis0_values: 1-d array, if None (default) use global average per subject: NI_arr.mean(axis=1)
    target: scalar, the desired target
    Returns
    -------
    The scaled array
    >>> import numpy as np
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_arr = np.array([[9., 11], [0, 2],  [4, 6]])
    >>> NI_arr
    array([[ 9., 11.],
           [ 0.,  2.],
           [ 4.,  6.]])
    >>> axis0_values = [10, 1, 5]
    >>> preproc.global_scaling(NI_arr, axis0_values, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    >>> preproc.global_scaling(NI_arr, axis0_values=None, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    """
    if axis0_values is None:
        axis0_values = NI_arr.mean(axis=1)
    gscaling = target / np.asarray(axis0_values)
    gscaling = gscaling.reshape([gscaling.shape[0]] + [1] * (NI_arr.ndim - 1))
    return gscaling * NI_arr


def make_participants_df(NI_filenames, id_regex=None):
    """
      Extract participant id from paths (by default, assumes it's BIDS format: /sub-<participant_id>/)
      If id_regex is given, use it to retrieve participant id (no BIDS format assumption)
      Parameters
      ----------
      NI_filenames : [str], filenames to Nifti images
      id_regex: str, regex expression used to extract <participant_id> from ni_path
      Returns
      -------
          participants: Dataframe, with 2 columns "participant_id", "ni_path"
    """
    match_filename_re = re.compile(id_regex or "/sub-([^/]+)/")
    pop_columns = ["participant_id", "ni_path"]
    NI_participants_df = pd.DataFrame([[match_filename_re.findall(NI_filename)[0]] + [NI_filename]
                                       for NI_filename in NI_filenames], columns=pop_columns)
    return NI_participants_df


def load_images(NI_participants_df, check=dict(), resampling=None, dtype=None):
    """
    Load images assuming paths contain a BIDS pattern to retrieve participant_id such /sub-<participant_id>/
    If id_regex is given, use it to retrieve participant id.
    Parameters
    ----------
    NI_participants_df : pandas DataFrame containing 'ni_path', path to the images to load
    check : dict, optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))
    resampling: float, factor to apply for resampling the images with nilearn.image.resample_img
    Returns
    -------
        NI_arr: ndarray, of shape (n_subjects, 1, image_shape). Shape should respect (n_subjects, n_channels, image_axis0, image_axis1, ...)
    Example
    -------
    >>> NI_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR017/ses-V1/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR033/ses-V1/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-STARTRA160489/ses-V1/mri/mwp1sub-STARTRA160489_ses-v1_T1w.nii']
    >>> NI_participants_df = make_participants_df(NI_filenames)
    >>> NI_arr = load_images(NI_participants_df, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> NI_arr.shape
    (3, 1, 121, 145, 121)
    >>> NI_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    """

    NI_imgs = [nibabel.load(NI_filename) for NI_filename in NI_participants_df.ni_path]
    ref_img = NI_imgs[0]
    # Check
    if 'shape' in check:
        assert ref_img.get_data().shape == check['shape']
    if 'zooms' in check:
        assert ref_img.header.get_zooms() == check['zooms']
    assert np.all([np.all(img.affine == ref_img.affine) for img in NI_imgs])
    assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in NI_imgs])

    ## Load image subjects x chanels (1) x image
    # Resampling
    if resampling is not None:
        from nilearn.image import resample_img
        target_affine=ref_img.affine[:3,:3] * resampling
        NI_arr = np.stack([np.expand_dims(resample_img(img, target_affine).get_data(), axis=0)
                           for img in NI_imgs])
    else:
        NI_arr = np.stack([np.expand_dims(img.get_data(), axis=0) for img in NI_imgs])

    if dtype is not None: # convert the np type
        NI_arr = NI_arr.astype(dtype)
    return NI_arr



def get_keys(filename):
    """
    Extract keys from bids filename. Check consistency of filename.

    Parameters
    ----------
    filename : str
        bids path

    Returns
    -------
    dict
        The minimum returned value is dict(participant_id=<match>,
                             session=<match, '' if empty>,
                             path=filename)

    Raises
    ------
    ValueError
        if match failed or inconsistent match.

    Examples
    --------
    >>> import nitk.bids
    >>> nitk.bids.get_keys('/dirname/sub-ICAAR017/ses-V1/mri/y_sub-ICAAR017_ses-V1_acq-s03_T1w.nii')
    {'participant_id': 'ICAAR017', 'session': 'V1'}
    """
    keys = OrderedDict()

    participant_id = participant_re.findall(filename)
    if len(set(participant_id)) != 1:
        raise ValueError('Found several or no participant id', participant_id, 'in path', filename)
    keys["participant_id"] = participant_id[0]

    session = session_re.findall(filename)
    if len(set(session)) > 1:
        raise ValueError('Found several sessions', session, 'in path', filename)

    elif len(set(session)) == 1:
        keys["session"] = session[0]

    else:
        keys["session"] = ''

    run = run_re.findall(filename)
    if len(set(run)) == 1:
        keys["run"] = run[0]

    else:
        keys["run"] = ''

    keys["ni_path"] = filename

    return keys


def diff_sets(a, b):
    """Compare sets

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    set
        diff between a and b.
    set
        a - b.
    set
        b - a.
    """
    a = set(a)
    b = set(b)
    return (a - b).union(b - a), a - b, b - a


def img_to_array(img_filenames, check_same_referential=True, expected=dict()):
    """
    Convert nii images to array (n_subjects, 1, , image_axis0, image_axis1, ...)
    Assume BIDS organisation of file to retrive participant_id, session and run.

    Parameters
    ----------
    img_filenames : [str]
        path to images

    check_same_referential : bool
        if True (default) check that all image have the same referential.

    expected : dict
        optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
        imgs_arr : array (n_subjects, 1, , image_axis0, image_axis1, ...)
            The array data structure (n_subjects, n_channels, image_axis0, image_axis1, ...)

        df : DataFrame
            With column: 'participant_id', 'session', 'run', 'path'

        ref_img : nii image
            The first image used to store referential and all information relative to the images.

    Example
    -------
    >>> from  nitk.image import img_to_array
    >>> import glob
    >>> img_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> imgs_arr, df, ref_img = img_to_array(img_filenames)
    >>> print(imgs_arr.shape)
    (171, 1, 121, 145, 121)
    >>> print(df.shape)
    (171, 3)
    >>> print(df.head())
      participant_id session                                               path
    0       ICAAR017      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    3  STARTLB160534      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    4       ICAAR048      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...

    """

    df = pd.DataFrame([pd.Series(get_keys(filename)) for filename in img_filenames])
    imgs_nii = [nibabel.load(filename) for filename in df.ni_path]

    ref_img = imgs_nii[0]

    # Check expected dimension
    if 'shape' in expected:
        assert ref_img.get_fdata().shape == expected['shape']
    if 'zooms' in expected:
        assert ref_img.header.get_zooms() == expected['zooms']

    if check_same_referential: # Check all images have the same transformation
        assert np.all([np.all(img.affine == ref_img.affine) for img in imgs_nii])
        assert np.all([np.all(img.get_fdata().shape == ref_img.get_fdata().shape) for img in imgs_nii])

    assert np.all([(not np.isnan(img.get_fdata()).any()) for img in imgs_nii])
    # Load image subjects x channels (1) x image
    imgs_arr = np.stack([np.expand_dims(img.get_fdata(), axis=0) for img in imgs_nii])

    return imgs_arr, df, ref_img


def ml_regression(data, y):
    """ Basic QC for age predictio

    data : dict of arrays (N, P)
    y : array (N, )
    """
    # sklearn for QC
    import sklearn.linear_model as lm
    from sklearn.model_selection import cross_validate
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lr = make_pipeline(preprocessing.StandardScaler(), lm.Ridge(alpha=10))

    res = list() #pd.DataFrame(columns= ["r2", "mae", "rmse"])
    for name, X, in sorted(data.items()):
        cv_res = cross_validate(estimator=lr, X=X, y=y, cv=cv,
                                n_jobs=5,
                                scoring=['r2', 'neg_mean_absolute_error',
                                         'neg_mean_squared_error'])
        r2 = cv_res['test_r2'].mean()
        rmse = np.sqrt(np.mean(-cv_res['test_neg_mean_squared_error']))
        mae = np.mean(-cv_res['test_neg_mean_absolute_error'])
        res.append([name, r2, mae, rmse])
        print("%s:\tCV R2:%.4f, MAE:%.4f, RMSE:%.4f" % (name, r2, mae, rmse))

    return pd.DataFrame(res, columns= ["data", "r2", "mae", "rmse"])


def ml_correlation_plot(data, y, output, study):
    # to understand why bas R2
    # sklearn for QC
    import sklearn.linear_model as lm
    from sklearn.model_selection import cross_val_predict
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    matplotlib.use( 'tkagg' )

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lr = make_pipeline(preprocessing.StandardScaler(), lm.Ridge(alpha=10))

    for name, X, in data.items():
        predicted = cross_val_predict(lr, X, y, cv=cv)
        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured (Age)')
        ax.set_ylabel('Predicted (Brain Age)')
        plt.title(name)
        plt.savefig(os.path.join(output, "{0}_{1}_corr_plot".format(study, name)))
        # plt.show()


def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df


