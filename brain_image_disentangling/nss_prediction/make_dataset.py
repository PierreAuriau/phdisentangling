#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Tuple

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold     

from config import Config

config = Config()

def generate_raw_data():
    """ Function to generate raw data
    /!\ Need the package makedataset --> see the repo phdisentangling/make_dataset
    """
    from makedataset.nii2npy import skeleton_nii2npy, cat12_nii2npy, freesurfer_concatnpy
    from makedataset.metadata import standardize_df
    
    # Load phenotype
    # Participant dataframe
    participant_df = pd.read_csv(os.path.join(config.study_dir, "participants_v-20240715.tsv"), sep="\t")
    assert len(participant_df) == 130, "130 subjects in the dataframe"
    participant_df = participant_df[participant_df["Study Subject ID"].notnull()]
    assert len(participant_df) == 127, "3 subjects without Study Subject ID"
    # Dataframe with NSS scores
    df_score_nss = pd.read_csv(os.path.join(config.study_dir, "sourcedata/AUSZ_2022_clinical_data_from_AntonIftimovici/DataAUSZviaSPSS_Gilles_21102021.csv"), sep=",")
    assert len(df_score_nss) == 119, "119 subjects in the dataframe"
    df_score_nss = df_score_nss[df_score_nss["StudySubjectID"].notnull()]
    assert len(df_score_nss) == 118, "1 subject without StudySubjectID"
    df_score_nss = df_score_nss[df_score_nss["NSS"].notnull()]
    assert len(df_score_nss) == 102, "16 subjects without NSS score"
    df_merged = pd.merge(participant_df, df_score_nss, left_on="Study Subject ID", right_on="StudySubjectID", validate="1:1")
    assert len(df_merged) == 99, "28 subjects without NSS scores/3 subjects without MRI"
    # Filling age, sex, TIV, diagnosis, site columns 
    df_merged["site"] = "AUSZ" # add site column
    df_merged["participant_id"] = df_merged["participant_id"].str.extract("sub-([a-zA-Z0-9]+)")[0] # remove sub-
    df_merged["tiv"] = None # add TIV
    for sbj in df_merged["participant_id"]:
        try:
            path2report = os.path.join(config.study_dir, f"derivatives/cat12-12.7_vbm/sub-{sbj}/ses-1/anat/report/cat_usub-{sbj}_ses-1_T1w.xml")
            if not os.path.exists(path2report):
                raise FileNotFoundError(f"file not found {path2report}")
            df = pd.read_xml(path2report)
            vol_tiv = df["vol_TIV"].iloc[7]
            df_merged.loc[df_merged["participant_id"] == sbj, "tiv"] = vol_tiv
        except BaseException as e:
            continue
    df_merged = df_merged[df_merged["tiv"].notnull()]
    assert len(df_merged) == 95, "4 subjects do not have the TIV"
    df_merged.loc[df_merged["age"].isnull(), "age"] = df_merged.loc[df_merged["age"].isnull(), "Age"]
    df_merged.loc[df_merged["sex"].isnull(), "sex"] = df_merged.loc[df_merged["sex"].isnull(), "Sex"]
    df_merged["sex"] = df_merged["sex"].replace({"m": "M", "f": "F"})
    df_merged.loc[df_merged["participant_id"] == "SR160602", "diagnosis"] = "control" # Group 3 == control
    assert df_merged["sex"].notnull().all(), "all subjects have sex column filled"
    assert df_merged["age"].notnull().all(), "all subjects have age column filled"
    assert df_merged["diagnosis"].notnull().all(), "all subjects have diagnosis column filled" 

    phenotype = standardize_df(df_merged, id_types=config.id_types)

    # Load QC
    qc_vbm_filename = os.path.join(config.study_dir, "derivatives", "cat12-12.7_vbm_qc", "qc.tsv")
    qc_vbm = pd.read_csv(qc_vbm_filename, sep='\t')
    qc_vbm = standardize_df(qc_vbm, id_types=config.id_types)
    qc_vbm = qc_vbm.drop("run", axis=1) # remove run
    qc_skel_filename = os.path.join(output_dir, "metadata", f"ausz_skeleton_qc.tsv")
    qc_skel = pd.read_csv(qc_skel_filename, sep='\t', dtype=config.id_types)
    qc = pd.merge(qc_vbm, qc_skel, how="outer", on=["participant_id", "session"], validate="1:1", suffixes=("_vbm", "_skeleton"))
    qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
    qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
    qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)

    skeleton_nii2npy(nii_path=nii_path, 
                    phenotype=phenotype, 
                    dataset_name="ausz", 
                    output_path=config.path2raw, 
                    qc=qc, 
                    sep=',',
                    data_type="float32",
                    id_types=config.id_types,
                    check = {"shape": (128, 152, 128), 
                             "voxel_size": (1.5, 1.5, 1.5),
                             "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
                             "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])}, 
                    side="F",
                    skeleton_size=True,
                    stored_data=True)

    cat12_nii2npy(nii_path=os.path.join(config.study_dir, "derivatives", "cat12-12.7_vbm", "sub-*/ses-*/anat/mri/mwp1*.nii"), 
                  phenotype=phenotype, 
                  dataset_name="ausz", 
                  output_path=config.path2raw, 
                  qc=qc, 
                  sep=',',
                  data_type="float32",
                  id_types=config.id_types,
                  check = {"shape": (121, 145, 121), 
                           "voxel_size": (1.5, 1.5, 1.5)})
    
    freesurfer_concatnpy(npy_path=os.path.join(config.study_dir, "derivatives", "freesurfer", "ses-1/sub-*/xhemi-textures.npy"), 
                         phenotype=phenotype, 
                         dataset_name="ausz", 
                         output_path=config.path2raw, 
                         qc=qc_vbm, 
                         sep=',',
                         dtype=np.float32,
                         id_types=config.id_types,
                         check = {"shape": (4, 327684), 
                                  "channels": ('thickness', 'curv', 'area', 'sulc')})




def combine_data_from_preprocessings():
    """ Function to get MRI data ready for models training
    * merge participant dataframes from the different preprocessings (skeleton, vbm and freesurfer)
    * re-index arrays to respect the new participant dataframe index. 
    * Save new arrays and dataframe
    """
    preprocessings = ("skeleton", "vbm", "freesurfer")
    preproc_index = "vbm" # pre-processing on which indexing is made

    folders = {"vbm": "cat12vbm", 
               "skeleton": "morphologist",
               "freesurfer": "freesurfer"}
    pd_files = {"vbm": "ausz_t1mri_mwp1_participants.csv",
                "skeleton": "ausz_t1mri_skeleton_participants.csv",
                "freesurfer": "ausz_t1mri"}
    npy_files = {"vbm": "ausz_t1mri_mwp1_gs-raw_data64.npy",
                 "skeleton": "ausz_t1mri_skeleton_data32.npy",
                 "freesurfer": "ausz..."} # FIXME

    # Loading metadata dataframes and image arrays for each pre-processing
    dico_df = {}
    dico_arr = {}
    for preproc in preprocessings:
        df = pd.read_csv(os.path.join(config.path2raw, folders[preproc], pd_files[preproc]), 
                         dtype=config.id_types)
        arr = np.load(os.path.join(config.path2raw, folders[preproc], npy_files[preproc]))
        df = df.reset_index() # add index
        dico_df[preproc] = df
        dico_arr[preproc] = arr
    # Merge dataframes
    df_merged = dico_df[preproc_index]
    for preproc, df in dico_df.items():
        if preproc != preproc_index:
            on = list(set(df_merged.columns) & set(df.columns))
            on.remove("index")
            if "ni_path" in on:
                on.remove("ni_path")
            df_merged = df_merged.merge(df, on=on, how="inner", validate="1:1", 
                                        suffixes=(f"_{preproc_index}", f"_{preproc}"))
    # Re-index dataframes and arrays
    for preproc in preprocessings:
        dico_df[preproc] = dico_df[preproc].iloc[df_merged[f"index_{preproc}"].tolist()]
        dico_arr[preproc] = dico_arr[preproc][df_merged[f"index_{preproc}"].values]
        dico_df[preproc] = dico_df[preproc].drop("index", axis=1)
        dico_df[preproc] = dico_df[preproc].reset_index(drop=True)
    # Reorder columns    
    cols = ["participant_id", "session", "sex", "age", "diagnosis", "tiv", 
            "skeleton_size", "site", "study", "ni_path_vbm", "ni_path_skeleton"]
    for c in df_merged.columns:
        if not (c in cols or c.startswith("index")): # drop indexes
            cols.append(c)
    df_merged = df_merged[cols]
    assert len(df_merged) == 95    
    # Saving new dataframes and array into interim folder
    for preproc in preprocessings:
        np.save(os.path.join(config.path2interim, npy_files[preproc]), dico_arr[preproc])
    df_merged.to_csv(os.path.join(config.path2interim, f"ausz_t1mri_participants.csv"), 
                     index=False, sep=",")
    # Transform arrays for model inputs
    for preproc, arr in dico_arr.items():
        if preproc == "skeleton":
            arr[arr > 0] = 1 # Binarize
            arr = np.pad(arr, pad_width=((0, 0), (0, 0), (0, 0), (4, 4), (0, 0)), mode='constant') # Padding
            arr = arr.astype(np.float32)
            assert np.all(arr.shape == (95, 128, 152, 128))
            assert set(np.unique(arr)).issubset({0, 1})
        elif preproc == "vbm":
            arr =  arr[..., :, 8:136, :] # Crop
            arr = np.pad(arr, pad_width=((0, 0), (0, 0), (3, 4), (0, 0), (3, 4)), mode='constant') # Padding
            arr = arr.astype(np.float32)
            assert np.all(arr.shape == (95, 128, 128, 128))
    dico_arr[preproc] = arr
    # Saving new dataframes and array into processed folder
    for preproc in preprocessings:
        np.save(os.path.join(config.path2processed, npy_files[preproc]), dico_arr[preproc])
    df_merged.to_csv(os.path.join(config.path2processed, f"ausz_t1mri_participants.csv"), index=False, sep=",")


def stratify_dataset():
    """ Function to split the dataset in train-test sets, according to metadata.
    Parameters are :
    config.stratify : labels on which is made the stratification ("NSS" for default)
    config.nb_folds : number of folds to create (10 for default)
    """
    # functions
    def discretize_continous_label(labels: str, bins: Union[str, int] = "sturges"):
        bin_edges = np.histogram_bin_edges(labels, bins=bins)
        # Discretizes the values according to these bins
        discretization = np.digitize(labels, bin_edges[1:], right=True)
        return discretization
    # Get subject metadata for all the studies
    metadata = pd.read_csv(os.path.join(config.path2processed, "ausz_t1mri_participants.csv"), dtype=config.id_types)
    scheme = metadata[config.unique_keys].copy(deep=True)
    sbj_to_strat = metadata[config.unique_keys + config.stratify]
    print(f"Number of subject to stratify : {len(sbj_to_strat)}")
    # Create arrays for splitting
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    # Discretize continuous labels
    y = sbj_to_strat[config.stratify].copy(deep=True).values
    if "age" in config.stratify:
        i_age = config.stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(float), bins="auto")
    if "NSS" in config.stratify:
        i_nss = config.stratify.index("NSS")
        y[:, i_nss] = discretize_continous_label(y[:, i_nss].astype(float), bins="auto")
    print(np.unique(y))

    # Stratification
    if len(config.stratify) > 1:    
        splitter = MultilabelStratifiedKFold(n_splits=config.nb_folds, shuffle=False)
    else:
        splitter = StratifiedKFold(n_splits=config.nb_folds, shuffle=False)
    gen = splitter.split(dummy_x, y)
    for f in range(config.nb_folds):
        train_index, test_index = next(gen)        
        scheme.loc[train_index][f"fold-{f}"] = "train"
        scheme.loc[test_index][f"fold-{f}"] = "test"    
    # Saving the schemes
    filename = os.path.join(config.path2schemes,  
                            "_".join([s.lower() for s in config.stratify]) + \
                                f"_stratified_10_fold_ausz.csv")
    scheme.to_csv(filename, sep=",", index=False)

def main():
    # generate_raw_data()
    combine_data_from_preprocessings()
    stratify_dataset()

if __name__ == "__main__":
    main()
