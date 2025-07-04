#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# FIXME : homogenize diagnosis
# FIXME : title in subplots
# FIXME : check if external set is ok

# CONSTANTS

PD_FILES = "%s_t1mri_participants.csv"
STUDIES = ("abide1", "abide2", "biobd", "bsnip1", 
           "cnp", "candi", "schizconnect-vip-prague")
UNIQUE_KEYS = ["participant_id", "session", "run", "study"]
ID_TYPES = {"participant_id": str,
            "session": int,
            "acq": int,
            "run": int}
EXTERNAL_SITES = (('UM', 'ABIDE2'), ('GU', 'ABIDE2'), 
                  ('mannheim', 'BIOBD'), ('geneve', 'BIOBD'), 
                  ('Hartford', 'BSNIP1'), ('Detroit', 'BSNIP1'))

# Functions

def discretize_continous_label(labels: str, bins: Union[str, int] = "sturges"):
    # Get an estimation of the best bin edges. 'Sturges' is conservative for pretty large datasets (N>1000).
    bin_edges = np.histogram_bin_edges(labels, bins=bins)
    # Discretizes the values according to these bins
    discretization = np.digitize(labels, bin_edges[1:], right=True)
    return discretization
    
def get_mask_from_df(source_df: pd.DataFrame, target_df: pd.DataFrame, keys: List):
    source_keys = source_df[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    assert len(set(source_keys)) == len(source_keys), f"Multiple identique identifiers found"
    target_keys = target_df[keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    mask = source_keys.isin(target_keys).values.astype(bool)
    return mask


def main():  
    path_to_analyse = "/neurospin/psy_sbox/analyses/2023_pauriau_EarlyBrainMarkersWithContrastiveAnalysis"
    raw = os.path.join(path_to_analyse, "data", "raw")
    processed = os.path.join(path_to_analyse, "data", "processed")
    root = os.path.join(path_to_analyse, "data", "root")
    test_size = 0.1
    val_size = 0.1
    stratify = ["age", "sex", "diagnosis", "site"]
    random_state = 0
    nb_folds = 3
    
    
    # Get subject metadata for all the studies
    metadata = pd.concat([pd.read_csv(os.path.join(processed, PD_FILES % db), dtype=ID_TYPES) for db in STUDIES], ignore_index=True, sort=False)
    print(f"Nb of sbj with metadata: {len(metadata)} | {len(metadata.drop_duplicates(subset=UNIQUE_KEYS))}")
    print(metadata.head())
    scheme = metadata[UNIQUE_KEYS].copy(deep=True)
    # Load previous schemes to keep same external test
    for target in ("asd", "bd", "scz"):
        pck = pickle.load(open(os.path.join(raw, "scheme", f"train_val_test_test-intra_{target}_stratified.pkl"), "rb"))
        df_test = pck["test"]
        if "run" not in df_test.columns:
            df_test["run"] = 1
        mask_test = get_mask_from_df(source_df=scheme, target_df=df_test, keys=UNIQUE_KEYS)

    """
    # merge previous schemes and save it in scheme df
    for split in ("train", "validation", "test_intra", "test"):
        for target in ("bd", "scz"):
            df_tmp = pck[target][split]
            df_tmp["run"] = 1 
            m = get_mask_from_df(source_df=scheme, target_df=df_tmp, keys=UNIQUE_KEYS)
            print(split, target, metadata.loc[m, "site"].unique())
        df_merge = pd.merge(pck["bd"][split], pck["scz"][split], on=("participant_id", "session", "study"), how="outer", validate="1:1")
        df_split = pd.concat((pck["asd"][split], df_merge), ignore_index=True, sort=False, join="outer", verify_integrity=True)
        df_split["run"] = df_split["run"].fillna(1).astype(int)

        mask = get_mask_from_df(source_df=scheme, target_df=df_split, keys=UNIQUE_KEYS)
        scheme.loc[mask, "set"] = split
        print("Split", split, "nb sbj:", mask.sum(), "/", len(df_split))
    print("Nb sbj not in splits:", scheme["set"].isnull().sum(), "/", len(scheme))

    path_to_scheme = os.path.join(processed, f"train_val_test_test-intra_stratified.csv")
    if os.path.exists(path_to_scheme):
        answer = input(f"There is already a scheme at {path_to_scheme}. Do you want to replace it ? (y/n)")
        if answer == "y":
            scheme.to_csv(path_to_scheme, sep=",", index=False)
    else:
        scheme.to_csv(path_to_scheme, sep=",", index=False)
    
    path_to_scheme = os.path.join(processed, f"train_val_test_test-intra_stratified.csv")
    scheme = pd.read_csv(path_to_scheme, sep=",", dtype=ID_TYPES)
    
    print(f"Metada and scheme have same order: {(scheme[UNIQUE_KEYS] == metadata[UNIQUE_KEYS]).all()}")
    """
    # Remove subjects from external sites
    # (keep the same as previous studies)
    mask_ext_sites = (metadata["site"] + metadata["study"]).isin([s[0] + s[1] for s in EXTERNAL_SITES])
    print(f"Nb of sbj in external sites : {mask_ext_sites.sum()}")
    sbj_to_strat = metadata.loc[~mask_ext_sites, UNIQUE_KEYS + stratify]    
    print(f"Number of subjects for stratification {len(sbj_to_strat)}")
    
    # Create arrays for splitting
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    
    # Discretize continuous labels
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))

    # Stratification
    print(f"Stratification on : {stratify} for internal test")
    print(f"Test_intra set")
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    gen = splitter.split(dummy_x, y)
    train_index, test_index = next(gen)
    df_test_intra = sbj_to_strat.iloc[test_index]
    mask_test_intra = get_mask_from_df(source_df=scheme, target_df=df_test_intra, keys=UNIQUE_KEYS)
    print(f"Nb of sbj in internal test: {mask_test_intra.sum()}")

    sbj_to_strat = sbj_to_strat.iloc[train_index]
    print(f"Nb of sbj left for train/val stratification {len(sbj_to_strat)}")
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))
    
    print("Train - validation sets")
    splitter = MultilabelStratifiedShuffleSplit(n_splits=nb_folds, test_size=val_size, random_state=random_state)
    gen = splitter.split(dummy_x, y)
    for f in range(nb_folds):
        train_index, val_index = next(gen)
        df_train = sbj_to_strat.iloc[train_index]
        mask_train = get_mask_from_df(source_df=scheme, target_df=df_train, keys=UNIQUE_KEYS)
        df_val = sbj_to_strat.iloc[val_index]
        mask_val = get_mask_from_df(source_df=scheme, target_df=df_val, keys=UNIQUE_KEYS)
        
        scheme.loc[mask_train, f"fold{f}"] = "train"
        scheme.loc[mask_val, f"fold{f}"] = "validation"
        scheme.loc[mask_test_intra, f"fold{f}"] = "test_intra"
        scheme.loc[mask_test, f"fold{f}"] = "test"
    
    print(scheme.head())
    # Sanity checks
    for fold in [f"fold{f}" for f in range(nb_folds)]:
        for split in ("train", "validation", "test", "test_intra"):
            print(f"Scheme: {fold} | Split {split}")
            print(f"Number of subjects {(scheme[fold] == split).sum()}")
            
        mask = (metadata["site"] + metadata["study"]).isin([s[0] + s[1] for s in EXTERNAL_SITES])

        print((scheme.loc[mask_ext_sites, fold].unique()))
        print(((scheme.loc[mask, fold] == "test") | (scheme.loc[mask, fold].isnull())).all())
        print("External acquisition sites are in train, validation or test_intra set !")
        
        fig, ax = plt.subplots(len(stratify), 1, figsize=(12, 6))
        fig.suptitle(f"{fold}")
        for i, s in enumerate(stratify):
            df = pd.concat((scheme, metadata[stratify]), axis=1)
            sns.histplot(data=df, x=s, hue=fold, stat="percent", kde=True, common_norm=False, ax=ax[i])
        fig.savefig(f"/neurospin/psy_sbox/analyses/2023_pauriau_EarlyBrainMarkersWithContrastiveAnalysis/figures/hist_{fold}.png")

    # Saving
    path_to_scheme = os.path.join(processed, f"train_val_test_test-intra_stratified.csv")
    if os.path.exists(path_to_scheme):
        answer = input(f"There is already a scheme at {path_to_scheme}. Do you want to replace it ? (y/n)")
        if answer == "y":
            scheme.to_csv(path_to_scheme, sep=",", index=False)
    
    # Make symlink to root
    os.symlink(path_to_scheme, os.path.join(root, f"train_val_test_test-intra_stratified.csv"))

def main2():
    path_to_analyse = "/neurospin/psy_sbox/analyses/2023_pauriau_EarlyBrainMarkersWithContrastiveAnalysis"
    raw = os.path.join(path_to_analyse, "data", "raw")
    processed = os.path.join(path_to_analyse, "data", "processed")
    root = os.path.join(path_to_analyse, "data", "root")
    val_size = 0.1
    stratify = ["age", "sex", "diagnosis", "site"]
    random_state = 0
    nb_folds = 3

    metadata = pd.concat([pd.read_csv(os.path.join(processed, PD_FILES % db), dtype=ID_TYPES) for db in STUDIES], ignore_index=True, sort=False)
    print(f"Nb of sbj with metadata: {len(metadata)} | {len(metadata.drop_duplicates(subset=UNIQUE_KEYS))}")
    print(metadata.head())
    scheme = metadata[UNIQUE_KEYS].copy(deep=True)
    # Load previous schemes
    external_sites = []
    for target in ("asd", "bd", "scz"):
        scheme[target] = None
        pck = pickle.load(open(os.path.join(raw, "scheme", f"train_val_test_test-intra_{target}_stratified.pkl"), "rb"))
        for split, df in pck.items():
            if "run" not in df.columns:
                df["run"] = 1
            mask = get_mask_from_df(source_df=scheme, target_df=df, keys=UNIQUE_KEYS)
            scheme.loc[mask, target] = split
        # get external sites
        df_test = pck["test"]
        mask_test = get_mask_from_df(source_df=metadata, target_df=df_test, keys=UNIQUE_KEYS)
        external_sites.extend([tuple(i) for i in metadata.loc[mask_test, ["site", "study"]].drop_duplicates().values])
    
    # Remove subjects from external sites
    # (keep the same as previous studies)
    mask_ext_sites = (metadata["site"] + metadata["study"]).isin([s[0] + s[1] for s in external_sites])
    print(f"Nb of sbj in external sites : {mask_ext_sites.sum()}")
    # Remove subjects from internal tests
    mask_test_intra = None
    for target in ("asd", "bd", "scz"):
        mask_test_intra |= (scheme[target] == "test_intra")
    
    sbj_to_strat = metadata.loc[~(mask_ext_sites | mask_test_intra), UNIQUE_KEYS + stratify]    
    print(f"Number of subjects for stratification {len(sbj_to_strat)}")

    # Create arrays for splitting
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    
    # Discretize continuous labels
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))

    # Stratification
    print(f"Stratification on : {stratify} for train/val")
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    gen = splitter.split(dummy_x, y)
    train_index, test_index = next(gen)
    df_test_intra = sbj_to_strat.iloc[test_index]
    mask_test_intra = get_mask_from_df(source_df=scheme, target_df=df_test_intra, keys=UNIQUE_KEYS)
    print(f"Nb of sbj in internal test: {mask_test_intra.sum()}")

    sbj_to_strat = sbj_to_strat.iloc[train_index]
    print(f"Nb of sbj left for train/val stratification {len(sbj_to_strat)}")
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))
    
    print("Train - validation sets")
    splitter = MultilabelStratifiedShuffleSplit(n_splits=nb_folds, test_size=val_size, random_state=random_state)
    gen = splitter.split(dummy_x, y)
    for f in range(nb_folds):
        train_index, val_index = next(gen)
        df_train = sbj_to_strat.iloc[train_index]
        mask_train = get_mask_from_df(source_df=scheme, target_df=df_train, keys=UNIQUE_KEYS)
        df_val = sbj_to_strat.iloc[val_index]
        mask_val = get_mask_from_df(source_df=scheme, target_df=df_val, keys=UNIQUE_KEYS)
        
        scheme.loc[mask_train, f"fold{f}"] = "train"
        scheme.loc[mask_val, f"fold{f}"] = "validation"
        scheme.loc[mask_test_intra, f"fold{f}"] = "test_intra"
        scheme.loc[mask_test, f"fold{f}"] = "test"
    
    print(scheme.head())
    # Sanity checks
    for fold in [f"fold{f}" for f in range(nb_folds)]:
        for split in ("train", "validation", "test", "test_intra"):
            print(f"Scheme: {fold} | Split {split}")
            print(f"Number of subjects {(scheme[fold] == split).sum()}")
            
        mask = (metadata["site"] + metadata["study"]).isin([s[0] + s[1] for s in EXTERNAL_SITES])

        print((scheme.loc[mask_ext_sites, fold].unique()))
        print(((scheme.loc[mask, fold] == "test") | (scheme.loc[mask, fold].isnull())).all())
        print("External acquisition sites are in train, validation or test_intra set !")
        
        fig, ax = plt.subplots(len(stratify), 1, figsize=(12, 6))
        fig.suptitle(f"{fold}")
        for i, s in enumerate(stratify):
            df = pd.concat((scheme, metadata[stratify]), axis=1)
            sns.histplot(data=df, x=s, hue=fold, stat="percent", kde=True, common_norm=False, ax=ax[i])
        fig.savefig(f"/neurospin/psy_sbox/analyses/2023_pauriau_EarlyBrainMarkersWithContrastiveAnalysis/figures/hist_{fold}.png")

    # Saving
    path_to_scheme = os.path.join(processed, f"train_val_test_test-intra_stratified.csv")
    if os.path.exists(path_to_scheme):
        answer = input(f"There is already a scheme at {path_to_scheme}. Do you want to replace it ? (y/n)")
        if answer == "y":
            scheme.to_csv(path_to_scheme, sep=",", index=False)
    
    # Make symlink to root
    os.symlink(path_to_scheme, os.path.join(root, f"train_val_test_test-intra_stratified.csv"))




if __name__ == "__main__":
    main()
