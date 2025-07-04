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
    # Parameters 
    path_to_analyse = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod"
    raw = os.path.join(path_to_analyse, "data", "raw")
    processed = os.path.join(path_to_analyse, "data", "processed")
    root = os.path.join(path_to_analyse, "data", "root")
    val_size = 0.1
    stratify = ["age", "sex", "diagnosis", "site"]
    random_state = 0
    nb_folds = 3
    
    # Get subject metadata for all the studies
    metadata = pd.concat([pd.read_csv(os.path.join(processed, PD_FILES % db), dtype=ID_TYPES) for db in STUDIES], ignore_index=True, sort=False)
    print(f"Nb of sbj with metadata: {len(metadata)} | {len(metadata.drop_duplicates(subset=UNIQUE_KEYS))}")
    print(metadata.head())
    scheme = metadata[UNIQUE_KEYS].copy(deep=True)
    
    # Load previous schemes
    mask_test_intra = False
    mask_test = False
    for target in ("asd", "bd", "scz"):
        pck = pickle.load(open(os.path.join(raw, "scheme", f"train_val_test_test-intra_{target}_stratified.pkl"), "rb"))
        df_test = pck["test"]
        df_test_intra = pck["test_intra"]
        if "run" not in df_test.columns:
            df_test["run"] = 1
        mask_test_intra |= get_mask_from_df(source_df=scheme, target_df=df_test_intra, keys=UNIQUE_KEYS)
        mask_test |= get_mask_from_df(source_df=scheme, target_df=df_test, keys=UNIQUE_KEYS)
    
    # Get subjects to stratify
    # 1. Remove subjects from external sites (keep the same as previous studies)
    mask_ext_sites = (metadata["site"] + metadata["study"]).isin([s[0] + s[1] for s in EXTERNAL_SITES])
    print(f"Nb of sbj in external sites : {mask_ext_sites.sum()}")
    sbj_to_strat = metadata.loc[~mask_ext_sites, UNIQUE_KEYS + stratify]    
    print(f"Number of subjects for stratification {len(sbj_to_strat)}")
    # 2. Remove subjects from internal test sets
    sbj_to_strat = sbj_to_strat[~mask_test_intra]
    
    # Create arrays for splitting
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    
    # Discretize continuous labels
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(np.float32))

    # Stratification
    
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
        fig.savefig(os.path.join(path_to_analyse, "figures", "hist_{fold}.png"))

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
