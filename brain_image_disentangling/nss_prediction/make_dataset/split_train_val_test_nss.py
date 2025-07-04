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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold     

# FIXME : change bins options
# FIXME : compute the difference between train/test sets distributions

# CONSTANTS

PD_FILES = "%s_t1mri_participants.csv"
STUDIES = ("ausz", )
UNIQUE_KEYS = ["participant_id", "session", "study"]
ID_TYPES = {"participant_id": str,
            "session": int,
            "acq": int,
            "run": int}

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
    processed = os.path.join(path_to_analyse, "data", "without_nss_scores_from_tableau", "processed")
    root = os.path.join(path_to_analyse, "data", "root")
    stratify = ["NSS"]
    nb_folds = 10
    
    
    # Get subject metadata for all the studies
    metadata = pd.concat([pd.read_csv(os.path.join(processed, PD_FILES % db), dtype=ID_TYPES) for db in STUDIES], ignore_index=True, sort=False)
    print(f"Nb of sbj with metadata: {len(metadata)} | {len(metadata.drop_duplicates(subset=UNIQUE_KEYS))}")
    print(metadata.head())
    scheme = metadata[UNIQUE_KEYS].copy(deep=True)

    sbj_to_strat = metadata[UNIQUE_KEYS + stratify].dropna(axis=0, how="any").copy(deep=True) # remove subjects without NSS
    if "sex" in stratify:
        sbj_to_strat["sex"] = sbj_to_strat["sex"].str.capitalize()
    print(f"Number of subject to stratify : {len(sbj_to_strat)}")
    print(sbj_to_strat.head())
    # Create arrays for splitting
    dummy_x = np.zeros((len(sbj_to_strat), 1, 128, 128, 128))
    
    # Discretize continuous labels
    y = sbj_to_strat[stratify].copy(deep=True).values
    if "age" in stratify:
        i_age = stratify.index("age")
        y[:, i_age] = discretize_continous_label(y[:, i_age].astype(float), bins="auto")
    if "NSS" in stratify:
        i_nss = stratify.index("NSS")
        y[:, i_nss] = discretize_continous_label(y[:, i_nss].astype(float), bins="auto")
    print(np.unique(y))

    # Stratification
    print("Train - test sets")
    if len(stratify) > 1:    
        splitter = MultilabelStratifiedKFold(n_splits=nb_folds, shuffle=False)
    else:
        splitter = StratifiedKFold(n_splits=nb_folds, shuffle=False)
    gen = splitter.split(dummy_x, y)
    for f in range(nb_folds):
        train_index, test_index = next(gen)
        df_train = sbj_to_strat.iloc[train_index]
        mask_train = get_mask_from_df(source_df=scheme, target_df=df_train, keys=UNIQUE_KEYS)
        df_test = sbj_to_strat.iloc[test_index]
        mask_test = get_mask_from_df(source_df=scheme, target_df=df_test, keys=UNIQUE_KEYS)
        
        scheme.loc[mask_train, f"fold{f}"] = "train"
        scheme.loc[mask_test, f"fold{f}"] = "test"
    
    print(scheme.head())
    # Sanity checks  
    for fold in [f"fold{f}" for f in range(nb_folds)]:
        for split in ("train", "test"):
            print(f"Scheme: {fold} | Split {split}")
            mask = scheme[fold] == split
            print(f"Number of subjects {mask.sum()}")
            for meta in stratify:
                if meta in ["sex", "diagnosis"]:
                    print(sbj_to_strat.loc[mask, meta].value_counts())
                else:
                    print(metadata.loc[mask, meta].mean(), metadata.loc[mask, meta].std())
        fig, ax = plt.subplots(len(stratify), 1, figsize=(6, 3*len(stratify)))
        fig.suptitle(f"{fold}")
        for i, s in enumerate(stratify):
            df = pd.concat((scheme, metadata[stratify]), axis=1)
            if len(stratify) == 1:
                sns.histplot(data=df, x=s, hue=fold, stat="percent", kde=True, common_norm=False, ax=ax)
            else:
                sns.histplot(data=df, x=s, hue=fold, stat="percent", kde=True, common_norm=False, ax=ax[i])
        fig.savefig(os.path.join(path_to_analyse, "figures", f"hist-ausz_{fold}.png"))
    

    # Saving
    path_to_scheme = os.path.join(processed,  "_".join([s.lower() for s in stratify]) + f"_stratified_10_fold_ausz.csv")
    if os.path.exists(path_to_scheme):
        answer = input(f"There is already a scheme at {path_to_scheme}. Do you want to replace it ? (y/n)")
        if answer == "y":
            scheme.to_csv(path_to_scheme, sep=",", index=False)
    else:
        scheme.to_csv(path_to_scheme, sep=",", index=False)
    
    # Make symlink to root
    if not os.path.exists(os.path.join(root, f"stratified_10_fold_ausz.csv")):
        os.symlink(path_to_scheme, os.path.join(root, "_".join([s.lower() for s in stratify]) + f"_stratified_10_fold_ausz.csv"))

if __name__ == "__main__":
    main()
