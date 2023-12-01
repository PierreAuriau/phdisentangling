#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create morphologist and deep_folding summaries.
Summaries are dataframes with participant id column and a
qc column (1 if the image has an output 0 else)
"""
import logging
import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from soma import aims

from makedataset.metadata import standardize_df
from makedataset.utils import get_keys, is_it_a_subject

ID_TYPES = {
    "participant_id": str,
    "session": int,
    "acq": int,
    "run": int
}


def make_morphologist_summary(morpho_dir, dataset_name, path_to_save,
                              analysis="default_analysis", labelling_session="deepcnn_session_auto",
                              id_regex="sub-([^_/]+)", ses_regex="ses-([^_/]+)", 
                              acq_regex="acq-([^_/]+)", run_regex="run-([^_/]+)",
                              check_voxel_size=False,
                              id_types=ID_TYPES):

    logger = logging.getLogger("morphologist_summary")
    
    sbj_list = [os.path.basename(f) for f in glob.glob(f"{morpho_dir}/*")  if is_it_a_subject(f)]
    qc = OrderedDict({"participant_id":[], "session":[], "acq":[], "run":[], "ni_path": [], "qc": [], "comment": []})
    
    for sbj in tqdm(sbj_list):
        mri_dirs = [f for f in glob.glob(os.path.join(morpho_dir, sbj, "t1mri", "*")) if not re.search(".minf$", f)]
        if len(mri_dirs) == 0:
            ni_path = os.path.join(morpho_dir, sbj)
            keys = get_keys(ni_path, id_regex, ses_regex, acq_regex, run_regex)
            keys["ni_path"] = ""
            assert (keys["session"] == "") & (keys["acq"] == "") & (keys["run"] == "")
            for k,v in keys.items():
                qc[k].append(v)
            qc["qc"].append(0)
            qc["comment"].append("No t1mri directory")
            continue
        for d in mri_dirs:
            try:
                ni_path = os.path.join(d, f"{sbj}.nii.gz")
                if not os.path.exists(ni_path):
                    ni_path = os.path.dirname(ni_path)
                    raise FileNotFoundError("No t1mri file")
                if not (os.path.exists(os.path.join(d, analysis, "folds", "3.1", f"R{sbj}.arg")) and \
                        os.path.exists(os.path.join(d, analysis, "folds", "3.1", f"L{sbj}.arg"))):
                    raise FileNotFoundError("No sulcus graphs") 
                if not (os.path.exists(os.path.join(d, analysis, "folds", "3.1", labelling_session, f"R{sbj}_{labelling_session}.arg")) and \
                        os.path.exists(os.path.join(d, analysis, "folds", "3.1", labelling_session, f"L{sbj}_{labelling_session}.arg"))):
                    raise FileNotFoundError("No labelled sulcus graphs") 
            except FileNotFoundError as e:		
                qc["qc"].append(0)
                qc["comment"].append(e)
            else:
                qc["qc"].append(1)
                qc["comment"].append("")
            finally:
                keys = get_keys(ni_path, id_regex, ses_regex, acq_regex, run_regex)
                if not re.search(".nii.gz$", keys["ni_path"]):
                    keys["ni_path"] = ""
                if re.search("/home/pa267054/neurospin", keys["ni_path"]):
                    keys["ni_path"] = keys["ni_path"][14:]
                for k,v in keys.items():
                    qc[k].append(v)
    df = pd.DataFrame(qc)
    if (df["session"] == "").all():
        df = df.drop("session", axis=1)
    if (df["run"] == "").all():
        df = df.drop("run",  axis=1)
    if (df["acq"] == "").all():
        df = df.drop("acq", axis=1)
    df = standardize_df(df, id_types=id_types)
    logger.info(f"{(df['qc'] == 0).sum()}/{len(df)} subjects do not have morphologist output.")
    
    if check_voxel_size:
        df["voxel_size"] = ""
        cnt = 0
        for i, sbj in df.iterrows:
            if sbj["qc"] == 1:
                img = aims.read(sbj["ni_path"])
                voxel_size = np.asarray(img.header()["voxel_size"])
                df.iloc[i]["voxel_size"] = voxel_size
                if np.any(voxel_size < 1):
                    df.iloc[i]["comment"] += "Be careful : MRI image has a voxel size under 1mm."
                    cnt += 1
        logger.info(f"Number of MRI images under 1mm resolution : {cnt}")
    
    path_to_qc = os.path.join(path_to_save, f"{dataset_name}_morphologist_summary.tsv")
    if os.path.exists(path_to_qc):
        ans = input(f"There is already a summary at : {path_to_save}. Do you want to replace it ? (y/n)")
        if ans == "y":
            df.to_csv(path_to_qc, sep="\t", index=False)
            logger.info(f"Summary saved at {path_to_save}")
    else:
        df.to_csv(path_to_qc, sep="\t", index=False)
        logger.info(f"Summary saved at {path_to_save}")
        
    return df


def make_deep_folding_summary(deep_folding_directories, side, dataset_name, path_to_save,
                              id_regex="sub-([^_/]+)", ses_regex="ses-([^_/]+)", 
                              acq_regex="acq-([^_/]+)", run_regex="run-([^_/]+)",
                              id_types=ID_TYPES):

    logger = logging.getLogger("deep_folding_summary")
    qc_df = None
    for name, directory in deep_folding_directories.items():
        file_list = [f for f in glob.glob(os.path.join(directory, side, "*"))  if not re.search('.minf$', f)]
        logger.info(f"Number of {name} files : {len(file_list)}")
    
        qc = OrderedDict({"participant_id":[], "session":[], "acq":[], "run":[], f"ni_path_{name}": []})
    
        for filename in tqdm(file_list):
            keys = get_keys(filename, id_regex, ses_regex, acq_regex, run_regex)
            for k,v in keys.items():
                if k == "ni_path":
                    qc[f"ni_path_{name}"].append(v)
                else:
                    qc[k].append(v)
        df = pd.DataFrame(qc)
        if qc_df is None:
            qc_df = df
        else:
            qc_df = pd.merge(qc_df, df, on=["participant_id", "session", "acq", "run"],
                             how="outer", validate="1:1")
    
    qc_df["qc"] = (qc_df[[f"ni_path_{k}" for k in deep_folding_directories.keys()]].notnull()).prod(axis=1)

    if (qc_df["session"] == "").all():
        qc_df = qc_df.drop("session", axis=1)
    if (qc_df["run"] == "").all():
        qc_df = qc_df.drop("run",  axis=1)
    if (qc_df["acq"] == "").all():
        qc_df = qc_df.drop("acq", axis=1)
    qc_df = standardize_df(qc_df, id_types=id_types)
    logger.info(f"{(qc_df['qc'] == 0).sum()}/{len(qc_df)} subjects do not have deep_folding output.")

    path_to_qc = os.path.join(path_to_save, f"{dataset_name}_deep_folding_summary.tsv")
    if os.path.exists(path_to_qc):
        ans = input(f"There is already a summary at : {path_to_save}. Do you want to replace it ? (y/n)")
        if ans == "y":
            qc_df.to_csv(path_to_qc, sep="\t", index=False)
            logger.info(f"QC saved at {path_to_save}")
        else:
            logger.warning("The qc file has not be saved")
    else:
        qc_df.to_csv(path_to_qc, sep="\t", index=False)
        logger.info(f"QC saved at {path_to_save}")
    return qc_df

def merge_skeleton_summaries(morphologist_df, deep_folding_df, dataset_name, path_to_save,
                             id_types=ID_TYPES):

    logger = logging.getLogger("skeleton_summary")
    unique_keys = list({"participant_id", "session", "acq", "run"} & set(morphologist_df.columns) & set(deep_folding_df.columns))
    
    morphologist_df = standardize_df(morphologist_df, id_types=id_types)
    deep_folding_df = standardize_df(deep_folding_df, id_types=id_types)
    df_merged = pd.merge(morphologist_df, deep_folding_df, on=unique_keys, 
                         how="outer", validate="1:1", suffixes=("_morphologist","_deep_folding"))
    df_merged["qc_deep_folding"] = df_merged["qc_deep_folding"].fillna(0)
    df_merged["qc_deep_folding"] = df_merged["qc_deep_folding"].astype(int)
    mask = (df_merged["qc_morphologist"] == 1) & (df_merged["qc_deep_folding"] == 0)
    df_merged.loc[mask, "comment"] = "deep_folding error"
    logger.info(f"{mask.sum()} subject(s) encountered a deep folding error")
    
    df_merged["qc"] = df_merged[["qc_morphologist", "qc_deep_folding"]].prod(axis=1).astype(int)
    end_cols = ["qc_morphologist", "qc_deep_folding", "qc", "comment"]
    df_merged = df_merged[[c for c in df_merged.columns if c not in end_cols] + end_cols]

    df_merged = standardize_df(df_merged, id_types=id_types)
    path_to_qc = os.path.join(path_to_save, f"{dataset_name}_skeleton_summary.tsv")
    if os.path.exists(path_to_qc):
        ans = input(f"There is already a summary at : {path_to_save}. Do you want to replace it ? (y/n)")
        if ans == "y":
            df_merged.to_csv(path_to_qc, sep="\t", index=False)
            logger.info(f"QC saved at {path_to_save}")
        else:
            logger.warning("The qc file has not be saved")
    else:
        df_merged.to_csv(path_to_qc, sep="\t", index=False)
        logger.info(f"QC saved at {path_to_save}")
    
    return df_merged

