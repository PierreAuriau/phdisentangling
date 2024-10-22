# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions to create participant dataframes.
"""
import logging
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

def merge_ni_df(ni_participants_df, participants_df, qc=None, tiv_columns=(), participant_columns=()):
    # FIXME : return roi df ?
    # FIXME : fill session and run columns ?
    """
    Select participants of ni_arr and ni_participants_df participants that are also in participants_df

    Parameters
    ----------
    ni_participants_df: DataFrame, with at least 2 columns: participant_id, "ni_path"
    participants_df: DataFrame, with 2 at least 1 columns participant_id
    qc: DataFrame, with at 2 columns participant_id and qc in [0, 1].
    participant_columns: columns that identify participant
    tiv_columns: columns that identify tiv
    Returns
    -------
     ni_participants_df (DataFrame) participants that are also in participants_df
    """
    logger = logging.getLogger("merge_ni_df")

    # 1) Extracts the session + run if available in participants_df/qc from <ni_path> in ni_participants_df
    unique_key_pheno = ["participant_id"]
    unique_key_qc = ["participant_id"]
    for key in ["session", "acq", "run"]:
        if key in participants_df:
            unique_key_pheno.append(key)
        if qc is not None and key in qc:
            unique_key_qc.append(key)
    logger.debug(f"Unique key phenotype dataframe : {unique_key_pheno}")
    logger.debug(f"Unique key qc dataframe : {unique_key_qc}")
    # 2) Keeps only the matching (participant_id, session, run) from both ni_participants_df and participants_df by
    #    preserving the order of ni_participants_df
    # !! Very import to have a clean index (to retrieve the order after the merge)
    # Create an "index" column
    for key in unique_key_pheno:
        try:
            assert ni_participants_df[key].dtype == participants_df[key].dtype, \
                logger.error(f"The {key} column does not have the same type in ni_participants_df and phenotype.")
        except KeyError as e:
            logger.error(f"The key {key} is in participants_df and not in ni_participants_df")
            raise KeyError(e)
    ni_participants_df = ni_participants_df.reset_index(drop=True).reset_index() # stores a clean index from 0..len(df)
    ni_participants_merged = pd.merge(ni_participants_df, participants_df, on=unique_key_pheno,
                                      how='inner', validate='m:1')
    logger.debug(f"ni_participants :\n{ni_participants_df.head()}"
                 f"\n\tdata types : {[(k, ni_participants_df[k].dtype) for k in unique_key_pheno]}")
    logger.debug(f"participants :\n{participants_df.head()}"
                 f"\n\tdata types : {[(k, participants_df[k].dtype) for k in unique_key_pheno]}")
    logger.debug(f"ni_participants_merged :\n {ni_participants_merged.head()}"
                 f"\n\tdata types : {[(k, ni_participants_merged[k].dtype) for k in unique_key_pheno]}")
    
    logger.info(f'--> {len(ni_participants_df)-len(ni_participants_merged)} {unique_key_pheno} have missing phenotype')

    assert len(ni_participants_merged) > 0, \
        logger.error("The merged dataframe is empty !"
                     f"\nni_participants_df :\n{ni_participants_df[unique_key_pheno].head()}"
                     f"\nparticipants_df :\n{participants_df[unique_key_pheno].head()}")
    
    # 3) If QC is available, filters out the (participant_id, session, run) who did not pass the QC
    if qc is not None:
        assert np.all(qc.qc.eq(0) | qc.qc.eq(1)), 'Unexpected value in qc.tsv'
        qc = qc.reset_index(drop=True) # removes an old index
        qc_val = qc.qc.values
        if np.all(qc_val==0):
            raise ValueError('No participant passed the QC !')
        else:
            keep = qc.loc[qc['qc'] == 1, unique_key_qc]
            init_len = len(ni_participants_merged)

            for key in unique_key_qc:
                try:
                    assert ni_participants_merged[key].dtype==keep[key].dtype, \
                    logger.error(f"The column {key} does not have the same type in qc and participants df.")
                except KeyError as e:
                    logger.warning(f"The key {key} is in qc and not in ni_participants_df")
                    raise KeyError(e)
            # Very important to have 1:1 correspondance between the QC and the ni_participant_array
            ni_participants_merged = pd.merge(ni_participants_merged, keep, on=unique_key_qc,
                                              how='inner', validate='1:1')
            logger.info(f'--> {init_len - len(ni_participants_merged)} {unique_key_qc} did not pass the QC')

            assert len(ni_participants_merged) > 0, \
                   logger.error(f"The merged dataframe is empty !"
                                f"\nni_participants_df :\n{ni_participants_merged[unique_key_qc].head()}"
                                f"\nQC :\n{keep[unique_key_qc].head()}")

    # Sanity check
    unique_key = unique_key_qc if set(unique_key_qc) >= set(unique_key_pheno) else unique_key_pheno
    assert len(ni_participants_merged.groupby(unique_key)) == len(ni_participants_merged), \
        logger.error(f"{len(ni_participants_merged)-len(ni_participants_merged.groupby(unique_key))} "
                     f"similar pairs {unique_key} found")

    # split rois and participants
    if 'session' not in ni_participants_merged:
        ni_participants_merged['session'] = 1
        logger.warning(f"Add session = 1 to participant dataframe.")
    if 'run' not in ni_participants_merged:
        ni_participants_merged['run'] = 1
        logger.warning(f"Add run = 1 to participant dataframe.")
    ni_participants = ni_participants_merged.drop(list(tiv_columns), axis=1)
    ni_participants = ni_participants.drop("index", axis=1)
    ni_rois = ni_participants_merged.drop(list(participant_columns), axis=1)
    ni_rois = ni_rois.drop("index", axis=1)    
    return ni_participants, ni_rois


def load_qc(qc_file, sep='\t'):
    """
    Functions which loads and merges qc_file depending the type of qc_file variable.

    Parameters
    ----------
    qc_file : DataFrame or str
        Quality Check Dataframe or path to Quality Check Dataframe.
        All Dataframes need a 'participant_id' and 'qc' columns
    sep : str, optional
        The separator to load the files (in case of qc_file is a path). The default is '\t'.

    Returns
    -------
    qc : DataFrame
        Quality Check Dataframe with a column participant_id and a qc column

    """
    logger = logging.getLogger("load_qc")

    if isinstance(qc_file, pd.DataFrame):
        qc = qc_file

    elif isinstance(qc_file, str):
        qc = pd.read_csv(qc_file, sep=sep)
        if len(qc.columns) < 2:
            ext = qc.split(".")[-1]
            if ext == "csv":
                sep = ","
            elif ext == "tsv":
                sep = "\t"
            else:
                raise ValueError(f"Unknown qc extension {ext}")
            qc = pd.read_csv(qc_file, sep=sep)
    else:
        raise ValueError("qc must be a Dataframe or a path towards a DataFrame")

    assert {'participant_id', 'qc'}.issubset(qc.columns), \
        logger.error("The qc dataframe misses a particpant_id or qc column")
    assert np.all(qc["qc"].eq(0) | qc["qc"].eq(1)), \
        logger.error("The qc column must contain only 0 and 1")
    return qc

def make_participants_df(ni_filenames, id_regex="/sub-([^/]+)/", session_regex='ses-([^_/]+)/',
                         acq_regex='acq-([^_/]+)/', run_regex='run-([^_/]+)\_.*nii'):
    """
      Extract participant id from paths (by default, assumes it's BIDS format: /sub-<participant_id>/)
      If id_regex is given, use it to retrieve participant id (no BIDS format assumption)
      Parameters
      ----------
      ni_filenames : [str], filenames to nifti images
      id_regex: str, regex expression used to extract <participant_id> from ni_path
      session_regex: str, regex expression used to extract <session> from ni_path
      acq_regex: str, regex expression used to extract <acquisition> from ni_path
      run_regex: str, regex expression used to extract <run> from ni_path
      Returns
      -------
          participants: Dataframe, with 2 columns "participant_id", "ni_path"
    """
    logger = logging.getLogger("make_participants_df")
    match_filename_re = re.compile(id_regex)
    pop_columns = ["participant_id", "ni_path"]
    ni_participants_df = pd.DataFrame([[match_filename_re.findall(ni_filename)[0]] + [ni_filename]
                                       for ni_filename in ni_filenames], columns=pop_columns)
    ni_participants_df['session'] = ni_participants_df.ni_path.str.extract(session_regex)[0]
    ni_participants_df['acq'] = ni_participants_df.ni_path.str.extract(acq_regex)[0]
    ni_participants_df['run'] = ni_participants_df.ni_path.str.extract(run_regex)[0]
    if ni_participants_df['session'].isna().all():
        ni_participants_df = ni_participants_df.drop('session', axis=1)
        logger.warning("Dropping session column in participant dataframe.")
        #ni_participants_df['session'] = ni_participants_df['session'].fillna(1)
        #logger.warning("Filling session column with 1 in participant dataframe.")
    if ni_participants_df['acq'].isna().all():
        ni_participants_df = ni_participants_df.drop('acq', axis=1)
        logger.warning("Dropping acquisition column in participant dataframe.")
    if ni_participants_df['run'].isna().all():
        ni_participants_df = ni_participants_df.drop('run', axis=1)
        logger.warning("Dropping run column in participant dataframe.")
        #ni_participants_df['run'] = ni_participants_df['run'].fillna(1)
        #logger.warning("Filling run column with 1 in participant dataframe.")
    return ni_participants_df

def standardize_df(df, id_types={"participant_id": str, "session": int, "acq": int, "run":int}):
    """Change data types of dataframe columns according to id_types."""
    if "TIV" in df.columns:
        df = df.rename(columns={'TIV': 'tiv'})
    if "Age" in df.columbs:
        df = df.rename(columns={"Age": "age"})
    if "Sex" in df.columns:
        df = df.rename(columns={"Sex": "sex"})
    if "session" in df.columns and is_string_dtype(df["session"]):
        df["session"] = df["session"].apply(lambda s: s[1:] if s.startswith(("V", "v")) else s)
    for col, t in id_types.items():
        if col in df.columns:
            df[col] = df[col].astype(t)
    return df
