# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sting formatting for output files.

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
import os

def output_cat12(dataset, output_path, modality='cat12vbm', mri_preproc='mwp1', scaling=None, ext=None):
    """
    Example
    -------
    output_path = "/neurospin/tmp/psy_sbox/all_studies/derivatives/arrays"
    dataset = 'localizer'
    modality='cat12vbm'
    mri_preproc='mwp1'
    scaling='gs'
    output_cat12(dataset, output_path, mri_preproc='mwp1', scaling="gs", ext='npy')
    output_cat12(dataset, output_path, mri_preproc='rois', scaling="gs", ext='tsv')
    output_cat12(dataset, output_path, mri_preproc='participants', ext='tsv')
    """
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "-" + scaling) + "." + ext)



def output_quasi_raw(dataset, output_path, modality='cat12vbm', mri_preproc='quasi_raw', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)

def output_dti(dataset, output_path, modality='dwi', mri_preproc='tbss', type=None, ext=None):
    # type data64, or data32
    return os.path.join(output_path, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if type is None else "_" + type) + "." + ext)

def output_skeleton(dataset, output_path, modality='morphologist', mri_preproc='skeleton', dtype=None, ext=None, side=None):
    # type data64, or data32
    return os.path.join(output_path, '_'.join(filter(None, (dataset, modality, side, mri_preproc, dtype))) + f".{ext}")

def output_freesurfer(dataset, output_path, modality="freesurfer", mri_preproc="textures", ext=None):
    if isinstance(mri_preproc, str):
        mri_preproc = [mri_preproc]
    return os.path.join(output_path, '_'.join(filter(None, (dataset, modality, *mri_preproc))) + f".{ext}")
