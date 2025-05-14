#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the BIOBD study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

"""

#Module import
import logging
import os
import json
import pandas as pd
import numpy as np
# deep_folding
from deep_folding.brainvisa import generate_skeletons
from deep_folding.brainvisa import generate_ICBM2009c_transforms
from deep_folding.brainvisa import remove_ventricle
from deep_folding.brainvisa import resample_files
from deep_folding.brainvisa import add_left_and_right_volumes
# Make dataset
from makedataset.logs import setup_logging
from makedataset.summary import make_morphologist_summary, \
                                  make_deep_folding_summary, \
                                  merge_skeleton_summaries, \
                                  ID_TYPES

study = "biobd"

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")


# Directories
neurospin = "/neurospin"
study_dir = os.path.join(neurospin, "psy_sbox", study)
morpho_dir = os.path.join(study_dir, "derivatives", "morphologist-2021")
path_to_data = os.path.join(neurospin, "dico", "data", "deep_folding", "current", "datasets")

# Output directory where to put all generated files
output_dir = os.path.join(path_to_data, study, "skeletons")

# Parameters
voxel_size = "1.5"
junction = "thin" # "wide" or "thin"
side = "F" # "F", "L" or "R"
do_skel = "True" # apply skeletonisation
bids = True

# Filenames
labelling_session = "deepcnn_session_auto"
skeleton_filename = "skeleton_generated"
without_ventricle_skeleton_filename = "skeleton_without_ventricle"
resampled_skeleton_filename = "resampled_skeleton"

# For debugging, set parallel=False, number_subjects=1 and verbosity=True
parallel = False
number_subjects = "all" # all subjects
verbosity = True

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, "subjects")

# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, "raw")

# Relative path to graph for hcp dataset
path_to_graph = "t1mri/*/default_analysis/folds/3.1"

# Output directory where to put transform files
transform_dir = os.path.join(path_to_data, study, "transforms")

# Output directory where to put skeleton files without ventricles
without_ventricle_skeleton_dir = os.path.join(output_dir, "without_ventricle", "raw")

# Output directory where to put resample skeleton files
resampled_skeleton_dir = os.path.join(output_dir, "without_ventricle", f"{voxel_size}mm")


# Morphologist summary
"""
path_to_morpho_df = os.path.join(output_dir, "metadata", f"{study}_morphologist_summary.tsv")
if os.path.exists(path_to_morpho_df):
    morphologist_df = pd.read_csv(path_to_morpho_df, sep="\t", dtype=ID_TYPES)
else:
    morphologist_df = make_morphologist_summary(morpho_dir=src_dir,
                                                dataset_name=study,
                                                path_to_save=os.path.join(path_to_data, study),
                                                labelling_session=labelling_session,
                                                id_regex="sub-([^_/\.]+)",
                                                ses_regex="ses-([^_/\.]+)",
                                                acq_regex="acq-([^_/\.]+)",
                                                run_regex="run-([^_/\.]+)")
"""

# DEEP FOLDING
for side in ("L", "R"):
    # Generate transform files
    argv = ["--src_dir", src_dir,
            "--output_dir", transform_dir,
            "--path_to_graph", path_to_graph,
            "--side", side,
            "--nb_subjects", number_subjects]
    if parallel:
        argv.append("--parallel")
    if bids:
        argv.append("--bids")
    if verbosity:
        argv.append("--verbose")

    generate_ICBM2009c_transforms.main(argv)

    # Generate skeletons from graphs
    argv = ["--src_dir", src_dir,
            "--output_dir", skeleton_dir,
            "--path_to_graph", path_to_graph,
            "--side", side,
            "--junction", junction,
            "--nb_subjects", number_subjects]
    if parallel:
        argv.append("--parallel")
    if bids:
        argv.append("--bids")
    if verbosity:
        argv.append("--verbose")

    generate_skeletons.main(argv)
    
    # Remove ventricles from the skeletons
    argv = ["--src_dir", skeleton_dir,
            "--output_dir", without_ventricle_skeleton_dir,
            "--path_to_graph", path_to_graph,
            "--morpho_dir", src_dir,
            "--side", side,
            "--labelling_session", labelling_session,
            "--src_filename", skeleton_filename,
            "--output_filename", without_ventricle_skeleton_filename,
            "--nb_subjects", number_subjects]
    if parallel:
        argv.append("--parallel")
    if bids:
        argv.append("--bids")
    if verbosity:
        argv.append("--verbose")

    remove_ventricle.main(argv)

    # Resample skeletons in the ICBM2099c template
    argv = ["--src_dir", without_ventricle_skeleton_dir,
            "--output_dir", resampled_skeleton_dir,
            "--side", side,
            "--input_type", "skeleton",
            "--transform_dir", transform_dir,
            "--out_voxel_size", voxel_size,
            "--src_filename", without_ventricle_skeleton_filename,
            "--output_filename", resampled_skeleton_filename,
            "--nb_subjects", number_subjects]
    if parallel:
        argv.append("--parallel")
    if verbosity:
        argv.append("--verbose")

    resample_files.main(argv)
    
# Add left and right hemispheres
argv = ["--src_dir", resampled_skeleton_dir,
        "--src_filename", resampled_skeleton_filename,
        "--output_filename", resampled_skeleton_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if verbosity:
    argv.append("--verbose")

add_left_and_right_volumes.main(argv)




# deep_folding summary
"""
df_directories = {"skeleton": skeleton_dir,
                  "transform": transform_dir,
                  "skeleton_without_ventricle": without_ventricle_skeleton_dir,
                  "resampled_skeleton": resampled_skeleton_dir
                  }
deep_folding_df = make_deep_folding_summary(deep_folding_directories=df_directories,
                                            side=side,
                                            dataset_name=study,
                                            path_to_save=os.path.join(path_to_data, study),
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")
"""
