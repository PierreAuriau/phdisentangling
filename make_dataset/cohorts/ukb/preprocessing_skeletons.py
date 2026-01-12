#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the SCHIZCONNECT-VIP-PRAGUE study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

"""

#Module import
import logging
import os
import re
import pandas as pd
import numpy as np
# deep_folding
from deep_folding.brainvisa import resample_files
from deep_folding.brainvisa import add_left_and_right_volumes
# Make dataset
from makedataset.logs import setup_logging

study = 'ukb'

setup_logging(level="info")
logger = logging.getLogger(f"make_{study}_skeleton_dataset")

# Directories
# Output directory where to put all generated files
output_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank"

# Parameters
voxel_size = "1.5"
side = "L" # "F", "L" or "R"

# Filenames
skeleton_filename = "skeleton_generated"
without_ventricle_skeleton_filename = "skeleton_generated_without_ventricle"
resampled_skeleton_filename = "resampled_skeleton_without_ventricle"

# For debugging, set parallel=False, number_subjects=1 and verbosity=True
parallel = True
number_of_subjects = "all" # all subjects
verbosity = True


# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, "skeletons", "raw")

# Output directory where to put transform files
transform_dir = os.path.join(output_dir, "transforms")

# Output directory where to put skeleton files without ventricles
without_ventricle_skeleton_dir = os.path.join(output_dir, "skeletons", "raw", "without_ventricle")

# Output directory where to put resample skeleton files
resampled_skeleton_dir = os.path.join(output_dir, "skeletons", f"{voxel_size}mm", "without_ventricle")


# DEEP FOLDING

# Resample skeletons in the ICBM2099c template
argv = ["--src_dir", without_ventricle_skeleton_dir,
        "--output_dir", resampled_skeleton_dir,
        "--side", side,
        "--input_type", "skeleton",
        "--transform_dir", transform_dir,
        "--out_voxel_size", voxel_size,
        "--src_filename", without_ventricle_skeleton_filename,
        "--output_filename", resampled_skeleton_filename,
        "--nb_subjects", number_of_subjects]
if parallel:
    argv.append("--parallel")
if verbosity:
    argv.append("--verbose")

# resample_files.main(argv)

argv[argv.index("L")] = "R"
# resample_files.main(argv)

argv = ["--src_dir", resampled_skeleton_dir,
        "--src_filename", resampled_skeleton_filename,
        "--output_filename", resampled_skeleton_filename,
        "--nb_subjects", number_of_subjects]
if parallel:
    argv.append("--parallel")
if verbosity:
    argv.append("--verbose")
add_left_and_right_volumes.main(argv)

