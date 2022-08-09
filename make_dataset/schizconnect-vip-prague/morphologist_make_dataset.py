#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pierre Auriau


Pipeline to create skeleton dataset for the SCHIZCONNECT-VIP-PRAGUE study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

"""

import os

#deep_folding
from deep_folding.brainvisa.generate_skeletons import generate_skeletons
from deep_folding.brainvisa.generate_ICBM2009c_transforms import generate_ICBM2009c_transforms
from deep_folding.brainvisa.resample_files import resample_files

######## TO DELETE ########
if os.path.exists('/home/pa267054/neurospin'):
    prefixe = '/home/pa267054'
else:
    prefixe = '/'
###########################

study = 'schizconnect-vip-prague'

# Directories
study_dir = os.path.join(prefixe, 'neurospin', 'psy_sbox', study)
morpho_dir = os.path.join(study_dir, 'derivatives', 'morphologist-2021')

output_dir = os.path.join(prefixe, "neurospin", "psy_sbox", "analyses", "202205_predict_neurodev", "data")


### Preprocessing with deep_folding ###

## Parameters

voxel_size = 1.5 #resampled voxel size

junction = "thin" # "wide" or "thin"

session = True
run = False

# For debugging, put parallel=False, number_subjects=1
parallel = True
number_subjects = -1 # all subjects

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, 'subjects')

# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, study, "skeleton", "raw")

# Relative path to graph for biobd dataset
path_to_graph = "t1mri/ses-*/default_analysis/folds/3.1"

# Output directory where to put transform files
transform_dir = os.path.join(output_dir, study, "skeleton", "transforms")

# Output directory where to pu resample skeleton files
resampled_skeleton_dir = os.path.join(output_dir, study, "skeleton", str(voxel_size)+"mm")

## Deep folding functions
for side in ["L", "R"]:
    
    #Generate skeletons from graphs
    generate_skeletons(src_dir=src_dir,
                       skeleton_dir=skeleton_dir,
                       path_to_graph=path_to_graph,
                       side=side,
                       junction=junction,
                       parallel=parallel,
                       number_subjects=number_subjects,
                       session=session,
                       run=run)
    
    #Generate transform files
    generate_ICBM2009c_transforms(src_dir=src_dir,
                                  transform_dir=transform_dir,
                                  path_to_graph=path_to_graph,
                                  side=side,
                                  parallel=parallel,
                                  number_subjects=number_subjects,
                                  session=session,
                                  run=run)
    
    #Resample skeletons in the ICBM2099c template
    resample_files(src_dir=skeleton_dir,
                   input_type="skeleton",
                   resampled_dir=resampled_skeleton_dir,
                   transform_dir=transform_dir,
                   side=side,
                   number_subjects=number_subjects,
                   out_voxel_size=voxel_size,
                   parallel=parallel)